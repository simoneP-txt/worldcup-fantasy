"""Tournament progression: P(team is still playing at round r).

Monte Carlo simulation of the whole World Cup:

1. Elo ratings replayed over ~45k historical internationals, updated
   with every completed 2026 match from the live feed, plus a small
   Klement-style economic adjustment (World Bank data).
2. A multinomial logistic regression (fit on the same history) turns an
   Elo gap into win/draw/loss probabilities per fixture.
3. Bookmaker odds, refreshed every run, override the model where the
   market knows more: match odds (h2h) pin the probabilities of the
   fixtures they cover; outright winner odds calibrate ratings via a
   second simulation pass (log-ratio nudge between simulated and
   market champion probabilities).
4. Each simulation plays out the remaining group fixtures (completed
   ones enter with their real result), ranks groups, allocates the 8
   best third-placed teams with FIFA's official slot table, and runs
   the official R32..Final bracket. KO draws go to whoever wins the
   coin weighted by Elo expected score (extra time/penalties).

The bracket template and third-place allocation were extracted from the
game's own bracket-predictor JS (src/wcfantasy/data/bracket_2026.json).

If external data is unavailable, falls back to the original price-based
heuristic so `advise` keeps working offline.
"""

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import feed_status
from ..data_sources import economics, intl_results, odds
from . import elo

log = logging.getLogger(__name__)

GROUP_ROUNDS = (1, 2, 3)
KO_ROUNDS = (4, 5, 6, 7, 8)  # R32, R16, QF, SF, F
_KO_STAGE_NAMES = {4: "R32", 5: "R16", 6: "QF", 7: "SF", 8: "F"}

HOST_NAMES = {"USA", "Mexico", "Canada"}

_BRACKET_FILE = Path(__file__).resolve().parents[1] / "data" / "bracket_2026.json"

N_SIMS = 2000

# Outright-odds calibration: Elo nudge = GAIN * ln(p_market / p_simulated),
# clamped. Kept moderate so the market corrects, not replaces, the model.
ODDS_NUDGE_GAIN = 60.0
ODDS_NUDGE_MAX = 80.0

# Filled by survival_curve on each run; advise reports it to the user.
last_run_info: dict = {}


# ---------------------------------------------------------------- fallback

def team_strength(players: pd.DataFrame) -> pd.Series:
    """Price-based strength, kept as offline fallback."""
    return (
        players.sort_values("price", ascending=False)
        .groupby("squadId")["price"]
        .apply(lambda s: s.head(11).mean())
    )


def _fallback_survival_curve(players: pd.DataFrame, squads: list[dict]) -> pd.DataFrame:
    strength = team_strength(players)
    pct = strength.rank(pct=True)
    per_round = 0.25 + 0.60 * pct
    eliminated = {s["id"] for s in squads if s.get("isEliminated")}
    rows = {}
    for squad_id in strength.index:
        alive = 0.0 if squad_id in eliminated else 1.0
        p = alive * min(1.0, (32 / 48) + 0.5 * (pct[squad_id] - 0.5))
        probs = {r: alive for r in GROUP_ROUNDS}
        for r in KO_ROUNDS:
            probs[r] = p
            p *= per_round[squad_id]
        rows[squad_id] = probs
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("squadId")


# ---------------------------------------------------------------- simulation

def _outcome_prob_fn(model: elo.EloModel):
    """Vectorizable (p_win, p_draw, p_loss) from a raw Elo diff.

    Manual softmax over the fitted coefficients: predict_proba is far
    too slow to call ~100k times inside the simulation loop.
    """
    coef = model.outcome_model.coef_.ravel()  # per class
    intercept = model.outcome_model.intercept_
    classes = list(model.outcome_model.classes_)  # sorted: [-1, 0, 1]
    i_loss, i_draw, i_win = classes.index(-1), classes.index(0), classes.index(1)

    def probs(diff: float) -> tuple[float, float, float]:
        z = coef * diff + intercept
        e = np.exp(z - z.max())
        p = e / e.sum()
        return p[i_win], p[i_draw], p[i_loss]

    return probs


def _build_elo(squads: list[dict], rounds: list[dict]) -> tuple[elo.EloModel, dict[int, str]]:
    history = intl_results.load()
    model = elo.fit(history)
    id_to_name = {s["id"]: intl_results.dataset_name(s["name"]) for s in squads}
    fifa_names = {s["id"]: s["name"] for s in squads}

    for rnd in rounds:
        for m in rnd.get("tournaments", []):
            if m.get("homeScore") is None or not feed_status.result_usable(m.get("status")):
                continue
            home_id, away_id = m["homeSquadId"], m["awaySquadId"]
            if home_id not in id_to_name or away_id not in id_to_name:
                continue
            model.update(
                id_to_name[home_id], id_to_name[away_id],
                int(m["homeScore"]), int(m["awayScore"]),
                neutral=fifa_names[home_id] not in HOST_NAMES,
            )
    return model, id_to_name


def _group_fixtures(rounds: list[dict]) -> list[dict]:
    return [
        m
        for rnd in rounds
        if rnd["id"] in GROUP_ROUNDS
        for m in rnd.get("tournaments", [])
    ]


def _resolve_ko_pair(h, a, actual_by_pair: dict, ko_winner) -> int:
    """Winner of a KO pairing (h, a): the real result if the feed has this
    exact matchup completed, else `ko_winner(h, a)`.

    Pure and unit-tested. The match is found by team pair in either
    orientation, and the winner is read from the MATCH's own home/away (mh,
    ma) - never from the lookup-key order - so a feed fixture stored as
    (a, h) still resolves correctly. This is the fix for the old positional
    zip, which could pin a real score onto the wrong simulated pairing.
    """
    m = actual_by_pair.get((h, a)) or actual_by_pair.get((a, h))
    if m and m.get("homeScore") is not None and feed_status.result_usable(m.get("status")):
        mh, ma = m["homeSquadId"], m["awaySquadId"]
        hs, as_ = int(m["homeScore"]), int(m["awayScore"])
        if hs != as_:
            return mh if hs > as_ else ma
        # decided in extra time / penalties
        hp = m.get("homePenaltyScore") or 0
        ap = m.get("awayPenaltyScore") or 0
        return mh if hp >= ap else ma
    return ko_winner(h, a)


def survival_curve(
    players: pd.DataFrame,
    squads: list[dict],
    rounds: list[dict],
    n_sims: int = N_SIMS,
    seed: int | None = 42,
) -> pd.DataFrame:
    """P(team still in the tournament at each round 1..8), per squadId."""
    try:
        return _simulate_with_market(squads, rounds, n_sims, seed)
    except Exception:
        # The broad catch keeps the tool working when a data source breaks,
        # but it can also hide a real bug behind a much weaker heuristic.
        # WCFANTASY_STRICT re-raises so failures are visible while debugging.
        if os.environ.get("WCFANTASY_STRICT"):
            raise
        log.exception("Monte Carlo simulation failed, using price heuristic")
        last_run_info.update(model="price heuristic (offline fallback)")
        last_run_info.pop("ratings", None)  # no stale opponent factors
        return _fallback_survival_curve(players, squads)


def _simulate_with_market(squads, rounds, n_sims, seed):
    model, id_to_name = _build_elo(squads, rounds)
    probs_fn = _outcome_prob_fn(model)

    try:
        econ = economics.elo_adjustments(squads)
    except Exception:
        log.exception("economic adjustment unavailable, using pure Elo")
        econ = {}
    rating = {
        s["id"]: model.rating(id_to_name[s["id"]]) + econ.get(s["id"], 0.0)
        for s in squads
    }

    h2h = outright = None
    try:
        h2h = odds.h2h_probabilities(squads)
        outright = odds.outright_probabilities(squads)
    except Exception:
        log.exception("bookmaker odds unavailable")
    last_run_info.update(
        model="Elo + economics",
        h2h_fixtures=len(h2h) if h2h else 0,
        outright_teams=len(outright) if outright else 0,
    )

    curve, champion = _simulate(squads, rounds, rating, probs_fn, h2h, n_sims, seed)
    # Exposed for the player-points opponent factor (fixture difficulty).
    last_run_info["ratings"] = rating

    if outright:
        nudged = nudge_ratings(rating, outright, champion)
        curve, champion = _simulate(squads, rounds, nudged, probs_fn, h2h, n_sims, seed)
        last_run_info["model"] = "Elo + economics + market calibration"
        last_run_info["ratings"] = nudged
    # P(wins the tournament) per squadId, for the `odds` command.
    last_run_info["champion"] = champion
    return curve


def nudge_ratings(
    rating: dict[int, float],
    outright: dict[int, float],
    champion: dict[int, float],
) -> dict[int, float]:
    """Elo ratings nudged toward the bookmaker outright probabilities.

    For each quoted team, shift its rating by GAIN * ln(p_market / p_sim),
    clamped to +-MAX, so the market corrects the model rather than replacing
    it. A team with no usable market or simulated probability, or one not in
    the rating table, is left unchanged. Pure and unit-tested.
    """
    nudged = dict(rating)
    for sid, p_market in outright.items():
        p_sim = champion.get(sid, 0.0)
        if p_market > 1e-4 and p_sim > 1e-4 and sid in nudged:
            delta = ODDS_NUDGE_GAIN * math.log(p_market / p_sim)
            nudged[sid] += max(-ODDS_NUDGE_MAX, min(ODDS_NUDGE_MAX, delta))
    return nudged


def _simulate(squads, rounds, rating, probs_fn, h2h, n_sims, seed):
    bracket = json.loads(_BRACKET_FILE.read_text(encoding="utf-8"))
    template = bracket["template"]
    third_alloc = bracket["third_place_allocation"]
    rng = np.random.default_rng(seed)

    squad_ids = [s["id"] for s in squads]
    groups: dict[str, list[int]] = {}
    for s in squads:
        groups.setdefault(s["group"], []).append(s["id"])
    fifa_names = {s["id"]: s["name"] for s in squads}

    def match_probs(h, a):
        """(p_home, p_draw, p_away): bookmaker if quoted, else Elo."""
        if h2h:
            if (h, a) in h2h:
                return h2h[(h, a)]
            if (a, h) in h2h:
                pl, pd_, pw = h2h[(a, h)]
                return pw, pd_, pl
        diff = rating[h] - rating[a]
        if fifa_names[h] in HOST_NAMES:
            diff += elo.HOME_ADVANTAGE
        elif fifa_names[a] in HOST_NAMES:
            diff -= elo.HOME_ADVANTAGE
        return probs_fn(diff)

    # Pre-sample group fixture outcomes for all sims at once.
    # outcome per sim: +1 home win, 0 draw, -1 away win.
    fixtures = _group_fixtures(rounds)
    fixture_outcomes = np.empty((len(fixtures), n_sims), dtype=np.int8)
    for f_idx, m in enumerate(fixtures):
        h, a = m["homeSquadId"], m["awaySquadId"]
        if m.get("homeScore") is not None and feed_status.result_usable(m.get("status")):
            hs, as_ = int(m["homeScore"]), int(m["awayScore"])
            out = 1 if hs > as_ else -1 if hs < as_ else 0
            fixture_outcomes[f_idx, :] = out
        else:
            pw, pd_, pl = match_probs(h, a)
            fixture_outcomes[f_idx] = rng.choice(
                [1, 0, -1], size=n_sims, p=[pw, pd_, pl]
            )

    # Real KO pairings/results from the feed override simulated ones.
    actual_ko: dict[int, list[dict]] = {
        rnd["id"]: [m for m in rnd.get("tournaments", []) if m.get("homeSquadId")]
        for rnd in rounds
        if rnd["id"] in KO_ROUNDS
    }

    reach_counts = {sid: dict.fromkeys(KO_ROUNDS, 0) for sid in squad_ids}
    champion_counts = dict.fromkeys(squad_ids, 0)

    def ko_winner(h, a):
        pw, pd_, pl = match_probs(h, a)
        u = rng.random()
        if u < pw:
            return h
        if u < pw + pl:
            return a
        # drawn after 90': extra time / penalties as Elo-weighted coin
        diff = rating[h] - rating[a]
        return h if rng.random() < elo.expected_score(diff) else a

    def resolve_pair(h, a, actual_by_pair):
        """Winner of one KO pairing: the real result when the feed has this
        exact matchup completed (either orientation), else a simulated one.
        Looking up by team pair - not by position - is what keeps a
        partially-played KO round from applying a real score to the wrong
        simulated pairing."""
        return _resolve_ko_pair(h, a, actual_by_pair, ko_winner)

    for sim in range(n_sims):
        # group standings
        points = dict.fromkeys(squad_ids, 0)
        for f_idx, m in enumerate(fixtures):
            out = fixture_outcomes[f_idx, sim]
            h, a = m["homeSquadId"], m["awaySquadId"]
            if out == 1:
                points[h] += 3
            elif out == -1:
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1

        # rank within groups; goal-difference tiebreaks approximated by
        # Elo plus noise (the feed has no simulated goal counts)
        def rank_key(sid):
            return points[sid] + (rating[sid] + rng.gumbel(0, 30)) / 1e5

        placed: dict[str, list[int]] = {
            g: sorted(members, key=rank_key, reverse=True)
            for g, members in groups.items()
        }
        thirds = {g: placed[g][2] for g in placed}
        best_third_groups = sorted(
            thirds, key=lambda g: rank_key(thirds[g]), reverse=True
        )[:8]
        alloc = third_alloc[",".join(sorted(g.upper() for g in best_third_groups))]

        def resolve(op):
            g = op["stageName"]
            if op["rank"] == 1:
                return placed[g][0]
            if op["rank"] == 2:
                return placed[g][1]
            return thirds[alloc[g.upper()].lower()]  # allocated third

        # knockout: R32 from template, later stages from previous winners
        prev_winners: list[int] = []
        for round_no in KO_ROUNDS:
            stage = _KO_STAGE_NAMES[round_no]
            actual = actual_ko.get(round_no, [])
            if actual and len(actual) == len(template[stage]):
                pairs = [(m["homeSquadId"], m["awaySquadId"]) for m in actual]
            elif stage == "R32":
                pairs = [
                    (resolve(t["opponent1"]), resolve(t["opponent2"]))
                    for t in template[stage]
                ]
            else:
                pairs = [
                    (prev_winners[t["opponent1"]["rank"] - 1],
                     prev_winners[t["opponent2"]["rank"] - 1])
                    for t in template[stage]
                ]

            # Real results are looked up by team pairing, not by position:
            # the simulated `pairs` need not line up with the order of the
            # completed matches in the feed (e.g. a half-played KO round).
            actual_by_pair = {
                (m["homeSquadId"], m["awaySquadId"]): m for m in actual
            }
            winners = []
            for h, a in pairs:
                reach_counts[h][round_no] += 1
                reach_counts[a][round_no] += 1
                winners.append(resolve_pair(h, a, actual_by_pair))
            prev_winners = winners
        champion_counts[prev_winners[0]] += 1

    eliminated = {s["id"] for s in squads if s.get("isEliminated")}
    rows = {}
    for sid in squad_ids:
        alive_group = 0.0 if sid in eliminated else 1.0
        probs = {r: alive_group for r in GROUP_ROUNDS}
        for r in KO_ROUNDS:
            probs[r] = reach_counts[sid][r] / n_sims
        rows[sid] = probs
    curve = pd.DataFrame.from_dict(rows, orient="index").rename_axis("squadId")
    champion = {sid: c / n_sims for sid, c in champion_counts.items()}
    return curve, champion


def future_rounds_from_curve(curve: pd.DataFrame, next_round: int) -> pd.Series:
    """Sum of P(still playing) over rounds >= next_round, per squadId."""
    future = [r for r in curve.columns if r >= next_round]
    return curve[future].sum(axis=1)


def advance_from_curve(curve: pd.DataFrame, next_round: int) -> pd.Series:
    """P(team reaches the round AFTER next_round), per squadId.

    This is exactly what progressing one knockout round means (winning the
    next match), so it is the right signal for the Qualification booster -
    not the multi-round 'expected to go deep' that future_rounds measures.
    Zero past the final (no round to advance into).
    """
    target = next_round + 1
    if target in curve.columns:
        return curve[target]
    return pd.Series(0.0, index=curve.index)


def expected_future_rounds(
    players: pd.DataFrame,
    squads: list[dict],
    rounds: list[dict],
    next_round: int,
) -> pd.Series:
    """Sum of P(still playing) over rounds >= next_round, per squadId."""
    return future_rounds_from_curve(survival_curve(players, squads, rounds), next_round)


def championship_probabilities(
    squads: list[dict],
    rounds: list[dict],
    n_sims: int = N_SIMS,
    seed: int | None = 42,
) -> dict[int, float]:
    """{squadId -> P(wins the tournament)} from the Monte Carlo.

    Market-calibrated when bookmaker outright odds are available (the same
    model `advise` uses). Returns {} if the simulation cannot run (e.g. the
    historical results are unavailable offline), so the caller degrades
    cleanly instead of crashing. Read `last_run_info['model']` for which
    model produced the numbers.
    """
    try:
        _simulate_with_market(squads, rounds, n_sims, seed)
    except Exception:
        log.exception("championship simulation failed")
        return {}
    return dict(last_run_info.get("champion", {}))
