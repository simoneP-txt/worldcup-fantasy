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

from .. import feed_status, tunables
from ..data_sources import economics, intl_results, odds
from ..tunables import ODDS_NUDGE_GAIN, ODDS_NUDGE_MAX  # noqa: F401 - re-exported
from . import elo

log = logging.getLogger(__name__)

GROUP_ROUNDS = (1, 2, 3)
KO_ROUNDS = (4, 5, 6, 7, 8)  # R32, R16, QF, SF, F
_KO_STAGE_NAMES = {4: "R32", 5: "R16", 6: "QF", 7: "SF", 8: "F"}
_STAGE_ROUND = {stage: rnd for rnd, stage in _KO_STAGE_NAMES.items()}
_STAGE_DISPLAY = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
    "SF": "Semi-finals", "F": "Final",
}

HOST_NAMES = {"USA", "Mexico", "Canada"}

_BRACKET_FILE = Path(__file__).resolve().parents[1] / "data" / "bracket_2026.json"

N_SIMS = 2000

# ODDS_NUDGE_GAIN / ODDS_NUDGE_MAX (the outright-odds Elo nudge) are tuning
# judgements, imported from tunables.py above.

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


def _sample_scoreline(outcomes, p_win: float, p_draw: float, rng) -> tuple:
    """Goals (home, away) consistent with each already-sampled outcome.

    Pure-ish (rng-driven), vectorised. Two independent Poissons whose means
    split GROUP_GOALS_PER_MATCH by the expected score E = p_win + p_draw/2,
    then the SIGN is pinned to the sampled outcome: the calibrated win/draw/loss
    stays authoritative and only the MARGIN is textured. A decisive game is
    forced to a >=1 margin, a draw to an equal score. Used solely for the group
    GD / goals-for tiebreaks, never to decide who won.
    """
    n = len(outcomes)
    expected = p_win + 0.5 * p_draw
    total = tunables.tuned("group_goals_per_match")
    hg = rng.poisson(total * expected, n)
    ag = rng.poisson(total * (1.0 - expected), n)
    hi = np.maximum(hg, ag)
    lo = np.minimum(hg, ag)
    decided = np.where(hi == lo, hi + 1, hi)  # a win/loss needs a real margin
    gf_home = np.select([outcomes == 1, outcomes == -1], [decided, lo], default=lo)
    gf_away = np.select([outcomes == 1, outcomes == -1], [lo, decided], default=lo)
    return gf_home.astype(np.int16), gf_away.astype(np.int16)


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
    """Survival curve from the market-calibrated Monte Carlo (P(reach r))."""
    curve, _champion, _ctx = _run_market_sim(squads, rounds, n_sims, seed)
    return curve


def _run_market_sim(squads, rounds, n_sims, seed):
    """Single market-calibrated Monte Carlo run.

    Returns (curve, champion, ctx) where `ctx` carries the match-probability
    machinery (final ratings, host map, outcome fn, bookmaker h2h) so callers
    that need per-fixture probabilities - the predicted bracket - can reuse the
    SAME run instead of simulating twice. `_simulate_with_market` is the thin
    curve-only wrapper kept for the existing callers (and the tests that patch
    it).
    """
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
        rating = nudged
    # P(wins the tournament) per squadId, for the `odds` command.
    last_run_info["champion"] = champion
    ctx = {
        "rating": rating,
        "fifa_names": {s["id"]: s["name"] for s in squads},
        "probs_fn": probs_fn,
        "h2h": h2h,
    }
    return curve, champion, ctx


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


def _match_probs(h, a, rating, fifa_names, probs_fn, h2h):
    """(p_home, p_draw, p_away): bookmaker h2h if quoted, else Elo + host edge.

    Shared by the simulation loop and the predicted-bracket display so the two
    never disagree on a fixture's probabilities. Pure and unit-tested.
    """
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


def _advance_probs(h, a, rating, fifa_names, probs_fn, h2h):
    """(P(home advances), P(away advances)) for one KO tie, summing to 1.

    The 90' draw is collapsed into extra time exactly as `ko_winner` does: the
    home team takes it with its Elo expected score on the RAW rating diff (no
    host edge in the shootout coin). Pure and unit-tested.
    """
    pw, pd_, _pl = _match_probs(h, a, rating, fifa_names, probs_fn, h2h)
    et = elo.expected_score(rating[h] - rating[a])
    p_home = pw + pd_ * et
    return p_home, 1.0 - p_home


def _modal(counter: dict):
    """Most-frequent key in an occupancy counter (argmax), or None if empty."""
    return max(counter, key=counter.get) if counter else None


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
        return _match_probs(h, a, rating, fifa_names, probs_fn, h2h)

    # Pre-sample group fixture outcomes for all sims at once.
    # outcome per sim: +1 home win, 0 draw, -1 away win.
    fixtures = _group_fixtures(rounds)
    n_fx = len(fixtures)
    fixture_outcomes = np.empty((n_fx, n_sims), dtype=np.int8)
    # Scoreline texture per fixture, for the GD / goals-for group tiebreaks (the
    # OUTCOME stays the calibrated one; completed matches carry their real goals).
    fixture_gf_home = np.zeros((n_fx, n_sims), dtype=np.int16)
    fixture_gf_away = np.zeros((n_fx, n_sims), dtype=np.int16)
    for f_idx, m in enumerate(fixtures):
        h, a = m["homeSquadId"], m["awaySquadId"]
        if m.get("homeScore") is not None and feed_status.result_usable(m.get("status")):
            hs, as_ = int(m["homeScore"]), int(m["awayScore"])
            fixture_outcomes[f_idx, :] = 1 if hs > as_ else -1 if hs < as_ else 0
            fixture_gf_home[f_idx, :] = hs
            fixture_gf_away[f_idx, :] = as_
        else:
            pw, pd_, pl = match_probs(h, a)
            outs = rng.choice([1, 0, -1], size=n_sims, p=[pw, pd_, pl])
            fixture_outcomes[f_idx] = outs
            fixture_gf_home[f_idx], fixture_gf_away[f_idx] = _sample_scoreline(
                outs, pw, pd_, rng
            )

    # Real KO pairings/results from the feed override simulated ones.
    actual_ko: dict[int, list[dict]] = {
        rnd["id"]: [m for m in rnd.get("tournaments", []) if m.get("homeSquadId")]
        for rnd in rounds
        if rnd["id"] in KO_ROUNDS
    }

    reach_counts = {sid: dict.fromkeys(KO_ROUNDS, 0) for sid in squad_ids}
    champion_counts = dict.fromkeys(squad_ids, 0)
    # Who fills each bracket slot's home/away side, per stage, across all sims:
    # occupancy[stage][idx] = [home_counter, away_counter]. The modal occupants
    # become the displayed pairing in `predicted_bracket`.
    occupancy = {
        stage: [[{}, {}] for _ in template[stage]]
        for stage in _KO_STAGE_NAMES.values()
    }

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
        goal_diff = dict.fromkeys(squad_ids, 0)
        goals_for = dict.fromkeys(squad_ids, 0)
        for f_idx, m in enumerate(fixtures):
            out = fixture_outcomes[f_idx, sim]
            h, a = m["homeSquadId"], m["awaySquadId"]
            hg = int(fixture_gf_home[f_idx, sim])
            ag = int(fixture_gf_away[f_idx, sim])
            if out == 1:
                points[h] += 3
            elif out == -1:
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1
            goal_diff[h] += hg - ag
            goal_diff[a] += ag - hg
            goals_for[h] += hg
            goals_for[a] += ag

        # FIFA group ranking: points, then goal difference, then goals scored.
        # The Elo + tiny noise final term stands in for head-to-head and the
        # drawing of lots, used only when those three are still level.
        def rank_key(sid):
            return (
                points[sid], goal_diff[sid], goals_for[sid],
                rating[sid] + rng.gumbel(0, 30) / 1e3,
            )

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
            stage_occ = occupancy[stage]
            winners = []
            for idx, (h, a) in enumerate(pairs):
                reach_counts[h][round_no] += 1
                reach_counts[a][round_no] += 1
                home_occ, away_occ = stage_occ[idx]
                home_occ[h] = home_occ.get(h, 0) + 1
                away_occ[a] = away_occ.get(a, 0) + 1
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
    # Stash the per-slot occupancy so `predicted_bracket` can read the modal
    # pairing without a second Monte Carlo (overwritten by the calibrated pass).
    last_run_info["bracket"] = {"occupancy": occupancy, "n_sims": n_sims}
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


# ---------------------------------------------------------- predicted bracket

def _ko_winner_from_result(m: dict) -> int:
    """Winning squadId of a COMPLETED KO feed match (penalties break a draw)."""
    mh, ma = m["homeSquadId"], m["awaySquadId"]
    hs, as_ = int(m["homeScore"]), int(m["awayScore"])
    if hs != as_:
        return mh if hs > as_ else ma
    hp = m.get("homePenaltyScore") or 0
    ap = m.get("awayPenaltyScore") or 0
    return mh if hp >= ap else ma


def _assemble_bracket(
    template, rounds, squads, occupancy, curve, champion, advance_fn, model_label
) -> dict:
    """Build the structured bracket schema from one finished simulation.

    Pure (no network, no sim): given the per-slot `occupancy` counters, the
    survival `curve`, the `champion` map and an `advance_fn(h, a) -> (p_home,
    p_away)`, it renders each tie. A slot whose modal pairing matches a feed KO
    match shows the ACTUAL teams (completed -> 100/0 + the real result;
    scheduled -> model head-to-head); otherwise the modal occupants with the
    model pass-through. Unit-tested with synthetic counters.
    """
    names = {s["id"]: s["name"] for s in squads}

    def p_reach(sid, round_no):
        if sid is None or sid not in curve.index or round_no not in curve.columns:
            return None
        return round(float(curve.loc[sid, round_no]), 4)

    def side(sid, round_no, p_advance, locked):
        return {
            "team": names.get(sid),
            "squadId": int(sid) if sid is not None else None,
            "p_reach": p_reach(sid, round_no),
            "p_advance": round(float(p_advance), 4),
            "locked": bool(locked),
        }

    rounds_out = []
    for stage in ("R32", "R16", "QF", "SF", "F"):
        round_no = _STAGE_ROUND[stage]
        actual = [
            m
            for r in rounds if r["id"] == round_no
            for m in r.get("tournaments", []) if m.get("homeSquadId")
        ]
        actual_by_pair: dict = {}
        for m in actual:
            actual_by_pair[(m["homeSquadId"], m["awaySquadId"])] = m
            actual_by_pair[(m["awaySquadId"], m["homeSquadId"])] = m
        stage_occ = occupancy.get(stage, [])

        ties = []
        for idx in range(len(template[stage])):
            home_occ, away_occ = stage_occ[idx] if idx < len(stage_occ) else ({}, {})
            mh, ma = _modal(home_occ), _modal(away_occ)
            feed_match = actual_by_pair.get((mh, ma)) if (mh and ma) else None

            result = None
            if feed_match is not None:
                home_id, away_id = feed_match["homeSquadId"], feed_match["awaySquadId"]
                completed = (
                    feed_match.get("homeScore") is not None
                    and feed_status.result_usable(feed_match.get("status"))
                )
                if completed:
                    winner = _ko_winner_from_result(feed_match)
                    ph = 1.0 if winner == home_id else 0.0
                    home = side(home_id, round_no, ph, True)
                    away = side(away_id, round_no, 1.0 - ph, True)
                    result = {
                        "homeScore": int(feed_match["homeScore"]),
                        "awayScore": int(feed_match["awayScore"]),
                    }
                    status = "completed"
                else:
                    ph, pa = advance_fn(home_id, away_id)
                    home = side(home_id, round_no, ph, False)
                    away = side(away_id, round_no, pa, False)
                    status = "scheduled"
            elif mh is not None and ma is not None:
                ph, pa = advance_fn(mh, ma)
                home = side(mh, round_no, ph, False)
                away = side(ma, round_no, pa, False)
                status = "predicted"
            else:  # degenerate: a slot no simulation ever filled
                home = side(mh, round_no, 0.0, False)
                away = side(ma, round_no, 0.0, False)
                status = "predicted"

            ties.append({
                "slot": idx + 1,
                "status": status,
                "home": home,
                "away": away,
                "result": result,
            })
        rounds_out.append({
            "stage": stage,
            "name": _STAGE_DISPLAY[stage],
            "round": round_no,
            "ties": ties,
        })

    champ_rows = sorted(
        (
            {"team": names.get(sid, str(sid)), "squadId": int(sid), "p": round(float(p), 5)}
            for sid, p in champion.items() if p > 0
        ),
        key=lambda row: row["p"],
        reverse=True,
    )
    winner = {"team": champ_rows[0]["team"], "p": champ_rows[0]["p"]} if champ_rows else None
    return {
        "available": True,
        "model": model_label,
        "rounds": rounds_out,
        "champion": champ_rows,
        "winner": winner,
    }


def predicted_bracket(
    squads: list[dict],
    rounds: list[dict],
    n_sims: int = N_SIMS,
    seed: int | None = 42,
) -> dict:
    """Full predicted knockout bracket (R32 -> Final) from one Monte Carlo.

    Reuses the SAME market-calibrated run as `advise`/`odds` (no second sim):
    each bracket slot shows its modal occupant, completed feed ties pin the real
    teams + result, and the under-tie pass-through % is the Elo/market
    head-to-head with the 90' draw collapsed into extra time. Returns {} if the
    simulation cannot run (e.g. offline), so the caller degrades cleanly like
    `championship_probabilities`.
    """
    try:
        curve, champion, ctx = _run_market_sim(squads, rounds, n_sims, seed)
    except Exception:
        if os.environ.get("WCFANTASY_STRICT"):
            raise
        log.exception("predicted bracket simulation failed")
        return {}

    occupancy = last_run_info.get("bracket", {}).get("occupancy", {})
    template = json.loads(_BRACKET_FILE.read_text(encoding="utf-8"))["template"]

    def advance_fn(h, a):
        return _advance_probs(
            h, a, ctx["rating"], ctx["fifa_names"], ctx["probs_fn"], ctx["h2h"]
        )

    return _assemble_bracket(
        template, rounds, squads, occupancy, curve, champion, advance_fn,
        last_run_info.get("model", "?"),
    )
