"""Captain, vice-captain, bench order and booster suggestions.

The ILP picks the captain (highest expected starter). Everything else
here is derived from the solved squad: vice-captain is the next-best
starter, bench order follows expected points. Booster advice applies
the official effects (rules.BOOSTERS) plus the user's strategy:
- nothing in round 1 (no data yet, too unpredictable);
- Wildcard late in the knockouts, when several squad players are out
  because their team was eliminated (the R32 window is unlimited and
  free anyway, so the Wildcard is only worth R16 or later);
- Qualification held for the semi-finals: by then the per-country cap
  (6 at SF) lets you stack starters from a likely finalist, and each
  one that reaches the final is +2.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from .. import fixtures, rules, tunables
from ..state import TeamState
from ..tunables import (
    CAPTAIN_CONTENDER_FLOOR,
    CAPTAIN_MIN_PREMIUM,
    CAPTAIN_SCORE_SD_FLOOR,
    CAPTAIN_SWITCH_COST,
    LINEUP_OPTION_WEIGHT,
)
from .squad import BENCH_WEIGHT, SquadSolution


def vice_captain(df: pd.DataFrame, solution: SquadSolution) -> int:
    starters = df[df["id"].isin(solution.starter_ids)]
    ranked = starters.sort_values("ep_match", ascending=False)
    return int(ranked[ranked["id"] != solution.captain_id].iloc[0]["id"])


def ordered_bench(df: pd.DataFrame, solution: SquadSolution) -> tuple[list[int], int]:
    """(outfield bench ids ordered 1st/2nd/3rd sub, bench GK id).

    Order is by EP: the bench order drives AUTO-subs (a DNP starter is replaced
    at round end by the highest-order bench player who played), and all bench
    players' minutes are known by then, so timing does not change it. The
    timing/option value lives in retime_lineup, which decides who benches."""
    bench = df[
        df["id"].isin(set(solution.squad_ids) - set(solution.starter_ids))
    ].sort_values("ep_match", ascending=False)
    outfield = [int(i) for i in bench[bench["position"] != "GK"]["id"]]
    gk = int(bench[bench["position"] == "GK"]["id"].iloc[0])
    return outfield, gk


# ----------------------------------------- temporal option-value of the XI
# A bench player can be brought on LIVE only by removing a starter whose match
# is already COMPLETE while the bench player has NOT yet kicked off. So a bench
# player who plays LATE keeps that reactive-substitution option; one who plays
# EARLY is auto-sub-only, and a strong EARLY player is better STARTED - his
# points bank and you could not react to him on the bench anyway. This mirrors
# the option-value captaincy. EP stays primary: the timing term only flips
# genuinely close, same-position calls (it never benches a clearly better
# player). Calibratable.
#
# The XI split is scored on THIS ROUND's ep_match, NOT the horizon-weighted
# squad score the ILP uses: who you OWN is a horizon bet (handled in squad
# selection), but who you START tonight only earns this round's points -
# benching a player does not forfeit his future rounds (he starts them when
# his fixture comes). Using the horizon here wrongly benched in-form players
# from weaker nations (a shorter expected run dragged their score down).
# LINEUP_OPTION_WEIGHT: see tunables.py (imported at top).


def _reactivity(squad: pd.DataFrame, kickoff: dict[int, object]) -> dict[int, float]:
    """Per squad player, the fraction of his squad-mates who kick off BEFORE
    him: how usable he is as a LIVE substitute (a late player can replace many
    already-completed starters, an early one almost none). Unknown kickoff -> 0
    so the timing term never moves a player we cannot place in time."""
    times = {int(r["id"]): kickoff.get(int(r["squadId"])) for _, r in squad.iterrows()}
    ids = list(times)
    react: dict[int, float] = {}
    for pid in ids:
        t = times[pid]
        if t is None or len(ids) < 2:
            react[pid] = 0.0
            continue
        earlier = sum(1 for q in ids if q != pid and times[q] is not None and times[q] < t)
        react[pid] = earlier / (len(ids) - 1)
    return react


def _start_premium(row, react: dict[int, float], option_weight: float) -> float:
    """Start value of a player for THIS round: ep_match net of the bench credit,
    minus the live-sub option a LATE player forfeits by starting, so close calls
    favour the earlier one. ep = ep_match * (1 - BENCH_WEIGHT - option_weight *
    reactivity). With option_weight 0 it ranks by ep_match (this round's points,
    deliberately NOT the horizon-weighted squad score)."""
    return float(row["ep_match"]) * (1.0 - BENCH_WEIGHT - option_weight * react.get(int(row["id"]), 0.0))


def retime_lineup(
    df: pd.DataFrame,
    solution: SquadSolution,
    rounds: list[dict],
    next_round: int,
    *,
    option_weight: float = LINEUP_OPTION_WEIGHT,
) -> list[tuple[int, int]]:
    """Re-pick the starting XI of the solved squad for the live-sub option value
    of kickoff order, MUTATING solution.starter_ids in place.

    Keeps the ILP's formation (same starter count per position) and swaps only
    WITHIN a position, so the lineup stays legal. Per position the highest
    `_start_premium` players start; with option_weight 0 it picks the
    this-round-EP-optimal XI for that formation (the horizon stays in squad
    selection, not the start/bench split). The GK split is left untouched (a
    keeper is auto-subbed only by the other keeper, so timing cannot help it).
    Returns the (in_id, out_id) swaps.
    """
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    kickoff = fixtures.kickoffs_by_squad(rnd) if rnd else {}
    squad = df[df["id"].isin(solution.squad_ids)]
    react = _reactivity(squad, kickoff)

    before = {int(i) for i in solution.starter_ids}
    new_starters: list[int] = []
    swaps: list[tuple[int, int]] = []
    for pos, group in squad.groupby("position"):
        rows = {int(r["id"]): r for _, r in group.iterrows()}
        started_here = [i for i in rows if i in before]
        k = len(started_here)
        if pos == "GK" or k == 0 or k == len(rows):
            new_starters.extend(started_here)
            continue
        ranked = sorted(rows, key=lambda i: _start_premium(rows[i], react, option_weight), reverse=True)
        keep = ranked[:k]
        new_starters.extend(keep)
        moved_in = [i for i in keep if i not in before]
        moved_out = [i for i in started_here if i not in keep]
        swaps.extend(zip(moved_in, moved_out))

    after = set(new_starters)
    solution.starter_ids = [int(i) for i in solution.squad_ids if int(i) in after]
    return swaps


def lineup_timing_note(df: pd.DataFrame, swaps: list[tuple[int, int]]) -> str | None:
    """Explain a timing-driven start/bench swap (the lineup analogue of
    captain_timing_note). None when the EP-optimal XI already respects timing."""
    if not swaps:
        return None
    names = df.set_index("id")["name"]
    pairs = ", ".join(
        f"{names.get(i, i)} (plays earlier) over {names.get(o, o)} (plays later)"
        for i, o in swaps
    )
    return (
        f"Timing/option value: start {pairs}. The EP is close, but an early "
        "starter's points bank and you can't react to him; keeping the later "
        "player on the bench preserves the live substitution (swap a completed "
        "starter for him)."
    )


# --------------------------------------------------- option-value captaincy
# A single-match captain score is volatile (a haul vs a quiet 2-pointer) and
# the armband is switchable live, unlimited times, from a COMPLETED player to
# one not yet played. So captaining the EARLIEST strong starter first and
# rolling the armband to a later, higher-EP one if he disappoints banks the
# MAX of the early realisation and the later expectation - worth more than
# captaining the single highest-EP starter outright, even when the early one
# rates a fair bit lower (the EP estimate is itself noisy). The constants that
# drive that estimate (CAPTAIN_SCORE_CV etc.) live in tunables.py, imported
# above, with their provenance and sensitivity notes.


@dataclass
class CaptainPlan:
    """Recommended captaincy for a round, accounting for the live option."""

    initial: int                # who to captain FIRST (earliest strong option)
    anchor: int                 # highest-EP starter (captain-outright baseline)
    sequence: list[int] = field(default_factory=list)  # strong options, kickoff order
    premium: float = 0.0        # doubled option gain over the anchor, net of cost


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _effective_cv(mean: float, cv: float, sd_floor: float) -> float:
    """Coefficient of variation, floored so a low-EP player still carries
    spread: the absolute SD floor expressed as a CV (sd_floor / mean)."""
    if mean <= 0:
        return cv
    return max(cv, sd_floor / mean)


def _expected_max(mean: float, floor: float, *, cv: float, sd_floor: float) -> float:
    """E[max(X, floor)] for X LOGNORMAL with this mean and effective CV.

    Lognormal, not Normal: a captain haul is a right tail, and the whole point
    of the live armband roll is to bank that upside - a symmetric Normal
    understates it. Closed form is the Black-Scholes call written on the mean
    (forward) F = mean and strike K = floor:
        E[max(X, K)] = F*Phi(d1) + K*Phi(-d2),
        d1 = (ln(F/K) + sigma**2/2)/sigma,  d2 = d1 - sigma.
    Degrades to max(mean, floor) when either is non-positive or sigma -> 0.
    """
    if mean <= 0 or floor <= 0:
        return max(mean, floor)
    sigma = math.sqrt(math.log(1.0 + _effective_cv(mean, cv, sd_floor) ** 2))
    if sigma <= 0:
        return max(mean, floor)
    d1 = (math.log(mean / floor) + 0.5 * sigma * sigma) / sigma
    d2 = d1 - sigma
    return mean * _norm_cdf(d1) + floor * _norm_cdf(-d2)


def _prob_below(mean: float, k: float, *, cv: float, sd_floor: float) -> float:
    """P(X < k) for X LOGNORMAL(mean, effective CV): the chance the earliest
    captain underperforms the continuation value k, i.e. you roll the armband."""
    if mean <= 0 or k <= 0:
        return float(mean < k)
    sigma = math.sqrt(math.log(1.0 + _effective_cv(mean, cv, sd_floor) ** 2))
    if sigma <= 0:
        return float(mean < k)
    d2 = (math.log(mean / k) - 0.5 * sigma * sigma) / sigma
    return _norm_cdf(-d2)


def sequential_captain_value(
    mus: list[float],
    *,
    cv: float | None = None,
    sd_floor: float = CAPTAIN_SCORE_SD_FLOOR,
) -> float:
    """EV (un-doubled) of optimally captaining players given in KICKOFF order.

    Policy: hold the armband on the first, then after each match keep it or
    roll to the next not-yet-played player. Backward recursion
    V_i = E[max(X_i, V_{i+1})], V_last = mu_last, X_i lognormal. Monotone
    (E[max(X, K)] >= max(mean, K)): starting earliest is >= captaining any
    single player outright. `cv` defaults to the calibratable captain-score CV.
    """
    if not mus:
        return 0.0
    if cv is None:
        cv = tunables.tuned("captain_score_cv")
    v = mus[-1]
    for mu in reversed(mus[:-1]):
        v = _expected_max(mu, v, cv=cv, sd_floor=sd_floor)
    return v


def plan_captain(
    candidates: list[tuple],
    *,
    cv: float | None = None,
    sd_floor: float = CAPTAIN_SCORE_SD_FLOOR,
    switch_cost: float = CAPTAIN_SWITCH_COST,
    contender_floor: float = CAPTAIN_CONTENDER_FLOOR,
    min_premium: float = CAPTAIN_MIN_PREMIUM,
) -> CaptainPlan | None:
    """Decide the initial captain from (id, ep, kickoff) tuples.

    Returns the highest-EP starter (the anchor) unless captaining an EARLIER
    strong starter first and rolling the armband to the anchor is worth more:
    the doubled option premium must beat the one-off live-switch cost. A
    `kickoff` of None means the time is unknown - such players cannot anchor a
    timed roll and only count as the fallback anchor. `cv` defaults to the
    calibratable captain-score CV.
    """
    cand = [(int(pid), float(ep), ko) for pid, ep, ko in candidates]
    if not cand:
        return None
    if cv is None:
        cv = tunables.tuned("captain_score_cv")
    anchor_id, top_ep, _ = max(cand, key=lambda c: c[1])
    if top_ep <= 0:
        return CaptainPlan(initial=anchor_id, anchor=anchor_id, sequence=[anchor_id])

    strong = sorted(
        (c for c in cand if c[2] is not None and c[1] >= contender_floor * top_ep),
        key=lambda c: (c[2], -c[1]),  # earliest kickoff; same slot -> higher EP first
    )
    if len(strong) < 2 or strong[0][0] == anchor_id:
        # only one real option, or the best already plays first: no roll to gain
        return CaptainPlan(
            initial=anchor_id, anchor=anchor_id,
            sequence=[c[0] for c in strong] or [anchor_id],
        )

    mus = [c[1] for c in strong]
    v_seq = sequential_captain_value(mus, cv=cv, sd_floor=sd_floor)
    v_rest = sequential_captain_value(mus[1:], cv=cv, sd_floor=sd_floor)
    p_roll = _prob_below(mus[0], v_rest, cv=cv, sd_floor=sd_floor)
    premium = 2.0 * (v_seq - top_ep) - p_roll * switch_cost
    if premium > min_premium:
        return CaptainPlan(
            initial=strong[0][0], anchor=anchor_id,
            sequence=[c[0] for c in strong], premium=premium,
        )
    return CaptainPlan(
        initial=anchor_id, anchor=anchor_id, sequence=[c[0] for c in strong],
    )


def recommend_captain(
    df: pd.DataFrame,
    solution: SquadSolution,
    rounds: list[dict],
    next_round: int,
    **kwargs,
) -> CaptainPlan:
    """CaptainPlan for the solved squad: option-value pick over the starters,
    falling back to the ILP captain when there is nothing to act on."""
    starters = df[df["id"].isin(solution.starter_ids)]
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    kickoff = fixtures.kickoffs_by_squad(rnd) if rnd else {}
    candidates = list(zip(
        starters["id"].astype(int),
        starters["ep_match"].astype(float),
        [kickoff.get(int(s)) for s in starters["squadId"]],
    ))
    plan = plan_captain(candidates, **kwargs)
    if plan is None:
        return CaptainPlan(
            initial=solution.captain_id, anchor=solution.captain_id,
            sequence=[solution.captain_id],
        )
    return plan


def captain_timing_note(
    df: pd.DataFrame,
    solution: SquadSolution,
    rounds: list[dict],
    next_round: int,
) -> str | None:
    """Explain an option-value captaincy: when the recommended initial captain
    is NOT the highest-EP starter, spell out the play and its deadline. None
    when the best starter is also the right one to captain first."""
    plan = recommend_captain(df, solution, rounds, next_round)
    if plan.initial == plan.anchor:
        return None
    names = df.set_index("id")["name"]
    early, anchor = names.get(plan.initial), names.get(plan.anchor)
    return (
        f"Timing/option value: captain {early} FIRST (the earliest of your "
        f"strong options); if he underdelivers, roll the armband live to "
        f"{anchor} before his kickoff. Banks the better of the two "
        f"(~+{plan.premium:.1f} captain pts vs captaining {anchor} outright) - "
        f"but only if you re-run `live` to move it before {anchor}'s match."
    )


def twelfth_man_pick(df: pd.DataFrame, solution: SquadSolution) -> pd.Series:
    """Best 12th Man: highest-EP player OUTSIDE the squad.

    The booster ignores budget and per-country caps, so the answer is
    simply the top expected scorer you do not already own.
    """
    outside = df[~df["id"].isin(solution.squad_ids)]
    return outside.sort_values("ep_match", ascending=False).iloc[0]


def _progressing_starters(starters: pd.DataFrame, next_round: int) -> int:
    """Starters favoured to WIN their next match (advance one knockout round).

    The Qualification +2 lands for advancing a SINGLE round, so the right
    signal is P(team reaches the following round) > 0.5 - the `p_advance`
    column. The older `future_rounds > 8 - next_round + 0.5` heuristic
    measured 'favoured to go all the way' and only coincided at the SF; it
    stays as a fallback for callers that do not carry p_advance.
    """
    if "p_advance" in starters:
        return int((starters["p_advance"] > 0.5).sum())
    threshold = 8 - next_round + 0.5
    return int((starters["future_rounds"] > threshold).sum())


def _dead_wood(squad: pd.DataFrame) -> pd.DataFrame:
    """Squad players whose team is eliminated or all but out (Wildcard fuel)."""
    return squad[
        squad["isEliminated"].fillna(False) | (squad["future_rounds"] < 0.5)
    ]


# Display names and the natural window of each booster, for the single-line
# "in hand" note when nothing is recommended this round.
_BOOSTER_LABELS = {
    "qualification": "Qualification Booster",
    "wildcard": "Wildcard",
    "clean_sheet_shield": "Clean Sheet Shield",
    "maximum_captain": "Maximum Captain",
    "twelfth_man": "12th Man",
}
_BOOSTER_HOLD_HINT = {
    "qualification": "for the semi-final",
    "wildcard": "for a late-knockout rebuild",
    "clean_sheet_shield": "for your strongest knockout defensive block",
    "maximum_captain": "for a high-ceiling round",
    "twelfth_man": "for a one-off extra scorer",
}


def _clean_sheet_shield_ev(starters: pd.DataFrame) -> dict | None:
    """EV breakdown of the Clean Sheet Shield for a starting XI, or None when
    there is no GK/DEF/MID clean-sheet block. Shared by the activation decision
    and the advice line so the two never disagree. Uses the per-team, Elo-derived
    P(concede exactly one) when expected_value supplied it, else the flat
    fallback. {ev, at_stake, n_block, top_team, top_n}."""
    block = starters[starters["position"].isin(["GK", "DEF", "MID"])].copy()
    if block.empty:
        return None
    cs = block["position"].map(rules.CLEAN_SHEET_POINTS).astype(float)
    p_play = block["p_play"].fillna(1.0) if "p_play" in block.columns else 1.0
    at_stake = float((cs * p_play).sum())
    if "p_concede_one" in block.columns:
        poc = block["p_concede_one"].fillna(rules.CLEAN_SHEET_SHIELD_CONCEDE_ONE_PROB)
        ev = float((cs * p_play * poc).sum())
    else:
        ev = at_stake * rules.CLEAN_SHEET_SHIELD_CONCEDE_ONE_PROB
    by_team = block.groupby("squadName")["id"].count().sort_values(ascending=False)
    return {
        "ev": ev,
        "at_stake": at_stake,
        "n_block": int(len(block)),
        "top_team": by_team.index[0],
        "top_n": int(by_team.iloc[0]),
    }


def planned_activation(
    unused: list[str],
    next_round: int,
    cap: pd.Series,
    starters: pd.DataFrame,
    squad: pd.DataFrame,
) -> str | None:
    """The SINGLE booster to ACTIVATE this round, by the user's strategy
    priority, or None.

    Decided up front so a lower-priority booster (Maximum Captain) can never
    preempt a higher-priority planned one (Qualification at the semi-final):
    that ordering bug is exactly what this guards against. Priority:
    Qualification at the SF -> a late-KO Wildcard rebuild -> the Clean Sheet
    Shield on a strong knockout block -> Maximum Captain.
    """
    if (
        "qualification" in unused
        and next_round == 7
        and _progressing_starters(starters, next_round) >= 6
    ):
        return "qualification"
    if (
        "wildcard" in unused
        and next_round >= 5
        and next_round not in rules.WILDCARD_EXCLUDED_ROUNDS
        and len(_dead_wood(squad)) >= 4
    ):
        return "wildcard"
    if "clean_sheet_shield" in unused and next_round in rules.CLEAN_SHEET_SHIELD_ROUNDS:
        info = _clean_sheet_shield_ev(starters)
        if info and info["ev"] >= rules.CLEAN_SHEET_SHIELD_MIN_EV:
            return "clean_sheet_shield"
    if "maximum_captain" in unused and cap["ep_match"] >= 5.5:
        return "maximum_captain"
    return None


def booster_advice(
    df: pd.DataFrame, state: TeamState, solution: SquadSolution, next_round: int
) -> list[str]:
    """A SINGLE recommendation for the round: the one booster to use (chosen by
    strategy priority among those still unused), or that none is worth it. No
    per-booster wall of HOLD/candidate lines - just the decision, with the key
    numbers, and one compact 'in hand' line when nothing is recommended.
    Everything except the Wildcard is reversible before lockout, so a USE is
    risk-free until the round locks."""
    unused = [b for b in rules.BOOSTERS if b not in state.boosters_used]
    if not unused:
        return ["All boosters already used — none left to play."]
    if next_round == 1:
        return [
            "This round: no booster (your strategy — round 1 is the most "
            "unpredictable of the tournament)."
        ]

    starters = df[df["id"].isin(solution.starter_ids)]
    squad = df[df["id"].isin(solution.squad_ids)]
    cap = df.loc[df["id"] == solution.captain_id].iloc[0]
    planned = planned_activation(unused, next_round, cap, starters, squad)

    if planned == "qualification":
        n = _progressing_starters(starters, next_round)
        return [
            f"This round: USE the Qualification Booster — your semi-final window: "
            f"{n}/11 starters favoured to progress, +2 each (1+ minute; the "
            "captain's +2 is not doubled)."
        ]
    if planned == "wildcard":
        dead = len(_dead_wood(squad))
        return [
            f"This round: USE the Wildcard — {dead} squad players are out; rebuild "
            "around the teams still alive. Irreversible once confirmed."
        ]
    if planned == "clean_sheet_shield":
        info = _clean_sheet_shield_ev(starters)
        return [
            f"This round: USE the Clean Sheet Shield — {info['n_block']} clean-sheet "
            f"starters (~{info['at_stake']:.0f} pts at stake, ~+{info['ev']:.1f} EV "
            f"if a team concedes exactly one), led by {info['top_team']} "
            f"({info['top_n']}). One-shot knockout booster."
        ]
    if planned == "maximum_captain":
        return [
            f"This round: USE Maximum Captain — high-ceiling round ({cap['name']} "
            f"~{cap['ep_match']:.1f} pts); it captains your best XI scorer in "
            "hindsight, never worse than picking one yourself."
        ]

    # Nothing worth activating: one concise line naming what is still in hand
    # and its window, instead of a HOLD line per booster. The Clean Sheet Shield
    # carries its current EV and the bar, so you can see how close it came and
    # tune CLEAN_SHEET_SHIELD_MIN_EV from real rounds.
    held = ", ".join(
        _booster_hold_note(b, next_round, starters)
        for b in unused
        if b in _BOOSTER_LABELS
    )
    return [
        f"This round: no booster worth it — keep what you have. In hand: {held}."
        if held
        else "This round: no booster worth it."
    ]


def _booster_hold_note(booster: str, next_round: int, starters: pd.DataFrame) -> str:
    """'<Name> (<window>)' for the in-hand line, with the Clean Sheet Shield's
    live EV and the activation bar appended at a KO round, so the number you
    would calibrate against is visible even when the shield is held."""
    label, hint = _BOOSTER_LABELS[booster], _BOOSTER_HOLD_HINT[booster]
    if booster == "clean_sheet_shield" and next_round in rules.CLEAN_SHEET_SHIELD_ROUNDS:
        info = _clean_sheet_shield_ev(starters)
        if info:
            return (
                f"{label} (~+{info['ev']:.1f} EV this round, bar "
                f"{rules.CLEAN_SHEET_SHIELD_MIN_EV:.0f} — {hint})"
            )
    return f"{label} ({hint})"


def clean_sheet_shield_advice(df: pd.DataFrame, solution: SquadSolution) -> str:
    """Detailed Clean Sheet Shield assessment for one round (a single string).

    The shield keeps the clean-sheet bonus of your GK/DEF/MID until a SECOND
    goal, so its value is the points those starters would otherwise lose to a
    single goal, weighted by P(the team concedes exactly one). Recommend it for
    the KO round with the biggest, most concentrated defensive block (correlated
    clean sheets pay together). booster_advice gives the one-line verdict; this
    is the breakdown behind it."""
    starters = df[df["id"].isin(solution.starter_ids)]
    info = _clean_sheet_shield_ev(starters)
    if info is None:
        return "HOLD Clean Sheet Shield: no clean-sheet starters this round."
    if info["ev"] >= rules.CLEAN_SHEET_SHIELD_MIN_EV:
        return (
            f"Clean Sheet Shield candidate: {info['n_block']} clean-sheet starters "
            f"(~{info['at_stake']:.0f} pts at stake, ~+{info['ev']:.1f} EV if a team "
            f"concedes exactly one), led by {info['top_team']} ({info['top_n']}). Use "
            "it in the KO round with your strongest, most favoured defensive block."
        )
    return (
        f"HOLD Clean Sheet Shield: only ~+{info['ev']:.1f} EV this round; save it for "
        "a KO round where you start more defenders from a favoured side."
    )
