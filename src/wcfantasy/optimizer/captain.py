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

from .. import fixtures, rules
from ..state import TeamState
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
LINEUP_OPTION_WEIGHT = 0.10


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
# rates a fair bit lower (the EP estimate is itself noisy). These constants
# drive that estimate; they are deliberate approximations, calibratable once
# real WC captain scores accumulate.
CAPTAIN_SCORE_CV = 0.65         # SD of a captain score as a fraction of its EP
CAPTAIN_SCORE_SD_FLOOR = 2.0    # min SD so low-EP players still carry spread
CAPTAIN_SWITCH_COST = 0.7       # pts forfeited by the FIRST live armband move
                                # (auto-sub + vice insurance), incurred once
CAPTAIN_CONTENDER_FLOOR = 0.70  # ignore starters below this fraction of the
                                # top EP as captain candidates (replaces the old
                                # hard 0.92 gate that rejected the option play)
CAPTAIN_MIN_PREMIUM = 0.1       # don't churn the armband for a negligible edge


@dataclass
class CaptainPlan:
    """Recommended captaincy for a round, accounting for the live option."""

    initial: int                # who to captain FIRST (earliest strong option)
    anchor: int                 # highest-EP starter (captain-outright baseline)
    sequence: list[int] = field(default_factory=list)  # strong options, kickoff order
    premium: float = 0.0        # doubled option gain over the anchor, net of cost


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _expected_max(mu: float, sd: float, floor: float) -> float:
    """E[max(X, floor)] for X ~ N(mu, sd**2) - the normal call-option value."""
    if sd <= 0:
        return max(mu, floor)
    d = (mu - floor) / sd
    return mu * _norm_cdf(d) + floor * _norm_cdf(-d) + sd * _norm_pdf(d)


def sequential_captain_value(
    mus: list[float],
    *,
    cv: float = CAPTAIN_SCORE_CV,
    sd_floor: float = CAPTAIN_SCORE_SD_FLOOR,
) -> float:
    """EV (un-doubled) of optimally captaining players given in KICKOFF order.

    Policy: hold the armband on the first, then after each match keep it or
    roll to the next not-yet-played player. Backward recursion
    V_i = E[max(X_i, V_{i+1})], V_last = mu_last. Monotone: the value of
    starting earliest is >= captaining any single player outright.
    """
    if not mus:
        return 0.0
    v = mus[-1]
    for mu in reversed(mus[:-1]):
        v = _expected_max(mu, max(cv * mu, sd_floor), v)
    return v


def plan_captain(
    candidates: list[tuple],
    *,
    cv: float = CAPTAIN_SCORE_CV,
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
    timed roll and only count as the fallback anchor.
    """
    cand = [(int(pid), float(ep), ko) for pid, ep, ko in candidates]
    if not cand:
        return None
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
    sd0 = max(cv * mus[0], sd_floor)
    p_roll = _norm_cdf((v_rest - mus[0]) / sd0) if sd0 > 0 else float(mus[0] < v_rest)
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
    Qualification at the SF -> a late-KO Wildcard rebuild -> Maximum Captain.
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
    if "maximum_captain" in unused and cap["ep_match"] >= 5.5:
        return "maximum_captain"
    return None


def booster_advice(
    df: pd.DataFrame, state: TeamState, solution: SquadSolution, next_round: int
) -> list[str]:
    unused = [b for b in rules.BOOSTERS if b not in state.boosters_used]
    if not unused:
        return []
    if next_round == 1:
        return [
            "No booster in round 1 (your strategy): no data yet, the "
            "opening round is the most unpredictable of the tournament."
        ]

    advice = []
    starters = df[df["id"].isin(solution.starter_ids)]
    squad = df[df["id"].isin(solution.squad_ids)]
    cap = df.loc[df["id"] == solution.captain_id].iloc[0]

    # Never two boosters in the same round: pick the one to ACTIVATE up
    # front by strategy priority; everyone else gets HOLD/info messaging.
    # Everything except the Wildcard can be deactivated before lockout, so
    # an ACTIVATE here is risk-free until the round locks.
    planned = planned_activation(unused, next_round, cap, starters, squad)

    if "maximum_captain" in unused:
        if planned == "maximum_captain":
            advice.append(
                f"ACTIVATE Maximum Captain: high-ceiling round ({cap['name']} "
                f"expected {cap['ep_match']:.1f} pts) and the booster captains "
                "your best XI scorer in hindsight, never worse than picking one."
            )
        elif cap["ep_match"] >= 5.5:
            advice.append(
                f"Maximum Captain candidate ({cap['name']} expected "
                f"{cap['ep_match']:.1f} pts), but a higher-priority booster is "
                "planned this round."
            )

    if "qualification" in unused and next_round in rules.QUALIFICATION_BOOSTER_ROUNDS:
        # +2 per starter (>=1 minute played) whose team progresses; the
        # captain's +2 is not doubled.
        progressing = _progressing_starters(starters, next_round)
        msg = (
            f"Qualification Booster: {progressing}/11 starters favoured to "
            "progress (+2 each, needs 1+ minute; captain's +2 not doubled)."
        )
        if planned == "qualification":
            # User strategy: fire it at the SF, when the 6-per-country cap
            # lets the squad stack a likely finalist.
            advice.append(
                "ACTIVATE " + msg + " Semi-final is your planned window: "
                "starters from a finalist bank the bonus twice (SF + final win)."
            )
        elif next_round < 7:
            advice.append(
                "HOLD Qualification Booster (your plan: semi-finals). " + msg
            )
        else:
            advice.append("Consider " + msg)

    if "wildcard" in unused:
        if next_round in rules.WILDCARD_EXCLUDED_ROUNDS:
            advice.append(
                "Wildcard not allowed this round (banned for matchday 1 "
                "and the R32; the R32 window is unlimited anyway)."
            )
        elif next_round >= 5:
            dead = _dead_wood(squad)
            if planned == "wildcard":
                advice.append(
                    f"ACTIVATE Wildcard (your plan): {len(dead)} squad players "
                    "are eliminated or done; rebuild around the teams still "
                    "alive. Careful: irreversible once confirmed."
                )
            elif len(dead) >= 4:
                advice.append(
                    f"Wildcard candidate round: {len(dead)} squad players "
                    "eliminated, but another booster is already advised."
                )
            else:
                advice.append(
                    f"HOLD Wildcard: only {len(dead)} squad players eliminated; "
                    "wait until the dead wood justifies a full rebuild."
                )
    if "twelfth_man" in unused and next_round >= 4:
        pick = twelfth_man_pick(df, solution)
        advice.append(
            f"12th Man candidate: {pick['name']} ({pick['squadName']}, "
            f"{pick['ep_match']:.1f} EP) - no budget or country limits, "
            "pick is locked for the round once it starts."
        )
    if "mystery" in unused and next_round >= rules.MYSTERY_AVAILABLE_FROM_ROUND:
        advice.append(
            "Mystery Booster: revealed now that the R32 is open; usable in "
            "one KO round incl. the final. Re-evaluate on its actual effect."
        )

    return advice or ["No booster this round: no standout opportunity."]
