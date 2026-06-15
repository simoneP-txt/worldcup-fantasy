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

from datetime import datetime, timezone

import pandas as pd

from .. import fixtures, rules
from ..state import TeamState
from .squad import SquadSolution


def vice_captain(df: pd.DataFrame, solution: SquadSolution) -> int:
    starters = df[df["id"].isin(solution.starter_ids)]
    ranked = starters.sort_values("ep_match", ascending=False)
    return int(ranked[ranked["id"] != solution.captain_id].iloc[0]["id"])


def ordered_bench(df: pd.DataFrame, solution: SquadSolution) -> tuple[list[int], int]:
    """(outfield bench ids ordered 1st/2nd/3rd sub, bench GK id)."""
    bench = df[
        df["id"].isin(set(solution.squad_ids) - set(solution.starter_ids))
    ].sort_values("ep_match", ascending=False)
    outfield = [int(i) for i in bench[bench["position"] != "GK"]["id"]]
    gk = int(bench[bench["position"] == "GK"]["id"].iloc[0])
    return outfield, gk


def captain_timing_note(
    df: pd.DataFrame,
    solution: SquadSolution,
    rounds: list[dict],
    next_round: int,
    tolerance: float = 0.92,
) -> str | None:
    """Option-value check on kickoff order.

    A live captain switch needs the old captain COMPLETED and the new
    one not yet played, so with two near-equal candidates the armband
    should go to whoever plays EARLIER: if he flops you move it to the
    later one, the reverse is locked out. Returns None when the gap is
    big enough that timing does not matter.
    """
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    if rnd is None:
        return None
    kickoff = fixtures.kickoffs_by_squad(rnd)

    starters = df[df["id"].isin(solution.starter_ids)]
    ranked = starters.sort_values("ep_match", ascending=False)
    cap_rows = ranked[ranked["id"] == solution.captain_id]
    contenders = ranked[ranked["id"] != solution.captain_id]
    if cap_rows.empty or contenders.empty:
        return None
    cap, runner = cap_rows.iloc[0], contenders.iloc[0]
    cap_ko = kickoff.get(int(cap["squadId"]))
    run_ko = kickoff.get(int(runner["squadId"]))
    if (
        cap_ko is None
        or run_ko is None
        or runner["ep_match"] < tolerance * cap["ep_match"]
    ):
        return None
    if run_ko < cap_ko:
        return (
            f"Timing: {runner['name']} ({runner['ep_match']:.1f} EP) is nearly "
            f"level with {cap['name']} and plays EARLIER: give him the armband "
            "first; keep it if he hauls, otherwise switch live to "
            f"{cap['name']} before his kickoff."
        )
    if cap_ko < run_ko:
        return (
            f"Timing: captain {cap['name']} plays before {runner['name']}; if "
            "he disappoints you can still move the armband to "
            f"{runner['name']} live before his match."
        )
    return None


def captain_by_timing(
    df: pd.DataFrame,
    solution: SquadSolution,
    rounds: list[dict],
    next_round: int,
    tolerance: float = 0.92,
) -> int:
    """Captain pick adjusted for kickoff order (option value).

    The ILP captains the highest-EP starter. But the armband is switchable
    live (old captain COMPLETED -> new one not yet played), so among the
    starters within `tolerance` of the top EP the captaincy should go to
    whoever plays EARLIEST: if he flops you can still move it to a later
    contender, never the reverse. Returns the chosen captain id, falling
    back to the ILP pick when there is no near-equal contender or no kickoff
    time to act on. captain_timing_note then explains the choice.
    """
    starters = df[df["id"].isin(solution.starter_ids)]
    if starters.empty:
        return solution.captain_id
    top_ep = float(starters["ep_match"].max())
    contenders = starters[starters["ep_match"] >= tolerance * top_ep]
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    if rnd is None or len(contenders) <= 1:
        return solution.captain_id

    kickoff = fixtures.kickoffs_by_squad(rnd)
    far_future = datetime.max.replace(tzinfo=timezone.utc)

    def sort_key(row: dict):
        # earliest kickoff first; unknown kickoff sorts last; higher EP
        # breaks a same-time tie so the stronger player keeps the armband.
        return (kickoff.get(int(row["squadId"]), far_future), -float(row["ep_match"]))

    best = min(contenders.to_dict("records"), key=sort_key)
    if kickoff.get(int(best["squadId"]), far_future) is far_future:
        return solution.captain_id  # no timing info: keep the ILP pick
    return int(best["id"])


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
