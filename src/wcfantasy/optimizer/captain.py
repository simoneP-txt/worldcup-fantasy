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

from datetime import datetime

import pandas as pd

from .. import rules
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


def _kickoff(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


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
    kickoff: dict[int, datetime] = {}
    for m in rnd.get("tournaments", []):
        when = _kickoff(m.get("date"))
        if when is None:
            continue
        for key in ("homeSquadId", "awaySquadId"):
            if m.get(key) is not None:
                kickoff[m[key]] = when

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


def twelfth_man_pick(df: pd.DataFrame, solution: SquadSolution) -> pd.Series:
    """Best 12th Man: highest-EP player OUTSIDE the squad.

    The booster ignores budget and per-country caps, so the answer is
    simply the top expected scorer you do not already own.
    """
    outside = df[~df["id"].isin(solution.squad_ids)]
    return outside.sort_values("ep_match", ascending=False).iloc[0]


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

    # Never two boosters in the same round: at most one ACTIVATE.
    # Everything except the Wildcard can be deactivated before lockout,
    # so an ACTIVATE here is risk-free until the round locks.
    activated = False

    if "maximum_captain" in unused and cap["ep_match"] >= 5.5:
        advice.append(
            f"ACTIVATE Maximum Captain: high-ceiling round ({cap['name']} "
            f"expected {cap['ep_match']:.1f} pts) and the booster captains "
            "your best XI scorer in hindsight, never worse than picking one."
        )
        activated = True

    if "qualification" in unused and next_round in rules.QUALIFICATION_BOOSTER_ROUNDS:
        # +2 per starter (>=1 minute played) whose team progresses; the
        # captain's +2 is not doubled. future_rounds > rounds left after
        # this one means the team is favoured to go through.
        threshold = 8 - next_round + 0.5
        progressing = (starters["future_rounds"] > threshold).sum()
        msg = (
            f"Qualification Booster: {progressing}/11 starters favoured to "
            "progress (+2 each, needs 1+ minute; captain's +2 not doubled)."
        )
        if next_round == 7 and not activated and progressing >= 6:
            # User strategy: fire it at the SF, when the 6-per-country cap
            # lets the squad stack a likely finalist.
            advice.append(
                "ACTIVATE " + msg + " Semi-final is your planned window: "
                "starters from a finalist bank the bonus twice (SF + final win)."
            )
            activated = True
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
            dead = squad[
                squad["isEliminated"].fillna(False) | (squad["future_rounds"] < 0.5)
            ]
            if len(dead) >= 4 and not activated:
                advice.append(
                    f"ACTIVATE Wildcard (your plan): {len(dead)} squad players "
                    "are eliminated or done; rebuild around the teams still "
                    "alive. Careful: irreversible once confirmed."
                )
                activated = True
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
