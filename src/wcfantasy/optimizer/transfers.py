"""Round-to-round transfer advice.

The allocation is stage-dependent (rules.FREE_TRANSFERS_BY_ROUND), plus
an optional carryover in the group stage. Unlimited windows (the R32)
re-solve the squad from scratch.

User strategy: never spend a voluntary -3 paid transfer. The plan stays
within the free allowance, and the Wildcard is the escape hatch for a
big rebuild (played only in an extraordinary, clearly-worth-it case).
The single exception is being FORCED past the allowance: when more
current players are unplayable (eliminated/transferred) than the
allowance can shed, those moves are unavoidable, so the planner takes
the minimum number that yields a valid squad and the report flags them
and points to the Wildcard.
"""

from dataclasses import dataclass, field

import pandas as pd

from .. import rules
from ..state import TeamState
from .squad import SquadSolution, build_squad

# How far past the free allowance the planner will search for FORCED
# moves (unplayable players) before giving up and rebuilding from
# scratch. Never used to take a voluntary paid transfer.
MAX_FORCED_TRANSFERS = 4


@dataclass
class TransferPlan:
    out_ids: list[int] = field(default_factory=list)
    in_ids: list[int] = field(default_factory=list)
    solution: SquadSolution | None = None
    free_allowance: int | None = None  # None = unlimited window
    n_paid: int = 0
    penalty_points: float = 0.0
    carried_next: int = 0  # transfers carried into the next group window


def free_transfers(next_round: int, carried: int = 0) -> int | None:
    """Free allocation for the round's window (None = unlimited)."""
    base = rules.FREE_TRANSFERS_BY_ROUND.get(next_round, 0)
    if base is None:
        return None
    if next_round in rules.CARRYOVER_TARGET_ROUNDS:
        base += min(carried, rules.TRANSFER_CARRYOVER_MAX)
    return base


def plan_transfers(
    players: pd.DataFrame,
    state: TeamState,
    budget: float,
    next_round: int,
    max_per_squad: int | None = None,
) -> TransferPlan:
    allowance = free_transfers(next_round, state.transfers_carried)

    if allowance is None:
        # Unlimited window (R32): full rebuild, every change is free.
        solution = build_squad(players, budget=budget, max_per_squad=max_per_squad)
        return _diff(state, solution, allowance=None, next_round=next_round)

    # Stay within the free allowance: no voluntary paid transfer. Walk
    # max_changes up only when the keep-constraint is INFEASIBLE, i.e. more
    # current players are unplayable (eliminated/transferred -> ep_match 0,
    # dropped from the pool) than the allowance can shed. The first feasible
    # level is the answer: the free allowance normally, or the minimum number
    # forced by unplayable players. We never go past it for a mere EP gain.
    best = None
    for changes in range(allowance, allowance + 1 + MAX_FORCED_TRANSFERS):
        try:
            best = build_squad(
                players,
                budget=budget,
                keep_ids=state.player_ids,
                max_changes=changes,
                max_per_squad=max_per_squad,
            )
        except RuntimeError:
            continue  # too few changes to shed unplayable players
        break
    if best is None:
        # Even the deepest level cannot keep enough of the squad (many teams
        # out): a free full rebuild is the only valid plan left (a Wildcard
        # round in practice).
        best = build_squad(players, budget=budget, max_per_squad=max_per_squad)
    return _diff(state, best, allowance=allowance, next_round=next_round)


def _diff(
    state: TeamState,
    solution: SquadSolution,
    allowance: int | None,
    next_round: int,
) -> TransferPlan:
    current = set(state.player_ids)
    new = set(solution.squad_ids)
    in_ids = sorted(new - current)
    used = len(in_ids)
    # The ILP may use fewer changes than allowed: count what it spent.
    n_paid = 0 if allowance is None else max(0, used - allowance)

    carried_next = 0
    # Only a round-2 window leaves something to carry (into round 3).
    if next_round == 2 and allowance is not None and used < allowance:
        carried_next = min(rules.TRANSFER_CARRYOVER_MAX, allowance - used)

    return TransferPlan(
        out_ids=sorted(current - new),
        in_ids=in_ids,
        solution=solution,
        free_allowance=allowance,
        n_paid=n_paid,
        penalty_points=n_paid * rules.EXTRA_TRANSFER_PENALTY,
        carried_next=carried_next,
    )
