"""Round-to-round transfer advice.

The allocation is stage-dependent (rules.FREE_TRANSFERS_BY_ROUND), plus
an optional carryover in the group stage. Unlimited windows (the R32)
re-solve the squad from scratch. Beyond the free allocation, each extra
transfer costs 3 points: an extra change is accepted only if its
marginal ILP objective gain beats the penalty (the objective is in
expected-points units, so the comparison is apples to apples up to the
bench/horizon weighting).
"""

from dataclasses import dataclass, field

import pandas as pd

from .. import rules
from ..state import TeamState
from .squad import SquadSolution, build_squad

# Never evaluate more than this many paid transfers beyond the free
# allocation: each one must individually beat the 3-point penalty, and
# in practice the marginal gain collapses after one or two.
MAX_PAID_TRANSFERS = 3


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

    best = None
    # Walk up from the free allocation. A level is infeasible when more
    # current players are unplayable (eliminated team -> ep_match 0, dropped
    # from the pool) than that many changes allow: those are forced moves,
    # so keep raising max_changes until a squad exists instead of crashing.
    # Beyond the allocation each extra change must beat the 3-point penalty.
    for changes in range(allowance, allowance + 1 + MAX_PAID_TRANSFERS):
        try:
            cand = build_squad(
                players,
                budget=budget,
                keep_ids=state.player_ids,
                max_changes=changes,
                max_per_squad=max_per_squad,
            )
        except RuntimeError:
            continue  # too few changes to shed unplayable players
        if best is None:
            best = cand
        elif cand.objective - best.objective > rules.EXTRA_TRANSFER_PENALTY:
            best = cand
        else:
            break
    if best is None:
        # Even the deepest paid level cannot keep enough of the squad (many
        # teams out): a free full rebuild is the only valid plan left.
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
