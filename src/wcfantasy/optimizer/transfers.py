"""Round-to-round transfer advice.

The allocation is stage-dependent (rules.FREE_TRANSFERS_BY_ROUND), plus
an optional carryover in the group stage. Unlimited windows (the R32)
re-solve the squad from scratch.

Strategy: stay within the free allowance by default, but take a voluntary
paid transfer (-3 points each, rules.EXTRA_TRANSFER_PENALTY) when it is
worth it on a CONSERVATIVE, next-round basis. The same swap would be free
at the next window, so paying -3 now only buys the upgrade for the current
round; an extra paid transfer is therefore accepted only when it lifts the
squad's NEXT-ROUND expected points (starting XI + captain double) by more
than the penalty. Horizon value is deliberately excluded from this test:
you can capture almost all of it for free next window, so it must not
justify a -3 today. This keeps paid transfers rare, and the Wildcard stays
the escape hatch for a big rebuild.

Forced moves are separate: when more current players are unplayable
(eliminated/transferred -> ep_match 0, dropped from the pool) than the
allowance can shed, those transfers are unavoidable. The planner first
takes the minimum number that yields a valid squad (the forced baseline),
then weighs voluntary paid transfers on top of it. The report flags forced
and voluntary paid transfers distinctly.
"""

from dataclasses import dataclass, field

import pandas as pd

from .. import rules
from ..state import TeamState
from .squad import SquadSolution, build_squad

# How far past the free allowance the planner will search for FORCED
# moves (unplayable players) before giving up and rebuilding from scratch.
MAX_FORCED_TRANSFERS = 4
# How many voluntary paid transfers (beyond the forced baseline) to weigh.
# Each must pay for itself in NEXT-ROUND points (see module docstring);
# realistically you would never pay for more than a couple.
MAX_VOLUNTARY_PAID = 3
# Require a strictly positive net improvement before taking a paid transfer,
# so a numerically-tied alternative never triggers a -3 hit for nothing.
NET_EPSILON = 1e-6

# Sentinel for plan_transfers' allowance_override: callers that already know the
# window's remaining free transfers (e.g. advise re-planning a round whose free
# transfers are already spent -> 0) pass it; the default recomputes the
# allowance from the round + carryover as usual.
_UNSET = object()


@dataclass
class TransferPlan:
    out_ids: list[int] = field(default_factory=list)
    in_ids: list[int] = field(default_factory=list)
    solution: SquadSolution | None = None
    free_allowance: int | None = None  # None = unlimited window
    n_paid: int = 0
    penalty_points: float = 0.0
    carried_next: int = 0  # transfers carried into the next group window
    n_forced_paid: int = 0       # paid transfers forced by unplayable players
    n_voluntary_paid: int = 0    # paid transfers chosen for a next-round gain
    nextround_gain: float = 0.0  # next-round EP added by the voluntary paid ones


def free_transfers(next_round: int, carried: int = 0) -> int | None:
    """Free allocation for the round's window (None = unlimited)."""
    base = rules.FREE_TRANSFERS_BY_ROUND.get(next_round, 0)
    if base is None:
        return None
    if next_round in rules.CARRYOVER_TARGET_ROUNDS:
        base += min(carried, rules.TRANSFER_CARRYOVER_MAX)
    return base


def _paid_count(state: TeamState, solution: SquadSolution, allowance: int | None) -> int:
    """Transfers in this solution beyond the free allowance (each costs -3)."""
    if allowance is None:
        return 0
    used = len(set(solution.squad_ids) - set(state.player_ids))
    return max(0, used - allowance)


def _next_round_ep(players: pd.DataFrame, solution: SquadSolution) -> float:
    """Expected points the squad scores NEXT round: the starting XI's
    ep_match plus the captain's ep_match again (the armband doubles). This
    is the basis for judging whether a voluntary paid transfer pays for its
    -3 (horizon value is excluded on purpose - see the module docstring)."""
    ep = players.set_index("id")["ep_match"]
    total = float(ep.reindex(solution.starter_ids).fillna(0.0).sum())
    if solution.captain_id in ep.index:
        total += float(ep.loc[solution.captain_id])
    return total


def plan_transfers(
    players: pd.DataFrame,
    state: TeamState,
    budget: float,
    next_round: int,
    max_per_squad: int | None = None,
    *,
    allowance_override: object = _UNSET,
    forced_only: bool = False,
) -> TransferPlan:
    """Plan the round's transfers under the rules and the strategy above.

    allowance_override pins the free allowance (e.g. 0 when the window's free
    transfers are already spent); the default derives it from the round +
    carryover. forced_only stops at the forced baseline (no voluntary paid
    search), so only unavoidable moves are proposed.
    """
    if allowance_override is _UNSET:
        allowance = free_transfers(next_round, state.transfers_carried)
    else:
        allowance = allowance_override  # type: ignore[assignment]

    if allowance is None:
        # Unlimited window (R32): full rebuild, every change is free.
        solution = build_squad(players, budget=budget, max_per_squad=max_per_squad)
        return _diff(state, solution, allowance=None, next_round=next_round)

    # Forced baseline: the minimum number of changes that yields a valid squad.
    # Normally just the free allowance; it grows only when more current players
    # are unplayable than the allowance can shed (build_squad raises until
    # enough changes are allowed to drop them).
    base = None
    base_changes = allowance
    for changes in range(allowance, allowance + 1 + MAX_FORCED_TRANSFERS):
        try:
            base = build_squad(
                players,
                budget=budget,
                keep_ids=state.player_ids,
                max_changes=changes,
                max_per_squad=max_per_squad,
            )
        except RuntimeError:
            continue  # too few changes to shed unplayable players
        base_changes = changes
        break
    if base is None:
        # Even the deepest level cannot keep enough of the squad (many teams
        # out): a free full rebuild is the only valid plan left (a Wildcard
        # round in practice).
        best = build_squad(players, budget=budget, max_per_squad=max_per_squad)
        return _diff(state, best, allowance=allowance, next_round=next_round)

    # Voluntary paid transfers on top of the forced baseline, judged on
    # NEXT-ROUND points only (the conservative rule). Each candidate level
    # gets a penalty-adjusted next-round score; the -3 * forced-paid term is
    # constant across candidates, so comparing these nets picks exactly the
    # extra transfers that pay for themselves next round.
    penalty = rules.EXTRA_TRANSFER_PENALTY
    champion = base
    champion_net = _next_round_ep(players, base) - penalty * _paid_count(state, base, allowance)
    if not forced_only:
        for changes in range(base_changes + 1, base_changes + 1 + MAX_VOLUNTARY_PAID):
            try:
                sol = build_squad(
                    players,
                    budget=budget,
                    keep_ids=state.player_ids,
                    max_changes=changes,
                    max_per_squad=max_per_squad,
                )
            except RuntimeError:
                break
            net = _next_round_ep(players, sol) - penalty * _paid_count(state, sol, allowance)
            if net > champion_net + NET_EPSILON:
                champion, champion_net = sol, net

    plan = _diff(state, champion, allowance=allowance, next_round=next_round)
    base_paid = _paid_count(state, base, allowance)
    plan.n_forced_paid = base_paid
    plan.n_voluntary_paid = max(0, plan.n_paid - base_paid)
    if plan.n_voluntary_paid:
        plan.nextround_gain = _next_round_ep(players, champion) - _next_round_ep(players, base)
    return plan


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
