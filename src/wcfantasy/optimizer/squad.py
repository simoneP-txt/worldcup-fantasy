"""Squad selection as an integer linear program (PuLP + CBC).

Decision variables per player: in squad, in starting XI, captain.
Objective: starters earn ep_next (next round) plus a horizon bonus for
future rounds; the captain earns ep_next once more (double points);
bench players count at a small weight (injury/rotation cover and the
12th Man booster later).

`build_squad` solves from scratch (initial team and Wildcard rounds).
`transfers.py` adds a constraint to keep all but K of the current squad.
"""

import warnings
from dataclasses import dataclass

import pandas as pd
import pulp

from .. import rules

BENCH_WEIGHT = 0.15
# Next round dominates; future rounds matter through ep_horizon at a
# discount so the optimizer still grabs likely-finalist players early.
HORIZON_WEIGHT = 0.35

_DEF_MIN, _DEF_MAX = min(f[0] for f in rules.FORMATIONS), max(f[0] for f in rules.FORMATIONS)
_MID_MIN, _MID_MAX = min(f[1] for f in rules.FORMATIONS), max(f[1] for f in rules.FORMATIONS)
_FWD_MIN, _FWD_MAX = min(f[2] for f in rules.FORMATIONS), max(f[2] for f in rules.FORMATIONS)


@dataclass
class SquadSolution:
    squad_ids: list[int]
    starter_ids: list[int]
    captain_id: int
    cost: float
    objective: float


def _player_score(row) -> float:
    return row.ep_match + HORIZON_WEIGHT * (row.ep_horizon - row.ep_match)


def build_squad(
    players: pd.DataFrame,
    budget: float = rules.BUDGET,
    keep_ids: list[int] | None = None,
    max_changes: int | None = None,
    max_per_squad: int | None = None,
) -> SquadSolution:
    """Pick the optimal 15, starting XI and captain under game rules.

    players must carry ep_match and ep_horizon (see player_points).
    With keep_ids/max_changes the solution differs from keep_ids by at
    most max_changes players (transfer mode). max_per_squad enforces the
    official per-country cap for the round (rules.max_per_squad).
    """
    pool = players[players["ep_match"] > 0].reset_index(drop=True)
    idx = pool.index

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    in_squad = prob.add_variable_dicts("squad", idx, cat="Binary")
    starter = prob.add_variable_dicts("starter", idx, cat="Binary")
    captain = prob.add_variable_dicts("captain", idx, cat="Binary")

    score = {i: _player_score(pool.loc[i]) for i in idx}
    prob += pulp.lpSum(
        starter[i] * score[i]
        + captain[i] * pool.loc[i, "ep_match"]
        + (in_squad[i] - starter[i]) * BENCH_WEIGHT * score[i]
        for i in idx
    )

    prob += pulp.lpSum(in_squad[i] * pool.loc[i, "price"] for i in idx) <= budget
    prob += pulp.lpSum(in_squad.values()) == rules.SQUAD_SIZE
    for pos, count in rules.SQUAD_COMPOSITION.items():
        prob += (
            pulp.lpSum(in_squad[i] for i in idx if pool.loc[i, "position"] == pos)
            == count
        )
    if max_per_squad is not None:
        for squad_id in pool["squadId"].unique():
            prob += (
                pulp.lpSum(in_squad[i] for i in idx if pool.loc[i, "squadId"] == squad_id)
                <= max_per_squad
            )

    prob += pulp.lpSum(starter.values()) == 11
    for i in idx:
        prob += starter[i] <= in_squad[i]
        prob += captain[i] <= starter[i]
    prob += pulp.lpSum(captain.values()) == 1

    def _count(pos):
        return pulp.lpSum(starter[i] for i in idx if pool.loc[i, "position"] == pos)

    prob += _count("GK") == 1
    prob += _count("DEF") >= _DEF_MIN
    prob += _count("DEF") <= _DEF_MAX
    prob += _count("MID") >= _MID_MIN
    prob += _count("MID") <= _MID_MAX
    prob += _count("FWD") >= _FWD_MIN
    prob += _count("FWD") <= _FWD_MAX

    if keep_ids is not None and max_changes is not None:
        kept = [i for i in idx if pool.loc[i, "id"] in set(keep_ids)]
        prob += pulp.lpSum(in_squad[i] for i in kept) >= rules.SQUAD_SIZE - max_changes

    with warnings.catch_warnings():
        # PULP_CBC_CMD is the CBC bundled with PuLP and the most portable
        # solver; COIN_CMD (PuLP 4.0's preferred path) needs an external
        # cbc binary on PATH, so keep CBC and mute the migration notice
        # instead of reducing portability.
        warnings.filterwarnings(
            "ignore", message="PULP_CBC_CMD is deprecated",
            category=DeprecationWarning,
        )
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"ILP not optimal: {pulp.LpStatus[status]}")

    squad_ids = [int(pool.loc[i, "id"]) for i in idx if in_squad[i].value() > 0.5]
    starter_ids = [int(pool.loc[i, "id"]) for i in idx if starter[i].value() > 0.5]
    captain_id = next(int(pool.loc[i, "id"]) for i in idx if captain[i].value() > 0.5)
    cost = float(sum(pool.loc[i, "price"] for i in idx if in_squad[i].value() > 0.5))
    return SquadSolution(
        squad_ids, starter_ids, captain_id, cost, float(pulp.value(prob.objective))
    )
