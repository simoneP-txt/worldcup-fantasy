import pandas as pd
import pytest

from wcfantasy import rules
from wcfantasy.optimizer.transfers import free_transfers, plan_transfers
from wcfantasy.state import TeamState

POSITIONS = (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3))


def _players(start_id: int, squad_id: int, ep: float, price: float) -> list[dict]:
    rows = []
    pid = start_id
    for pos, n in POSITIONS:
        for _ in range(n):
            rows.append({
                "id": pid, "name": f"p{pid}", "squadId": squad_id,
                "position": pos, "price": price,
                "ep_match": ep, "ep_horizon": ep * 2,
            })
            pid += 1
    return rows


@pytest.fixture
def current_team():
    # a full valid 15 (ids 1..15), cheap and mediocre
    return TeamState(player_ids=list(range(1, 16)))


def _pool(star_ep: float) -> pd.DataFrame:
    # current 15 spread over squads 1-3 plus 3 star MIDs elsewhere
    rows = _players(1, 1, ep=2.0, price=4.0)
    for i, r in enumerate(rows):
        r["squadId"] = 1 + i % 3
    for k in range(3):
        rows.append({
            "id": 101 + k, "name": f"star{k}", "squadId": 4 + k,
            "position": "MID", "price": 9.0,
            "ep_match": star_ep, "ep_horizon": star_ep * 2,
        })
    return pd.DataFrame(rows)


def test_free_transfers_allocation_and_carryover():
    assert free_transfers(2) == 2
    assert free_transfers(3) == 2
    assert free_transfers(3, carried=1) == 3
    assert free_transfers(2, carried=1) == 2  # nothing carries into round 2
    assert free_transfers(4) is None  # unlimited R32
    assert free_transfers(5) == 4
    assert free_transfers(8) == 6


def test_unlimited_window_is_full_rebuild(current_team):
    pool = _pool(star_ep=12.0)
    plan = plan_transfers(pool, current_team, budget=105.0, next_round=4)
    assert plan.free_allowance is None
    assert plan.n_paid == 0
    assert plan.penalty_points == 0.0
    # all 3 stars come in despite being more than any limited allocation
    assert {101, 102, 103} <= set(plan.solution.squad_ids)


def test_caps_at_free_allowance_no_voluntary_paid(current_team):
    # 3 huge upgrades available, but only 2 free transfers: the user never
    # pays a voluntary -3, so advise takes exactly 2 and leaves the third
    # (the Wildcard is the route for a bigger rebuild)
    pool = _pool(star_ep=12.0)
    plan = plan_transfers(pool, current_team, budget=100.0, next_round=2)
    assert plan.free_allowance == 2
    assert len(plan.in_ids) == 2
    assert plan.n_paid == 0
    assert plan.penalty_points == 0.0


def test_no_paid_transfer_for_marginal_gains(current_team):
    # stars barely better: never worth the 3-point penalty
    pool = _pool(star_ep=2.2)
    plan = plan_transfers(pool, current_team, budget=100.0, next_round=2)
    assert len(plan.in_ids) <= 2
    assert plan.n_paid == 0


def test_infeasible_keep_constraint_does_not_crash(current_team):
    # three current MIDs become unplayable (ep 0 -> dropped from the pool):
    # more forced changes than the 2 free transfers must not raise
    pool = _pool(star_ep=12.0)
    pool.loc[pool["id"].isin([8, 9, 10]), ["ep_match", "ep_horizon"]] = 0.0
    plan = plan_transfers(pool, current_team, budget=100.0, next_round=2)
    assert plan.solution is not None
    assert len(plan.solution.squad_ids) == rules.SQUAD_SIZE
    assert {8, 9, 10}.isdisjoint(set(plan.solution.squad_ids))


def test_carryover_reported_when_transfers_unused(current_team):
    # pool == current squad only: no improvement possible in round 2
    pool = pd.DataFrame(_players(1, 1, ep=2.0, price=4.0))
    for i in pool.index:  # spread squads so no constraint binds
        pool.loc[i, "squadId"] = 1 + i % 5
    plan = plan_transfers(pool, current_team, budget=100.0, next_round=2)
    assert plan.in_ids == []
    assert plan.carried_next == 1
