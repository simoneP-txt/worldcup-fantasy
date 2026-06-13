import pandas as pd
import pytest

from wcfantasy import rules
from wcfantasy.optimizer.squad import build_squad


@pytest.fixture
def pool():
    rows = []
    pid = 1
    # 8 fake nations x (1 GK, 3 DEF, 3 MID, 2 FWD) = 72 players
    for squad_id in range(1, 9):
        for pos, n in (("GK", 1), ("DEF", 3), ("MID", 3), ("FWD", 2)):
            for k in range(n):
                price = 4.0 + (squad_id % 4) + k
                rows.append({
                    "id": pid, "name": f"p{pid}", "squadId": squad_id,
                    "position": pos, "price": price,
                    "ep_match": price * 0.5, "ep_horizon": price * 2.0,
                })
                pid += 1
    return pd.DataFrame(rows)


def test_build_squad_respects_rules(pool):
    sol = build_squad(pool, budget=rules.BUDGET)
    squad = pool[pool["id"].isin(sol.squad_ids)]

    assert len(sol.squad_ids) == rules.SQUAD_SIZE
    assert sol.cost <= rules.BUDGET
    assert squad["position"].value_counts().to_dict() == rules.SQUAD_COMPOSITION

    starters = pool[pool["id"].isin(sol.starter_ids)]
    assert len(starters) == 11
    assert set(sol.starter_ids) <= set(sol.squad_ids)
    assert (starters["position"] == "GK").sum() == 1
    assert sol.captain_id in sol.starter_ids


def test_build_squad_respects_country_cap(pool):
    cap = rules.max_per_squad(1)  # 3 in the group stage
    sol = build_squad(pool, budget=rules.BUDGET, max_per_squad=cap)
    squad = pool[pool["id"].isin(sol.squad_ids)]
    assert squad.groupby("squadId").size().max() <= cap


def test_transfer_constraint_limits_changes(pool):
    base = build_squad(pool, budget=rules.BUDGET)
    # force cheap squad then allow 2 changes from it
    constrained = build_squad(
        pool, budget=rules.BUDGET, keep_ids=base.squad_ids, max_changes=2
    )
    kept = set(base.squad_ids) & set(constrained.squad_ids)
    assert len(kept) >= rules.SQUAD_SIZE - 2
