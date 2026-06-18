import pandas as pd
import pytest

from wcfantasy import rules
from wcfantasy.optimizer import squad
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


def _gk_choice_pool() -> pd.DataFrame:
    """Budget-tight pool whose only real decisions are the backup GK (cheap
    weak vs expensive strong) and one MID slot (cheap vs a 1.0m-dearer, higher
    -EP upgrade). The budget of 62 affords exactly ONE of the two upgrades."""
    rows = []

    def add(pid, pos, price, ep):
        rows.append({
            "id": pid, "name": f"p{pid}", "squadId": pid, "position": pos,
            "price": price, "ep_match": ep, "ep_horizon": ep,
        })

    add(1, "GK", 5.0, 6.0)    # nailed #1 keeper, always starts
    add(2, "GK", 4.0, 0.5)    # cheap backup
    add(3, "GK", 5.0, 5.0)    # expensive strong backup (the trap)
    for k in range(5):
        add(10 + k, "DEF", 4.0, 2.0)
    for k in range(5):
        add(20 + k, "MID", 4.0, 3.0)
    add(25, "MID", 5.0, 3.4)  # outfield upgrade: +1.0m for +0.4 EP, and starts
    for k in range(3):
        add(30 + k, "FWD", 4.0, 3.0)
    return pd.DataFrame(rows)


def test_backup_gk_is_cheap_not_expensive():
    # The second keeper barely ever plays, so the budget belongs in the XI:
    # take the cheap backup and the outfield upgrade, not the dear backup GK.
    sol = build_squad(_gk_choice_pool(), budget=62.0)
    assert 2 in sol.squad_ids and 3 not in sol.squad_ids  # cheap backup, not dear
    assert 25 in sol.squad_ids                            # saving spent in the XI
    assert 1 in sol.starter_ids                           # the #1 keeper starts


def test_uniform_bench_weight_would_overpay_for_the_backup_gk(monkeypatch):
    # With the backup keeper weighted like an outfield sub (the prior bug),
    # the ILP burns the budget on the strong, expensive backup instead.
    monkeypatch.setattr(squad, "BENCH_GK_WEIGHT", squad.BENCH_WEIGHT)
    sol = build_squad(_gk_choice_pool(), budget=62.0)
    assert 3 in sol.squad_ids and 25 not in sol.squad_ids


def _gk_pplay_pool() -> pd.DataFrame:
    """Two backup-keeper options: a DEAD one (lost his starting spot, p_play
    ~0) whose score is inflated by a deep-running team (horizon), and a cheaper
    one who actually plays. Plus a clear #1 keeper and generous cheap outfield
    so the budget never binds - the only decision is which backup to carry."""
    rows = []

    def add(pid, pos, price, ep_match, ep_horizon, p_play):
        rows.append({
            "id": pid, "name": f"p{pid}", "squadId": pid, "position": pos,
            "price": price, "ep_match": ep_match, "ep_horizon": ep_horizon,
            "p_play": p_play,
        })

    add(1, "GK", 5.0, 4.0, 4.0, 0.95)    # nailed #1, starts
    add(2, "GK", 5.0, 0.3, 4.0, 0.05)    # dead but horizon-inflated (a finalist)
    add(3, "GK", 4.0, 1.0, 1.0, 0.95)    # cheaper, actually plays
    for k in range(5):
        add(10 + k, "DEF", 4.0, 2.0, 2.0, 0.95)
    for k in range(5):
        add(20 + k, "MID", 4.0, 3.0, 3.0, 0.95)
    for k in range(3):
        add(30 + k, "FWD", 4.0, 3.0, 3.0, 0.95)
    return pd.DataFrame(rows)


def test_dead_backup_gk_not_credited_bench_value():
    # The dead 5.0m keeper must not be kept over the cheaper playing one just
    # because his team runs deep: P(plays) scaling zeroes his bench value.
    sol = build_squad(_gk_pplay_pool(), budget=rules.BUDGET)
    assert 3 in sol.squad_ids and 2 not in sol.squad_ids  # cheap live, not dead
    assert 1 in sol.starter_ids                            # #1 keeper starts


def test_backup_gk_kept_when_actually_available():
    # Same pool but the would-be backup is fully available: now the higher
    # -score keeper is the right bench pick, confirming p_play drives the flip.
    pool = _gk_pplay_pool()
    pool.loc[pool["id"] == 2, "p_play"] = 0.95
    sol = build_squad(pool, budget=rules.BUDGET)
    assert 2 in sol.squad_ids and 3 not in sol.squad_ids


def test_among_dead_backups_the_cheapest_is_kept():
    # Both backups are dead (~0 bench value): the price tiebreak frees budget
    # by keeping the cheaper one, not the dearer (horizon-inflated) keeper.
    pool = _gk_pplay_pool()
    pool.loc[pool["id"] == 3, "p_play"] = 0.05  # the cheap keeper is dead too
    sol = build_squad(pool, budget=rules.BUDGET)
    assert 3 in sol.squad_ids and 2 not in sol.squad_ids  # cheaper dead kept
