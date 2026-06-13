"""Sanity checks that rules.py encodes the official 'How to Play' tables."""

from wcfantasy import rules


def test_per_country_cap_by_stage():
    assert rules.MAX_PER_SQUAD_BY_ROUND == {
        1: 3, 2: 3, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 8
    }
    assert rules.max_per_squad(1) == 3
    assert rules.max_per_squad(8) == 8
    # out-of-range rounds clamp instead of crashing
    assert rules.max_per_squad(0) == 3
    assert rules.max_per_squad(99) == 8


def test_transfer_allocation_table():
    assert rules.FREE_TRANSFERS_BY_ROUND[1] is None  # pre-tournament build
    assert rules.FREE_TRANSFERS_BY_ROUND[4] is None  # R32 unlimited
    assert rules.FREE_TRANSFERS_BY_ROUND[2] == 2
    assert rules.FREE_TRANSFERS_BY_ROUND[3] == 2
    assert rules.FREE_TRANSFERS_BY_ROUND[5] == 4
    assert rules.FREE_TRANSFERS_BY_ROUND[6] == 4
    assert rules.FREE_TRANSFERS_BY_ROUND[7] == 5
    assert rules.FREE_TRANSFERS_BY_ROUND[8] == 6
    assert rules.EXTRA_TRANSFER_PENALTY == 3.0
    assert rules.CARRYOVER_TARGET_ROUNDS == (3,)


def test_booster_constraints():
    assert rules.WILDCARD_EXCLUDED_ROUNDS == (1, 4)
    assert rules.IRREVERSIBLE_BOOSTERS == ("wildcard",)
    assert rules.QUALIFICATION_BOOSTER_ROUNDS == (4, 5, 6, 7, 8)
    assert set(rules.BOOSTERS) == {
        "maximum_captain", "twelfth_man", "wildcard", "qualification", "mystery"
    }


def test_scoring_table():
    assert rules.GOAL_POINTS == {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5}
    assert rules.CLEAN_SHEET_POINTS == {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0}
    assert rules.SCORING_ALL["assist"] == 3
    assert rules.SCORING_ALL["red_card"] == -2
    # appearance read as cumulative: 2 points total for a 60+ outing
    assert rules.SCORING_ALL["appearance"] + rules.SCORING_ALL["appearance_60min"] == 2
    assert rules.SCOUTING_BONUS == 2
    assert rules.SCOUTING_BONUS_MAX_OWNERSHIP == 5.0


def test_prices_are_fixed():
    assert rules.PRICES_ARE_FIXED is True
