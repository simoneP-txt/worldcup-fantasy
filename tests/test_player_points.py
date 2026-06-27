import pandas as pd
import pytest

from wcfantasy.data_sources.club_stats import match_name, normalize_name
from wcfantasy.models.player_points import (
    NAILED_STARTER_CEILING,
    clean_sheet_adjustment,
    clean_sheet_probability,
    ep_basis_and_confidence,
    expected_points_per_match,
    fixture_adjusted_ep,
    opponent_factor,
    project_club_ep,
    starter_probability,
    team_clean_sheet,
)

_CS_ROUNDS = [{"id": 1, "tournaments": [{"homeSquadId": 1, "awaySquadId": 2}]}]
_CS_RATINGS = {1: 1800.0, 2: 1400.0}  # squad 1 much stronger


def _club_row(position: str, **stats) -> dict:
    row = {
        "name": "x", "nation": "ITA", "position": position,
        "games": 10, "minutes": 900, "goals": 0, "assists": 0,
        "shots_on_target": 0, "tackles": 0, "chances_created": 0,
        "saves": 0, "clean_sheets": 0, "goals_conceded": 0,
        "yellow": 0, "red": 0, "penalties_won": 0, "penalties_conceded": 0,
    }
    row.update(stats)
    return row


def test_project_club_ep_forward():
    # 0.5 goals/90 * 5 pts + 2 SoT/90 / 2 + 2 appearance pts = 5.5
    club = pd.DataFrame([_club_row("FWD", goals=5, shots_on_target=20)])
    assert project_club_ep(club).iloc[0] == pytest.approx(5.5)


def test_project_club_ep_goalkeeper():
    # 3 saves/90 -> +1; 1 GA/90 -> first goal free; 40% CS rate -> +2;
    # appearance 2: total 5.0
    club = pd.DataFrame([
        _club_row("GK", saves=30, goals_conceded=10, clean_sheets=4)
    ])
    assert project_club_ep(club).iloc[0] == pytest.approx(5.0)


def test_project_club_ep_midfielder():
    # 3 tackles/90 -> +1; 2 chances/90 -> +1; flat 0.30 CS * 1 pt;
    # appearance 2: total 4.3
    club = pd.DataFrame([_club_row("MID", tackles=30, chances_created=20)])
    assert project_club_ep(club).iloc[0] == pytest.approx(4.3)


def test_goal_points_follow_position():
    same = {"goals": 9, "minutes": 810, "games": 9}  # 1 goal per 90
    fwd = pd.DataFrame([_club_row("FWD", **same)])
    mid = pd.DataFrame([_club_row("MID", **same)])
    # MID goal is worth 6 vs FWD 5, but MID also gets the flat CS term
    diff = project_club_ep(mid).iloc[0] - project_club_ep(fwd).iloc[0]
    assert diff == pytest.approx(1.0 + 0.30)


def _players(percent_selected: float, squad_id: int = 1) -> dict:
    # distinct squad ids keep everyone rank 1 in his depth chart, so
    # the starter multiplier stays neutral unless a test wants it
    return {
        "price": 8.0, "percentSelected": percent_selected,
        "matchesPlayed": 0, "form": 0.0,
        "status": "playing", "isEliminated": False,
        "squadId": squad_id, "position": "MID", "name": f"p{squad_id}",
    }


def test_scouting_bonus_only_for_differentials():
    players = pd.DataFrame([_players(2.0, 1), _players(50.0, 2)])
    ep = expected_points_per_match(players)
    assert ep.iloc[0] > ep.iloc[1]  # identical except ownership


def test_club_prior_shrunk_toward_price_prior():
    # ownership 8% -> no scouting bonus, no matches -> pure prior;
    # price prior here is 3.98, club projection 9.0, shrunk halfway
    players = pd.DataFrame([_players(8.0, 1), _players(8.0, 2)])
    club_ep = pd.Series([9.0, None], index=players.index, dtype=float)
    ep = expected_points_per_match(players, club_ep)
    # rank-1 starters, so the depth-chart ceiling (0.93) scales the prior
    assert ep.iloc[0] == pytest.approx(NAILED_STARTER_CEILING * (0.5 * 9.0 + 0.5 * 3.98))
    assert ep.iloc[1] == pytest.approx(NAILED_STARTER_CEILING * 3.98)  # price fallback


def test_unavailable_players_are_zeroed():
    benched = _players(2.0) | {"status": "injured"}
    players = pd.DataFrame([benched])
    assert expected_points_per_match(players).iloc[0] == 0.0


def test_starter_probability_follows_in_squad_price_rank():
    rows = [
        _players(30.0) | {"price": 5.0, "position": "GK"},  # the #1
        _players(5.0) | {"price": 4.5, "position": "GK"},   # backup
        _players(8.0) | {"price": 4.0, "position": "GK"},   # third choice
        _players(2.0, 2) | {"price": 4.0, "position": "GK"},  # other team's #1
    ]
    probs = starter_probability(pd.DataFrame(rows))
    c = NAILED_STARTER_CEILING  # the table is scaled by the ceiling
    assert probs.iloc[0] == pytest.approx(1.0 * c)
    assert probs.iloc[1] == pytest.approx(0.10 * c)
    assert probs.iloc[2] == pytest.approx(0.03 * c)
    assert probs.iloc[3] == pytest.approx(1.0 * c)


def test_availability_factor_discounts_a_doubtful_player():
    # same player, one flagged doubtful (0.4): his EP must drop ~60%
    players = pd.DataFrame([_players(8.0, 1), _players(8.0, 2)])
    players.loc[0, "id"] = 100
    players.loc[1, "id"] = 200
    fit = expected_points_per_match(players)
    hurt = expected_points_per_match(players, availability={100: 0.4})
    assert hurt.iloc[0] < fit.iloc[0]
    assert hurt.iloc[1] == fit.iloc[1]  # the other player is untouched
    assert hurt.iloc[0] == pytest.approx(0.4 * fit.iloc[0], rel=0.05)


def test_ruled_out_player_collapses_to_almost_zero():
    players = pd.DataFrame([_players(8.0, 1)])
    players.loc[0, "id"] = 100
    out = expected_points_per_match(players, availability={100: 0.0})
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_observed_starts_override_price_rank():
    # the price #1 GK (more expensive + more owned) did NOT play; the cheaper
    # keeper started. Observed minutes must flip the depth chart.
    rows = [
        _players(30.0) | {"price": 5.0, "position": "GK", "id": 1},  # price #1
        _players(5.0) | {"price": 4.5, "position": "GK", "id": 2},   # cheaper
    ]
    df = pd.DataFrame(rows)
    base = starter_probability(df)                       # price rank only
    assert base.iloc[0] > base.iloc[1]                   # price favours #1
    out = starter_probability(df, observed_starts={1: 0.05, 2: 0.95})
    assert out.iloc[0] == pytest.approx(0.05)            # observed bench
    assert out.iloc[1] == pytest.approx(0.95)            # observed starter


def test_backup_keeper_never_outranks_the_number_one():
    # Joan Garcia scenario: a 3rd-choice keeper with a stellar club
    # season must stay far below the actual #1 of the same nation.
    number_one = _players(20.0) | {"price": 5.0, "position": "GK"}
    third_choice = _players(3.0) | {"price": 4.0, "position": "GK"}
    players = pd.DataFrame([number_one, third_choice])
    club_ep = pd.Series([4.0, 9.0], index=players.index)  # backup shines at club
    ep = expected_points_per_match(players, club_ep)
    assert ep.iloc[0] > 5 * ep.iloc[1]


def test_opponent_factor_rewards_easier_fixtures():
    players = pd.DataFrame([{"squadId": 1}, {"squadId": 2}])
    rounds = [{"id": 1, "tournaments": [{"homeSquadId": 1, "awaySquadId": 2}]}]
    ratings = {1: 1800.0, 2: 1400.0}  # team 1 much stronger
    factor = opponent_factor(players, rounds, next_round=1, ratings=ratings)
    assert factor.iloc[0] > 1.0  # strong team vs weak opponent -> boosted
    assert factor.iloc[1] < 1.0  # weak team vs strong opponent -> discounted
    assert factor.iloc[0] + factor.iloc[1] == pytest.approx(2.0)  # symmetric


def test_opponent_factor_neutral_without_ratings():
    players = pd.DataFrame([{"squadId": 1}, {"squadId": 2}])
    rounds = [{"id": 1, "tournaments": [{"homeSquadId": 1, "awaySquadId": 2}]}]
    assert (opponent_factor(players, rounds, 1, ratings=None) == 1.0).all()


# --------------------------------------------- opponent-aware clean sheet model

def test_clean_sheet_probability_tilts_around_base():
    assert clean_sheet_probability(0.5, 0.30, 1.0) == pytest.approx(0.30)  # even
    assert clean_sheet_probability(1.0, 0.30, 1.0) == pytest.approx(0.60)  # winner
    assert clean_sheet_probability(0.0, 0.30, 1.0) == pytest.approx(0.0)   # loser
    assert 0.0 <= clean_sheet_probability(0.9, 0.30, 5.0) <= 1.0           # clipped


def test_team_clean_sheet_strong_side_reads_higher_and_concede_one_valid():
    players = pd.DataFrame([{"squadId": 1, "position": "DEF"},
                            {"squadId": 2, "position": "DEF"}])
    p_cs, p_one = team_clean_sheet(players, _CS_ROUNDS, 1, _CS_RATINGS)
    assert p_cs.iloc[0] > 0.30 > p_cs.iloc[1]            # strong vs weak
    assert (p_one >= 0).all() and (p_one <= 1).all()
    # Poisson-consistent: P(1) = -ln(p_cs) * p_cs
    import numpy as np
    assert p_one.iloc[0] == pytest.approx(-np.log(p_cs.iloc[0]) * p_cs.iloc[0])


def test_team_clean_sheet_flat_without_ratings():
    players = pd.DataFrame([{"squadId": 1, "position": "DEF"},
                            {"squadId": 2, "position": "DEF"}])
    p_cs, _ = team_clean_sheet(players, _CS_ROUNDS, 1, ratings=None)
    assert (p_cs == 0.30).all()  # no opponent signal -> the flat base rate


def test_clean_sheet_adjustment_only_moves_def_mid():
    players = pd.DataFrame([
        {"squadId": 1, "position": "DEF"},
        {"squadId": 1, "position": "MID"},
        {"squadId": 1, "position": "FWD"},
        {"squadId": 1, "position": "GK"},
    ])
    p_cs = pd.Series([0.50, 0.50, 0.50, 0.50], index=players.index)
    adj = clean_sheet_adjustment(players, p_cs)
    assert adj.iloc[0] == pytest.approx((0.50 - 0.30) * 5)  # DEF
    assert adj.iloc[1] == pytest.approx((0.50 - 0.30) * 1)  # MID
    assert adj.iloc[2] == 0.0  # FWD (no CS points)
    assert adj.iloc[3] == 0.0  # GK (prior already uses its own club CS rate)


def test_fixture_adjusted_ep_boosts_easy_floors_hard_and_never_resurrects():
    players = pd.DataFrame([
        {"squadId": 1, "position": "DEF"},  # strong side, available
        {"squadId": 2, "position": "DEF"},  # weak side, available
        {"squadId": 2, "position": "DEF"},  # weak side, ruled out (base 0)
    ])
    base = pd.Series([2.0, 2.0, 0.0], index=players.index)
    ep, _ = fixture_adjusted_ep(players, base, _CS_ROUNDS, 1, _CS_RATINGS)
    assert ep.iloc[0] > 2.0            # easy fixture + CS tilt lifts it
    assert ep.iloc[1] == pytest.approx(0.5)  # hard fixture floored, not below
    assert ep.iloc[2] == 0.0           # unavailable stays 0 (no resurrection)


def test_ep_basis_and_confidence_tiers():
    players = pd.DataFrame([
        {"matchesPlayed": 5},  # >= full -> observed form (high)
        {"matchesPlayed": 0},  # no minutes, club-matched -> club proxy (medium)
        {"matchesPlayed": 0},  # no minutes, no club -> price only (low)
        {"matchesPlayed": 2},  # half of full, no club -> partial form (medium)
    ])
    club_matched = pd.Series([False, True, False, False], index=players.index)
    basis, level = ep_basis_and_confidence(players, club_matched, full=4.0)
    assert level.tolist() == ["high", "medium", "low", "medium"]
    assert basis.iloc[0] == "observed WC form"
    assert basis.iloc[2] == "price prior only"


def test_normalize_name_strips_accents():
    assert normalize_name("Ousmane Dembélé") == "ousmane dembele"
    assert normalize_name("Çalhanoğlu, Hakan!") == "calhanoglu  hakan"


def test_match_name_token_containment_and_cutoff():
    pool = ["Kylian Mbappé", "Achraf Hakimi", "Lamine Yamal"]
    assert match_name("Mbappé", pool) == "Kylian Mbappé"
    assert match_name("Lamine Yamal", pool) == "Lamine Yamal"
    assert match_name("Zzyzx Quux", pool) is None
