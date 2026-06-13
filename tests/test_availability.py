import pandas as pd

from wcfantasy.data_sources import availability


SQUADS = [
    {"id": 10, "name": "Switzerland"},
    {"id": 47, "name": "USA"},
    {"id": 41, "name": "Spain"},
]


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "name": "Rubén Vargas", "squadId": 10, "position": "MID"},
        {"id": 2, "name": "Granit Xhaka", "squadId": 10, "position": "MID"},
        {"id": 3, "name": "Christian Pulisic", "squadId": 47, "position": "MID"},
        {"id": 4, "name": "Lamine Yamal", "squadId": 41, "position": "MID"},
    ])


def test_injury_weight_known_and_unknown_statuses():
    assert availability.injury_weight("Questionable") == 0.5
    assert availability.injury_weight("doubtful") == 0.4
    # anything in the feed but unrecognised is a hard out
    assert availability.injury_weight("Missing Fixture") == 0.0
    assert availability.injury_weight("Out") == 0.0


def test_build_injury_factors_matches_team_and_player():
    raw = [
        {"team": {"name": "Switzerland"}, "player": {"name": "Ruben Vargas"},
         "type": "Questionable"},
        {"team": {"name": "United States"}, "player": {"name": "Christian Pulisic"},
         "type": "Missing Fixture"},
    ]
    factors = availability.build_injury_factors(raw, SQUADS, _players())
    assert factors[1] == 0.5   # Vargas doubtful, fuzzy team "United States"->USA
    assert factors[3] == 0.0   # Pulisic out
    assert 2 not in factors and 4 not in factors  # unflagged = fit


def test_build_injury_factors_keeps_the_worst_of_duplicates():
    raw = [
        {"team": {"name": "Switzerland"}, "player": {"name": "Ruben Vargas"},
         "type": "Questionable"},   # 0.5
        {"team": {"name": "Switzerland"}, "player": {"name": "Ruben Vargas"},
         "type": "Out"},            # 0.0 -> wins
    ]
    factors = availability.build_injury_factors(raw, SQUADS, _players())
    assert factors[1] == 0.0


def test_build_injury_factors_ignores_unmatched():
    raw = [
        {"team": {"name": "Atlantis"}, "player": {"name": "Nobody"}, "type": "Out"},
        {"team": {"name": "Spain"}, "player": {"name": "Totally Unknown"},
         "type": "Out"},
    ]
    assert availability.build_injury_factors(raw, SQUADS, _players()) == {}


def test_build_fixture_id_map_matches_team_names():
    raw = [
        {"fixture": {"id": 111},
         "teams": {"home": {"name": "Switzerland"}, "away": {"name": "United States"}}},
        {"fixture": {"id": 222},
         "teams": {"home": {"name": "Spain"}, "away": {"name": "Atlantis"}}},  # away unmatched
    ]
    fmap = availability.build_fixture_id_map(raw, SQUADS)
    assert fmap == {(10, 47): 111}  # Switzerland=10, USA=47; Spain pair dropped


def test_availability_factors_off_without_key(monkeypatch, tmp_path):
    # no key, no cache -> {} and a clean "off" note, never raises
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(availability, "KEY_FILE", tmp_path / "missing.txt")
    monkeypatch.setattr(availability, "INJURIES_CACHE", tmp_path / "injuries.json")
    assert availability.availability_factors(SQUADS, _players()) == {}
    assert availability.last_run_info["flagged"] == 0
