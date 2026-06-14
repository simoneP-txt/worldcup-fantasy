import pandas as pd

from wcfantasy.data_sources import availability


SQUADS = [
    {"id": 10, "name": "Switzerland", "abbr": "SUI"},
    {"id": 47, "name": "USA", "abbr": "USA"},
    {"id": 41, "name": "Spain", "abbr": "ESP"},
]


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "name": "Rubén Vargas", "squadId": 10, "position": "MID"},
        {"id": 2, "name": "Granit Xhaka", "squadId": 10, "position": "MID"},
        {"id": 3, "name": "Christian Pulisic", "squadId": 47, "position": "MID"},
        {"id": 4, "name": "Lamine Yamal", "squadId": 41, "position": "MID"},
    ])


# ----------------------------------------- API-Football injuries (optional)

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


# --------------------------------------------------------------- aggregator

def test_merge_keeps_the_lowest_factor_per_player():
    merged = availability._merge({1: 0.5}, {1: 0.0, 2: 0.4}, {3: 0.0})
    assert merged == {1: 0.0, 2: 0.4, 3: 0.0}


def test_availability_factors_off_when_all_sources_empty(monkeypatch, tmp_path):
    # no key, FIFA/Wikipedia patched empty -> {} and a clean count, never raises
    monkeypatch.setattr(availability.fifa_match, "suspension_factors", lambda *a, **k: {})
    monkeypatch.setattr(
        availability.wikipedia_squads, "withdrawal_factors", lambda *a, **k: {}
    )
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(availability, "KEY_FILE", tmp_path / "missing.txt")
    monkeypatch.setattr(availability, "INJURIES_CACHE", tmp_path / "injuries.json")
    assert availability.availability_factors(SQUADS, _players()) == {}
    assert availability.last_run_info["flagged"] == 0


def test_availability_factors_merges_suspensions_and_withdrawals(monkeypatch, tmp_path):
    monkeypatch.setattr(
        availability.fifa_match, "suspension_factors", lambda *a, **k: {1: 0.0}
    )
    monkeypatch.setattr(
        availability.wikipedia_squads, "withdrawal_factors", lambda *a, **k: {2: 0.0}
    )
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(availability, "KEY_FILE", tmp_path / "missing.txt")
    out = availability.availability_factors(SQUADS, _players(), rounds=[], round_no=2)
    assert out == {1: 0.0, 2: 0.0}
    assert availability.last_run_info["suspensions"] == 1
    assert availability.last_run_info["withdrawals"] == 1
    assert availability.last_run_info["flagged"] == 2
