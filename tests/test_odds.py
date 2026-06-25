from wcfantasy.data_sources import odds


SQUADS = [
    {"id": 1, "name": "Korea Republic"},
    {"id": 2, "name": "USA"},
    {"id": 3, "name": "Côte d'Ivoire"},
    {"id": 4, "name": "Netherlands"},
]


def test_implied_removes_overround():
    probs = odds._implied({"A": 2.0, "Draw": 3.0, "B": 4.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # raw 1/2 = 0.5 deflated by the ~1.083 book total
    assert probs["A"] < 0.5
    assert probs["A"] > probs["Draw"] > probs["B"]


def test_match_team_overrides_and_fuzzy():
    assert odds.match_team("South Korea", SQUADS) == 1
    assert odds.match_team("United States", SQUADS) == 2
    assert odds.match_team("Ivory Coast", SQUADS) == 3
    assert odds.match_team("Netherland", SQUADS) == 4  # fuzzy
    assert odds.match_team("Atlantis", SQUADS) is None


def test_world_cup_sport_keys_picks_soccer_not_cricket(monkeypatch):
    # cricket is FIRST in the list: the old "world_cup in key" filter grabbed
    # it for the h2h market. The fix must pick the men's soccer World Cup.
    sports = [
        {"key": "cricket_icc_world_cup", "has_outrights": False},
        {"key": "rugbyleague_world_cup", "has_outrights": False},
        {"key": "soccer_fifa_club_world_cup", "has_outrights": False},
        {"key": "soccer_fifa_world_cup_women", "has_outrights": False},
        {"key": "soccer_fifa_world_cup_winner", "has_outrights": True},
        {"key": "soccer_fifa_world_cup", "has_outrights": False},
    ]
    monkeypatch.setattr(odds, "_get", lambda *a, **k: sports)
    h2h, outright = odds._world_cup_sport_keys()
    assert h2h == "soccer_fifa_world_cup"
    assert outright == "soccer_fifa_world_cup_winner"


def test_world_cup_sport_keys_none_when_no_sport_list(monkeypatch):
    monkeypatch.setattr(odds, "_get", lambda *a, **k: None)
    assert odds._world_cup_sport_keys() == (None, None)
