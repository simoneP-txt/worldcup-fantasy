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
