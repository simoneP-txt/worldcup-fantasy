import pandas as pd

from wcfantasy.data_sources import wikipedia_squads


SQUADS = [
    {"id": 10, "name": "Switzerland", "abbr": "SUI"},
    {"id": 47, "name": "USA", "abbr": "USA"},
]


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "name": "Rubén Vargas", "squadId": 10, "position": "MID"},
        {"id": 2, "name": "Granit Xhaka", "squadId": 10, "position": "MID"},
        {"id": 3, "name": "Christian Pulisic", "squadId": 47, "position": "MID"},
    ])


def test_parse_withdrawals_flags_injured_player_only():
    wikitext = (
        "Some intro paragraph.\n"
        "On 5 June 2026, [[Ruben Vargas]] was ruled out of the tournament with "
        "a knee injury and replaced by [[Some Reserve]].\n"
        "[[Granit Xhaka]] will captain the side.\n"
    )
    out = wikipedia_squads.parse_withdrawals(wikitext, SQUADS, _players())
    assert out == {1: 0.0}          # Vargas, injury line (accent-insensitive)
    # Xhaka has no injury phrase; "Some Reserve" is not a fantasy player.


def test_parse_withdrawals_ignores_unrelated_lines():
    wikitext = (
        "[[Christian Pulisic]] scored twice in the warm-up match.\n"
        "[[Granit Xhaka]] trained fully on Tuesday.\n"
    )
    assert wikipedia_squads.parse_withdrawals(wikitext, SQUADS, _players()) == {}


def test_parse_withdrawals_needs_both_verb_and_injury_noun():
    # a withdrawal verb without an injury/fitness noun nearby must not fire
    wikitext = "[[Christian Pulisic]] was ruled out as captain by the new coach.\n"
    assert wikipedia_squads.parse_withdrawals(wikitext, SQUADS, _players()) == {}


def test_parse_withdrawals_never_flags_the_replacement():
    # the healthy replacement (after "replaced by") must never be flagged
    wikitext = (
        "[[Granit Xhaka]] withdrew injured and was replaced by "
        "[[Christian Pulisic]] on 1 June.\n"
    )
    out = wikipedia_squads.parse_withdrawals(wikitext, SQUADS, _players())
    assert out == {2: 0.0}        # only Xhaka (the subject), not Pulisic
