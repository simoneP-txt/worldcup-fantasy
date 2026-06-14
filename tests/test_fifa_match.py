import pandas as pd

from wcfantasy.data_sources import fifa_match


SQUADS = [
    {"id": 10, "name": "Switzerland", "abbr": "SUI"},
    {"id": 47, "name": "USA", "abbr": "USA"},
    {"id": 28, "name": "Mexico", "abbr": "MEX"},
]


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "name": "Rubén Vargas", "squadId": 10, "position": "MID"},
        {"id": 2, "name": "Granit Xhaka", "squadId": 10, "position": "MID"},
        {"id": 3, "name": "Breel Embolo", "squadId": 10, "position": "FWD"},
        {"id": 4, "name": "Christian Pulisic", "squadId": 47, "position": "MID"},
        {"id": 5, "name": "César Montes", "squadId": 28, "position": "DEF"},
    ])


def _named(text: str) -> list[dict]:
    return [{"Locale": "en-GB", "Description": text}]


def _player(idp, name, status):
    return {"IdPlayer": idp, "PlayerName": _named(name), "Status": status}


# --------------------------------------------------------------- lineups

def test_parse_lineup_factors_starter_bench_and_omitted():
    match = {
        "HomeTeam": {
            "IdCountry": "SUI",
            "Players": [
                _player("a", "Ruben Vargas", fifa_match.STARTER_STATUS),
                _player("b", "Granit Xhaka", fifa_match.BENCH_STATUS),
            ],
        },
        # USA lineup not published yet: must be left untouched.
        "AwayTeam": {"IdCountry": "USA", "Players": []},
    }
    factors = fifa_match.parse_lineup_factors(match, SQUADS, _players())
    assert factors[1] == 1.0                       # Vargas named starter
    assert factors[2] == fifa_match.BENCH_FACTOR   # Xhaka named sub
    assert factors[3] == 0.0                       # Embolo (SUI) left out of the 26
    assert 4 not in factors                        # USA lineup empty -> untouched
    assert 5 not in factors                        # Mexico not in this match


# --------------------------------------------------------------- bookings

def test_classify_bookings_yellows_reds_and_double_yellow():
    bookings = [
        {"IdPlayer": "x", "Card": fifa_match.YELLOW_CARD},          # single yellow
        {"IdPlayer": "y", "Card": fifa_match.YELLOW_CARD},          # two yellows
        {"IdPlayer": "y", "Card": fifa_match.YELLOW_CARD},          # -> sending off
        {"IdPlayer": "z", "Card": 3},                              # straight red
        {"IdCoach": "c", "Card": fifa_match.YELLOW_CARD},          # coach -> ignored
    ]
    singles, reds = fifa_match.classify_bookings(bookings)
    assert singles == {"x"}
    assert reds == {"y", "z"}


def test_normalize_match_collects_present_and_cards():
    match = {
        "HomeTeam": {
            "IdCountry": "SUI",
            "Players": [
                _player("a", "Ruben Vargas", 1),
                _player("b", "Granit Xhaka", 2),
            ],
            "Bookings": [{"IdPlayer": "a", "Card": 3}],
        },
        "AwayTeam": {"IdCountry": "USA", "Players": [], "Bookings": []},
    }
    entries = fifa_match.normalize_match(match, round_no=1)
    home = next(e for e in entries if e["abbr"] == "SUI")
    assert home["present"] == {"a", "b"}
    assert home["reds"] == {"a"}
    assert home["names"]["a"] == ("Ruben Vargas", "SUI")


# ------------------------------------------------------------ suspensions

def _entry(rnd, present, yellows=(), reds=()):
    return {
        "abbr": "SUI", "round": rnd, "present": set(present),
        "yellows": set(yellows), "reds": set(reds),
    }


def test_red_card_in_last_match_bans_next_round():
    entries = [_entry(1, present={"p1", "p2"}, reds={"p1"})]
    banned = fifa_match.derive_suspensions(
        entries, yellow_ban=2, reset_after_rounds=(3, 6)
    )
    assert banned == {"p1"}


def test_served_ban_is_cleared_when_player_misses_a_later_match():
    entries = [
        _entry(1, present={"p1", "p2"}, reds={"p1"}),
        _entry(2, present={"p2"}),  # p1 absent -> ban served
    ]
    banned = fifa_match.derive_suspensions(
        entries, yellow_ban=2, reset_after_rounds=(3, 6)
    )
    assert banned == set()


def test_two_accumulated_yellows_ban_next_round():
    entries = [
        _entry(1, present={"p1"}, yellows={"p1"}),
        _entry(2, present={"p1"}, yellows={"p1"}),  # second yellow -> ban pending
    ]
    banned = fifa_match.derive_suspensions(
        entries, yellow_ban=2, reset_after_rounds=(3, 6)
    )
    assert banned == {"p1"}


def test_yellow_tally_resets_after_the_group_stage():
    entries = [
        _entry(1, present={"p1"}, yellows={"p1"}),   # 1 yellow, group stage
        _entry(4, present={"p1"}, yellows={"p1"}),   # round > 3 -> tally reset
    ]
    banned = fifa_match.derive_suspensions(
        entries, yellow_ban=2, reset_after_rounds=(3, 6)
    )
    assert banned == set()


def test_resolve_to_fantasy_bridges_by_abbr_then_name():
    out = fifa_match.resolve_to_fantasy(
        {"a"}, {"a": ("Ruben Vargas", "SUI")}, SQUADS, _players()
    )
    assert out == {1: 0.0}


def test_resolve_to_fantasy_drops_unmappable():
    out = fifa_match.resolve_to_fantasy(
        {"q"}, {"q": ("Nobody At All", "SUI")}, SQUADS, _players()
    )
    assert out == {}
