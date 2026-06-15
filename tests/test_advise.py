import pandas as pd

from wcfantasy import advise, state
from wcfantasy.advise import fixture_schedule, next_round_number, squad_comparison
from wcfantasy.optimizer.squad import SquadSolution


def test_next_round_number_skips_done_and_live_rounds():
    # the real feed says 'complete' (not 'completed') and 'playing'; a live
    # round is locked at its first kickoff, so advise looks past it too
    rounds = [
        {"id": 1, "status": "complete"},
        {"id": 2, "status": "playing"},
        {"id": 3, "status": "scheduled"},
    ]
    assert next_round_number(rounds) == 3  # round 2 live = locked, look ahead
    rounds[1]["status"] = "complete"
    assert next_round_number(rounds) == 3
    rounds[2]["status"] = "complete"
    assert next_round_number(rounds) == 3  # all done: fall back to last round

ROUNDS = [{
    "id": 1,
    "tournaments": [
        {"homeSquadId": 2, "awaySquadId": 9, "date": "2026-06-13T02:00:00+01:00",
         "homeSquadName": "France", "awaySquadName": "Senegal"},
        {"homeSquadId": 1, "awaySquadId": 3, "date": "2026-06-11T20:00:00+01:00",
         "homeSquadName": "Spain", "awaySquadName": "Qatar"},
        {"homeSquadId": 4, "awaySquadId": 5, "date": "2026-06-12T20:00:00+01:00",
         "homeSquadName": "Brazil", "awaySquadName": "Ghana"},  # none of mine
    ],
}]


def test_fixture_schedule_filters_sorts_and_converts_to_italian_time():
    df = pd.DataFrame([
        {"id": 1, "name": "Yamal", "squadId": 1},
        {"id": 2, "name": "Raya", "squadId": 1},
        {"id": 3, "name": "Mbappe", "squadId": 2},
    ])
    solution = SquadSolution([1, 2, 3], [1, 2, 3], 1, 0.0, 0.0)
    lines = fixture_schedule(df, solution, vice_id=3, rounds=ROUNDS, next_round=1)

    assert len(lines) == 2  # Brazil-Ghana dropped: no players of mine
    # chronological, with +01:00 feed times shifted to CEST (+02:00)
    assert lines[0].startswith("Thu 11 Jun 21:00")
    assert "Spain-Qatar" in lines[0]
    assert "Yamal (C)" in lines[0] and "Raya" in lines[0]
    assert lines[1].startswith("Sat 13 Jun 03:00")
    assert "Mbappe (V)" in lines[1]


def test_squad_comparison_overlap_gap_add_and_drop():
    df = pd.DataFrame([
        {"id": 1, "name": "A", "squadName": "T1", "position": "GK", "ep_match": 4.0},
        {"id": 2, "name": "B", "squadName": "T2", "position": "DEF", "ep_match": 5.0},
        {"id": 3, "name": "C", "squadName": "T3", "position": "MID", "ep_match": 6.0},
        {"id": 4, "name": "D", "squadName": "T4", "position": "FWD", "ep_match": 2.0},
    ])
    ideal = SquadSolution([1, 2, 3], [1, 2, 3], 3, 0.0, 0.0)
    lines = squad_comparison(df, ideal, current_ids=[1, 2, 4])
    text = " ".join(lines)
    assert "own 2/15" in text                       # overlap {1, 2}
    assert "gap +4.0" in text                        # 15 ideal - 11 current
    assert any(l.strip().startswith("+ C") for l in lines)  # bring in C
    assert any(l.strip().startswith("- D") for l in lines)  # drop D


def test_squad_comparison_handles_id_missing_from_feed():
    df = pd.DataFrame([
        {"id": 1, "name": "A", "squadName": "T1", "position": "GK", "ep_match": 4.0},
    ])
    ideal = SquadSolution([1], [1], 1, 0.0, 0.0)
    lines = squad_comparison(df, ideal, current_ids=[1, 999])  # 999 not in feed
    assert any("999" in l and "not in feed" in l for l in lines)


def test_use_booster_records_dedups_and_rejects_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "MY_TEAM_FILE", tmp_path / "my_team.json")
    advise.use_booster("wildcard")
    assert state.load().boosters_used == ["wildcard"]
    advise.use_booster("WILDCARD")  # case-insensitive, no duplicate
    assert state.load().boosters_used == ["wildcard"]
    advise.use_booster("maximum-captain")  # dash -> underscore
    assert state.load().boosters_used == ["wildcard", "maximum_captain"]
    advise.use_booster("nonsense")  # unknown booster ignored
    assert state.load().boosters_used == ["wildcard", "maximum_captain"]
