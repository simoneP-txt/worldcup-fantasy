import pandas as pd

from wcfantasy.optimizer.captain import captain_timing_note
from wcfantasy.optimizer.squad import SquadSolution

ROUNDS = [{
    "id": 1,
    "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 3,
         "date": "2026-06-11T20:00:00+01:00"},
        {"homeSquadId": 2, "awaySquadId": 4,
         "date": "2026-06-13T20:00:00+01:00"},
    ],
}]


def _df(runner_ep: float) -> pd.DataFrame:
    rows = [
        {"id": 1, "name": "Early Star", "squadId": 1, "position": "FWD",
         "ep_match": 7.0},
        {"id": 2, "name": "Late Star", "squadId": 2, "position": "MID",
         "ep_match": runner_ep},
    ]
    rows += [
        {"id": 10 + k, "name": f"filler{k}", "squadId": 3 + k,
         "position": "MID", "ep_match": 3.0}
        for k in range(9)
    ]
    return pd.DataFrame(rows)


def _solution(captain_id: int, df: pd.DataFrame) -> SquadSolution:
    ids = [int(i) for i in df["id"]]
    return SquadSolution(ids, ids[:11], captain_id, 0.0, 0.0)


def test_reassures_when_captain_plays_first():
    df = _df(runner_ep=6.8)
    note = captain_timing_note(df, _solution(1, df), ROUNDS, 1)
    assert "plays before" in note


def test_suggests_earlier_armband_when_captain_plays_late():
    df = _df(runner_ep=7.2)  # ILP captain is the late player now
    note = captain_timing_note(df, _solution(2, df), ROUNDS, 1)
    assert "EARLIER" in note and "Early Star" in note


def test_silent_when_the_gap_is_clear():
    df = _df(runner_ep=3.0)  # no contender close to the captain
    note = captain_timing_note(df, _solution(1, df), ROUNDS, 1)
    assert note is None
