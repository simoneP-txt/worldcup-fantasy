import pandas as pd

from wcfantasy.optimizer.captain import (
    _progressing_starters,
    booster_advice,
    captain_by_timing,
    captain_timing_note,
    planned_activation,
)
from wcfantasy.optimizer.squad import SquadSolution
from wcfantasy.state import TeamState

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


# ----------------------------------------------------- captain by kickoff order

def test_captain_by_timing_prefers_earlier_kickoff_among_near_equals():
    # Early Star (id 1, plays 11 Jun) vs Late Star (id 2, 13 Jun), within
    # tolerance. The ILP captained the late one; timing moves it to the early.
    df = _df(runner_ep=6.8)
    assert captain_by_timing(df, _solution(2, df), ROUNDS, 1) == 1


def test_captain_by_timing_keeps_ilp_pick_without_a_near_equal_contender():
    df = _df(runner_ep=3.0)  # only the early star is a contender
    assert captain_by_timing(df, _solution(1, df), ROUNDS, 1) == 1


def test_captain_by_timing_does_not_downgrade_a_clear_best():
    # Late Star is clearly the best and below-tolerance Early Star must not
    # steal the armband just because he plays first.
    df = _df(runner_ep=10.0)
    assert captain_by_timing(df, _solution(2, df), ROUNDS, 1) == 2


# ----------------------------------------------------- progressing starters

def test_progressing_starters_uses_p_advance_when_present():
    # P(win next match) > 0.5 is the signal; next_round is irrelevant here.
    starters = pd.DataFrame({
        "p_advance": [0.60, 0.55, 0.40, 0.51],
        "future_rounds": [0.0, 0.0, 0.0, 0.0],
    })
    assert _progressing_starters(starters, next_round=7) == 3


def test_progressing_starters_falls_back_to_future_rounds():
    # No p_advance column: old heuristic, SF threshold 8 - 7 + 0.5 = 1.5.
    starters = pd.DataFrame({"future_rounds": [2.0, 1.0, 1.6]})
    assert _progressing_starters(starters, next_round=7) == 2


# ------------------------------------------------------------ booster advice

def _squad_df(cap_ep: float = 8.0, future: float = 2.0, n_dead: int = 0) -> pd.DataFrame:
    """A full 15-man squad (ids 1..15), captain = id 11, plus two players
    OUTSIDE the squad (ids 100..101) so the 12th Man pick has candidates.
    `future` sets future_rounds for everyone; `n_dead` marks the first n
    squad players eliminated (Wildcard fuel)."""
    rows = []
    for i in range(1, 16):
        pos = "GK" if i in (1, 15) else "DEF" if i <= 5 else "MID" if i <= 10 else "FWD"
        rows.append({
            "id": i, "name": f"p{i}", "squadName": f"t{i}", "squadId": i,
            "position": pos, "ep_match": 4.0,
            "future_rounds": future, "isEliminated": False,
        })
    for i in (100, 101):  # outside the squad (12th Man pool)
        rows.append({
            "id": i, "name": f"out{i}", "squadName": f"t{i}", "squadId": i,
            "position": "FWD", "ep_match": 5.0,
            "future_rounds": future, "isEliminated": False,
        })
    df = pd.DataFrame(rows)
    df.loc[df["id"] == 11, "ep_match"] = cap_ep  # captain
    for i in range(1, n_dead + 1):
        df.loc[df["id"] == i, "isEliminated"] = True
    return df


def _full_solution(captain_id: int = 11) -> SquadSolution:
    ids = list(range(1, 16))  # the 15-man squad, not the 12th-Man outsiders
    return SquadSolution(ids, ids[:11], captain_id, 0.0, 0.0)


def test_qualification_takes_priority_over_max_captain_at_sf():
    # SF, strong captain (>5.5) AND most starters progressing: the planned
    # Qualification booster must win, Maximum Captain must NOT preempt it.
    df = _squad_df(cap_ep=8.0, future=2.0, n_dead=0)
    lines = booster_advice(df, TeamState(), _full_solution(), next_round=7)
    assert any(l.startswith("ACTIVATE") and "Qualification" in l for l in lines)
    assert not any("ACTIVATE Maximum Captain" in l for l in lines)


def test_wildcard_takes_priority_over_max_captain_with_dead_wood():
    # R16 with 4 eliminated squad players: the rebuild beats a strong captain.
    df = _squad_df(cap_ep=8.0, future=2.0, n_dead=4)
    lines = booster_advice(df, TeamState(), _full_solution(), next_round=5)
    assert any(l.startswith("ACTIVATE Wildcard") for l in lines)
    assert not any("ACTIVATE Maximum Captain" in l for l in lines)


def test_max_captain_activates_when_no_higher_priority_booster():
    # QF, teams alive but not favourites (future 1.0: not dead wood, below
    # the 2.5 progression threshold) and no dead wood: Maximum Captain is it.
    df = _squad_df(cap_ep=8.0, future=1.0, n_dead=0)
    lines = booster_advice(df, TeamState(), _full_solution(), next_round=6)
    assert any(l.startswith("ACTIVATE Maximum Captain") for l in lines)


def test_planned_activation_priority_order():
    df = _squad_df(cap_ep=8.0, future=2.0, n_dead=4)
    starters = df[df["id"].isin(range(1, 12))]
    cap = df.loc[df["id"] == 11].iloc[0]
    all_three = ["qualification", "wildcard", "maximum_captain"]
    # SF: qualification wins even with dead wood and a strong captain
    assert planned_activation(all_three, 7, cap, starters, df) == "qualification"
    # R16: wildcard wins over Maximum Captain
    assert planned_activation(["wildcard", "maximum_captain"], 5, cap, starters, df) == "wildcard"
    # QF, no dead wood, nothing progressing: Maximum Captain
    quiet = _squad_df(cap_ep=8.0, future=0.0, n_dead=0)
    quiet_starters = quiet[quiet["id"].isin(range(1, 12))]
    quiet_cap = quiet.loc[quiet["id"] == 11].iloc[0]
    assert planned_activation(["maximum_captain"], 6, quiet_cap, quiet_starters, quiet) == "maximum_captain"
