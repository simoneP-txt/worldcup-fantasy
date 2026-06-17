import pandas as pd

from wcfantasy.optimizer.captain import (
    _progressing_starters,
    booster_advice,
    captain_timing_note,
    plan_captain,
    planned_activation,
    recommend_captain,
    sequential_captain_value,
)
from wcfantasy.optimizer.squad import SquadSolution
from wcfantasy.state import TeamState

ROUNDS = [{
    "id": 1,
    "tournaments": [
        # squad 1 (Early) plays 11 Jun, squad 2 (Late) plays 13 Jun
        {"homeSquadId": 1, "awaySquadId": 3,
         "date": "2026-06-11T20:00:00+01:00"},
        {"homeSquadId": 2, "awaySquadId": 4,
         "date": "2026-06-13T20:00:00+01:00"},
    ],
}]


def _df2(early_ep: float, late_ep: float) -> pd.DataFrame:
    """Two captain candidates (Early plays first, Late plays later) plus
    nine fillers with no kickoff so only the two matter."""
    rows = [
        {"id": 1, "name": "Early Star", "squadId": 1, "position": "FWD",
         "ep_match": early_ep},
        {"id": 2, "name": "Late Star", "squadId": 2, "position": "FWD",
         "ep_match": late_ep},
    ]
    rows += [
        {"id": 10 + k, "name": f"filler{k}", "squadId": 30 + k,
         "position": "MID", "ep_match": 2.0}
        for k in range(9)
    ]
    return pd.DataFrame(rows)


def _solution(captain_id: int, df: pd.DataFrame) -> SquadSolution:
    ids = [int(i) for i in df["id"]]
    return SquadSolution(ids, ids[:11], captain_id, 0.0, 0.0)


# ----------------------------------------------------- option-value captaincy

def test_sequential_captain_value_beats_single_and_is_monotone():
    # the option to roll the armband can only add value
    assert sequential_captain_value([7.0]) == 7.0
    assert sequential_captain_value([5.0, 7.0]) >= 7.0
    assert sequential_captain_value([5.0, 7.0]) >= sequential_captain_value([7.0])


def test_recommend_captain_starts_earlier_for_option_value():
    # Late Star is the higher EP but plays later; captaining the slightly
    # weaker Early Star first (and rolling to the Late Star if he flops) wins.
    df = _df2(early_ep=5.8, late_ep=7.0)
    plan = recommend_captain(df, _solution(2, df), ROUNDS, 1)
    assert plan.initial == 1 and plan.anchor == 2 and plan.premium > 0


def test_recommend_captain_keeps_best_when_it_plays_first():
    df = _df2(early_ep=7.0, late_ep=5.8)  # the best already kicks off first
    assert recommend_captain(df, _solution(1, df), ROUNDS, 1).initial == 1


def test_recommend_captain_ignores_a_too_weak_early_option():
    # Early Star at 3.0 is below the contender floor of 0.7*8.0: the armband
    # stays on the clear best, not wasted on a near-certain costly roll.
    df = _df2(early_ep=3.0, late_ep=8.0)
    assert recommend_captain(df, _solution(2, df), ROUNDS, 1).initial == 2


def test_plan_captain_falls_back_without_kickoff_times():
    plan = plan_captain([(1, 5.8, None), (2, 7.0, None)])
    assert plan.initial == 2  # no times to roll on: captain the best outright


def test_captain_timing_note_explains_option_when_starting_earlier():
    df = _df2(early_ep=5.8, late_ep=7.0)
    note = captain_timing_note(df, _solution(2, df), ROUNDS, 1)
    assert note is not None and "Early Star" in note and "FIRST" in note


def test_captain_timing_note_silent_when_best_plays_first():
    df = _df2(early_ep=7.0, late_ep=5.8)
    assert captain_timing_note(df, _solution(1, df), ROUNDS, 1) is None


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
