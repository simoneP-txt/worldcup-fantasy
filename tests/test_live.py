from datetime import datetime

import pandas as pd
import pytest

from wcfantasy import live
from wcfantasy.models import player_points
from wcfantasy.state import TeamState


# ------------------------------------------------------------ feed parsing

def test_match_phase_mapping():
    # 'complete' and 'playing' are the values the real feed uses
    assert live.match_phase({"status": "complete", "homeScore": 2}) == "completed"
    assert live.match_phase({"status": "completed", "homeScore": 2}) == "completed"
    assert live.match_phase({"status": "playing", "homeScore": 1}) == "live"
    assert live.match_phase({"status": "live", "homeScore": 1}) == "live"
    assert live.match_phase({"status": "scheduled"}) == "scheduled"
    assert live.match_phase({}) == "scheduled"
    # unknown status but a score present: assume in progress
    assert live.match_phase({"status": "2H", "homeScore": 0}) == "live"


def test_current_live_round():
    rounds = [
        {"id": 1, "tournaments": [{"status": "completed", "homeScore": 1}] * 4},
        {"id": 2, "tournaments": [
            {"status": "completed", "homeScore": 0},
            {"status": "scheduled"},
        ]},
        {"id": 3, "tournaments": [{"status": "scheduled"}] * 2},
    ]
    assert live.current_live_round(rounds)["id"] == 2

    all_done_or_pending = [
        {"id": 1, "tournaments": [{"status": "completed", "homeScore": 1}]},
        {"id": 2, "tournaments": [{"status": "scheduled"}]},
    ]
    assert live.current_live_round(all_done_or_pending) is None


def test_player_round_points_schemas():
    assert live.player_round_points({"roundPoints": {"2": 7}}, 2) == 7
    assert live.player_round_points(
        {"roundPoints": [{"round": 2, "points": 5}]}, 2
    ) == 5
    assert live.player_round_points({"roundPoints": [3, 8]}, 2) == 8
    assert live.player_round_points({"roundPoints": [3]}, 2) is None
    assert live.player_round_points({}, 2) is None
    assert live.player_round_points(None, 2) is None


# ------------------------------------------------------------ formations

def test_formation_valid_after_swap():
    xi_442 = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
    assert live.formation_valid_after_swap(xi_442, "DEF", "MID")   # 3-5-2
    assert live.formation_valid_after_swap(xi_442, "FWD", "DEF")   # 5-4-1
    assert live.formation_valid_after_swap(xi_442, "GK", "GK")
    assert not live.formation_valid_after_swap(xi_442, "GK", "DEF")
    assert not live.formation_valid_after_swap(xi_442, "DEF", "GK")

    xi_352 = ["GK"] + ["DEF"] * 3 + ["MID"] * 5 + ["FWD"] * 2
    assert not live.formation_valid_after_swap(xi_352, "DEF", "MID")  # 2-6-2


# ------------------------------------------------------------ advice EV

@pytest.fixture
def squad_df():
    rows = []
    # 11 starters (squads 1-11), 4 bench (squads 12-15); ids = squadId
    for i in range(1, 16):
        pos = "GK" if i in (1, 15) else "DEF" if i <= 5 else "MID" if i <= 10 else "FWD"
        rows.append({
            "id": i, "name": f"p{i}", "squadName": f"team{i}", "squadId": i,
            "position": pos, "ep_match": 3.0 + (0.1 * i),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def team():
    return TeamState(
        player_ids=list(range(1, 16)),
        starters=list(range(1, 12)),
        captain_id=11,
        vice_captain_id=10,
        bench_order=[12, 13, 14],
        bench_gk_id=15,
    )


def test_captain_switch_when_candidate_beats_banked_points(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"  # captain played
    points = {11: 2.0}  # disappointing return
    lines = live.captain_advice(squad_df, team, phases, points, change_cost=0.0)
    assert any(line.startswith("SWITCH") for line in lines)


def test_captain_kept_when_banked_points_beat_candidates(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"
    points = {11: 12.0}  # captain hauled
    lines = live.captain_advice(squad_df, team, phases, points, change_cost=0.0)
    assert any(line.startswith("KEEP") for line in lines)


def test_captain_change_cost_flips_marginal_switch(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"
    # best candidate is p10 with ep 4.0: marginal vs 3.8 banked
    points = {11: 3.8}
    cheap = live.captain_advice(squad_df, team, phases, points, change_cost=0.0)
    pricey = live.captain_advice(squad_df, team, phases, points, change_cost=1.0)
    assert any(line.startswith("SWITCH") for line in cheap)
    assert any(line.startswith("KEEP") for line in pricey)


def test_captain_locked_while_playing(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "live"
    lines = live.captain_advice(squad_df, team, phases, points={}, change_cost=0.0)
    assert any("locked" in line for line in lines)


def test_substitution_suggested_for_underperformer(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[6] = "completed"  # MID starter done with 1 pt
    points = {6: 1.0}
    lines = live.substitution_advice(squad_df, team, phases, points, change_cost=0.0)
    assert any(line.startswith("SUB p6") for line in lines)


def test_dnp_left_to_auto_subs_when_no_manual_change_yet(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[6] = "completed"
    points = {6: 0.0}  # likely DNP
    lines = live.substitution_advice(squad_df, team, phases, points, change_cost=0.5)
    assert any("auto-sub" in line for line in lines)
    assert not any(line.startswith("SUB ") for line in lines)


def test_no_substitution_when_starter_banked_enough(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[6] = "completed"
    points = {6: 9.0}
    lines = live.substitution_advice(squad_df, team, phases, points, change_cost=0.0)
    assert lines == []


def test_advice_carries_the_kickoff_deadline(squad_df, team):
    # every suggestion expires when the incoming player's match starts
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"
    phases[6] = "completed"
    points = {11: 2.0, 6: 1.0}
    kickoffs = {
        i: datetime.fromisoformat("2026-06-13T18:00:00+01:00")
        for i in range(1, 16)
    }
    cap = live.captain_advice(squad_df, team, phases, points, 0.0, kickoffs)
    subs = live.substitution_advice(squad_df, team, phases, points, 0.0, kickoffs)
    # kickoff 18:00 +01:00 shown in Italian time (CEST, +02:00) -> 19:00
    assert any("13 Jun 19:00" in line for line in cap)
    assert any("13 Jun 19:00" in line for line in subs)


def test_round_outlook_lists_only_players_still_to_play(squad_df, team):
    # add the columns point_probability needs (price/ownership depth chart)
    squad_df = squad_df.assign(price=6.0, percentSelected=10.0)
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"   # captain already played -> excluded
    phases[6] = "live"         # in progress -> excluded
    lines = live.round_outlook(squad_df, team, phases, availability=None)
    names = " ".join(lines)
    assert "p11" not in names and "p6" not in names
    assert "p1" in names  # a starter still to play is listed
    assert all("to play" in line for line in lines)


def test_round_outlook_discounts_an_injured_player(squad_df, team):
    squad_df = squad_df.assign(price=6.0, percentSelected=10.0)
    phases = {i: "scheduled" for i in range(1, 16)}
    healthy = player_points.starter_probability(squad_df, availability=None)
    hurt = player_points.starter_probability(squad_df, availability={1: 0.2})
    healthy.index = squad_df["id"]
    hurt.index = squad_df["id"]
    assert hurt[1] < healthy[1]
    assert hurt[2] == healthy[2]


def test_round_kickoffs_parses_dates():
    rnd = {"tournaments": [
        {"homeSquadId": 1, "awaySquadId": 2,
         "date": "2026-06-11T20:00:00+01:00"},
        {"homeSquadId": 3, "awaySquadId": 4, "date": None},  # TBD match
    ]}
    kickoffs = live.round_kickoffs(rnd)
    assert set(kickoffs) == {1, 2}
    assert kickoffs[1].hour == 20
