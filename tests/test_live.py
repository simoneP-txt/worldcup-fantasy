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


def test_round_outlook_matches_groups_by_fixture(squad_df, team):
    # captain id 11 (squad 11), vice id 10 (squad 10), bench id 12 (squad 12)
    squad_df = squad_df.assign(price=6.0, percentSelected=10.0)
    rnd = {"id": 1, "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 12, "homeSquadName": "team1",
         "awaySquadName": "team12", "date": "2026-06-12T20:00:00+01:00",
         "status": "scheduled"},
        {"homeSquadId": 10, "awaySquadId": 11, "homeSquadName": "team10",
         "awaySquadName": "team11", "date": "2026-06-11T20:00:00+01:00",
         "status": "scheduled"},
    ]}
    kickoffs = live.round_kickoffs(rnd)
    matches = live.round_outlook_matches(
        squad_df, team, rnd, availability=None, kickoffs=kickoffs
    )
    # chronological: the 10-11 match (11 Jun) precedes the 1-12 match (12 Jun)
    assert [m["home"] for m in matches] == ["team10", "team1"]
    assert matches[0]["away"] == "team11"
    assert matches[0]["kickoff"].endswith("21:00")  # CEST = +02:00
    cap = next(p for m in matches for p in m["players"] if p["name"] == "p11")
    assert cap["is_captain"] and cap["role"] == "XI"
    vice = next(p for m in matches for p in m["players"] if p["name"] == "p10")
    assert vice["is_vice"]
    sub = next(p for m in matches for p in m["players"] if p["name"] == "p12")
    assert sub["role"] == "bench"
    assert all("p_play" in p for m in matches for p in m["players"])


def test_round_outlook_matches_skips_played_fixtures(squad_df, team):
    squad_df = squad_df.assign(price=6.0, percentSelected=10.0)
    rnd = {"id": 1, "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 2, "homeSquadName": "team1",
         "awaySquadName": "team2", "date": "2026-06-11T20:00:00+01:00",
         "status": "complete", "homeScore": 1, "awayScore": 0},
        {"homeSquadId": 3, "awaySquadId": 4, "homeSquadName": "team3",
         "awaySquadName": "team4", "date": "2026-06-12T20:00:00+01:00",
         "status": "scheduled"},
    ]}
    kickoffs = live.round_kickoffs(rnd)
    matches = live.round_outlook_matches(
        squad_df, team, rnd, availability=None, kickoffs=kickoffs
    )
    # only the still-to-play fixture survives
    assert len(matches) == 1 and matches[0]["home"] == "team3"


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


# ------------------------------------------------------------ recording

def test_resolve_my_player_matches_within_squad():
    df = pd.DataFrame([
        {"id": 1, "name": "Lionel Messi", "squadName": "ARG", "squadId": 1, "position": "FWD"},
        {"id": 2, "name": "Kylian Mbappe", "squadName": "FRA", "squadId": 2, "position": "FWD"},
        {"id": 3, "name": "Jude Bellingham", "squadName": "ENG", "squadId": 3, "position": "MID"},
        {"id": 9, "name": "Someone Else", "squadName": "XXX", "squadId": 9, "position": "DEF"},
    ])
    t = TeamState(player_ids=[1, 2, 3])  # id 9 is NOT in my squad
    assert live.resolve_my_player("Messi", df, t) == 1            # substring
    assert live.resolve_my_player("kylian mbappe", df, t) == 2    # exact, case-insensitive
    assert live.resolve_my_player("Bellinghma", df, t) == 3       # fuzzy typo
    with pytest.raises(ValueError):
        live.resolve_my_player("Someone Else", df, t)             # not in MY squad
    with pytest.raises(ValueError):
        live.resolve_my_player("zzzzz", df, t)                    # no match


def test_record_substitution_outfield_updates_starters_and_bench(squad_df, team):
    msg = live.record_substitution(team, out_id=6, in_id=12, df=squad_df)
    assert 12 in team.starters and 6 not in team.starters
    assert 6 in team.bench_order and 12 not in team.bench_order
    assert team.bench_gk_id == 15  # untouched
    assert "OUT" in msg and "IN" in msg


def test_record_substitution_gk_updates_bench_gk(squad_df, team):
    live.record_substitution(team, out_id=1, in_id=15, df=squad_df)
    assert 15 in team.starters and 1 not in team.starters
    assert team.bench_gk_id == 1


def test_record_substitution_rejects_illegal_swap(squad_df, team):
    with pytest.raises(ValueError):
        live.record_substitution(team, out_id=1, in_id=12, df=squad_df)  # GK <-> FWD


def test_record_substitution_rejects_non_members(squad_df, team):
    with pytest.raises(ValueError):
        live.record_substitution(team, out_id=12, in_id=13, df=squad_df)  # out is a bench player
    with pytest.raises(ValueError):
        live.record_substitution(team, out_id=6, in_id=7, df=squad_df)    # in is a starter


def test_record_captain_sets_captain(squad_df, team):
    live.record_captain(team, 10, squad_df)
    assert team.captain_id == 10


def test_record_captain_rejects_non_starter(squad_df, team):
    with pytest.raises(ValueError):
        live.record_captain(team, 12, squad_df)  # 12 is on the bench


def test_record_captain_swaps_vice_on_conflict(squad_df, team):
    # captain=11, vice=10; promoting the vice to captain moves the old
    # captain (11) into the vice slot so the two never coincide
    live.record_captain(team, 10, squad_df)
    assert team.captain_id == 10 and team.vice_captain_id == 11


def test_apply_recorded_changes_sub_and_captain_marks_round(squad_df, team):
    lines = live.apply_recorded_changes(
        team, squad_df, round_no=2, sub_out="p6", sub_in="p12", captain="p10"
    )
    assert team.manual_changes_round == 2
    assert 12 in team.starters and 6 in team.bench_order
    assert team.captain_id == 10
    assert lines


def test_apply_recorded_changes_requires_both_sub_args(squad_df, team):
    with pytest.raises(ValueError):
        live.apply_recorded_changes(team, squad_df, 2, sub_out="p6")


def test_apply_recorded_changes_bare_mark_only_flags_the_round(squad_df, team):
    lines = live.apply_recorded_changes(team, squad_df, 3, mark_changed=True)
    assert team.manual_changes_round == 3
    assert any("manual change" in line for line in lines)
    # nothing structural moved
    assert team.captain_id == 11 and team.starters == list(range(1, 12))


def test_apply_recorded_changes_rejects_captain_equals_vice(squad_df, team):
    with pytest.raises(ValueError):
        live.apply_recorded_changes(team, squad_df, 2, captain="p9", vice="p9")


def test_recorded_captain_change_stops_resuggestion(squad_df, team):
    # Reproduces the reported bug: live re-proposes the same captain switch.
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[11] = "completed"   # captain done with a low return
    points = {11: 2.0}
    before = live.captain_advice(squad_df, team, phases, points, change_cost=0.0)
    assert any(line.startswith("SWITCH") for line in before)
    # record the executed switch to the suggested candidate (best scheduled = p10)
    live.apply_recorded_changes(team, squad_df, round_no=1, captain="p10")
    after = live.captain_advice(squad_df, team, phases, points, change_cost=0.0)
    assert not any(line.startswith("SWITCH") for line in after)
    assert any("not played yet" in line for line in after)


def test_recorded_sub_stops_resuggestion(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    phases[6] = "completed"
    points = {6: 1.0}
    before = live.substitution_advice(squad_df, team, phases, points, change_cost=0.0)
    assert any(line.startswith("SUB p6") for line in before)
    live.apply_recorded_changes(team, squad_df, round_no=1, sub_out="p6", sub_in="p12")
    after = live.substitution_advice(squad_df, team, phases, points, change_cost=0.0)
    assert not any(line.startswith("SUB p6") for line in after)


# ------------------------------------------------- proactive (sequential) captaincy

def test_live_proactive_suggests_earlier_captain(squad_df, team):
    # Captain (id 11) has the highest EP but kicks off LAST; id 10 is nearly
    # as good and plays earlier -> captain him first, roll to 11 if he flops.
    phases = {i: "scheduled" for i in range(1, 16)}
    kickoffs = {
        10: datetime.fromisoformat("2026-06-20T18:00:00+00:00"),
        11: datetime.fromisoformat("2026-06-21T18:00:00+00:00"),
    }
    lines = live.captain_advice(
        squad_df, team, phases, points={}, change_cost=0.0, kickoffs=kickoffs
    )
    assert any("MOVE the armband to p10" in line for line in lines)


def test_live_proactive_keeps_captain_when_he_plays_first(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    kickoffs = {
        11: datetime.fromisoformat("2026-06-20T18:00:00+00:00"),
        10: datetime.fromisoformat("2026-06-21T18:00:00+00:00"),
    }
    lines = live.captain_advice(
        squad_df, team, phases, points={}, change_cost=0.0, kickoffs=kickoffs
    )
    assert any("plays first among your strong options" in line for line in lines)


def test_live_proactive_nothing_to_do_without_kickoffs(squad_df, team):
    phases = {i: "scheduled" for i in range(1, 16)}
    lines = live.captain_advice(
        squad_df, team, phases, points={}, change_cost=0.0, kickoffs=None
    )
    assert any("nothing to do" in line for line in lines)
