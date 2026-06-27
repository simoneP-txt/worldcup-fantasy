"""Offline tests for the web service layer (no network, no server).

Synthetic FIFA feeds + stubbed heavy model deps (progression / availability /
club stats), mirroring how the existing model tests stay offline. Verifies the
structured shapes the web API serves and the local-state side effects.
"""

import json

import pandas as pd
import pytest

from wcfantasy import advise, fifa_client, service, settings, state
from wcfantasy.data_sources import availability, club_stats
from wcfantasy.data_sources import odds as odds_mod
from wcfantasy.models import player_points, progression


# --------------------------------------------------------------- synthetic data

def make_feeds(round_one="scheduled"):
    """8 nations x (2 GK, 5 DEF, 5 MID, 3 FWD) raw players + squads + 8 rounds.

    `round_one` selects round 1's match phases: 'scheduled' (nothing started)
    or 'live' (one completed + one scheduled, i.e. a live round)."""
    groups = "ABCDEFGH"
    squads, players = [], []
    pid = 1
    for i in range(8):
        sid = i + 1
        squads.append({
            "id": sid, "name": f"Team{sid}", "group": groups[i],
            "isEliminated": False, "abbr": f"T{sid}",
        })
        for pos, n, base in (("GK", 2, 5.0), ("DEF", 5, 4.5), ("MID", 5, 5.5), ("FWD", 3, 6.0)):
            for k in range(n):
                players.append({
                    "id": pid, "knownName": f"P{pid}", "squadId": sid,
                    "position": pos, "price": base + 0.2 * k, "status": "playing",
                    "percentSelected": 5.0,
                    "stats": {"totalPoints": 0, "avgPoints": 0.0, "form": 0.0, "roundPoints": {}},
                })
                pid += 1

    if round_one == "live":
        r1 = [
            {"homeSquadId": 1, "awaySquadId": 2, "homeSquadName": "Team1",
             "awaySquadName": "Team2", "date": "2026-06-12T18:00:00+00:00",
             "status": "complete", "homeScore": 1, "awayScore": 0},
            {"homeSquadId": 3, "awaySquadId": 4, "homeSquadName": "Team3",
             "awaySquadName": "Team4", "date": "2026-06-13T18:00:00+00:00",
             "status": "scheduled"},
        ]
    else:
        r1 = [
            {"homeSquadId": 1, "awaySquadId": 2, "homeSquadName": "Team1",
             "awaySquadName": "Team2", "date": "2026-06-12T18:00:00+00:00",
             "status": "scheduled"},
        ]
    rounds = [{"id": r, "status": "scheduled", "tournaments": []} for r in range(1, 9)]
    rounds[0]["tournaments"] = r1
    return players, squads, rounds


def install_feeds(monkeypatch, players, squads, rounds):
    def fake_load(feed):
        return {"players": players, "squads": squads, "rounds": rounds}[feed]
    monkeypatch.setattr(fifa_client, "load", fake_load)


def stub_heavy_model(monkeypatch):
    """Replace the network/CPU-heavy deps so compute_advice runs offline."""
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {})
    monkeypatch.setattr(availability, "availability_factors", lambda *a, **k: {})
    monkeypatch.setattr(availability, "observed_start_factors", lambda *a, **k: {})
    monkeypatch.setattr(club_stats, "load", lambda: None)

    def fake_curve(players_df, squads, rounds, *a, **k):
        progression.last_run_info.update(model="test-model")
        sids = [s["id"] for s in squads]
        data = {sid: {r: 0.5 for r in range(1, 9)} for sid in sids}
        return pd.DataFrame.from_dict(data, orient="index").rename_axis("squadId")

    monkeypatch.setattr(progression, "survival_curve", fake_curve)


def make_team():
    return state.TeamState(
        player_ids=list(range(1, 16)),
        starters=list(range(1, 12)),
        captain_id=1,
        vice_captain_id=2,
        bench_order=[12, 13, 14],
        bench_gk_id=15,
        bank=8.0,
        round_no=1,
    )


@pytest.fixture
def state_file(monkeypatch, tmp_path):
    f = tmp_path / "my_team.json"
    monkeypatch.setattr(state, "MY_TEAM_FILE", f)
    return f


@pytest.fixture
def settings_file(monkeypatch, tmp_path):
    f = tmp_path / "ui_settings.json"
    monkeypatch.setattr(settings, "UI_SETTINGS_FILE", f)
    return f


# --------------------------------------------------------------- read-only

def test_team_snapshot_empty(state_file):
    snap = service.team_snapshot()
    assert snap["has_team"] is False
    assert snap["players"] == []
    # booster metadata is always present for the boosters screen
    assert {b["key"] for b in snap["boosters"]}
    assert any(b["irreversible"] for b in snap["boosters"])  # the wildcard


def test_team_snapshot_groups_and_roles(monkeypatch, state_file):
    players, squads, rounds = make_feeds()
    install_feeds(monkeypatch, players, squads, rounds)
    state.save(make_team())

    snap = service.team_snapshot()
    assert snap["has_team"] and snap["feeds_available"]
    assert len(snap["players"]) == 15
    by_id = {p["id"]: p for p in snap["players"]}
    assert by_id[1]["role"] == "CAPTAIN" and by_id[1]["is_captain"]
    assert by_id[2]["role"] == "VICE" and by_id[2]["is_vice"]
    assert by_id[15]["role"] == "bench GK" and by_id[15]["is_bench_gk"]
    assert by_id[12]["bench_slot"] == 1
    starters = [p for p in snap["players"] if p["is_starter"]]
    assert len(starters) == 11


def test_team_snapshot_feeds_missing(monkeypatch, state_file):
    def missing(feed):
        raise FileNotFoundError(feed)
    monkeypatch.setattr(fifa_client, "load", missing)
    state.save(make_team())
    snap = service.team_snapshot()
    assert snap["has_team"] is True
    assert snap["feeds_available"] is False


def test_status_line(monkeypatch, settings_file):
    players, squads, rounds = make_feeds()
    install_feeds(monkeypatch, players, squads, rounds)
    monkeypatch.setattr(player_points, "CALIBRATION_FILE", settings_file.parent / "nope.json")
    monkeypatch.setattr(odds_mod, "api_key", lambda: None)
    s = service.status_line()
    assert s["feeds_available"] is True
    assert s["freshness"]
    assert s["calibration_on"] is False
    assert s["odds_key_set"] is False
    assert s["language"] == "en"


def test_status_line_handles_missing_feeds(monkeypatch, settings_file):
    def missing(feed):
        raise FileNotFoundError(feed)
    monkeypatch.setattr(fifa_client, "load", missing)
    monkeypatch.setattr(odds_mod, "api_key", lambda: "abc")
    s = service.status_line()
    assert s["feeds_available"] is False
    assert s["freshness"] is None
    assert s["odds_key_set"] is True


# --------------------------------------------------------------- settings

def test_settings_round_trip(settings_file):
    assert service.get_settings()["language"] == "en"
    out = service.set_settings("it")
    assert out["language"] == "it"
    assert service.get_settings()["language"] == "it"


def test_set_settings_rejects_unknown_language(settings_file):
    with pytest.raises(ValueError):
        service.set_settings("xx")


# --------------------------------------------------------------- boosters

def test_set_booster_records_and_dedups(state_file):
    state.save(make_team())
    out = service.set_booster("wildcard")
    assert out["status"] == "recorded" and out["boosters_used"] == ["wildcard"]
    again = service.set_booster("WILDCARD")
    assert again["status"] == "already"
    bad = service.set_booster("nonsense")
    assert bad["status"] == "usage"


# --------------------------------------------------------------- apply

def test_apply_advice_applied(monkeypatch, state_file, tmp_path):
    players, squads, rounds = make_feeds()  # all scheduled -> no live round
    install_feeds(monkeypatch, players, squads, rounds)
    advice_file = tmp_path / "last_advice.json"
    monkeypatch.setattr(advise, "LAST_ADVICE_FILE", advice_file)
    advice_file.write_text(json.dumps({
        "round": 2, "squad_ids": list(range(1, 16)), "starter_ids": list(range(1, 12)),
        "captain_id": 1, "vice_captain_id": 2, "bench_order": [12, 13, 14],
        "bench_gk_id": 15, "bank": 3.0, "transfers_carried": 0,
    }))
    out = service.apply_advice(force=False)
    assert out["status"] == "applied" and out["round"] == 2
    assert state.load().player_ids == list(range(1, 16))


def test_apply_advice_blocked_over_live_round(monkeypatch, state_file, tmp_path):
    players, squads, rounds = make_feeds(round_one="live")  # round 1 is live
    install_feeds(monkeypatch, players, squads, rounds)
    advice_file = tmp_path / "last_advice.json"
    monkeypatch.setattr(advise, "LAST_ADVICE_FILE", advice_file)
    advice_file.write_text(json.dumps({
        "round": 2, "squad_ids": list(range(1, 16)), "starter_ids": list(range(1, 12)),
        "captain_id": 1, "vice_captain_id": 2, "bench_order": [12, 13, 14],
        "bench_gk_id": 15, "bank": 3.0, "transfers_carried": 0,
    }))
    blocked = service.apply_advice(force=False)
    assert blocked["status"] == "blocked" and blocked["live_round"] == 1
    forced = service.apply_advice(force=True)
    assert forced["status"] == "applied"


# --------------------------------------------------------------- record live

def test_record_live_mark_changed(monkeypatch, state_file):
    players, squads, rounds = make_feeds(round_one="live")
    install_feeds(monkeypatch, players, squads, rounds)
    state.save(make_team())
    out = service.record_live(mark_changed=True)
    assert out["status"] == "ok" and out["round_no"] == 1
    assert state.load().manual_changes_round == 1


def test_record_live_captain_switch(monkeypatch, state_file):
    players, squads, rounds = make_feeds(round_one="live")
    install_feeds(monkeypatch, players, squads, rounds)
    state.save(make_team())
    out = service.record_live(captain="P3")  # P3 is a starter (id 3)
    assert out["status"] == "ok"
    assert state.load().captain_id == 3


def test_record_live_bad_name_is_an_error(monkeypatch, state_file):
    players, squads, rounds = make_feeds(round_one="live")
    install_feeds(monkeypatch, players, squads, rounds)
    state.save(make_team())
    out = service.record_live(captain="Nobody")
    assert out["status"] == "error" and "Nobody" in out["message"]


def test_record_live_no_live_round(monkeypatch, state_file):
    players, squads, rounds = make_feeds()  # all scheduled
    install_feeds(monkeypatch, players, squads, rounds)
    state.save(make_team())
    out = service.record_live(mark_changed=True)
    assert out["status"] == "error"


def test_record_live_no_team(monkeypatch, state_file):
    players, squads, rounds = make_feeds(round_one="live")
    install_feeds(monkeypatch, players, squads, rounds)
    out = service.record_live(mark_changed=True)
    assert out["status"] == "error"


# --------------------------------------------------------------- slow (offline)

def test_advise_plan_initial_build(monkeypatch, state_file, tmp_path):
    players, squads, rounds = make_feeds()
    install_feeds(monkeypatch, players, squads, rounds)
    stub_heavy_model(monkeypatch)
    monkeypatch.setattr(advise, "LAST_ADVICE_FILE", tmp_path / "last_advice.json")

    logs = []
    plan = service.advise_plan(from_scratch=False, log=logs.append)

    assert plan["mode"] == "initial" and plan["round"] == 1
    assert plan["applyable"] is True
    assert len(plan["squad_rows"]) == 15
    assert plan["captain"]["id"] in [r["id"] for r in plan["squad_rows"]]
    assert any("Refreshing FIFA feeds" in line for line in logs)
    # JSON-serializable and persisted for `apply`
    json.dumps(plan)
    assert (tmp_path / "last_advice.json").exists()


def test_live_state_no_team_offline(monkeypatch, state_file):
    players, squads, rounds = make_feeds(round_one="live")
    install_feeds(monkeypatch, players, squads, rounds)
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {})
    out = service.live_state()
    assert out["status"] == "no_team"


def test_calibrate_preview_no_matches_played(monkeypatch):
    players, squads, rounds = make_feeds()  # roundPoints empty -> nobody played
    install_feeds(monkeypatch, players, squads, rounds)
    out = service.calibrate_preview(write=False)
    assert out["n_played"] == 0
    assert out["written"] is False
    assert out["refusal_reason"]


def test_fetch_feeds(monkeypatch):
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {"players": True, "rounds": False})
    out = service.fetch_feeds()
    assert out["feeds"] == {"players": "updated", "rounds": "unchanged"}


def test_predicted_bracket_offline(monkeypatch):
    players, squads, rounds = make_feeds()
    install_feeds(monkeypatch, players, squads, rounds)
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {})
    monkeypatch.setattr(
        progression, "predicted_bracket",
        lambda squads, rounds, *a, **k: {
            "available": True, "model": "test-model",
            "rounds": [{"stage": "R32", "name": "Round of 32", "round": 4, "ties": []}],
            "champion": [{"team": "Team1", "squadId": 1, "p": 0.2}],
            "winner": {"team": "Team1", "p": 0.2},
        },
    )
    out = service.predicted_bracket()
    assert out["available"] is True
    assert {"rounds", "champion", "winner", "model"} <= out.keys()
    json.dumps(out)  # JSON-serializable for the web API


def test_predicted_bracket_unavailable_offline(monkeypatch):
    players, squads, rounds = make_feeds()
    install_feeds(monkeypatch, players, squads, rounds)
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {})
    monkeypatch.setattr(progression, "predicted_bracket", lambda *a, **k: {})
    out = service.predicted_bracket()
    assert out["available"] is False and out["message"]
    assert out["rounds"] == [] and out["champion"] == [] and out["winner"] is None
