"""Offline tests for the FastAPI app (TestClient, no real network).

Sync endpoints are exercised against the real service with synthetic state;
the slow job endpoints are stubbed to fast functions so we test the wiring and
the job lifecycle (start -> poll -> done) rather than the model.
"""

import time

import pytest
from fastapi.testclient import TestClient

from wcfantasy import fifa_client, jobs, server, service, settings, state


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    jobs.clear()
    monkeypatch.setattr(state, "MY_TEAM_FILE", tmp_path / "my_team.json")
    monkeypatch.setattr(settings, "UI_SETTINGS_FILE", tmp_path / "ui_settings.json")
    yield
    jobs.clear()


@pytest.fixture
def client():
    return TestClient(server.create_app())


def _poll(client, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        snap = r.json()
        if snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.02)
    raise AssertionError("job did not finish")


# --------------------------------------------------------------- read-only

def test_get_team_empty(client):
    r = client.get("/api/team")
    assert r.status_code == 200
    assert r.json()["has_team"] is False


def test_get_status(client, monkeypatch):
    def missing(feed):
        raise FileNotFoundError(feed)
    monkeypatch.setattr(fifa_client, "load", missing)
    from wcfantasy.data_sources import odds as odds_mod
    monkeypatch.setattr(odds_mod, "api_key", lambda: None)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["feeds_available"] is False
    assert body["language"] == "en"


def test_settings_round_trip_and_validation(client):
    assert client.get("/api/settings").json()["language"] == "en"
    r = client.post("/api/settings", json={"language": "it"})
    assert r.status_code == 200 and r.json()["language"] == "it"
    bad = client.post("/api/settings", json={"language": "xx"})
    assert bad.status_code == 400


# --------------------------------------------------------------- jobs

def test_fetch_job_lifecycle(client, monkeypatch):
    monkeypatch.setattr(fifa_client, "refresh", lambda force=False: {"players": True})
    r = client.post("/api/fetch")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    snap = _poll(client, job_id)
    assert snap["status"] == "done"
    assert snap["result"]["feeds"] == {"players": "updated"}
    assert any("Refreshing" in line for line in snap["logs"])


def test_advise_job_uses_service(client, monkeypatch):
    def fake_advise(log):
        log("planning")
        return {"mode": "initial", "round": 1}
    monkeypatch.setattr(service, "advise_plan", fake_advise)
    job_id = client.post("/api/advise").json()["job_id"]
    snap = _poll(client, job_id)
    assert snap["status"] == "done" and snap["result"]["mode"] == "initial"


def test_advise_from_scratch_passes_flag(client, monkeypatch):
    seen = {}
    def fake_advise(*, from_scratch=False, log=None):
        seen["from_scratch"] = from_scratch
        return {"mode": "from_scratch_comparison"}
    monkeypatch.setattr(service, "advise_plan", fake_advise)
    job_id = client.post("/api/advise/from-scratch").json()["job_id"]
    snap = _poll(client, job_id)
    assert snap["status"] == "done"
    assert seen["from_scratch"] is True


def test_calibrate_passes_write_flag(client, monkeypatch):
    seen = {}
    def fake_cal(*, write=False, log=None):
        seen["write"] = write
        return {"written": write, "n_played": 0}
    monkeypatch.setattr(service, "calibrate_preview", fake_cal)
    job_id = client.post("/api/calibrate", json={"write": True}).json()["job_id"]
    snap = _poll(client, job_id)
    assert snap["status"] == "done" and seen["write"] is True


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


# --------------------------------------------------------------- sync wiring

def test_apply_passes_force(client, monkeypatch):
    seen = {}
    def fake_apply(*, force=False):
        seen["force"] = force
        return {"status": "applied", "round": 2, "message": "ok"}
    monkeypatch.setattr(service, "apply_advice", fake_apply)
    r = client.post("/api/apply", json={"force": True})
    assert r.status_code == 200 and r.json()["status"] == "applied"
    assert seen["force"] is True


def test_apply_defaults_force_false(client, monkeypatch):
    seen = {}
    def fake_apply(*, force=False):
        seen["force"] = force
        return {"status": "applied"}
    monkeypatch.setattr(service, "apply_advice", fake_apply)
    client.post("/api/apply")
    assert seen["force"] is False


def test_booster_passes_name(client, monkeypatch):
    seen = {}
    def fake_booster(name):
        seen["name"] = name
        return {"status": "recorded", "name": name}
    monkeypatch.setattr(service, "set_booster", fake_booster)
    r = client.post("/api/booster", json={"name": "wildcard"})
    assert r.status_code == 200
    assert seen["name"] == "wildcard"


def test_record_passes_fields(client, monkeypatch):
    seen = {}
    def fake_record(**kwargs):
        seen.update(kwargs)
        return {"status": "ok", "messages": []}
    monkeypatch.setattr(service, "record_live", fake_record)
    r = client.post("/api/live/record", json={"captain": "P3", "mark_changed": True})
    assert r.status_code == 200
    assert seen["captain"] == "P3" and seen["mark_changed"] is True
    assert seen["sub_out"] is None and seen["vice"] is None
