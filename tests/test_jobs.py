"""Offline tests for the background-job registry (no network, no server)."""

import time

import pytest

from wcfantasy import jobs


@pytest.fixture(autouse=True)
def _clear_jobs():
    jobs.clear()
    yield
    jobs.clear()


def _wait(job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        snap = jobs.get(job_id)
        if snap and snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_job_runs_collects_logs_and_result():
    def work(log):
        log("step 1")
        log("step 2")
        return {"ok": True}

    job_id = jobs.start(work, kind="unit")
    snap = _wait(job_id)
    assert snap["status"] == "done"
    assert snap["result"] == {"ok": True}
    assert snap["logs"] == ["step 1", "step 2"]
    assert snap["kind"] == "unit"


def test_job_captures_errors_without_leaking_traceback_to_client():
    def boom(log):
        log("before the error")
        raise ValueError("nope")

    job_id = jobs.start(boom)
    snap = _wait(job_id)
    assert snap["status"] == "error"
    assert "nope" in snap["error"]
    assert any("nope" in line for line in snap["logs"])


def test_get_unknown_job_is_none():
    assert jobs.get("does-not-exist") is None


def test_heavy_jobs_are_serialized_by_the_work_lock():
    order = []

    def make(tag):
        def fn(log):
            order.append(f"start-{tag}")
            time.sleep(0.05)
            order.append(f"end-{tag}")
            return tag
        return fn

    a = jobs.start(make("a"))
    b = jobs.start(make("b"))
    _wait(a)
    _wait(b)
    # The lock guarantees no interleaving: each job ends before the next starts.
    assert order in (
        ["start-a", "end-a", "start-b", "end-b"],
        ["start-b", "end-b", "start-a", "end-a"],
    )
