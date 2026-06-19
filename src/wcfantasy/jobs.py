"""In-process background-job registry for the local web server.

The slow commands (advise / fetch / odds / calibrate) refresh the FIFA
feeds and run the Monte Carlo / ILP, taking seconds to minutes (the first
FBref pull, minutes). The web API starts them as background jobs and the
frontend polls for progress, so the UI never blocks.

A single global lock serializes the heavy jobs on purpose: they share
module-global diagnostics (each model's `last_run_info`) that would race if
run concurrently, and one CPU-bound job at a time is plenty for a local
single user. A started job therefore sits in 'queued' until it acquires the
lock, then flips to 'running'.

Each job exposes a `log(message)` sink (passed to the worker as the `log`
keyword) so progress lines accumulate and the client can render them live.
The worker's return value becomes the job result (must be JSON-friendly);
any exception is captured as an error string (the full traceback goes to the
server log, never to the client).
"""

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Only one heavy job runs at a time (see module docstring).
_WORK_LOCK = threading.Lock()
_REGISTRY_LOCK = threading.Lock()
_JOBS: dict[str, "Job"] = {}
# Cap the registry so a long-lived server does not leak finished jobs.
MAX_JOBS = 100


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    logs: list[str] = field(default_factory=list)
    result: object = None
    error: str | None = None
    created: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_log(self, message: str) -> None:
        with self._lock:
            self.logs.append(str(message))

    def _set(self, **values) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        """Thread-safe JSON-friendly view of the job for the polling client."""
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "logs": list(self.logs),
                "result": self.result,
                "error": self.error,
                "created": self.created,
            }


def _prune_locked() -> None:
    """Drop the oldest FINISHED jobs once the registry exceeds MAX_JOBS.

    Assumes _REGISTRY_LOCK is held. Running/queued jobs are never dropped.
    """
    if len(_JOBS) <= MAX_JOBS:
        return
    for job_id in list(_JOBS):
        if len(_JOBS) <= MAX_JOBS:
            break
        if _JOBS[job_id].status in ("done", "error"):
            del _JOBS[job_id]


def _run(job: Job, fn: Callable[..., object]) -> None:
    # Serialize heavy work: the job waits here (status 'queued') until free.
    with _WORK_LOCK:
        job._set(status="running")
        try:
            result = fn(log=job.add_log)
            job._set(status="done", result=result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as text
            job.add_log(f"ERROR: {exc}")
            job._set(status="error", error=str(exc))
            traceback.print_exc()  # full detail to the server log only


def start(fn: Callable[..., object], *, kind: str = "job") -> str:
    """Run `fn(log=...)` in a background thread; return its job id.

    `fn` must accept a `log` keyword (a callable taking one string) and
    return a JSON-serializable result. Poll `get(job_id)` for progress.
    """
    job = Job(
        id=uuid.uuid4().hex,
        kind=kind,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with _REGISTRY_LOCK:
        _JOBS[job.id] = job
        _prune_locked()
    threading.Thread(target=_run, args=(job, fn), daemon=True).start()
    return job.id


def get(job_id: str) -> dict | None:
    """Snapshot of a job by id, or None if unknown."""
    with _REGISTRY_LOCK:
        job = _JOBS.get(job_id)
    return job.snapshot() if job else None


def clear() -> None:
    """Drop all jobs (used by tests for isolation)."""
    with _REGISTRY_LOCK:
        _JOBS.clear()
