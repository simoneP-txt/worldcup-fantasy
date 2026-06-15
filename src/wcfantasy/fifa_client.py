"""Download FIFA fantasy feeds with checksum-based caching.

Every advise run calls `refresh()` first, so decisions always use data
updated to the last played match. Each changed feed is also snapshotted
under data/raw/ with a timestamp, building the historical dataset
(prices, ownership, round points over time) for the ML models.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import FEEDS, FIFA_DATA_API_BASE, FIFA_JSON_BASE, HTTP_HEADERS, RAW_DIR

log = logging.getLogger(__name__)

_LATEST = RAW_DIR / "latest"


def _request_with_retry(url: str, **kwargs) -> requests.Response:
    """GET with 3 attempts and exponential backoff.

    Retries BOTH non-200 responses and transport errors (the latter used to
    propagate on the first try, with no retry). Returns the final 200
    Response, or raises: the last HTTP error when the server answered, else
    a RuntimeError when the host was never reachable.
    """
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=20, **kwargs)
            if resp.status_code == 200:
                return resp
            log.warning("GET %s -> %s (attempt %d)", url, resp.status_code, attempt + 1)
        except requests.RequestException as exc:
            resp = None
            log.warning("GET %s failed: %s (attempt %d)", url, exc, attempt + 1)
        if attempt < 2:
            time.sleep(2**attempt)
    if resp is not None:
        resp.raise_for_status()
    raise RuntimeError(f"unreachable after retries: {url}")


def _fetch(feed: str) -> bytes:
    return _request_with_retry(f"{FIFA_JSON_BASE}{feed}.json").content


def fifa_api_get(path: str, **params) -> dict | list:
    """GET an api.fifa.com/api/v3 endpoint (the official Match Centre feed).

    Same browser-UA and retry policy as the fantasy feeds, but returns
    parsed JSON. Raises on failure so optional callers (availability) can
    wrap it and degrade cleanly to {}. `path` is appended to
    FIFA_DATA_API_BASE and must start with '/'.
    """
    return _request_with_retry(f"{FIFA_DATA_API_BASE}{path}", params=params).json()


def _latest_path(feed: str) -> Path:
    return _LATEST / f"{feed}.json"


def refresh(force: bool = False) -> dict:
    """Update local copies of all feeds. Returns {feed: changed?}.

    Uses checksums.json to skip unchanged feeds; snapshots changed ones.
    """
    _LATEST.mkdir(parents=True, exist_ok=True)

    remote_sums = json.loads(_fetch("checksums"))
    _latest_path("checksums").write_bytes(json.dumps(remote_sums).encode())

    local_sums_file = _LATEST / "local_checksums.json"
    local_sums = (
        json.loads(local_sums_file.read_text()) if local_sums_file.exists() else {}
    )

    changed = {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for feed in FEEDS:
        if feed == "checksums":
            continue
        known = local_sums.get(feed)
        remote = remote_sums.get(feed)  # squads has no checksum entry
        needs = force or remote is None or known != remote or not _latest_path(feed).exists()
        if not needs:
            changed[feed] = False
            continue
        data = _fetch(feed)
        _latest_path(feed).write_bytes(data)
        snap_dir = RAW_DIR / feed
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / f"{feed}_{stamp}.json").write_bytes(data)
        if remote is not None:
            local_sums[feed] = remote
        changed[feed] = True

    local_sums_file.write_text(json.dumps(local_sums))
    return changed


def load(feed: str) -> list | dict:
    """Load the latest local copy of a feed (call refresh() first)."""
    path = _latest_path(feed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run refresh() first")
    return json.loads(path.read_text(encoding="utf-8"))


def freshness_report(rounds: list) -> str:
    """Human-readable summary of how current the data is."""
    finished = [
        m
        for r in rounds
        for m in r.get("tournaments", [])
        if m.get("status") not in (None, "scheduled") and m.get("homeScore") is not None
    ]
    if not finished:
        nxt = min(
            (m for r in rounds for m in r.get("tournaments", [])),
            key=lambda m: m["date"],
        )
        return (
            "No matches played yet: the tournament opens with "
            f"{nxt['homeSquadName']}-{nxt['awaySquadName']} on {nxt['date']}"
        )
    last = max(finished, key=lambda m: m["date"])
    return (
        f"Data updated to {last['homeSquadName']} {last['homeScore']}"
        f"-{last['awayScore']} {last['awaySquadName']} ({last['date']})"
    )
