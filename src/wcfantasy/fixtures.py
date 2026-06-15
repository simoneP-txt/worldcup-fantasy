"""Shared fixture-time helpers.

The fantasy rounds feed gives each match an ISO `date`. Several modules
need it parsed and, often, a {squadId -> kickoff} map. Every datetime
returned here is tz-aware (a feed date without an explicit offset is read
as UTC) so comparisons and sorting never mix naive and aware datetimes -
the kind of TypeError that hid in the per-module copies this replaces.
"""

from datetime import datetime, timezone


def parse_kickoff(value) -> datetime | None:
    """ISO date string -> tz-aware datetime (naive read as UTC), or None."""
    try:
        when = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def kickoffs_by_squad(rnd: dict) -> dict[int, datetime]:
    """{squadId -> kickoff datetime} for one round's fixtures (tz-aware)."""
    out: dict[int, datetime] = {}
    for m in rnd.get("tournaments", []):
        when = parse_kickoff(m.get("date"))
        if when is None:
            continue
        for key in ("homeSquadId", "awaySquadId"):
            if m.get(key) is not None:
                out[m[key]] = when
    return out
