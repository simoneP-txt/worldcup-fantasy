"""Canonical FIFA feed status vocabularies, in one place.

The feed marks matches and rounds with status strings. Verified at
matchday 1 (2026-06-12) the live values are 'complete' / 'playing' /
'scheduled'; the extra synonyms below are defensive against schema
drift. Centralized here so a vocabulary change is a one-file edit
instead of a hunt across advise / live / progression.
"""

# A match (or round) that is over and carries a final result.
DONE = frozenset(
    {"complete", "completed", "finished", "fulltime", "full-time", "played"}
)
# A match (or round) currently being played.
LIVE = frozenset(
    {
        "live", "inprogress", "in-progress", "playing",
        "started", "firsthalf", "secondhalf", "halftime",
    }
)
# Not started yet.
PENDING = frozenset({"scheduled", "postponed"})


def _norm(status) -> str:
    return str(status or "").strip().lower()


def is_done(status) -> bool:
    return _norm(status) in DONE


def is_live(status) -> bool:
    return _norm(status) in LIVE


def is_locked(status) -> bool:
    """A round/match that can no longer be edited: done or already live."""
    return is_done(status) or is_live(status)


def result_usable(status) -> bool:
    """A status whose score should be applied as a real result.

    Matches the historical literal check in the progression model
    (anything that is not None and not 'scheduled'); kept strict so a
    'postponed' row carrying a score behaves exactly as before.
    """
    return status is not None and str(status).strip().lower() != "scheduled"
