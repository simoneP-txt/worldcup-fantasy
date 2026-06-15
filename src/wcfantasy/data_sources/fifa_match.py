"""Official FIFA Match Centre feed (api.fifa.com/api/v3).

Free and key-less: the same data that powers the FIFA app/site, reached
with the browser User-Agent the fantasy client already uses. It carries
two PRE-match availability signals the fantasy feed (play.fifa.com) lacks
- the fantasy feed only exposes data OBSERVED at full time:

1. Confirmed lineups (~1h before kickoff). The match-detail `Players`
   array lists the 26-man squad with `Status` 1 = starting XI, 2 = bench.
   A fantasy player named a starter scores 1.0, a named sub BENCH_FACTOR,
   any squad member NOT listed (left out of the matchday squad) 0.0.
   Strongest signal, but only close to kickoff (mainly for `live`).
2. Suspensions (known days ahead). Derived from the `Bookings` of
   COMPLETED matches via the official disciplinary rules in rules.py: a
   sending-off, or a second accumulated yellow, bans the next match.

Team bridge is exact: FIFA's own 3-letter code (squads.json `abbr`
== api.fifa.com `IdCountry`), so no fuzzy team matching is needed. Player
bridge is fuzzy within the matched squad (club_stats.match_name), which
correctly rejects a wrong same-surname match (verified: FIFA's
"Sphephelo Sithole" is not confused with the feed's "Yaya Sithole").

Optional, same contract as the other data_sources: every public function
returns {} on any failure (network, schema drift) and the model falls
back to the price depth chart alone.

`Card` codes verified against the live feed (MEX-RSA, 2026-06-13): 1 =
single yellow, 2 = sending-off (Montes, Zwane and Sithole each carried a
lone Card 2 and were correctly flagged). The straight-red code is assumed
to be 3 and is included in RED_LIKE_CODES; confirm if one ever appears.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .. import fifa_client, rules
from ..config import EXTERNAL_DIR, WC_COMPETITION_ID, WC_SEASON_ID
from .club_stats import match_name

log = logging.getLogger(__name__)

# Player `Status` in the match-detail Players array.
STARTER_STATUS = 1
BENCH_STATUS = 2
# A confirmed-lineup substitute still tends to get some minutes.
BENCH_FACTOR = 0.5
# A squad member NOT on the published list was left out -> 0.0, but only
# trust that when EVERY published name bridged to a fantasy player: an
# unbridged published name could BE one of those "omitted" players under a
# different spelling, so reading silence as 'dropped' would falsely rule
# out a real starter. Above this many unresolved names, skip the 0.0s for
# that side (keep the positive signals, fall back to the depth chart).
MAX_UNRESOLVED_FOR_OMISSION = 0

# `MatchStatus` in the calendar/detail feeds: 0 = completed, 1 = scheduled
# (other values cover live/abandoned and are ignored for suspensions).
MATCH_COMPLETED = 0

# `Card` codes in Bookings. Verified against the live feed: 1 = single
# yellow, 2 = sending-off. 3 (assumed straight red) is included for
# safety. Two yellows for one player in one match are also treated as a
# sending-off regardless of code.
YELLOW_CARD = 1
RED_LIKE_CODES = frozenset({2, 3})
# Two yellows in a SINGLE match are always a sending-off (laws of the
# game), independent of the cross-match accumulation ban threshold.
SAME_MATCH_YELLOWS_FOR_RED = 2

CALENDAR_CACHE = EXTERNAL_DIR / "fifa_calendar.json"

last_run_info: dict = {}


# ----------------------------------------------------------- network (cached)

def _match_cache(id_match) -> Path:
    return EXTERNAL_DIR / f"fifa_match_{id_match}.json"


def _fetch_calendar() -> list | None:
    """All season matches (cheap, one call). Refetched each run because
    statuses change; falls back to the cached copy on failure."""
    try:
        data = fifa_client.fifa_api_get(
            "/calendar/matches",
            idCompetition=WC_COMPETITION_ID,
            idSeason=WC_SEASON_ID,
            count=500,
            language="en",
        )
        results = data.get("Results", []) if isinstance(data, dict) else []
        CALENDAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CALENDAR_CACHE.write_text(json.dumps(results), encoding="utf-8")
        return results
    except Exception as exc:  # network, schema, anything: degrade cleanly
        log.warning("FIFA calendar fetch failed (%s)", exc)
        if CALENDAR_CACHE.exists():
            return json.loads(CALENDAR_CACHE.read_text(encoding="utf-8"))
        return None


def _fetch_match(id_stage, id_match, *, cache: bool = True) -> dict | None:
    """One match detail (Players + Bookings).

    `cache=True` (suspensions path) keeps a permanent on-disk copy: a
    COMPLETED match is immutable once played. `cache=False` (the lineups
    path for a not-yet-finished match) always refetches and never writes,
    because the pre-match detail mutates as the confirmed XI is published
    ~1h before kickoff - caching an early, lineup-less payload would hide
    the lineup from every later `live` run.
    """
    cache_file = _match_cache(id_match)
    if cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    try:
        data = fifa_client.fifa_api_get(
            f"/live/football/{WC_COMPETITION_ID}/{WC_SEASON_ID}/{id_stage}/{id_match}",
            language="en",
        )
        if isinstance(data, dict):
            if cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        return None
    except Exception as exc:
        log.warning("FIFA match %s fetch failed (%s)", id_match, exc)
        return None


# ------------------------------------------------------------- name bridges

def abbr_to_squad(squads: list[dict]) -> dict[str, int]:
    """{FIFA 3-letter code -> fantasy squadId} from squads.json `abbr`."""
    return {s["abbr"]: s["id"] for s in squads if s.get("abbr")}


def squad_roster(players: pd.DataFrame) -> dict[int, list[tuple[str, int]]]:
    """{squadId -> [(player name, fantasy id)]}."""
    return {
        int(sid): list(zip(grp["name"], grp["id"]))
        for sid, grp in players.groupby("squadId")
    }


def _team_name(text_block) -> str:
    """First localized description from a FIFA Name/PlayerName array."""
    if isinstance(text_block, list) and text_block:
        return text_block[0].get("Description", "")
    return ""


# --------------------------------------------------------- pure: lineups

def parse_lineup_factors(
    match: dict, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fantasy id -> availability} from one match detail's lineups.

    Pure (no network), unit-tested. For each side whose lineup is
    published (non-empty Players), each PUBLISHED name is resolved to a
    fantasy player: named starter 1.0, named sub BENCH_FACTOR. A squad
    member no published name resolved to is ruled out (0.0) ONLY when every
    published name bridged (see MAX_UNRESOLVED_FOR_OMISSION) - otherwise a
    spelling miss on a real starter must not read as 'dropped', so only the
    positive signals are kept and the rest fall back to the depth chart.
    Sides with no published lineup are left untouched.
    """
    by_abbr = abbr_to_squad(squads)
    by_squad = squad_roster(players)
    factors: dict[int, float] = {}
    for side in ("HomeTeam", "AwayTeam"):
        block = match.get(side) or {}
        squad_players = block.get("Players") or []
        if not squad_players:
            continue
        sid = by_abbr.get(block.get("IdCountry"))
        if sid is None or sid not in by_squad:
            continue
        roster = by_squad[sid]
        roster_names = [n for n, _ in roster]
        id_of = {n: int(i) for n, i in roster}

        published = [
            (_team_name(p.get("PlayerName")), 1.0)
            for p in squad_players
            if p.get("Status") == STARTER_STATUS
        ] + [
            (_team_name(p.get("PlayerName")), BENCH_FACTOR)
            for p in squad_players
            if p.get("Status") == BENCH_STATUS
        ]

        resolved: dict[int, float] = {}
        unresolved = 0
        for api_name, value in published:
            matched = match_name(api_name, roster_names, cutoff=0.82)
            if matched is None:
                unresolved += 1
                continue
            pid = id_of[matched]
            resolved[pid] = max(resolved.get(pid, 0.0), value)
        factors.update(resolved)

        if unresolved <= MAX_UNRESOLVED_FOR_OMISSION:
            for _, pid in roster:
                if int(pid) not in resolved:
                    factors[int(pid)] = 0.0
    return factors


# ------------------------------------------------------- pure: suspensions

def classify_bookings(bookings: list[dict]) -> tuple[set, set]:
    """(yellow player ids, sending-off player ids) from a match's Bookings.

    Only player cards count (coach/staff cards are ignored). Two yellows
    for the same player in one match collapse to a sending-off.
    """
    yellows = Counter()
    reds: set = set()
    for b in bookings or []:
        pid = b.get("IdPlayer")
        if not pid:
            continue
        card = b.get("Card")
        if card in RED_LIKE_CODES:
            reds.add(pid)
        elif card == YELLOW_CARD:
            yellows[pid] += 1
    singles = set()
    for pid, count in yellows.items():
        (reds if count >= SAME_MATCH_YELLOWS_FOR_RED else singles).add(pid)
    return singles, reds


def normalize_match(match: dict, round_no: int) -> list[dict]:
    """One entry per side: {abbr, round, present, yellows, reds, names}.

    `present` is the set of FIFA player ids in the matchday squad (a
    suspended player is absent from it - that is how a served ban is
    detected). `names` maps each FIFA player id to (name, abbr) so the
    final suspended set can be bridged to fantasy ids.
    """
    entries = []
    for side in ("HomeTeam", "AwayTeam"):
        block = match.get(side) or {}
        abbr = block.get("IdCountry")
        if not abbr:
            continue
        squad_players = block.get("Players") or []
        present = {p.get("IdPlayer") for p in squad_players if p.get("IdPlayer")}
        names = {
            p["IdPlayer"]: (_team_name(p.get("PlayerName")), abbr)
            for p in squad_players
            if p.get("IdPlayer")
        }
        yellows, reds = classify_bookings(block.get("Bookings") or [])
        entries.append(
            {
                "abbr": abbr,
                "round": round_no,
                "present": present,
                "yellows": yellows,
                "reds": reds,
                "names": names,
            }
        )
    return entries


def derive_suspensions(
    entries: list[dict],
    *,
    yellow_ban: int,
    reset_after_rounds: tuple[int, ...],
) -> set:
    """FIFA player ids banned for their team's NEXT match.

    Pure, unit-tested. Walks each team's completed matches in round order
    and tracks bans the way the laws do:
    - a sending-off, or the `yellow_ban`-th accumulated single yellow,
      sets a pending ban (and clears the yellow tally);
    - a pending ban is cleared once the player is absent from a later
      matchday squad (the ban was served);
    - the single-yellow tally resets after the rounds in
      `reset_after_rounds` (group stage, quarter-finals).
    A ban still pending after the last completed match applies next round.
    """
    by_team: dict = defaultdict(list)
    for e in entries:
        by_team[e["abbr"]].append(e)

    suspended: set = set()
    for matches in by_team.values():
        matches.sort(key=lambda e: e["round"])
        yellows: Counter = Counter()
        pending: set = set()
        applied_resets: set = set()
        for e in matches:
            for boundary in reset_after_rounds:
                if e["round"] > boundary and boundary not in applied_resets:
                    yellows.clear()
                    applied_resets.add(boundary)
            # A pending player absent from this matchday squad served it.
            for pid in list(pending):
                if pid not in e["present"]:
                    pending.discard(pid)
            for pid in e["reds"]:
                pending.add(pid)
                yellows[pid] = 0
            for pid in e["yellows"]:
                yellows[pid] += 1
                if yellows[pid] >= yellow_ban:
                    pending.add(pid)
                    yellows[pid] = 0
        suspended.update(pending)
    return suspended


def resolve_to_fantasy(
    fifa_ids: set, id_names: dict, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fantasy id -> 0.0} for each suspended FIFA player id we can map.

    Bridges by exact team (abbr) then a fuzzy name match within the squad;
    unmappable ids are dropped (degrade cleanly).
    """
    by_abbr = abbr_to_squad(squads)
    by_squad = squad_roster(players)
    out: dict[int, float] = {}
    for fid in fifa_ids:
        name, abbr = id_names.get(fid, ("", None))
        sid = by_abbr.get(abbr)
        if sid is None or sid not in by_squad:
            continue
        roster = by_squad[sid]
        matched = match_name(name, [n for n, _ in roster])
        if matched is None:
            continue
        pid = next(i for n, i in roster if n == matched)
        out[int(pid)] = 0.0
    return out


# --------------------------------------------------------- orchestration

def _round_index(rounds: list[dict], squads: list[dict]) -> dict[tuple[str, str], int]:
    """{(home abbr, away abbr) -> round id} from the fantasy rounds feed.

    The fantasy fixtures already carry homeSquadAbbr/awaySquadAbbr (FIFA
    codes), so this needs no id lookups.
    """
    out: dict[tuple[str, str], int] = {}
    for r in rounds:
        for m in r.get("tournaments", []):
            h, a = m.get("homeSquadAbbr"), m.get("awaySquadAbbr")
            if h and a:
                out[(h, a)] = r["id"]
    return out


def lineup_factors(
    rnd: dict, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """Confirmed-lineup availability for one round's fixtures (live use).

    Maps each fantasy fixture to its FIFA match (via the exact abbr code),
    reads the published lineups and merges them. {} when the feed is
    unreachable so the caller falls back to the suspensions signal.
    """
    calendar = _fetch_calendar()
    if not calendar:
        return {}
    # {(home abbr, away abbr) -> (idStage, idMatch, MatchStatus)} from the
    # calendar. The status decides caching: a completed match is immutable
    # (cache it), a scheduled/live one must be refetched so a confirmed
    # lineup published after an earlier run is not missed.
    fmap: dict[tuple[str, str], tuple] = {}
    for m in calendar:
        h = (m.get("Home") or {}).get("IdCountry")
        a = (m.get("Away") or {}).get("IdCountry")
        if h and a and m.get("IdMatch") is not None:
            fmap[(h, a)] = (m.get("IdStage"), m.get("IdMatch"), m.get("MatchStatus"))

    factors: dict[int, float] = {}
    for m in rnd.get("tournaments", []):
        h, a = m.get("homeSquadAbbr"), m.get("awaySquadAbbr")
        ref = fmap.get((h, a)) or fmap.get((a, h))
        if ref is None:
            continue
        detail = _fetch_match(ref[0], ref[1], cache=ref[2] == MATCH_COMPLETED)
        if detail:
            factors.update(parse_lineup_factors(detail, squads, players))
    return factors


def suspension_factors(
    rounds: list[dict], target_round: int, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fantasy id -> 0.0} for players suspended for `target_round`.

    Pulls every completed match before the target round from the FIFA
    feed, derives the bans (derive_suspensions) and bridges them to
    fantasy ids. {} when the feed is unreachable.
    """
    calendar = _fetch_calendar()
    if not calendar:
        last_run_info.update(suspensions=0, reachable=False)
        return {}

    round_of = _round_index(rounds, squads)
    entries: list[dict] = []
    id_names: dict = {}
    for m in calendar:
        if m.get("MatchStatus") != MATCH_COMPLETED:
            continue
        h = (m.get("Home") or {}).get("IdCountry")
        a = (m.get("Away") or {}).get("IdCountry")
        rno = round_of.get((h, a)) or round_of.get((a, h))
        if rno is None or rno >= target_round:
            continue
        detail = _fetch_match(m.get("IdStage"), m.get("IdMatch"))
        if not detail:
            continue
        for entry in normalize_match(detail, rno):
            id_names.update(entry.pop("names"))
            entries.append(entry)

    banned = derive_suspensions(
        entries,
        yellow_ban=rules.YELLOW_CARDS_FOR_BAN,
        reset_after_rounds=rules.YELLOW_RESET_AFTER_ROUNDS,
    )
    factors = resolve_to_fantasy(banned, id_names, squads, players)
    last_run_info.update(suspensions=len(factors), reachable=True)
    return factors
