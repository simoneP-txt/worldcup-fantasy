"""Player availability: one factor in [0, 1] = P(the player takes the pitch).

The fantasy feed flags only `playing`/`transferred`: nothing about
fitness, suspensions or who actually starts (a doubtful player looks
identical to a nailed starter). This module fills that gap and feeds the
starter probability, so every advise/transfer decision discounts a
doubtful, injured or banned player automatically.

It aggregates the free signals (strongest first); any source saying a
player is less available wins (the minimum factor):

1. Confirmed lineups (~1h before kickoff), from the official FIFA Match
   Centre feed (data_sources.fifa_match): named starter 1.0, named sub
   0.5, omitted 0.0. Decisive, but only close to kickoff -> `live`. Kept
   separate in `live_lineup_factors` so it can override the rest.
2. Suspensions (days ahead), derived from FIFA bookings (fifa_match): a
   sending-off or a second accumulated yellow -> 0.0 for the next round.
3. Injury withdrawals (days ahead), from the Wikipedia squads page
   (data_sources.wikipedia_squads): the only free source for fitness the
   FIFA feeds lack -> 0.0.
4. API-Football injuries (optional, days ahead): kept for when a paid key
   is configured. Inert on the free plan (no 2026 season access), so it
   contributes {} and the other signals carry the load.

Optional, same contract as the other data_sources: returns {} on any
failure (no key, network, rate limit, schema drift) and the model falls
back to the price depth chart alone. The API-Football key is read from
API_FOOTBALL_KEY or api_football_key.txt (gitignored).
"""

import json
import logging
import os

import pandas as pd

import requests

from ..config import EXTERNAL_DIR, PROJECT_ROOT
from . import fifa_match, wikipedia_squads
from .club_stats import match_name
from .names import EXTERNAL_TO_FIFA

log = logging.getLogger(__name__)

# Filled on each run; advise reports it to the user.
last_run_info: dict = {}


# ----------------------------------------- API-Football injuries (optional)

BASE = "https://v3.football.api-sports.io"
KEY_FILE = PROJECT_ROOT / "api_football_key.txt"
INJURIES_CACHE = EXTERNAL_DIR / "injuries.json"

WORLD_CUP_LEAGUE_ID = 1
SEASON = 2026

# Availability weight by the injury `type` string the API returns.
# Anything present in the feed but not listed is treated as a hard out:
# being flagged at all is already a bad sign.
STATUS_WEIGHT = {
    "questionable": 0.5,
    "doubtful": 0.4,
    "coach decision": 0.35,
}
DEFAULT_INJURY_WEIGHT = 0.0


def api_key() -> str | None:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def _get(path: str, cache, **params) -> list | None:
    key = api_key()
    if not key:
        return None
    try:
        resp = requests.get(
            f"{BASE}{path}", params=params,
            headers={"x-apisports-key": key}, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("response", [])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (requests.RequestException, ValueError) as exc:
        log.warning("api-football %s failed (%s)", path, exc)
        if cache.exists():
            log.warning("using stale cached %s", cache.name)
            return json.loads(cache.read_text(encoding="utf-8"))
        return None


def injury_weight(type_str) -> float:
    return STATUS_WEIGHT.get(str(type_str).strip().lower(), DEFAULT_INJURY_WEIGHT)


def _match_squad(team_name: str, squad_names: dict[str, int]) -> int | None:
    name = EXTERNAL_TO_FIFA.get(team_name, team_name)
    if name in squad_names:
        return squad_names[name]
    best = match_name(name, list(squad_names), cutoff=0.85)
    return squad_names.get(best) if best else None


def _squad_name_index(players: pd.DataFrame) -> dict[int, list[tuple[str, int]]]:
    return {
        int(sid): list(zip(grp["name"], grp["id"]))
        for sid, grp in players.groupby("squadId")
    }


def build_injury_factors(
    raw: list[dict], squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fifa player id: availability factor} from an injuries response.

    Pure (no network), so it is unit-tested directly. A player flagged
    twice keeps the worst (lowest) factor.
    """
    squad_names = {s["name"]: s["id"] for s in squads}
    by_squad = _squad_name_index(players)
    factors: dict[int, float] = {}
    for item in raw:
        team_name = (item.get("team") or {}).get("name", "")
        player_name = (item.get("player") or {}).get("name", "")
        sid = _match_squad(team_name, squad_names)
        if sid is None or sid not in by_squad:
            continue
        names = [n for n, _ in by_squad[sid]]
        matched = match_name(player_name, names)
        if matched is None:
            continue
        pid = next(i for n, i in by_squad[sid] if n == matched)
        weight = injury_weight(item.get("type"))
        factors[int(pid)] = min(factors.get(int(pid), 1.0), weight)
    return factors


def _api_football_injuries(
    squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """API-Football injuries; {} without a key or on the free plan."""
    raw = _get("/injuries", INJURIES_CACHE, league=WORLD_CUP_LEAGUE_ID, season=SEASON)
    return build_injury_factors(raw, squads, players) if raw else {}


# --------------------------------------------------------------- aggregator

def _merge(*sources: dict[int, float]) -> dict[int, float]:
    """Combine availability dicts, keeping the lowest factor per player:
    any source flagging a player as less available wins."""
    out: dict[int, float] = {}
    for source in sources:
        for pid, value in source.items():
            out[pid] = min(out.get(pid, 1.0), value)
    return out


def availability_factors(
    squads: list[dict],
    players: pd.DataFrame,
    rounds: list[dict] | None = None,
    round_no: int | None = None,
) -> dict[int, float]:
    """P(plays) in [0, 1] for flagged players; absent id = assume fit.

    Merges suspensions (FIFA), injury withdrawals (Wikipedia) and, when a
    paid key is set, API-Football injuries. Suspensions need the fixture
    calendar and target round; pass `rounds`/`round_no` (advise and live
    do). {} when nothing is flagged so the caller cleanly falls back to
    the depth chart alone.
    """
    suspensions = (
        fifa_match.suspension_factors(rounds, round_no, squads, players)
        if rounds is not None and round_no is not None
        else {}
    )
    withdrawals = wikipedia_squads.withdrawal_factors(squads, players)
    injuries = _api_football_injuries(squads, players)

    merged = _merge(suspensions, withdrawals, injuries)
    last_run_info.update(
        flagged=len(merged),
        suspensions=len(suspensions),
        withdrawals=len(withdrawals),
        injuries=len(injuries),
        fifa_reachable=fifa_match.last_run_info.get("reachable", False),
    )
    return merged


def live_lineup_factors(
    rnd: dict, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """Confirmed-lineup availability for one round's fixtures (live use).

    Delegates to the FIFA Match Centre feed: published lineups (~1h before
    kickoff) are the strongest signal and override the injuries/suspension
    factors in `live`. {} when no lineup is out yet or the feed is
    unreachable, so the caller falls back to the other signals.
    """
    return fifa_match.lineup_factors(rnd, squads, players)


def observed_start_factors(
    rounds: list[dict], target_round: int, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fantasy id -> P(starts)} from who actually played the completed
    matches (FIFA Match Centre). This is the data-driven depth chart that
    supersedes the game's price rank in the player-points model. {} when the
    feed is unreachable or nothing has been played yet, so the caller falls
    back to the price prior. Delegates to fifa_match; same clean-degrade
    contract as the other signals here."""
    return fifa_match.observed_start_factors(rounds, target_round, squads, players)
