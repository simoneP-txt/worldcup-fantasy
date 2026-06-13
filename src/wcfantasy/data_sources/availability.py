"""Player availability from API-Football (injuries + confirmed lineups).

The FIFA feed flags only `playing`/`transferred`: nothing about
injuries, suspensions or fitness (a doubtful player looks identical to a
nailed starter). This module fills that gap with an availability factor
in [0, 1] = P(the player actually takes the pitch), fed into the starter
probability so every advise/transfer decision discounts a doubtful or
injured player automatically.

Two signals, strongest first:

1. Confirmed lineups (~1h before kickoff): a named starter is ~1.0, a
   benched player gets `BENCH_FACTOR`, an omitted one 0.0. Decisive, but
   only available close to kickoff (mainly useful to `live`).
2. The injuries feed (days ahead): a flagged player is discounted by the
   weight of his injury `type` (out/doubtful/questionable).

Optional, same contract as the other data_sources: returns {} on any
failure (no key, network, rate limit, schema drift) and the model falls
back to the price depth chart alone. Key from the API_FOOTBALL_KEY env
var or api_football_key.txt in the project root (gitignored).

API-Football ids and response shapes are marked TODO(verify): confirm
them on the first real call (the World Cup league id especially).
"""

import json
import logging
import os

import pandas as pd

import requests

from ..config import EXTERNAL_DIR, PROJECT_ROOT
from .club_stats import match_name
from .names import EXTERNAL_TO_FIFA

log = logging.getLogger(__name__)

BASE = "https://v3.football.api-sports.io"
KEY_FILE = PROJECT_ROOT / "api_football_key.txt"
INJURIES_CACHE = EXTERNAL_DIR / "injuries.json"
FIXTURES_CACHE = EXTERNAL_DIR / "af_fixtures.json"

# TODO(verify) on the first keyed call: GET /leagues?search=world cup.
# 1 is API-Football's historical id for the men's World Cup.
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
# A confirmed-lineup substitute still tends to get some minutes.
BENCH_FACTOR = 0.5

# Filled on each run; advise reports the count to the user.
last_run_info: dict = {}


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


def availability_factors(
    squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """P(plays) in [0, 1] for flagged players; absent id = assume fit.

    {} when no key/data so the caller cleanly falls back to the depth
    chart alone.
    """
    raw = _get("/injuries", INJURIES_CACHE, league=WORLD_CUP_LEAGUE_ID, season=SEASON)
    if not raw:
        last_run_info["flagged"] = 0
        last_run_info["source"] = "off (no API_FOOTBALL_KEY)" if api_key() is None else "no data"
        return {}
    factors = build_injury_factors(raw, squads, players)
    last_run_info["flagged"] = len(factors)
    last_run_info["source"] = "api-football injuries"
    return factors


def lineup_factors(
    fixtures: list[dict], squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """Confirmed-lineup availability for a set of fixture ids (live use).

    Overrides the injuries signal once a lineup is published: named
    starters 1.0, named subs BENCH_FACTOR, everyone else in those squads
    0.0. Squads with no published lineup are left untouched (absent).
    """
    by_squad = _squad_name_index(players)
    squad_names = {s["name"]: s["id"] for s in squads}
    factors: dict[int, float] = {}
    for fx in fixtures:
        lineups = _get(
            "/fixtures/lineups", EXTERNAL_DIR / f"lineup_{fx}.json", fixture=fx
        )
        if not lineups:
            continue
        for team_block in lineups:
            team_name = (team_block.get("team") or {}).get("name", "")
            sid = _match_squad(team_name, squad_names)
            if sid is None or sid not in by_squad:
                continue
            roster = by_squad[sid]
            named = {
                "start": [p.get("player", {}).get("name", "") for p in team_block.get("startXI", [])],
                "sub": [p.get("player", {}).get("name", "") for p in team_block.get("substitutes", [])],
            }
            for name, pid in roster:
                if match_name_in(name, named["start"]):
                    factors[int(pid)] = 1.0
                elif match_name_in(name, named["sub"]):
                    factors[int(pid)] = BENCH_FACTOR
                else:
                    factors[int(pid)] = 0.0
    return factors


def match_name_in(fifa_name: str, api_names: list[str]) -> bool:
    return bool(api_names) and match_name(fifa_name, api_names, cutoff=0.82) is not None


def build_fixture_id_map(
    raw: list[dict], squads: list[dict]
) -> dict[tuple[int, int], int]:
    """{(home squadId, away squadId): API-Football fixture id} from a
    /fixtures response. Pure (no network), so it is unit-tested."""
    squad_names = {s["name"]: s["id"] for s in squads}
    out: dict[tuple[int, int], int] = {}
    for item in raw:
        fx = item.get("fixture") or {}
        teams = item.get("teams") or {}
        hid = _match_squad((teams.get("home") or {}).get("name", ""), squad_names)
        aid = _match_squad((teams.get("away") or {}).get("name", ""), squad_names)
        if hid is not None and aid is not None and fx.get("id") is not None:
            out[(hid, aid)] = int(fx["id"])
    return out


def _world_cup_fixture_ids(squads: list[dict]) -> dict[tuple[int, int], int]:
    raw = _get("/fixtures", FIXTURES_CACHE, league=WORLD_CUP_LEAGUE_ID, season=SEASON)
    return build_fixture_id_map(raw, squads) if raw else {}


def live_lineup_factors(
    rnd: dict, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """Confirmed-lineup availability for the fixtures of one round.

    Maps each fixture in the round to its API-Football id and reads the
    published lineups (lineup_factors), the strongest availability signal
    (~1h before kickoff). Returns {} when no key/data so the caller falls
    back to the injuries feed; note the API-Football free plan has no
    access to the current World Cup season, so this stays empty there.
    """
    fid_map = _world_cup_fixture_ids(squads)
    if not fid_map:
        return {}
    fixture_ids = []
    for m in rnd.get("tournaments", []):
        h, a = m.get("homeSquadId"), m.get("awaySquadId")
        fid = fid_map.get((h, a)) or fid_map.get((a, h))
        if fid is not None:
            fixture_ids.append(fid)
    return lineup_factors(fixture_ids, squads, players) if fixture_ids else {}
