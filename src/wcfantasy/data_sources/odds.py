"""Bookmaker odds from The Odds API (the-odds-api.com).

Fetched FRESH on every advise run (2 requests/run against a 500/month
free quota), so prices reflect lineups and news up to launch time. The
last good response is cached only as an offline fallback.

Key setup: free signup at https://the-odds-api.com, then either set the
ODDS_API_KEY environment variable or put the key in odds_api_key.txt in
the project root (gitignored). Without a key everything returns None
and the model runs on Elo alone.

Implied probabilities are de-overrounded per bookmaker (divide by the
book's total), then averaged across bookmakers.
"""

import difflib
import json
import logging
import os

import requests

from ..config import EXTERNAL_DIR, HTTP_HEADERS, PROJECT_ROOT
from .intl_results import dataset_name
from .names import EXTERNAL_TO_FIFA

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"
KEY_FILE = PROJECT_ROOT / "odds_api_key.txt"
H2H_CACHE = EXTERNAL_DIR / "odds_h2h.json"
OUTRIGHT_CACHE = EXTERNAL_DIR / "odds_outright.json"


def api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def _get(path: str, cache, **params) -> list | None:
    key = api_key()
    if not key:
        return None
    try:
        resp = requests.get(
            f"{BASE}{path}", params={"apiKey": key, **params},
            headers=HTTP_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except requests.RequestException as exc:
        log.warning("odds fetch %s failed (%s)", path, exc)
        if cache.exists():
            log.warning("using stale cached odds from %s", cache.name)
            return json.loads(cache.read_text(encoding="utf-8"))
        return None


def _world_cup_sport_keys() -> tuple[str | None, str | None]:
    """(h2h sport key, outright sport key) discovered from the API."""
    sports = _get("/sports", EXTERNAL_DIR / "odds_sports.json", all="true")
    if not sports:
        return None, None
    h2h = outright = None
    for s in sports:
        k = s["key"]
        if "world_cup" not in k or "women" in k:
            continue
        if s.get("has_outrights") or "winner" in k:
            outright = outright or k
        else:
            h2h = h2h or k
    return h2h, outright


def match_team(odds_name: str, squads: list[dict]) -> int | None:
    """Map a bookmaker team name to a squadId."""
    fifa = EXTERNAL_TO_FIFA.get(odds_name, odds_name)
    by_name = {s["name"]: s["id"] for s in squads}
    if fifa in by_name:
        return by_name[fifa]
    by_dataset = {dataset_name(s["name"]): s["id"] for s in squads}
    if odds_name in by_dataset:
        return by_dataset[odds_name]
    close = difflib.get_close_matches(odds_name, by_name.keys(), n=1, cutoff=0.85)
    if close:
        return by_name[close[0]]
    # expected for non-qualified teams in outright markets
    log.debug("odds team name not matched: %r", odds_name)
    return None


def _implied(prices: dict[str, float]) -> dict[str, float]:
    """De-overround one bookmaker's decimal prices into probabilities."""
    inv = {name: 1.0 / p for name, p in prices.items() if p and p > 1.0}
    total = sum(inv.values())
    return {name: v / total for name, v in inv.items()} if total else {}


def h2h_probabilities(squads: list[dict]) -> dict[tuple[int, int], tuple[float, float, float]] | None:
    """{(home_id, away_id): (p_home, p_draw, p_away)} in the bookmaker's
    orientation; the caller flips the tuple if its fixture is reversed."""
    sport, _ = _world_cup_sport_keys()
    if not sport:
        return None
    # 3 regions cost 3 credits/call; worth it for coverage vs 500/month.
    events = _get(f"/sports/{sport}/odds", H2H_CACHE,
                  regions="eu,uk,us", markets="h2h", oddsFormat="decimal")
    if not events:
        return None

    out = {}
    for ev in events:
        home_id = match_team(ev["home_team"], squads)
        away_id = match_team(ev["away_team"], squads)
        if home_id is None or away_id is None:
            continue
        samples = []
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market["outcomes"]}
                probs = _implied(prices)
                if {ev["home_team"], ev["away_team"], "Draw"} <= set(probs):
                    samples.append((
                        probs[ev["home_team"]], probs["Draw"], probs[ev["away_team"]]
                    ))
        if samples:
            n = len(samples)
            avg = tuple(sum(s[i] for s in samples) / n for i in range(3))
            out[(home_id, away_id)] = avg
    return out or None


def outright_probabilities(squads: list[dict]) -> dict[int, float] | None:
    """{squadId: implied P(win tournament)}, averaged across bookmakers."""
    _, sport = _world_cup_sport_keys()
    if not sport:
        return None
    events = _get(f"/sports/{sport}/odds", OUTRIGHT_CACHE,
                  regions="eu,uk,us", markets="outrights", oddsFormat="decimal")
    if not events:
        return None

    samples: dict[int, list[float]] = {}
    for ev in events:
        for book in ev.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "outrights":
                    continue
                prices = {o["name"]: o["price"] for o in market["outcomes"]}
                for name, prob in _implied(prices).items():
                    sid = match_team(name, squads)
                    if sid is not None:
                        samples.setdefault(sid, []).append(prob)
    if not samples:
        return None
    return {sid: sum(v) / len(v) for sid, v in samples.items()}
