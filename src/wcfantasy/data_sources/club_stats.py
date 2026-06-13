"""25/26 club season stats from FBref (via soccerdata): the quality
prior for the player points model.

The FIFA feeds carry no pre-tournament signal beyond price and
ownership. This module provides per-player club season stats in a
normalized schema, cached in data/external/, plus FIFA->FBref name
matching (normalized fuzzy match with a manual overrides file - the
part that needs hand care as mismatches surface during the tournament).

soccerdata is optional and FBref scraping is slow and rate-limited:
load() returns None on any failure and the caller falls back to the
price heuristic, the same pattern as the progression model.

Normalized schema (season totals per player):
    name, nation, position (GK/DEF/MID/FWD), games, minutes, goals,
    assists, shots_on_target, tackles, chances_created, saves,
    clean_sheets, goals_conceded, yellow, red, penalties_won,
    penalties_conceded

soccerdata only exposes the standard/keeper/shooting/playing_time/misc
player season tables (no passing/defense), and the 25/26 Big-5 pages
carry no xG/xAG columns at all: tackles come from the misc table
(tackles won) and chances_created is proxied from assists (a chance
creator converts roughly 1 assist per 7 key passes).
"""

import hashlib
import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from ..config import EXTERNAL_DIR

log = logging.getLogger(__name__)

CACHE_FILE = EXTERNAL_DIR / "club_stats_2526.csv"
# {fifa player name: fbref player name}, maintained by hand when the
# fuzzy match gets someone wrong or misses them.
OVERRIDES_FILE = EXTERNAL_DIR / "club_name_overrides.json"
# The fuzzy name match is O(players x club names) and dominates an advise
# run; its result is cached here, keyed by a hash of the inputs.
MATCH_CACHE_FILE = EXTERNAL_DIR / "club_match_cache.json"

SEASON = "2526"
LEAGUES = "Big 5 European Leagues Combined"

NUMERIC_COLUMNS = (
    "games", "minutes", "goals", "assists", "shots_on_target", "tackles",
    "chances_created", "saves", "clean_sheets", "goals_conceded",
    "yellow", "red", "penalties_won", "penalties_conceded",
)

# Key passes are not reachable (no passing table, no xAG on the 25/26
# pages): proxy them from assists, ~7 key passes per assist.
CHANCES_PER_ASSIST = 7.0


def load(refresh: bool = False) -> pd.DataFrame | None:
    """Cached club stats, fetching from FBref only when asked/missing."""
    if CACHE_FILE.exists() and not refresh:
        return pd.read_csv(CACHE_FILE)
    df = _fetch()
    if df is not None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)
    return df


def _fetch() -> pd.DataFrame | None:
    try:
        import soccerdata  # noqa: PLC0415 - optional, heavy import
    except ImportError:
        log.warning("soccerdata not installed: no club stats prior")
        return None
    try:
        fbref = soccerdata.FBref(leagues=LEAGUES, seasons=SEASON)
        standard = _flat(fbref.read_player_season_stats("standard"))
        shooting = _flat(fbref.read_player_season_stats("shooting"))
        misc = _flat(fbref.read_player_season_stats("misc"))
        keeper = _flat(fbref.read_player_season_stats("keeper"))
    except Exception:
        log.exception("FBref fetch failed: no club stats prior")
        return None
    return _normalize(standard, shooting, misc, keeper)


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten soccerdata's (table, column) MultiIndex columns."""
    out = df.reset_index()
    out.columns = [
        "_".join(str(p) for p in col if p and not str(p).startswith("Unnamed"))
        if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def _first(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(0.0, index=df.index)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index)


def _normalize(standard, shooting, misc, keeper) -> pd.DataFrame:
    """Map FBref tables into the module's flat schema."""
    base = pd.DataFrame({
        "name": standard["player"],
        "nation": _col(standard, "nation"),
        "position": _col(standard, "pos").map(_position),
        "games": _first(standard, "Playing Time_MP", "games"),
        "minutes": _first(standard, "Playing Time_Min", "minutes"),
        "goals": _first(standard, "Performance_Gls", "goals"),
        "assists": _first(standard, "Performance_Ast", "assists"),
        "yellow": _first(standard, "Performance_CrdY", "cards_yellow"),
        "red": _first(standard, "Performance_CrdR", "cards_red"),
        "chances_created": CHANCES_PER_ASSIST
        * _first(standard, "assists", "Performance_Ast"),
    })
    # Mid-season movers appear once per club: collapse to one row BEFORE
    # merging the (already player-summed) extra tables, or their totals
    # would be double-counted.
    numeric = [c for c in base.columns if c not in ("name", "nation", "position")]
    base = base.groupby("name", as_index=False).agg(
        {**{c: "sum" for c in numeric}, "nation": "first", "position": "first"}
    )

    def merge(table: pd.DataFrame, mapping: dict[str, tuple[str, ...]]):
        cols = {"player": table["player"]}
        for target, candidates in mapping.items():
            cols[target] = _first(table, *candidates)
        extra = pd.DataFrame(cols).groupby("player", as_index=False).sum()
        return base.merge(extra, left_on="name", right_on="player", how="left").drop(
            columns="player"
        )

    base = merge(shooting, {"shots_on_target": ("Standard_SoT", "shots_on_target")})
    base = merge(misc, {
        "tackles": ("Performance_TklW", "tackles_won"),
        "penalties_won": ("Performance_PKwon", "pens_won"),
        "penalties_conceded": ("Performance_PKcon", "pens_conceded"),
    })
    base = merge(keeper, {
        "saves": ("Performance_Saves", "saves"),
        "clean_sheets": ("Performance_CS", "clean_sheets"),
        "goals_conceded": ("Performance_GA", "goals_conceded"),
    })
    for col in NUMERIC_COLUMNS:
        base[col] = pd.to_numeric(base.get(col), errors="coerce").fillna(0.0)
    return base[base["minutes"] > 0].reset_index(drop=True)


def _position(pos) -> str:
    p = str(pos or "")
    if p.startswith("GK"):
        return "GK"
    if p.startswith("D"):
        return "DEF"
    if p.startswith("F"):
        return "FWD"
    return "MID"


# --------------------------------------------------------- name matching

def normalize_name(name: str) -> str:
    """Accent-strip, lowercase, drop punctuation: 'Dembélé' -> 'dembele'."""
    flat = unicodedata.normalize("NFKD", str(name))
    flat = "".join(c for c in flat if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", flat.lower()).strip()


def _overrides() -> dict[str, str]:
    if OVERRIDES_FILE.exists():
        return json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
    return {}


def match_name(fifa_name: str, club_names: list[str], cutoff: float = 0.82) -> str | None:
    """Best FBref name for a FIFA name, None below the cutoff.

    Token-set containment first (handles 'L. Messi' vs 'Lionel Messi'),
    then a plain similarity ratio over normalized strings.
    """
    target = normalize_name(fifa_name)
    target_tokens = set(target.split())
    best, best_score = None, 0.0
    for cand in club_names:
        norm = normalize_name(cand)
        tokens = set(norm.split())
        if target_tokens and (target_tokens <= tokens or tokens <= target_tokens):
            score = 0.95
        else:
            score = SequenceMatcher(None, target, norm).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= cutoff else None


def match_players(fifa: pd.DataFrame, club: pd.DataFrame) -> dict[int, int]:
    """{fifa player id: club row index}. Position must agree; manual
    overrides win over the fuzzy match.

    Cached by a hash of the inputs that affect the match (player
    id/name/position + club names): an unchanged feed reuses the result
    instead of re-running the O(players x names) fuzzy pass.
    """
    key = _match_cache_key(fifa, club)
    cached = _load_match_cache(key)
    if cached is not None:
        return cached
    result = _compute_match_players(fifa, club)
    _save_match_cache(key, result)
    return result


def _compute_match_players(fifa: pd.DataFrame, club: pd.DataFrame) -> dict[int, int]:
    overrides = _overrides()
    result: dict[int, int] = {}
    by_pos = {pos: grp for pos, grp in club.groupby("position")}
    for _, row in fifa.iterrows():
        pool = by_pos.get(row["position"])
        if pool is None or pool.empty:
            continue
        names = pool["name"].tolist()
        wanted = overrides.get(row["name"])
        matched = wanted if wanted in names else match_name(row["name"], names)
        if matched is None:
            continue
        result[int(row["id"])] = int(pool.index[pool["name"] == matched][0])
    return result


def _match_cache_key(fifa: pd.DataFrame, club: pd.DataFrame) -> str:
    src = "|".join(
        f"{int(pid)}:{name}:{pos}"
        for pid, name, pos in zip(fifa["id"], fifa["name"], fifa["position"])
    )
    src += "##" + "|".join(map(str, club["name"].tolist()))
    return hashlib.md5(src.encode("utf-8")).hexdigest()


def _load_match_cache(key: str) -> dict[int, int] | None:
    if not MATCH_CACHE_FILE.exists():
        return None
    try:
        blob = json.loads(MATCH_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if blob.get("key") != key:
        return None
    return {int(k): int(v) for k, v in blob.get("map", {}).items()}


def _save_match_cache(key: str, result: dict[int, int]) -> None:
    MATCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MATCH_CACHE_FILE.write_text(
        json.dumps({"key": key, "map": {str(k): v for k, v in result.items()}}),
        encoding="utf-8",
    )
