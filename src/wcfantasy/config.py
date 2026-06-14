from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
STATE_DIR = DATA_DIR / "state"

FIFA_JSON_BASE = "https://play.fifa.com/json/fantasy/"
# The CDN returns 403 without a browser-like User-Agent.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

FEEDS = ("players", "rounds", "squads", "checksums")

# Official FIFA Match Centre feed (api.fifa.com/api/v3): the same data
# that powers the FIFA app/site, free and key-less with the browser
# User-Agent above. Distinct from the fantasy feeds (play.fifa.com): it
# carries the PRE-match confirmed lineups and per-match bookings the
# fantasy feed lacks. The men's World Cup is competition 17; the 2026
# tournament is season 285023 (both verified against the live feed,
# 2026-06-13).
FIFA_DATA_API_BASE = "https://api.fifa.com/api/v3"
WC_COMPETITION_ID = "17"
WC_SEASON_ID = "285023"

MY_TEAM_FILE = STATE_DIR / "my_team.json"
