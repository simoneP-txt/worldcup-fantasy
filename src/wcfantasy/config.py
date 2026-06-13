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

MY_TEAM_FILE = STATE_DIR / "my_team.json"
