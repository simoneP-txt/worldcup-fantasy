"""Historical international match results.

Source: github.com/martj42/international_results (same data as the
Kaggle "International football results from 1872" dataset), ~49k
matches, refreshed by the maintainer through the tournament. Cached in
data/external/ and re-downloaded when older than MAX_AGE.
"""

import logging
import time

import pandas as pd
import requests

from ..config import EXTERNAL_DIR, HTTP_HEADERS

log = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CSV_PATH = EXTERNAL_DIR / "intl_results.csv"
MAX_AGE_SECONDS = 12 * 3600

# FIFA squads.json name -> dataset name, for the teams that differ.
NAME_OVERRIDES = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


def dataset_name(fifa_name: str) -> str:
    return NAME_OVERRIDES.get(fifa_name, fifa_name)


def load(refresh: bool = True) -> pd.DataFrame:
    """Return played matches as a DataFrame sorted by date.

    Drops unplayed fixtures (NA scores) and World Cup 2026 rows: those
    matches are applied from the live FIFA feed instead, which both
    updates faster and avoids double-counting.
    """
    stale = (
        not CSV_PATH.exists()
        or time.time() - CSV_PATH.stat().st_mtime > MAX_AGE_SECONDS
    )
    if refresh and stale:
        try:
            resp = requests.get(URL, headers=HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            CSV_PATH.write_bytes(resp.content)
        except requests.RequestException as exc:
            if not CSV_PATH.exists():
                raise
            log.warning("results download failed (%s), using cached copy", exc)

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[~((df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01"))]
    return df.sort_values("date").reset_index(drop=True)
