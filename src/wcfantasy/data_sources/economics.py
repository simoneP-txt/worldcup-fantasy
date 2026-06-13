"""Country economic indicators (World Bank API).

Inspired by Joachim Klement's World Cup models, which blend economic
factors (wealth, population => talent pool and infrastructure) with
football data and called recent winners well. Treated with care: that
track record may be luck, so the output is a SMALL Elo adjustment in
the progression model, not a driver. Weights live in this module.

Data is cached in data/external/ for 7 days; a failed download falls
back to the cached copy, and the caller treats a missing file as zero
adjustment everywhere.
"""

import json
import logging
import math
import time

import requests

from ..config import EXTERNAL_DIR, HTTP_HEADERS

log = logging.getLogger(__name__)

CACHE = EXTERNAL_DIR / "worldbank_econ.json"
MAX_AGE_SECONDS = 7 * 24 * 3600

GDP_PC = "NY.GDP.PCAP.CD"  # GDP per capita, current US$
POPULATION = "SP.POP.TOTL"

# Elo points per standard deviation of the (log-scale) indicator.
GDP_ELO_PER_SD = 20.0
POP_ELO_PER_SD = 10.0

# FIFA squad name -> ISO3 (England/Scotland share the UK aggregate).
ISO3 = {
    "Algeria": "DZA", "Argentina": "ARG", "Australia": "AUS",
    "Austria": "AUT", "Belgium": "BEL", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Cabo Verde": "CPV", "Canada": "CAN",
    "Colombia": "COL", "Congo DR": "COD", "Croatia": "HRV",
    "Curaçao": "CUW", "Czechia": "CZE", "Côte d'Ivoire": "CIV",
    "Ecuador": "ECU", "Egypt": "EGY", "England": "GBR", "France": "FRA",
    "Germany": "DEU", "Ghana": "GHA", "Haiti": "HTI", "IR Iran": "IRN",
    "Iraq": "IRQ", "Japan": "JPN", "Jordan": "JOR", "Korea Republic": "KOR",
    "Mexico": "MEX", "Morocco": "MAR", "Netherlands": "NLD",
    "New Zealand": "NZL", "Norway": "NOR", "Panama": "PAN",
    "Paraguay": "PRY", "Portugal": "PRT", "Qatar": "QAT",
    "Saudi Arabia": "SAU", "Scotland": "GBR", "Senegal": "SEN",
    "South Africa": "ZAF", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "CHE", "Tunisia": "TUN", "Türkiye": "TUR", "USA": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB",
}


def _fetch_indicator(indicator: str, iso_codes: list[str]) -> dict[str, float]:
    """Most recent non-empty value per ISO3 code (mrnev= rejects
    multi-country requests, so query a year range and keep the latest)."""
    url = (
        f"https://api.worldbank.org/v2/country/{';'.join(iso_codes)}"
        f"/indicator/{indicator}?format=json&date=2018:2025&per_page=500"
    )
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    out: dict[str, tuple[str, float]] = {}
    for row in payload[1]:
        if row.get("value") is None:
            continue
        iso, year = row["countryiso3code"], row["date"]
        if iso not in out or year > out[iso][0]:
            out[iso] = (year, float(row["value"]))
    return {iso: value for iso, (_, value) in out.items()}


def load_indicators(refresh: bool = True) -> dict[str, dict[str, float]]:
    """{iso3: {gdp_pc, population}}, cached weekly."""
    stale = not CACHE.exists() or time.time() - CACHE.stat().st_mtime > MAX_AGE_SECONDS
    if refresh and stale:
        try:
            codes = sorted(set(ISO3.values()))
            gdp = _fetch_indicator(GDP_PC, codes)
            pop = _fetch_indicator(POPULATION, codes)
            data = {
                c: {"gdp_pc": gdp.get(c), "population": pop.get(c)} for c in codes
            }
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except requests.RequestException as exc:
            if not CACHE.exists():
                raise
            log.warning("World Bank download failed (%s), using cache", exc)
    return json.loads(CACHE.read_text(encoding="utf-8"))


def elo_adjustments(squads: list[dict]) -> dict[int, float]:
    """squadId -> Elo bonus from z-scored log GDP per capita and population."""
    data = load_indicators()

    def logs(key):
        vals = {}
        for s in squads:
            iso = ISO3.get(s["name"])
            v = (data.get(iso) or {}).get(key) if iso else None
            if v:
                vals[s["id"]] = math.log(v)
        return vals

    out: dict[int, float] = {}
    for key, weight in (("gdp_pc", GDP_ELO_PER_SD), ("population", POP_ELO_PER_SD)):
        vals = logs(key)
        if len(vals) < 2:
            continue
        mean = sum(vals.values()) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals.values()) / len(vals)) ** 0.5
        for sid, v in vals.items():
            out[sid] = out.get(sid, 0.0) + weight * (v - mean) / sd
    return out
