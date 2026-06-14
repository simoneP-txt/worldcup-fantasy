"""Tournament withdrawals from the Wikipedia squads page (MediaWiki API).

The only free, license-clean source for fitness injuries the FIFA feeds
do not carry: FIFA publishes lineups and bookings, never a medical
bulletin. The "2026 FIFA World Cup squads" article notes players ruled out
through injury (pre-tournament replacements and in-tournament withdrawals)
in prose; this maps those to a 0.0 availability factor.

Best-effort and deliberately conservative: wikitext is unstructured and
varies, so the parser only fires on an explicit injury/withdrawal phrase
near a linked player name, and a missed note simply leaves that player on
his depth-chart prior. Returns {} on any failure (network, schema), like
the other data_sources. Content is CC BY-SA; programmatic use via the
public MediaWiki API is permitted.
"""

import logging
import re

import pandas as pd

import requests

from ..config import EXTERNAL_DIR, HTTP_HEADERS
from .club_stats import match_name

log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE = "2026 FIFA World Cup squads"
CACHE = EXTERNAL_DIR / "wikipedia_squads.json"

# The SUBJECT who withdrew, captured tightly to avoid the replacement
# trap: "[[Injured]] withdrew injured and was replaced by [[Healthy]]"
# must flag only the injured player, never the healthy replacement. So
# the linked name must sit immediately before a withdrawal verb, with an
# injury noun in the same clause (no sentence break). "Replaced" is
# deliberately NOT a trigger verb - it points at the wrong player.
WITHDRAWAL = re.compile(
    r"\[\[([^\]|#]+?)(?:\|[^\]]*)?\]\]"          # [[Subject]] or [[Target|Display]]
    r"[^.\[\]]{0,40}"                              # short gap, no break / new link
    r"(?:withdrew|was ruled out|ruled out|withdrawn|pulled out|left the squad)"
    r"[^.]{0,60}?"                                 # up to the injury noun, same clause
    r"(?:injur|fitness|torn|fracture|surgery|ruptur|ligament|operation)",
    re.IGNORECASE,
)
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.IGNORECASE | re.DOTALL)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")

# Defensive cap: a real tournament flags a few dozen withdrawals at most.
# Far more means the wikitext format drifted and the parser is matching
# noise, so distrust the whole pass and return {} (degrade to the other
# signals) rather than flood the optimizer with false 0.0s.
MAX_PLAUSIBLE_WITHDRAWALS = 40


def _strip_markup(wikitext: str) -> str:
    """Drop citations/templates and flatten newlines so the withdrawal
    regex sees clean prose (citation titles also mention injuries)."""
    text = _REF.sub(" ", wikitext)
    text = _TEMPLATE.sub(" ", text)
    return text.replace("\n", " ")


def parse_withdrawals(
    wikitext: str, squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """{fantasy id -> 0.0} for players noted as withdrawn through injury.

    Pure (no network), unit-tested. Conservative by design: only the
    player named immediately before a withdrawal verb (with an injury noun
    in the same clause) is flagged, and only if he fuzzy-matches a fantasy
    player. Pre-tournament replacements drop the injured player from the
    squad, so this usually flags nobody currently - it exists to catch a
    genuine in-tournament season-ending injury to a player still in the
    feed.
    """
    names = list(players["name"])
    ids = list(players["id"])
    out: dict[int, float] = {}
    for subject in WITHDRAWAL.findall(_strip_markup(wikitext)):
        matched = match_name(subject.strip(), names, cutoff=0.9)
        if matched is None:
            continue
        pid = ids[names.index(matched)]
        out[int(pid)] = 0.0
    if len(out) > MAX_PLAUSIBLE_WITHDRAWALS:
        log.warning(
            "Wikipedia withdrawals parser matched %d players (> %d): the page "
            "format likely drifted, discarding the pass", len(out),
            MAX_PLAUSIBLE_WITHDRAWALS,
        )
        return {}
    return out


def _fetch_wikitext() -> str | None:
    """Raw wikitext of the squads article; cached, falls back on failure."""
    try:
        resp = requests.get(
            API_URL,
            params={
                "action": "parse",
                "page": PAGE,
                "prop": "wikitext",
                "format": "json",
                "formatversion": "2",
            },
            headers=HTTP_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["parse"]["wikitext"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(text, encoding="utf-8")
        return text
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("Wikipedia squads fetch failed (%s)", exc)
        if CACHE.exists():
            return CACHE.read_text(encoding="utf-8")
        return None


def withdrawal_factors(
    squads: list[dict], players: pd.DataFrame
) -> dict[int, float]:
    """P(plays)=0.0 for players Wikipedia notes as injured/withdrawn; {}
    on any failure so the caller falls back to the other signals."""
    text = _fetch_wikitext()
    if not text:
        return {}
    return parse_withdrawals(text, squads, players)
