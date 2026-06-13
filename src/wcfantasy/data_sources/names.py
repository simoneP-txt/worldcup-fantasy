"""Shared team-name bridges to the FIFA squads.json names.

Both the bookmaker feed (The Odds API) and API-Football spell some
national teams differently from FIFA, and the mapping is identical for
the two, so it lives here once instead of being copied per module. Grow
it as new mismatches surface during the tournament.

(intl_results maps the OTHER direction - FIFA name -> the martj42
historical dataset - and keeps its own table.)
"""

# External provider name -> FIFA squads.json name.
EXTERNAL_TO_FIFA = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
}
