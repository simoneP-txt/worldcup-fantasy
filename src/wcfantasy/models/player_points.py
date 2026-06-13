"""Expected fantasy points per player.

Signal priority:

1. 25/26 club season stats (FBref via data_sources.club_stats),
   projected per-match through the OFFICIAL scoring table - the prior
   for everyone confidently name-matched;
2. price/ownership heuristic where no club match exists (prices are
   fixed: the price is the experts' pre-tournament judgement, a decent
   prior but never a market signal);
3. observed World Cup form takes over with weight proportional to
   matches played.

On top of the per-match figure, `expected_value` adds the scouting
bonus EV for sub-5%-owned players, scales the next round by an
Elo-based opponent factor, and weights the horizon by the expected
future rounds of the player's national team (progression model).

National-team starter status matters as much as quality: a 3rd-choice
keeper with a stellar club season (e.g. a Joan Garcia type) must not
outrank the actual #1. The game's own prices embed the experts' depth
chart - within each (squad, position) the expected starters are priced
above the backups - so the price rank becomes a starter probability
that scales the prior. On top of it, an `availability` factor from
data_sources.availability (injuries / confirmed lineups) discounts
doubtful or injured players: P(plays) = depth-chart starter prob x
availability. Observed WC form gradually overrides the inference once
real minutes are on the board.

Known approximations, documented on purpose: club numbers are used as
if they transferred 1:1 to the national team, and DEF/MID clean-sheet
probability is a flat league-typical rate (FBref has no outfield CS).
"""

import json

import numpy as np
import pandas as pd

from .. import rules
from ..config import EXTERNAL_DIR, STATE_DIR
from ..data_sources import availability as availability_mod
from ..data_sources import club_stats
from . import elo, progression

# Tunable model weights. Built-in defaults below; a calibration run
# (wcfantasy.calibrate) may write data/state/calibration.json, and the
# advise / live entrypoints call apply_calibration() to layer it on top.
# Tests call the model functions directly, so they always see these
# defaults (apply_calibration is never invoked at import).
_WEIGHT_DEFAULTS = {
    # Price fallback prior: a 4.5$m squad player ~2 pts, an 11$m star ~5.5.
    "price_slope": 0.55,
    "price_intercept": -0.5,
    # Ownership adds at most ~1 pt for the most-picked players.
    "ownership_weight": 0.01,
    # The club projection runs hotter than the price prior (different
    # scoring environment, generous proxies) and only Big-5 players have
    # one, so shrink it halfway toward the price prior: the scales stay
    # comparable while the club ordering survives.
    "club_prior_weight": 0.5,
    # DEF/MID clean-sheet probability without team-level data.
    "flat_clean_sheet_rate": 0.30,
    # Opponent factor range: ~0.75 vs far stronger, 1.0 even, ~1.25 weaker.
    "opponent_spread": 0.5,
    # Matches of observed WC form before form fully replaces the prior.
    "matches_for_full_form": 4.0,
}

CALIBRATION_FILE = STATE_DIR / "calibration.json"

WEIGHTS = dict(_WEIGHT_DEFAULTS)


def apply_calibration() -> dict:
    """Layer calibration.json over the built-in defaults into WEIGHTS.

    Called by the advise / live entrypoints at runtime (not at import),
    so a written calibration takes effect for real runs while tests that
    call the model functions directly keep the defaults. Idempotent;
    unknown or non-numeric keys are ignored, and an absent/corrupt file
    leaves the defaults untouched.
    """
    weights = dict(_WEIGHT_DEFAULTS)
    if CALIBRATION_FILE.exists():
        try:
            blob = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            weights.update(
                {k: float(v) for k, v in blob.get("weights", {}).items()
                 if k in _WEIGHT_DEFAULTS}
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    WEIGHTS.clear()
    WEIGHTS.update(weights)
    return WEIGHTS


# P(starts for the national team) by within-(squad, position) price
# rank. Rank 1 keeps full value; the table lengths roughly match how
# many of each position a 26-man squad carries, anything deeper gets
# the floor. Tuned so an XI of true starters always dominates backups.
STARTER_PROB_BY_RANK = {
    "GK": (1.0, 0.10, 0.03),
    "DEF": (1.0, 0.95, 0.90, 0.80, 0.55, 0.30, 0.15, 0.08),
    "MID": (1.0, 0.95, 0.90, 0.80, 0.60, 0.35, 0.18, 0.10),
    "FWD": (1.0, 0.90, 0.65, 0.35, 0.15),
}
STARTER_PROB_FLOOR = 0.05
# No starter is ever a literal certainty (late knocks, rotation,
# suspensions): the depth-chart probability is capped so a nailed rank-1
# starter reads ~0.93, not 1.0. Applied to the whole table as a single
# multiplier, which preserves the depth-chart ordering.
NAILED_STARTER_CEILING = 0.93
STARTER_OVERRIDES_FILE = EXTERNAL_DIR / "starter_overrides.json"

# Filled by expected_value on each run; advise reports it to the user.
last_run_info: dict = {}


def price_prior(players: pd.DataFrame) -> pd.Series:
    return (
        WEIGHTS["price_intercept"]
        + WEIGHTS["price_slope"] * players["price"]
        + WEIGHTS["ownership_weight"] * players["percentSelected"]
    ).clip(lower=0.5)


def starter_probability(
    players: pd.DataFrame, availability: dict[int, float] | None = None
) -> pd.Series:
    """P(player plays for his national team) in [0, 1].

    Depth chart from the game's own prices (within (squad, position) the
    experts price expected starters above backups; percentSelected
    breaks ties) times an availability factor from injuries / confirmed
    lineups. starter_overrides.json still wins over the inference for
    manual depth-chart corrections.
    """
    key = players["price"] + players["percentSelected"] / 1000.0
    rank = key.groupby([players["squadId"], players["position"]]).rank(
        ascending=False, method="first"
    )

    def lookup(position: str, r: float) -> float:
        table = STARTER_PROB_BY_RANK.get(position, ())
        idx = int(r) - 1
        if idx < len(table):
            return table[idx] * NAILED_STARTER_CEILING
        return STARTER_PROB_FLOOR

    out = pd.Series(
        [lookup(p, r) for p, r in zip(players["position"], rank)],
        index=players.index,
    )

    if STARTER_OVERRIDES_FILE.exists() and "name" in players.columns:
        overrides = json.loads(STARTER_OVERRIDES_FILE.read_text(encoding="utf-8"))
        forced = players["name"].map(overrides)
        out = forced.fillna(out)

    if availability:
        # Injury / lineup factor multiplies the depth-chart probability:
        # a nailed starter (1.0) ruled out (0.0) drops to 0.0, a fit
        # backup keeps his small chance. Absent id = assumed fit (1.0).
        out = out * players["id"].map(availability).fillna(1.0)
    return out


def project_club_ep(club: pd.DataFrame) -> pd.Series:
    """Per-match expected points for each club-stats row, projected
    through the official scoring table (rules.py)."""
    games = club["games"].clip(lower=1)
    minutes = club["minutes"].clip(lower=1)
    avg_min = minutes / games
    share = (avg_min / 90).clip(upper=1.0)

    def per90(col: str) -> pd.Series:
        return club[col] / minutes * 90

    common = (
        club["position"].map(rules.GOAL_POINTS) * per90("goals")
        + rules.SCORING_ALL["assist"] * per90("assists")
        + rules.SCORING_ALL["yellow_card"] * per90("yellow")
        + rules.SCORING_ALL["red_card"] * per90("red")
        + rules.SCORING_ALL["penalty_won"] * per90("penalties_won")
        + rules.SCORING_ALL["penalty_conceded"] * per90("penalties_conceded")
    )

    gk = club["position"] == "GK"
    mid = club["position"] == "MID"
    fwd = club["position"] == "FWD"
    extra = pd.Series(0.0, index=club.index)
    extra[gk] = (
        per90("saves")[gk] / rules.SAVES_PER_POINT
        + rules.GOAL_CONCEDED_AFTER_FIRST
        * (per90("goals_conceded")[gk] - 1).clip(lower=0)
    )
    extra[mid] = (
        per90("tackles")[mid] / rules.TACKLES_PER_POINT
        + per90("chances_created")[mid] / rules.CHANCES_CREATED_PER_POINT
    )
    extra[fwd] = per90("shots_on_target")[fwd] / rules.SHOTS_ON_TARGET_PER_POINT

    cs_prob = pd.Series(WEIGHTS["flat_clean_sheet_rate"], index=club.index)
    cs_prob[gk] = (club["clean_sheets"] / games)[gk]
    cs_points = (
        club["position"].map(rules.CLEAN_SHEET_POINTS)
        * cs_prob
        * (avg_min >= 60)  # clean sheet requires 60+ minutes
    )

    appearance = rules.SCORING_ALL["appearance"] + rules.SCORING_ALL[
        "appearance_60min"
    ] * (avg_min / 60).clip(upper=1.0)

    return (appearance + share * (common + extra) + cs_points).clip(lower=0.5)


def club_prior(players: pd.DataFrame) -> pd.Series:
    """Per-match EP from club stats, NaN where no confident match."""
    club = club_stats.load()
    if club is None or club.empty:
        return pd.Series(np.nan, index=players.index)
    projected = project_club_ep(club)
    mapping = club_stats.match_players(players, club)
    out = players["id"].map(mapping).map(projected)
    out.index = players.index
    return out


def expected_points_per_match(
    players: pd.DataFrame,
    club_ep: pd.Series | None = None,
    availability: dict[int, float] | None = None,
) -> pd.Series:
    prior = price_prior(players)
    if club_ep is not None:
        w = WEIGHTS["club_prior_weight"]
        blended = w * club_ep + (1 - w) * prior
        prior = blended.fillna(prior)
    # Depth-chart starter prob scales the prior and IS overridden by
    # observed form: real minutes reveal the true lineup, so a "backup"
    # who keeps playing earns full value. The injury / availability factor
    # is separate (applied to the final EP below), because an injury is
    # fresh news that past form must NOT override.
    prior = prior * starter_probability(players, None)

    full = WEIGHTS["matches_for_full_form"]
    played = players["matchesPlayed"].clip(upper=full)
    blend = played / full  # full trust in observed form after `full` matches
    observed = players["form"].where(players["form"] > 0, prior)
    ep = (1 - blend) * prior + blend * observed

    # Scouting bonus EV: +2 when scoring >4 pts while owned by <5%.
    # P(>4 pts) is a crude increasing proxy of ep, capped: good enough
    # to nudge the optimizer toward differentials, not to dominate.
    p_gt4 = ((ep - 2.0) / 6.0).clip(0.0, 0.6)
    differential = players["percentSelected"] < rules.SCOUTING_BONUS_MAX_OWNERSHIP
    ep = ep + rules.SCOUTING_BONUS * p_gt4 * differential

    # Availability (injuries / confirmed lineups) discounts the whole EP,
    # form included: a ruled-out player is worth ~0 no matter his form.
    if availability:
        ep = ep * players["id"].map(availability).fillna(1.0)

    unavailable = players["status"].ne("playing") | players["isEliminated"].fillna(False)
    return ep.mask(unavailable, 0.0)


def opponent_factor(
    players: pd.DataFrame, rounds: list[dict], next_round: int
) -> pd.Series:
    """Scale next-round EP by fixture difficulty (Elo expected score).

    Uses the ratings the progression model leaves in last_run_info; 1.0
    everywhere when they are unavailable (offline fallback)."""
    neutral = pd.Series(1.0, index=players.index)
    ratings = progression.last_run_info.get("ratings")
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    if not ratings or rnd is None:
        return neutral

    opponent: dict[int, int] = {}
    for m in rnd.get("tournaments", []):
        h, a = m.get("homeSquadId"), m.get("awaySquadId")
        if h is not None and a is not None:
            opponent[h] = a
            opponent[a] = h

    def factor(squad_id: int) -> float:
        opp = opponent.get(squad_id)
        if opp is None or squad_id not in ratings or opp not in ratings:
            return 1.0
        expected = elo.expected_score(ratings[squad_id] - ratings[opp])
        return 1.0 + WEIGHTS["opponent_spread"] * (expected - 0.5)

    return players["squadId"].map(factor).astype(float)


def expected_value(
    players: pd.DataFrame, squads: list[dict], rounds: list[dict], next_round: int
) -> pd.DataFrame:
    """Adds ep_match, future_rounds and ep_horizon columns.

    ep_match is next-round specific (opponent factor applied);
    ep_horizon = base ep * expected future rounds of the player's team,
    the total points a player is expected to add from here to the end.
    """
    out = players.copy()
    # Progression first: it also leaves the Elo ratings in last_run_info
    # for the opponent factor below.
    future = progression.expected_future_rounds(players, squads, rounds, next_round)
    club_ep = club_prior(players)
    last_run_info["club_matched"] = int(club_ep.notna().sum())
    availability = availability_mod.availability_factors(squads, players)
    base = expected_points_per_match(players, club_ep, availability)
    out["ep_match"] = base * opponent_factor(players, rounds, next_round)
    out["future_rounds"] = out["squadId"].map(future).fillna(0.0)
    out["ep_horizon"] = base * out["future_rounds"]
    # P(plays) for the report: depth chart x availability. The depth-chart
    # part scaled the prior; the availability part scaled the final EP.
    out["p_play"] = starter_probability(players, availability)
    return out
