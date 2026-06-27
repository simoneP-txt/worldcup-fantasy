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

from .. import rules, tunables
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
    players: pd.DataFrame,
    availability: dict[int, float] | None = None,
    observed_starts: dict[int, float] | None = None,
) -> pd.Series:
    """P(player plays for his national team) in [0, 1].

    Depth chart, strongest signal last:
    1. the game's price rank within (squad, position) - a WEAK pre-tournament
       prior (it is near-flat for GKs and ranked the wrong keeper for Spain
       and Egypt, so it must never be the final word once data exists);
    2. `observed_starts` (who actually played the completed matches, from the
       FIFA Match Centre) overrides the price rank wherever we have it;
    3. starter_overrides.json (a deliberate manual correction) wins over both.
    The result is then scaled by an availability factor (injuries / confirmed
    lineups). Absent id = assumed fit (1.0).
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

    # Observed minutes beat the market prior: real starts reveal the true
    # depth chart the price only guesses at.
    if observed_starts and "id" in players.columns:
        observed = players["id"].map(observed_starts)
        out = observed.fillna(out)

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
    observed_starts: dict[int, float] | None = None,
) -> pd.Series:
    prior = price_prior(players)
    if club_ep is not None:
        w = WEIGHTS["club_prior_weight"]
        blended = w * club_ep + (1 - w) * prior
        prior = blended.fillna(prior)
    # Depth-chart starter prob scales the prior. The price rank is only a
    # weak prior; `observed_starts` (who actually played) supersedes it, and
    # observed form takes over the quality side separately below. The injury
    # / availability factor is applied to the final EP (not here), because an
    # injury is fresh news that past form must NOT override.
    prior = prior * starter_probability(players, None, observed_starts)

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


def _next_round_opponents(rounds: list[dict], next_round: int) -> dict[int, int]:
    """{squadId -> opponent squadId} for next_round's fixtures (both directions)."""
    rnd = next((r for r in rounds if r["id"] == next_round), None)
    opponent: dict[int, int] = {}
    if rnd is None:
        return opponent
    for m in rnd.get("tournaments", []):
        h, a = m.get("homeSquadId"), m.get("awaySquadId")
        if h is not None and a is not None:
            opponent[h] = a
            opponent[a] = h
    return opponent


def opponent_factor(
    players: pd.DataFrame,
    rounds: list[dict],
    next_round: int,
    ratings: dict[int, float] | None = None,
) -> pd.Series:
    """Scale next-round EP by fixture difficulty (Elo expected score).

    `ratings` (squadId -> Elo) comes from the progression run; pass it
    explicitly instead of reaching into another module's globals. 1.0
    everywhere when ratings are unavailable (offline fallback)."""
    neutral = pd.Series(1.0, index=players.index)
    opponent = _next_round_opponents(rounds, next_round)
    if not ratings or not opponent:
        return neutral

    def factor(squad_id: int) -> float:
        opp = opponent.get(squad_id)
        if opp is None or squad_id not in ratings or opp not in ratings:
            return 1.0
        expected = elo.expected_score(ratings[squad_id] - ratings[opp])
        return 1.0 + WEIGHTS["opponent_spread"] * (expected - 0.5)

    return players["squadId"].map(factor).astype(float)


def clean_sheet_probability(expected_score: float, base_rate: float, spread: float) -> float:
    """P(clean sheet) tilted around base_rate by the Elo expected score E in
    [0, 1]: base*(1 + spread*(2E - 1)), clipped to [0, 1]. An even tie (E=0.5)
    reads the base; a near-certain winner up to ~base*(1+spread), a near-certain
    loser down to ~base*(1-spread). Replaces the flat rate for DEF/MID."""
    return min(1.0, max(0.0, base_rate * (1.0 + spread * (2.0 * expected_score - 1.0))))


def team_clean_sheet(
    players: pd.DataFrame,
    rounds: list[dict],
    next_round: int,
    ratings: dict[int, float] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Per-player (P clean sheet, P concede exactly one) for next_round, by each
    squad's opponent (Elo). Even or unknown fixtures fall back to the flat base
    rate (WEIGHTS['flat_clean_sheet_rate']). P(concede exactly one) is the
    Poisson value consistent with P(0)=p_cs: lambda = -ln(p_cs), P(1)=lambda*p_cs
    - the scenario the Clean Sheet Shield converts into a clean sheet, so the
    shield and the defensive EP share one opponent-aware model."""
    base = WEIGHTS["flat_clean_sheet_rate"]
    opponent = _next_round_opponents(rounds, next_round)

    def pcs(squad_id) -> float:
        opp = opponent.get(squad_id)
        if not ratings or opp is None or squad_id not in ratings or opp not in ratings:
            return base
        expected = elo.expected_score(ratings[squad_id] - ratings[opp])
        return clean_sheet_probability(
            expected, base, tunables.tuned("clean_sheet_opponent_spread")
        )

    p_cs = players["squadId"].map(pcs).astype(float)
    p_cs.index = players.index
    lam = -np.log(p_cs.clip(lower=1e-6, upper=1.0))
    p_one = (lam * p_cs).clip(lower=0.0, upper=1.0)
    return p_cs, p_one


def clean_sheet_adjustment(players: pd.DataFrame, p_cs: pd.Series) -> pd.Series:
    """Opponent-aware clean-sheet correction (points) for DEF/MID - the
    positions whose prior used the flat rate. (p_cs - base) * clean-sheet
    points, 0 for GK (its prior already uses the keeper's own club CS rate) and
    FWD (no CS points). It adds the defender-specific opponent tilt the blanket
    opponent factor under-weights; 0 everywhere when ratings are unavailable
    (p_cs == base)."""
    base = WEIGHTS["flat_clean_sheet_rate"]
    pts = players["position"].map(rules.CLEAN_SHEET_POINTS).fillna(0.0)
    defmid = players["position"].isin(["DEF", "MID"]).astype(float)
    return ((p_cs - base) * pts * defmid).astype(float)


def fixture_adjusted_ep(
    players: pd.DataFrame,
    base: pd.Series,
    rounds: list[dict],
    next_round: int,
    ratings: dict[int, float] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Apply next-round fixture context to a per-match base EP: opponent
    difficulty (opponent_factor) plus the DEF/MID clean-sheet tilt
    (clean_sheet_adjustment). Returns (ep_match, p_concede_one).

    Available players (base > 0) keep the 0.5 floor expected_points_per_match
    uses; unavailable ones (base == 0) stay 0 - the clean-sheet tilt never
    resurrects a ruled-out or eliminated player, and never drives a real starter
    out of the ILP's ep_match>0 pool. Shared by advise (expected_value) and live
    so the two value still-to-play players identically."""
    p_cs, p_one = team_clean_sheet(players, rounds, next_round, ratings)
    ep = base * opponent_factor(players, rounds, next_round, ratings)
    ep = ep + clean_sheet_adjustment(players, p_cs).where(base > 0, 0.0)
    ep = ep.where(base <= 0, ep.clip(lower=0.5))
    return ep, p_one


def ep_basis_and_confidence(
    players: pd.DataFrame,
    club_matched: pd.Series,
    *,
    full: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Per player, (basis label, confidence level) describing WHAT backs his EP.

    The point estimate hides a stack of approximations whose strength varies by
    player: observed WC minutes are firm, a club projection transferred to the
    national team is a proxy, and a price-only prior is the weakest (no club
    match, no minutes). This surfaces that so the user knows when to trust a
    number. `full` is the matches needed for observed form to fully take over
    (WEIGHTS['matches_for_full_form'] by default). No model decision reads these
    columns - they are purely informational.
    """
    full = full if full is not None else WEIGHTS["matches_for_full_form"]
    form_frac = (players["matchesPlayed"].astype(float).clip(upper=full) / full)
    matched = club_matched.reindex(players.index).fillna(False).astype(bool)
    basis, level = [], []
    for frac, has_club in zip(form_frac, matched):
        if frac >= 1.0:
            basis.append("observed WC form"); level.append("high")
        elif has_club and frac > 0:
            basis.append("club proxy + partial form"); level.append("medium")
        elif has_club:
            basis.append("club proxy"); level.append("medium")
        elif frac >= 0.5:
            basis.append("partial WC form"); level.append("medium")
        elif frac > 0:
            basis.append("scant WC form + price prior"); level.append("low")
        else:
            basis.append("price prior only"); level.append("low")
    return (
        pd.Series(basis, index=players.index),
        pd.Series(level, index=players.index),
    )


def expected_value(
    players: pd.DataFrame, squads: list[dict], rounds: list[dict], next_round: int
) -> pd.DataFrame:
    """Adds ep_match, future_rounds and ep_horizon columns.

    ep_match is next-round specific (opponent factor applied);
    ep_horizon = base ep * expected future rounds of the player's team,
    the total points a player is expected to add from here to the end.
    """
    out = players.copy()
    # Progression first: it computes the Elo ratings used by the opponent
    # factor. Run the (seeded) simulation once and derive everything from the
    # one survival curve - the horizon weight, the next-round advance
    # probability for the Qualification booster - then read the ratings it
    # produced and pass them down explicitly (no hidden cross-module coupling).
    curve = progression.survival_curve(players, squads, rounds)
    future = progression.future_rounds_from_curve(curve, next_round)
    advance = progression.advance_from_curve(curve, next_round)
    ratings = progression.last_run_info.get("ratings")
    club_ep = club_prior(players)
    last_run_info["club_matched"] = int(club_ep.notna().sum())
    availability = availability_mod.availability_factors(
        squads, players, rounds, next_round
    )
    # Observed starts (who actually played the completed matches) supersede
    # the price depth chart in starter_probability. {} before any match.
    observed = availability_mod.observed_start_factors(
        rounds, next_round, squads, players
    )
    last_run_info["observed_start_ids"] = set(observed)
    base = expected_points_per_match(players, club_ep, availability, observed)
    # Next-round fixture context: opponent difficulty + the DEF/MID clean-sheet
    # tilt (both Elo-driven), and the per-team P(concede exactly one) the Clean
    # Sheet Shield advice reads.
    ep_match, p_concede_one = fixture_adjusted_ep(players, base, rounds, next_round, ratings)
    out["ep_match"] = ep_match
    out["p_concede_one"] = p_concede_one
    out["future_rounds"] = out["squadId"].map(future).fillna(0.0)
    # P(team wins its next match), for the Qualification booster (advancing
    # ONE round, not going deep). Distinct from future_rounds on purpose.
    out["p_advance"] = out["squadId"].map(advance).fillna(0.0)
    out["ep_horizon"] = base * out["future_rounds"]
    # P(plays) for the report: depth chart x availability. The depth-chart
    # part scaled the prior; the availability part scaled the final EP.
    out["p_play"] = starter_probability(players, availability, observed)
    # What backs each EP (observed form / club proxy / price-only): purely
    # informational, so the user knows how much the number is leaning on
    # approximations this round.
    basis, confidence = ep_basis_and_confidence(players, club_ep.notna())
    out["ep_basis"] = basis
    out["ep_confidence"] = confidence
    return out
