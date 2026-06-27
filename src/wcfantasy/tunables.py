"""Hand-tuned model constants in one place.

These are the JUDGEMENT calls of the engine: numbers that change real
decisions (which player to own, who to captain or sub, when to fire a
booster) but that are NOT game rules (those live in rules.py). Each scalar
below carries a PROVENANCE line (where the value came from) and a SENSITIVITY
line (what moves if you change it).

A SUBSET of these is CALIBRATABLE: the scalar is the hand-picked DEFAULT, and
once enough World Cup data accrues `calibrate.py` may fit a better value and
write it to the 'tunables' block of calibration.json. `apply_calibration()`
(called by the advise / live entrypoints, never at import) layers that fitted
value into the live `TUNED` dict, and the consumers read it via `tuned(name)`
at call time. Same contract as player_points.WEIGHTS: tests call the model
functions directly and see the built-in defaults. The rest stay pure
hand-tuned scalars. (The clean-sheet BASE rate and the opponent EP spread are
in player_points.WEIGHTS, the other calibratable home; only the clean-sheet
opponent SPREAD lives here.)

Imports only the config path (for the calibration file) - still effectively a
leaf, so squad / captain / live / rules / progression re-export these names and
existing references and test monkeypatch points keep working unchanged.
"""

import json

from .config import STATE_DIR

# --------------------------------------------------------------- squad ILP

# Bench players count in the squad objective at a fraction of a starter
# (rotation / injury cover and the 12th Man booster).
# PROVENANCE: hand-picked.  SENSITIVITY: raising it makes the ILP pay up for
# a strong bench at the expense of the starting XI.
BENCH_WEIGHT = 0.15

# The backup GK is worth far less than an outfield sub: you start exactly one
# keeper and the #2 only scores when the #1 plays zero minutes with no manual
# change that round (~P(DNP)).
# PROVENANCE: hand-picked.  SENSITIVITY: at BENCH_WEIGHT the ILP overpays for
# an expensive backup keeper instead of spending on the XI (regression-tested
# in test_optimizer).
BENCH_GK_WEIGHT = 0.03

# Microscopic price tiebreak among ~worthless backup keepers: prefer the
# cheapest so the budget is freed for the XI.
# PROVENANCE: hand-picked.  SENSITIVITY: must stay far below any real
# bench-value gap, or it would start ranking keepers by price.
BENCH_GK_PRICE_TIEBREAK = 0.005

# Future rounds enter the squad score at this discount so the optimizer grabs
# likely-finalist players early without mortgaging the next round.
# PROVENANCE: hand-picked.  SENSITIVITY: higher => more weight on deep runs
# (who to OWN), lower => almost purely next-round.
HORIZON_WEIGHT = 0.35

# ----------------------------------------------- captain / lineup option value

# Captain match scores are right-skewed (a haul is a fat upside tail), so the
# value of the live armband roll is modelled on a LOGNORMAL with this
# coefficient of variation (SD / mean) - see captain._expected_max.
# PROVENANCE: hand-picked pending real WC captain-score history (calibratable).
# SENSITIVITY: higher cv => fatter tail => the roll-the-armband premium grows.
CAPTAIN_SCORE_CV = 0.65

# Floor on the captain-score spread so a low-EP player still carries
# optionality. Expressed as an absolute SD, converted to an effective
# cv = max(cv, sd_floor / mean) inside the lognormal helpers.
# PROVENANCE: hand-picked.  SENSITIVITY: raises the option value of weak early
# starters specifically.
CAPTAIN_SCORE_SD_FLOOR = 2.0

# Points forfeited by the FIRST live armband move (it cancels the auto-sub
# insurance and the vice-captain double), charged once.
# PROVENANCE: order-of-magnitude estimate.  SENSITIVITY: higher => the planner
# is more reluctant to set up a roll.
CAPTAIN_SWITCH_COST = 0.7

# Ignore starters below this fraction of the top EP as captain candidates.
# PROVENANCE: replaced the old hard 0.92 gate that rejected the option play.
# SENSITIVITY: lower => more (weaker) early options considered for the roll.
CAPTAIN_CONTENDER_FLOOR = 0.70

# Don't churn the armband for a doubled-option premium below this.
# PROVENANCE: hand-picked.  SENSITIVITY: a deadband against EP noise.
CAPTAIN_MIN_PREMIUM = 0.1

# Weight of the live-sub option a LATE starter forfeits by starting (the XI
# timing term). 0 => rank the XI purely by this-round EP.
# PROVENANCE: hand-picked.  SENSITIVITY: only flips genuinely close,
# same-position start/bench calls.
LINEUP_OPTION_WEIGHT = 0.10

# ------------------------------------------------------------------- live

# Rough P(a still-to-play starter ends with no minutes): late knock, surprise
# rotation. Used ONLY to price the auto-sub insurance the first manual change
# forfeits, never the captain/sub EP comparison itself.
# PROVENANCE: hand-picked.  SENSITIVITY: scales that one insurance cost.
DNP_PROB = 0.06

# ----------------------------------------------------- clean sheet shield booster

# Minimum one-round EV (clean-sheet points saved x P(concede exactly one))
# before the shield is RECOMMENDED to use, rather than held for a stronger KO
# round. A real KO defensive block tends to land ~+5..+9 EV, so the old 2.5
# fired on almost anything; 6.0 means "a full, favoured block" and keeps the
# one-shot booster for a standout round. This is a DECISION / opportunity-cost
# threshold (one shield across up to 5 KO rounds), NOT a quantity fit from
# observed points - so calibrate.py does not fit it; tune it by hand from the
# per-round EVs the advice now prints (see the Clean Sheet Shield note).
# PROVENANCE: hand-picked.  SENSITIVITY: higher => the shield is recommended
# only on stronger blocks / later rounds; lower => it fires more readily.
CLEAN_SHEET_SHIELD_MIN_EV = 6.0

# Fallback P(a team concedes EXACTLY one goal) in a KO tie - the scenario the
# shield converts into a clean sheet. Used only when no opponent-aware value is
# supplied; player_points now passes a per-team estimate where it has one.
# PROVENANCE: rough league-typical guess.  SENSITIVITY: scales the shield EV.
CLEAN_SHEET_SHIELD_CONCEDE_ONE_PROB = 0.30

# --------------------------------------------------- defensive / clean-sheet model

# How strongly the opponent (Elo expected score) bends the clean-sheet rate
# around its base (the base itself is player_points.WEIGHTS["flat_clean_sheet
# _rate"], which calibrate.py may fit). 0 => flat base everywhere (the old
# behaviour); 1 => a team certain to win reads ~2x the base, a certain loser
# ~0. The same per-team P(clean sheet) also drives the Clean Sheet Shield via a
# Poisson-consistent P(concede exactly one) = -ln(p_cs) * p_cs.
# PROVENANCE: hand-picked.  SENSITIVITY: the opponent spread of CS probability,
# hence every DEF/MID clean-sheet contribution and the shield EV.
CLEAN_SHEET_OPPONENT_SPREAD = 1.0

# ------------------------------------------- progression: group scoreline texture

# Average goals in a group-stage match, split between the two sides by their
# (market-or-Elo) expected score so the simulated GROUP RANKING can use goal
# difference and goals-for - FIFA's first tiebreakers - instead of an Elo coin.
# The match OUTCOME (win/draw/loss) stays the calibrated one; this only paints a
# plausible scoreline of the right sign for the tiebreak.
# PROVENANCE: long-run WC group-stage average.  SENSITIVITY: only the GD /
# goals-for spread of the group ranking, never the qualification probabilities.
GROUP_GOALS_PER_MATCH = 2.6

# --------------------------------------------- progression: outright-odds nudge

# Elo nudge = GAIN * ln(p_market / p_simulated), clamped to +-MAX, so the
# bookmaker market CORRECTS the model rather than replacing it.
# PROVENANCE: hand-picked.  SENSITIVITY: GAIN = how hard the market pulls,
# MAX = the most any one team's rating can move per run.
ODDS_NUDGE_GAIN = 60.0
ODDS_NUDGE_MAX = 80.0


# --------------------------------------------------------- calibratable subset

# The tunables `calibrate.py` may fit from observed World Cup data. Each keeps
# its hand-picked scalar above as the DEFAULT (and the re-exported name, used as
# a fallback / explicit-arg value); a calibration run layers a fitted value into
# TUNED, which consumers read via tuned() at call time. See the module docstring.
_CALIBRATABLE_DEFAULTS = {
    "captain_score_cv": CAPTAIN_SCORE_CV,
    "clean_sheet_opponent_spread": CLEAN_SHEET_OPPONENT_SPREAD,
    "group_goals_per_match": GROUP_GOALS_PER_MATCH,
}
CALIBRATABLE = frozenset(_CALIBRATABLE_DEFAULTS)
TUNED = dict(_CALIBRATABLE_DEFAULTS)

CALIBRATION_FILE = STATE_DIR / "calibration.json"


def tuned(name: str) -> float:
    """Current value of a calibratable tunable: the fitted one when a
    calibration is loaded, else the hand-picked default. Consumers call this at
    run time so a written calibration takes effect without re-importing."""
    return TUNED.get(name, _CALIBRATABLE_DEFAULTS[name])


def apply_calibration() -> dict:
    """Layer the 'tunables' block of calibration.json over the defaults into
    TUNED. Mirrors player_points.apply_calibration: called by the advise / live
    entrypoints at runtime (not at import), idempotent, and resilient to an
    absent or corrupt file (defaults untouched). Unknown or non-numeric keys are
    ignored. Returns TUNED."""
    values: dict = {}
    if CALIBRATION_FILE.exists():
        try:
            blob = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            values = blob.get("tunables", {}) or {}
        except (json.JSONDecodeError, OSError):
            values = {}
    merged = dict(_CALIBRATABLE_DEFAULTS)
    merged.update(
        {k: float(v) for k, v in values.items()
         if k in _CALIBRATABLE_DEFAULTS and isinstance(v, (int, float))}
    )
    TUNED.clear()
    TUNED.update(merged)
    return TUNED
