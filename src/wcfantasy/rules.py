"""Game rules for FIFA Fantasy (World Cup 2026).

Source: the official "How to Play" page (full text provided by the user
on 2026-06-10), plus the in-game onboarding and FAQ. Rounds are numbered
1..8: 1-3 group-stage matchdays, 4 R32, 5 R16, 6 QF, 7 SF, 8 Final.
Items marked TODO(verify) still need an in-app check at matchday 1.
"""

BUDGET = 100.0  # $m, group stage
# Applied automatically once round 3 locks and R32 transfers open.
KNOCKOUT_BUDGET_BONUS = 5.0  # $m

# Player prices are FIXED for the whole tournament (official rule). The
# price is the experts' pre-tournament judgement of scoring potential,
# never a market signal: there is no buy-low/sell-high play.
PRICES_ARE_FIXED = True

SQUAD_SIZE = 15
SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # official

# Max players from the same country, by round (official stage table).
MAX_PER_SQUAD_BY_ROUND = {1: 3, 2: 3, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 8}


def max_per_squad(round_no: int) -> int:
    return MAX_PER_SQUAD_BY_ROUND[min(max(round_no, 1), 8)]


# Valid starting formations (DEF-MID-FWD, always 1 GK, 11 starters).
# Official picker list; a new team defaults to 4-4-2.
FORMATIONS = [
    (3, 4, 3), (3, 5, 2),
    (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 3, 2), (5, 4, 1),
]

CAPTAIN_MULTIPLIER = 2
# The vice-captain scores double ONLY if the captain plays no minute AND
# no manual change was made during the live round: any manual sub or
# (vice-)captain change forfeits both auto-subs and the vice insurance.
HAS_VICE_CAPTAIN = True

# Bench is ordered: first/second/third outfield sub (auto-sub priority)
# plus the backup GK.
BENCH_OUTFIELD_SLOTS = 3

# Substitution rules (official):
# - manual, during a live round: a starter whose match is COMPLETED can
#   be swapped with a bench player whose match has NOT started; the new
#   lineup must keep a valid formation;
# - automatic, at round end: DNP starters are replaced from the bench in
#   priority order, ONLY if no manual change was made during the round.

# ------------------------------------------------------------- transfers

# Free transfers per round window (None = unlimited). Round 1 is the
# pre-tournament build: unlimited edits until the first lockout. The R32
# window is unlimited too (de facto a free full rebuild).
FREE_TRANSFERS_BY_ROUND = {1: None, 2: 2, 3: 2, 4: None, 5: 4, 6: 4, 7: 5, 8: 6}

# Group stage only: 1 unused transfer carries over to the next group
# round. Only the round-3 window can receive a carryover (round 2 has no
# limited window before it, and nothing carries into the unlimited R32).
TRANSFER_CARRYOVER_MAX = 1
CARRYOVER_TARGET_ROUNDS = (3,)

# Exceeding the allocation is allowed: each extra transfer costs 3
# points, deducted at lockout.
EXTRA_TRANSFER_PENALTY = 3.0

# Transfers confirmed during a live round only apply from the next round.

# --------------------------------------------------------- suspensions

# Disciplinary suspensions (official). A sending-off (straight red or two
# yellows in one match), or a second yellow accumulated across matches in
# the same window, bans the player from the next match. FIFA wipes single
# yellow cards after the group stage and again after the quarter-finals,
# so a player can never miss the final for an early-round caution.
# Used to discount a known-suspended player's expected points to ~0.
YELLOW_CARDS_FOR_BAN = 2
# Round numbers after which the accumulated single-yellow tally resets:
# 3 = end of the group stage, 6 = end of the quarter-finals.
YELLOW_RESET_AFTER_ROUNDS = (3, 6)

# -------------------------------------------------------------- boosters

# One booster per round max, each usable once in the tournament. All of
# them can be deactivated before the round locks out, EXCEPT the
# Wildcard, which is final once confirmed.
BOOSTERS = {
    "maximum_captain": (
        "Double points to the highest-scoring player of your starting XI, "
        "auto-assigned as captain for the round (hindsight-optimal captain)"
    ),
    "twelfth_man": (
        "One additional scoring player for the round, picked from OUTSIDE "
        "your squad with no budget or country restriction; cannot be "
        "substituted, captained or transferred"
    ),
    "wildcard": (
        "Unlimited transfers for one round; NOT allowed for matchday 1 or "
        "the R32; irreversible once confirmed"
    ),
    "qualification": (
        "+2 points to each starting-XI player who progresses to the next "
        "knockout round or wins the final (needs >= 1 minute played; the "
        "captain's +2 is NOT doubled). Knockout stage only"
    ),
    "clean_sheet_shield": (
        "Knockout stage only: your goalkeeper, defenders and midfielders keep "
        "the clean-sheet bonus until their team concedes a SECOND goal "
        "(normally the first goal conceded already loses it). One knockout "
        "round, Final included. Revealed when the R32 opened (formerly the "
        "hidden 'Mystery' booster)."
    ),
}
MAX_BOOSTERS_PER_ROUND = 1
IRREVERSIBLE_BOOSTERS = ("wildcard",)
WILDCARD_EXCLUDED_ROUNDS = (1, 4)
QUALIFICATION_BOOSTER_ROUNDS = (4, 5, 6, 7, 8)  # knockout only
QUALIFICATION_BONUS = 2
QUALIFICATION_MIN_MINUTES = 1

# Clean Sheet Shield (revealed when the R32 opened; formerly the hidden
# "Mystery" booster). Knockout stage only: a player's clean-sheet bonus
# survives until his team concedes a SECOND goal.
CLEAN_SHEET_SHIELD_ROUNDS = (4, 5, 6, 7, 8)
CLEAN_SHEET_SHIELD_AVAILABLE_FROM_ROUND = 4
CLEAN_SHEET_SHIELD_GOALS_ALLOWED = 1
# Rough P(a team concedes EXACTLY one goal) in a knockout tie - the scenario
# the shield converts into a clean sheet. Calibratable once KO results accrue.
CLEAN_SHEET_SHIELD_CONCEDE_ONE_PROB = 0.30
# Minimum one-round EV (clean-sheet pts saved x that probability) worth a
# "candidate" call rather than holding the shield for a stronger round.
CLEAN_SHEET_SHIELD_MIN_EV = 2.5

# Old booster keys mapped onto a renamed one, so a recorded/typed legacy
# name still resolves after the Mystery reveal.
BOOSTER_ALIASES = {"mystery": "clean_sheet_shield"}

# --------------------------------------------------------------- scoring

# Official scoring table. Appearance points are CUMULATIVE (FPL-style):
# +1 for any minutes played, +1 more at 60 minutes, so a 60+ appearance
# is worth 2. VERIFIED at matchday 1 (2026-06-12): in the live feed the
# 60+ starters sit on 2 points, the short-cameo subs on 1.
SCORING_ALL = {
    "appearance": 1,        # any minutes played
    "appearance_60min": 1,  # additional point at 60+ minutes
    "assist": 3,
    "yellow_card": -1,
    "red_card": -2,
    "own_goal": -2,
    "penalty_won": 2,
    "penalty_conceded": -1,
}

GOAL_POINTS = {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5}
# Clean sheet requires 60+ minutes played.
CLEAN_SHEET_POINTS = {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0}
# GK and DEF: first goal conceded is free, each additional one is -1.
GOAL_CONCEDED_AFTER_FIRST = -1


def clean_sheet_points(position: str, goals_conceded: int, *, shield: bool = False) -> int:
    """Clean-sheet points for a player who played 60+ minutes. The bonus is
    normally lost on the FIRST goal conceded; with the Clean Sheet Shield
    booster active it survives until the second
    (CLEAN_SHEET_SHIELD_GOALS_ALLOWED). FWD always score 0."""
    allowed = CLEAN_SHEET_SHIELD_GOALS_ALLOWED if shield else 0
    return CLEAN_SHEET_POINTS.get(position, 0) if goals_conceded <= allowed else 0
PENALTY_SAVE_POINTS = 3  # GK, shootouts excluded
SAVES_PER_POINT = 3      # GK: +1 every 3 saves
TACKLES_PER_POINT = 3    # MID: +1 every 3 tackles
CHANCES_CREATED_PER_POINT = 2  # MID: +1 every 2 chances created
SHOTS_ON_TARGET_PER_POINT = 2  # FWD: +1 every 2 shots on target

# Bonus points.
DIRECT_FREE_KICK_BONUS = 1
# +2 if a player scores more than 4 points in a match while owned by
# fewer than 5% of all teams: cheap upside on differential picks.
SCOUTING_BONUS = 2
SCOUTING_BONUS_MIN_POINTS = 4
SCOUTING_BONUS_MAX_OWNERSHIP = 5.0  # percentSelected, in percent
