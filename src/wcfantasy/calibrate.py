"""Offline calibration of the player-points priors against real WC data.

Re-runnable at the end of every round. It reads the LATEST local feeds
(run `advise` or `fetch` first so they are current — no network here),
compares the model's pre-form prior against the points players have
actually scored so far, and proposes refit weights.

Dry-run by default: it only prints a report. Pass `--write` to persist
the proposal to data/state/calibration.json, which player_points layers
on top of its built-in defaults via apply_calibration() (absent file =
pure defaults, so nothing changes until you write one).

It is built to stay inert until there is data: with fewer than
MIN_SAMPLES played-player observations it refuses to write and only
prints the report. The intended first real run is at the end of round 2
(2026-06-24); a peek after round 1 (2026-06-18) is fine as a dry run.

What it fits (deliberately a SMALL, stable set — two rounds is little
data, so we do not touch everything):
  - price_slope / price_intercept: least-squares of observed
    points-per-match on price;
  - club_prior_weight: the club/price blend weight that best matches
    observed, ranking (corr) first then error (mae).
Each candidate is GATED: it enters the proposal only when it beats the
current default on the metrics that drive squad choice (is_improvement).
A least-squares fit can lower the average bias while RAISING the typical
error on the right-skewed points distribution, and an MAE-only blend
search collapses to a noise-driven corner early in the tournament — the
gate rejects both, so the proposal stays empty until a refit genuinely
helps. The rest of WEIGHTS keeps its default. See CALIBRATION.md.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console

from . import feed_status, features, fifa_client, tunables
from .models import elo, player_points

console = Console()

CALIBRATION_FILE = player_points.CALIBRATION_FILE

# A player must have played at least this many WC matches before his
# observed average is trusted as a calibration target.
MIN_MATCHES = 1
# Below this many observations the fit is too noisy to persist.
MIN_SAMPLES = 30

# A refit weight is proposed only when it clears these margins against the
# default. corr (ranking) leads because the optimizer orders players; mae is
# the secondary check. The epsilons stop round-to-round noise from flipping a
# weight in or out of the proposal.
CORR_EPS = 0.005
MAE_EPS = 0.01

# Calibratable TUNABLES (tunables.py). Each has its own sample gate so it stays
# inert until the data supports it, plus a minimum delta from the default so a
# near-identical fit does not churn the file. Deliberately strict: early-round
# samples are thin, so most stay unproposed for the first rounds.
MIN_GROUP_MATCHES = 16      # completed group matches before the goals mean is trusted
GROUP_GOALS_EPS = 0.1       # don't rewrite group goals for a sub-0.1 change
MIN_CV_MATCHES = 3          # matches a player needs before his score CV is usable
MIN_CV_PLAYERS = 15         # qualifying players before a captain CV is proposed
CAPTAIN_CALIBER_MEAN = 3.0  # only count players averaging captain-worthy points
CV_EPS = 0.05               # don't rewrite captain CV for a sub-0.05 change
MIN_CS_OBS = 60             # team-match clean-sheet observations before fitting the spread
CS_SPREAD_MIN = 0.05        # a fitted spread below this is treated as "flat"


def observed_per_match(df: pd.DataFrame) -> pd.Series:
    """Real points-per-appearance for players who have played (the
    calibration target); NaN for everyone who has not played yet."""
    played = df["matchesPlayed"] >= MIN_MATCHES
    return df["avgPoints"].astype(float).where(played)


def fit_price_linear(
    price: pd.Series, observed: pd.Series
) -> tuple[float, float] | None:
    """Least-squares (slope, intercept) of observed ~ intercept + slope*price.

    Ownership weight is intentionally left at its default: early in the
    tournament it is too collinear with price and too noisy to refit.
    """
    mask = observed.notna() & price.notna()
    if mask.sum() < 2:
        return None
    design = np.column_stack([price[mask].to_numpy(float), np.ones(int(mask.sum()))])
    (slope, intercept), *_ = np.linalg.lstsq(design, observed[mask].to_numpy(float), rcond=None)
    return float(slope), float(intercept)


def fit_club_weight(
    club_ep: pd.Series, price_prior: pd.Series, observed: pd.Series
) -> float | None:
    """Blend weight w in [0, 1] for w*club + (1-w)*price that best tracks
    observed, prioritising ranking (corr) then absolute error (mae) — the
    order the optimizer cares about. None when fewer than three players have
    both a club projection and minutes. Selection only; whether the winning
    w actually beats the 50/50 default is decided by is_improvement."""
    mask = observed.notna() & club_ep.notna() & price_prior.notna()
    if mask.sum() < 3:
        return None
    club, price, obs = club_ep[mask], price_prior[mask], observed[mask]
    best_w, best_key = None, None
    for w in np.linspace(0.0, 1.0, 21):
        m = metrics(w * club + (1 - w) * price, obs)
        # rank by corr (missing -> worst), break ties on lower mae
        key = (m.get("corr", float("-inf")), -m.get("mae", float("inf")))
        if best_key is None or key > best_key:
            best_w, best_key = float(w), key
    return best_w


def metrics(pred: pd.Series, obs: pd.Series) -> dict:
    """MAE, bias and correlation of a prediction against observed points."""
    mask = pred.notna() & obs.notna()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    err = pred[mask] - obs[mask]
    out = {"n": n, "mae": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3)}
    if n > 1 and pred[mask].std() > 0 and obs[mask].std() > 0:
        out["corr"] = round(float(np.corrcoef(pred[mask], obs[mask])[0, 1]), 3)
    return out


def is_improvement(
    refit: dict, base: dict, *, corr_eps: float = CORR_EPS, mae_eps: float = MAE_EPS
) -> bool:
    """True when `refit` beats `base` on the metrics that drive squad choice.

    Ranking (corr) leads, absolute error (mae) is secondary: a candidate must
    improve at least one beyond its epsilon while worsening NEITHER beyond it.
    This rejects a least-squares price fit that only removes level bias (mae
    rises, corr flat) and a blend whose lower mae comes at the cost of ranking.
    Returns False if either metric is missing (too small / no spread).
    """
    rc, bc = refit.get("corr"), base.get("corr")
    rm, bm = refit.get("mae"), base.get("mae")
    if None in (rc, bc, rm, bm):
        return False
    no_worse = rc >= bc - corr_eps and rm <= bm + mae_eps
    better = rc > bc + corr_eps or rm < bm - mae_eps
    return no_worse and better


def _rejection_reason(
    refit: dict, base: dict, *, corr_eps: float = CORR_EPS, mae_eps: float = MAE_EPS
) -> str:
    """Human-readable explanation of why a candidate failed is_improvement."""
    rc, bc = refit.get("corr"), base.get("corr")
    rm, bm = refit.get("mae"), base.get("mae")
    if None in (rc, bc, rm, bm):
        return "corr/mae missing (sample too small or no spread)"
    if rc < bc - corr_eps:
        return f"corr drops {bc:.3f} -> {rc:.3f} (ranking would get worse)"
    if rm > bm + mae_eps:
        return f"mae rises {bm:.3f} -> {rm:.3f} with no corr gain"
    return "no meaningful gain on corr or mae over the default"


def _price_prior_with(df: pd.DataFrame, slope: float, intercept: float) -> pd.Series:
    """The deployed price prior under a candidate (slope, intercept): same
    formula player_points uses, ownership term unchanged, so default vs refit
    differ ONLY in the fitted coefficients (a fair head-to-head)."""
    own_w = player_points.WEIGHTS["ownership_weight"]
    return (intercept + slope * df["price"] + own_w * df["percentSelected"]).clip(lower=0.5)


# ----------------------------------------------- tunable (tunables.py) fits

def _completed_matches(rounds, only_rounds=None):
    """Yield (round id, match) for every completed match carrying a score."""
    for r in rounds:
        if only_rounds is not None and r.get("id") not in only_rounds:
            continue
        for m in r.get("tournaments", []):
            if m.get("homeScore") is None or not feed_status.result_usable(m.get("status")):
                continue
            yield r.get("id"), m


def fit_group_goals(rounds) -> tuple[float, int] | None:
    """Empirical GROUP_GOALS_PER_MATCH: the mean total goals in a completed
    group-stage match. None below MIN_GROUP_MATCHES completed games (too few to
    trust the average over the hand-picked default)."""
    totals = [
        int(m["homeScore"]) + int(m["awayScore"])
        for _, m in _completed_matches(rounds, only_rounds=(1, 2, 3))
    ]
    if len(totals) < MIN_GROUP_MATCHES:
        return None
    return float(np.mean(totals)), len(totals)


def _player_match_points(stats) -> list[float]:
    """Per-round points a player has scored, from the feed's roundPoints (a dict
    keyed by round number, with a list fallback)."""
    rp = (stats or {}).get("roundPoints")
    if isinstance(rp, dict):
        return [float(v) for v in rp.values() if isinstance(v, (int, float))]
    if isinstance(rp, list):
        return [float(v) for v in rp if isinstance(v, (int, float))]
    return []


def fit_captain_cv(players_raw) -> tuple[float, int] | None:
    """Empirical CAPTAIN_SCORE_CV: the median coefficient of variation (SD/mean)
    of per-match points among captain-caliber players - those averaging at least
    CAPTAIN_CALIBER_MEAN points over at least MIN_CV_MATCHES matches. That is
    exactly the match-to-match score dispersion the option-value model assumes
    for the kind of player you would captain. None until MIN_CV_PLAYERS such
    players have a multi-match history, so it stays inert early on."""
    cvs = []
    for p in players_raw:
        pts = _player_match_points(p.get("stats"))
        if len(pts) < MIN_CV_MATCHES:
            continue
        arr = np.asarray(pts, dtype=float)
        mean = float(arr.mean())
        if mean < CAPTAIN_CALIBER_MEAN:
            continue
        cvs.append(float(arr.std(ddof=1) / mean))
    if len(cvs) < MIN_CV_PLAYERS:
        return None
    return float(np.median(cvs)), len(cvs)


def fit_clean_sheet_spread(rounds, squads, base_rate) -> tuple[float, int] | None:
    """Empirical CLEAN_SHEET_OPPONENT_SPREAD: how strongly the opponent's Elo
    bends the clean-sheet rate. Model cs ~ base*(1 + spread*(2E-1)); least
    squares of the residual (cs - base) on base*(2E-1), where E is the Elo
    expected score and cs is 1 when the opponent failed to score. Needs the Elo
    ratings (local match history), so it degrades to None if those are
    unavailable. Returned ONLY when the fit beats the flat baseline on MAE and is
    meaningfully non-zero; None below MIN_CS_OBS team-match observations."""
    try:
        from .models import progression  # local: heavy Elo / history deps
        model, id_to_name = progression._build_elo(squads, rounds)
    except Exception:
        return None
    rating = {
        s["id"]: model.rating(id_to_name[s["id"]])
        for s in squads
        if s["id"] in id_to_name
    }

    xs, ys, observed = [], [], []
    for _, m in _completed_matches(rounds):
        h, a = m.get("homeSquadId"), m.get("awaySquadId")
        if h not in rating or a not in rating:
            continue
        hs, as_ = int(m["homeScore"]), int(m["awayScore"])
        for team, opp, conceded in ((h, a, as_), (a, h, hs)):
            e = elo.expected_score(rating[team] - rating[opp])
            cs = 1.0 if conceded == 0 else 0.0
            xs.append(base_rate * (2.0 * e - 1.0))
            ys.append(cs - base_rate)
            observed.append(cs)
    if len(xs) < MIN_CS_OBS:
        return None
    x, y, cs = np.asarray(xs), np.asarray(ys), np.asarray(observed)
    denom = float((x * x).sum())
    if denom <= 0:
        return None
    spread = max(0.0, min(2.0, float((x * y).sum() / denom)))
    mae_flat = float(np.abs(base_rate - cs).mean())
    pred = np.clip(base_rate + spread * x, 0.0, 1.0)
    mae_fit = float(np.abs(pred - cs).mean())
    if spread < CS_SPREAD_MIN or mae_fit > mae_flat - MAE_EPS:
        return None
    return spread, len(xs)


def propose_tunables(
    rounds, squads, players_raw, base_rate, *, emit
) -> tuple[dict, dict, list[str]]:
    """Gated fits for the calibratable tunables (tunables.py). Returns
    (proposals, diagnostics, rejections): each proposal is accepted only when
    its estimator clears its sample gate AND differs from the current default by
    more than a churn epsilon (the clean-sheet spread self-gates on MAE)."""
    proposals: dict = {}
    diag: dict = {}
    rejections: list[str] = []

    gg = fit_group_goals(rounds)
    if gg is None:
        emit(f"Group goals/match: need >= {MIN_GROUP_MATCHES} completed group "
             "matches; skipping.")
    else:
        value, n = gg
        diag["group_goals"] = {"value": round(value, 3), "n": n}
        emit(f"Group goals/match: {value:.2f} from {n} completed group matches "
             f"(default {tunables.GROUP_GOALS_PER_MATCH}).")
        if abs(value - tunables.GROUP_GOALS_PER_MATCH) > GROUP_GOALS_EPS:
            proposals["group_goals_per_match"] = round(value, 3)
            emit("  -> accepted.")
        else:
            rejections.append(
                f"group_goals_per_match: within {GROUP_GOALS_EPS} of the default")

    cc = fit_captain_cv(players_raw)
    if cc is None:
        emit(f"Captain score CV: need >= {MIN_CV_PLAYERS} players with "
             f">= {MIN_CV_MATCHES} matches; skipping.")
    else:
        value, n = cc
        diag["captain_cv"] = {"value": round(value, 3), "n": n}
        emit(f"Captain score CV: {value:.2f} from {n} multi-match players "
             f"(default {tunables.CAPTAIN_SCORE_CV}).")
        if abs(value - tunables.CAPTAIN_SCORE_CV) > CV_EPS:
            proposals["captain_score_cv"] = round(value, 3)
            emit("  -> accepted.")
        else:
            rejections.append(f"captain_score_cv: within {CV_EPS} of the default")

    cs = fit_clean_sheet_spread(rounds, squads, base_rate)
    if cs is None:
        emit("Clean-sheet opponent spread: too few clean-sheet observations or "
             "no improvement over the flat rate; skipping.")
    else:
        value, n = cs
        diag["cs_spread"] = {"value": round(value, 3), "n": n}
        proposals["clean_sheet_opponent_spread"] = round(value, 3)
        emit(f"Clean-sheet opponent spread: {value:.2f} from {n} team-matches "
             f"(beats the flat baseline; default "
             f"{tunables.CLEAN_SHEET_OPPONENT_SPREAD}).")

    return proposals, diag, rejections


def compute_calibration(
    *, write: bool = False, log=None
) -> dict:
    """Calibration preview as structured data (no console formatting).

    Reads the LATEST local feeds (run `advise`/`fetch` first; no network here),
    compares the un-tuned model against observed points and proposes refit
    weights. A weight is proposed ONLY when it beats the default on the metrics
    that drive squad choice (corr first, then mae); the report lists what was
    fit and, for anything rejected, why. With `write` and enough samples it
    persists calibration.json; otherwise it only previews. `log` receives the
    human-readable report lines so a caller (the CLI, or a web job) can surface
    them. The same dry-run/refusal rules as the old `run()` apply.
    """
    emit = log or (lambda _msg: None)

    players_raw = fifa_client.load("players")
    squads = fifa_client.load("squads")
    df = features.player_table(players_raw, squads)

    obs = observed_per_match(df)
    n_played = int(obs.notna().sum())
    result: dict = {
        "n_played": n_played,
        "min_matches": MIN_MATCHES,
        "min_samples": MIN_SAMPLES,
        "default_price_metrics": None,
        "refit_price_metrics": None,
        "price_fit": None,
        "price_accepted": False,
        "club_default_metrics": None,
        "club_blend": None,
        "club_accepted": False,
        "proposed_weights": {},
        "rejections": [],
        "tunables_diag": {},
        "proposed_tunables": {},
        "tunable_rejections": [],
        "written": False,
        "refusal_reason": None,
    }
    emit(f"Players with >= {MIN_MATCHES} WC match played: {n_played}")
    if n_played == 0:
        emit("No World Cup matches played yet: nothing to calibrate.")
        result["refusal_reason"] = "no World Cup matches played yet"
        return result

    weights: dict[str, float] = {}

    # --- price prior: refit only if it beats the default on corr/mae --------
    price_now = player_points.price_prior(df)
    default_price_metrics = metrics(price_now, obs)
    result["default_price_metrics"] = default_price_metrics
    emit(f"Default price prior vs observed: {default_price_metrics}")

    base_price = price_now  # what the club blend sits on; swapped in if refit lands
    fit = fit_price_linear(df["price"], obs)
    if fit:
        slope, intercept = fit
        refit_price = _price_prior_with(df, slope, intercept)
        refit_metrics = metrics(refit_price, obs)
        result["price_fit"] = {"slope": round(slope, 4), "intercept": round(intercept, 4)}
        result["refit_price_metrics"] = refit_metrics
        emit(
            f"Refit price prior vs observed:   {refit_metrics} "
            f"(slope {slope:.3f}, intercept {intercept:.3f})"
        )
        if is_improvement(refit_metrics, default_price_metrics):
            weights["price_slope"] = round(slope, 4)
            weights["price_intercept"] = round(intercept, 4)
            base_price = refit_price
            result["price_accepted"] = True
            emit("  -> accepted: improves the metrics that drive squad choice.")
        else:
            reason = _rejection_reason(refit_metrics, default_price_metrics)
            result["rejections"].append(f"price slope/intercept: {reason}")
            emit(f"  -> rejected ({reason}); keeping the default price prior.")

    # --- club/price blend: propose a new weight only if it beats 50/50 ------
    club_ep = player_points.club_prior(df)
    n_club = int(club_ep.notna().sum())
    if n_club == 0:
        emit("No club projections available: skipping the club-weight fit.")
    else:
        default_blend = 0.5 * club_ep + 0.5 * base_price
        default_blend_metrics = metrics(default_blend, obs)
        result["club_default_metrics"] = default_blend_metrics
        w = fit_club_weight(club_ep, base_price, obs)
        if w is not None:
            best_blend = w * club_ep + (1 - w) * base_price
            best_metrics = metrics(best_blend, obs)
            result["club_blend"] = {"w": round(w, 3), "n_club": n_club, "metrics": best_metrics}
            emit(
                f"Club/price blend ({n_club} club-matched): default w=0.50 -> "
                f"{default_blend_metrics}; best w={w:.2f} -> {best_metrics}"
            )
            if abs(w - 0.5) > 1e-9 and is_improvement(best_metrics, default_blend_metrics):
                weights["club_prior_weight"] = round(w, 3)
                result["club_accepted"] = True
                emit("  -> accepted: the blend beats the 50/50 default on the metrics.")
            else:
                reason = (
                    "best blend is the 50/50 default"
                    if abs(w - 0.5) <= 1e-9
                    else _rejection_reason(best_metrics, default_blend_metrics)
                )
                result["rejections"].append(f"club_prior_weight: {reason}; keeping default 0.50")
                emit(f"  -> rejected ({reason}); keeping default 0.50.")

    result["proposed_weights"] = weights
    emit(f"Proposed weights: {weights or '(none — defaults already best on corr/mae)'}")

    # --- calibratable tunables (tunables.py): each gated, inert until its data
    # supports it (so most stay unproposed for the first rounds) -------------
    rounds = fifa_client.load("rounds")
    proposed_tunables, tdiag, trej = propose_tunables(
        rounds, squads, players_raw,
        player_points.WEIGHTS["flat_clean_sheet_rate"], emit=emit,
    )
    result["proposed_tunables"] = proposed_tunables
    result["tunables_diag"] = tdiag
    result["tunable_rejections"] = trej
    emit(f"Proposed tunables: {proposed_tunables or '(none — defaults kept)'}")

    if not write:
        emit(
            f"Dry run. Re-run with --write to persist to {CALIBRATION_FILE} "
            "(advise/live then pick it up automatically)."
        )
        result["refusal_reason"] = "dry run (use --write to persist)"
        return result
    if n_played < MIN_SAMPLES:
        emit(
            f"Refusing to write: only {n_played} observations "
            f"(need >= {MIN_SAMPLES}). Wait for more matches, then re-run."
        )
        result["refusal_reason"] = (
            f"only {n_played} observations (need >= {MIN_SAMPLES})"
        )
        return result
    if not weights and not proposed_tunables:
        emit("Nothing to write: no weight or tunable beat the defaults.")
        result["refusal_reason"] = "no candidate beat the defaults"
        return result

    blob = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": n_played,
        "metrics_before": default_price_metrics,
        "weights": weights,
        "tunables": proposed_tunables,
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    result["written"] = True
    emit(f"Wrote {CALIBRATION_FILE}. advise/live use it from the next run.")
    return result


def run(write: bool = False) -> None:
    compute_calibration(write=write, log=console.print)
