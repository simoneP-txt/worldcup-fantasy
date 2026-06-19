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
  - club_prior_weight: the club/price blend weight that minimises MAE of
    the blended prior vs observed, among players with a club projection.
The rest of WEIGHTS keeps its default. See CALIBRATION.md.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console

from . import features, fifa_client
from .models import player_points

console = Console()

CALIBRATION_FILE = player_points.CALIBRATION_FILE

# A player must have played at least this many WC matches before his
# observed average is trusted as a calibration target.
MIN_MATCHES = 1
# Below this many observations the fit is too noisy to persist.
MIN_SAMPLES = 30


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
    """Blend weight w in [0, 1] minimising MAE of w*club + (1-w)*price vs
    observed, over players who have both a club projection and minutes."""
    mask = observed.notna() & club_ep.notna() & price_prior.notna()
    if mask.sum() < 3:
        return None
    best_w, best_mae = None, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        blended = w * club_ep[mask] + (1 - w) * price_prior[mask]
        mae = float((blended - observed[mask]).abs().mean())
        if mae < best_mae:
            best_w, best_mae = float(w), mae
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


def compute_calibration(
    *, write: bool = False, log=None
) -> dict:
    """Calibration preview as structured data (no console formatting).

    Reads the LATEST local feeds (run `advise`/`fetch` first; no network here),
    compares the un-tuned model against observed points and proposes refit
    weights. With `write` and enough samples it persists calibration.json;
    otherwise it only previews. `log` receives the human-readable report lines
    so a caller (the CLI, or a web job) can surface them. Returns the
    structured result the web view renders. The same dry-run/refusal rules as
    the old `run()` apply.
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
        "club_blend": None,
        "proposed_weights": {},
        "written": False,
        "refusal_reason": None,
    }
    emit(f"Players with >= {MIN_MATCHES} WC match played: {n_played}")
    if n_played == 0:
        emit("No World Cup matches played yet: nothing to calibrate.")
        result["refusal_reason"] = "no World Cup matches played yet"
        return result

    # Baseline always uses the built-in defaults (calibrate never calls
    # apply_calibration), so 'before' is the un-tuned model.
    price_now = player_points.price_prior(df)
    club_ep = player_points.club_prior(df)
    default_price_metrics = metrics(price_now, obs)
    result["default_price_metrics"] = default_price_metrics
    emit(f"Default price prior vs observed: {default_price_metrics}")

    weights: dict[str, float] = {}
    refit_price = price_now
    fit = fit_price_linear(df["price"], obs)
    if fit:
        slope, intercept = fit
        weights["price_slope"] = round(slope, 4)
        weights["price_intercept"] = round(intercept, 4)
        refit_price = (intercept + slope * df["price"]).clip(lower=0.5)
        result["price_fit"] = {"slope": round(slope, 4), "intercept": round(intercept, 4)}
        result["refit_price_metrics"] = metrics(refit_price, obs)
        emit(
            f"Refit price prior vs observed:   {result['refit_price_metrics']} "
            f"(slope {slope:.3f}, intercept {intercept:.3f})"
        )

    n_club = int(club_ep.notna().sum())
    if n_club == 0:
        emit("No club projections available: skipping the club-weight fit.")
    else:
        w = fit_club_weight(club_ep, refit_price, obs)
        if w is not None:
            weights["club_prior_weight"] = round(w, 3)
            blended = w * club_ep + (1 - w) * refit_price
            blend_metrics = metrics(blended, obs)
            result["club_blend"] = {"w": round(w, 3), "n_club": n_club, "metrics": blend_metrics}
            emit(
                f"Best club/price blend: w={w:.2f} "
                f"(default 0.50) -> {blend_metrics}, {n_club} club-matched"
            )

    result["proposed_weights"] = weights
    emit(f"Proposed weights: {weights or '(none)'}")

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
    if not weights:
        emit("Nothing to write: no weight could be fit.")
        result["refusal_reason"] = "no weight could be fit"
        return result

    blob = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": n_played,
        "metrics_before": default_price_metrics,
        "weights": weights,
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    result["written"] = True
    emit(f"Wrote {CALIBRATION_FILE}. advise/live use it from the next run.")
    return result


def run(write: bool = False) -> None:
    compute_calibration(write=write, log=console.print)
