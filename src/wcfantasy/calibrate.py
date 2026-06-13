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


def run(write: bool = False) -> None:
    players_raw = fifa_client.load("players")
    squads = fifa_client.load("squads")
    df = features.player_table(players_raw, squads)

    obs = observed_per_match(df)
    n_played = int(obs.notna().sum())
    console.print(f"Players with >= {MIN_MATCHES} WC match played: [bold]{n_played}[/bold]")
    if n_played == 0:
        console.print("No World Cup matches played yet: nothing to calibrate.")
        return

    # Baseline always uses the built-in defaults (calibrate never calls
    # apply_calibration), so 'before' is the un-tuned model.
    price_now = player_points.price_prior(df)
    club_ep = player_points.club_prior(df)
    console.print(f"Default price prior vs observed: {metrics(price_now, obs)}")

    weights: dict[str, float] = {}
    refit_price = price_now
    fit = fit_price_linear(df["price"], obs)
    if fit:
        slope, intercept = fit
        weights["price_slope"] = round(slope, 4)
        weights["price_intercept"] = round(intercept, 4)
        refit_price = (intercept + slope * df["price"]).clip(lower=0.5)
        console.print(
            f"Refit price prior vs observed:   {metrics(refit_price, obs)} "
            f"(slope {slope:.3f}, intercept {intercept:.3f})"
        )

    n_club = int(club_ep.notna().sum())
    if n_club == 0:
        console.print("No club projections available: skipping the club-weight fit.")
    else:
        w = fit_club_weight(club_ep, refit_price, obs)
        if w is not None:
            weights["club_prior_weight"] = round(w, 3)
            blended = w * club_ep + (1 - w) * refit_price
            console.print(
                f"Best club/price blend: w={w:.2f} "
                f"(default 0.50) -> {metrics(blended, obs)}, {n_club} club-matched"
            )

    console.print(f"\n[bold]Proposed weights:[/bold] {weights or '(none)'}")

    if not write:
        console.print(
            f"Dry run. Re-run with [bold]--write[/bold] to persist to {CALIBRATION_FILE} "
            "(advise/live then pick it up automatically)."
        )
        return
    if n_played < MIN_SAMPLES:
        console.print(
            f"[yellow]Refusing to write[/yellow]: only {n_played} observations "
            f"(need >= {MIN_SAMPLES}). Wait for more matches, then re-run."
        )
        return
    if not weights:
        console.print("[yellow]Nothing to write[/yellow]: no weight could be fit.")
        return

    blob = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": n_played,
        "metrics_before": metrics(price_now, obs),
        "weights": weights,
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {CALIBRATION_FILE}. advise/live use it from the next run.")
