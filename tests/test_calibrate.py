"""Offline tests for the calibration scaffold (pure fit helpers and the
calibration.json -> WEIGHTS loader). No network, no feeds."""

import json

import pandas as pd
import pytest

from wcfantasy import calibrate
from wcfantasy.models import player_points


def test_observed_per_match_requires_a_played_match():
    df = pd.DataFrame({"matchesPlayed": [0, 2], "avgPoints": [5.0, 3.0]})
    obs = calibrate.observed_per_match(df)
    assert pd.isna(obs.iloc[0])  # never played -> no target
    assert obs.iloc[1] == 3.0


def test_fit_price_linear_recovers_a_clean_line():
    price = pd.Series([4.0, 6.0, 8.0, 10.0, 12.0])
    observed = 0.5 * price + 1.0  # exact line
    slope, intercept = calibrate.fit_price_linear(price, observed)
    assert slope == pytest.approx(0.5, abs=1e-6)
    assert intercept == pytest.approx(1.0, abs=1e-6)


def test_fit_price_linear_needs_two_points():
    assert calibrate.fit_price_linear(pd.Series([5.0]), pd.Series([3.0])) is None


def test_fit_club_weight_prefers_the_better_signal():
    observed = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    club = observed.copy()  # club projection is a perfect predictor
    price_prior = pd.Series([5.0] * 5)  # price prior is useless here
    assert calibrate.fit_club_weight(club, price_prior, observed) == pytest.approx(1.0)


def test_metrics_zero_error_for_a_perfect_prediction():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    m = calibrate.metrics(s, s)
    assert m["mae"] == 0.0 and m["bias"] == 0.0 and m["corr"] == 1.0


def test_apply_calibration_layers_file_over_defaults(tmp_path, monkeypatch):
    cal_file = tmp_path / "calibration.json"
    monkeypatch.setattr(player_points, "CALIBRATION_FILE", cal_file)

    # No file: WEIGHTS are the built-in defaults.
    player_points.apply_calibration()
    assert player_points.WEIGHTS["price_slope"] == player_points._WEIGHT_DEFAULTS["price_slope"]

    # File present: known keys override, unknown keys ignored.
    cal_file.write_text(json.dumps({"weights": {"price_slope": 0.9, "bogus": 1}}))
    player_points.apply_calibration()
    assert player_points.WEIGHTS["price_slope"] == 0.9
    assert "bogus" not in player_points.WEIGHTS
    # untouched keys keep their default
    assert player_points.WEIGHTS["club_prior_weight"] == player_points._WEIGHT_DEFAULTS["club_prior_weight"]


def test_apply_calibration_survives_a_corrupt_file(tmp_path, monkeypatch):
    cal_file = tmp_path / "calibration.json"
    cal_file.write_text("{ not json")
    monkeypatch.setattr(player_points, "CALIBRATION_FILE", cal_file)
    player_points.apply_calibration()  # must not raise
    assert player_points.WEIGHTS == player_points._WEIGHT_DEFAULTS
