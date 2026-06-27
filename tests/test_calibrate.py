"""Offline tests for the calibration scaffold (pure fit helpers and the
calibration.json -> WEIGHTS loader). No network, no feeds."""

import json

import pandas as pd
import pytest

from wcfantasy import calibrate, tunables
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


def test_fit_club_weight_keeps_price_when_club_only_adds_noise():
    # Price already tracks observed; club is pure noise -> the corr-first
    # selector should not lean on the club projection.
    observed = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 7.0])
    price_prior = observed.copy()
    club = pd.Series([7.0, 1.0, 9.0, 2.0, 5.0, 8.0, 0.0])
    assert calibrate.fit_club_weight(club, price_prior, observed) == pytest.approx(0.0)


def test_is_improvement_accepts_a_clear_win():
    base = {"mae": 2.0, "bias": -0.5, "corr": 0.30}
    refit = {"mae": 1.8, "bias": 0.0, "corr": 0.34}
    assert calibrate.is_improvement(refit, base)


def test_is_improvement_rejects_a_corr_regression():
    # Lower mae but worse ranking: the optimizer cares about ordering.
    base = {"mae": 2.0, "corr": 0.30}
    refit = {"mae": 1.5, "corr": 0.25}
    assert not calibrate.is_improvement(refit, base)


def test_is_improvement_rejects_a_bias_only_fit():
    # The real round-1 shape: least squares zeroes the bias but raises mae
    # while corr is flat -> nothing that changes player ordering, so reject.
    base = {"mae": 1.965, "bias": -0.549, "corr": 0.305}
    refit = {"mae": 2.128, "bias": -0.0, "corr": 0.305}
    assert not calibrate.is_improvement(refit, base)


def test_is_improvement_needs_both_metrics():
    assert not calibrate.is_improvement({"mae": 1.0}, {"mae": 2.0, "corr": 0.3})


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


# ------------------------------------------------ calibratable tunable fits

def test_fit_group_goals_averages_completed_group_matches():
    sparse = [{"id": 1, "tournaments": [
        {"homeScore": 2, "awayScore": 1, "status": "complete"},
        {"homeScore": 0, "awayScore": 0, "status": "scheduled"},  # no score: skip
    ]}]
    assert calibrate.fit_group_goals(sparse) is None  # below MIN_GROUP_MATCHES

    rounds = [
        {"id": 1, "tournaments": [
            {"homeScore": 1, "awayScore": 1, "status": "complete"} for _ in range(20)
        ]},
        {"id": 4, "tournaments": [  # a KO blow-out, must be ignored
            {"homeScore": 6, "awayScore": 0, "status": "complete"}
        ]},
    ]
    value, n = calibrate.fit_group_goals(rounds)
    assert n == 20 and value == pytest.approx(2.0)


def test_fit_captain_cv_uses_multimatch_caliber_players():
    caliber = {"stats": {"roundPoints": {"1": 6, "2": 2, "3": 4}}}  # mean 4, cv 0.5
    assert calibrate.fit_captain_cv([caliber]) is None  # below the player gate
    value, n = calibrate.fit_captain_cv([caliber] * 20)
    assert n == 20 and value == pytest.approx(0.5)
    low = {"stats": {"roundPoints": {"1": 1, "2": 0, "3": 1}}}  # mean < caliber
    assert calibrate.fit_captain_cv([low] * 20) is None


def test_fit_clean_sheet_spread_positive_when_favourites_keep_sheets(monkeypatch):
    from wcfantasy.models import progression

    class _Model:
        def rating(self, name):
            return 2000.0 if name == "Strong" else 1400.0

    monkeypatch.setattr(
        progression, "_build_elo",
        lambda squads, rounds: (_Model(), {1: "Strong", 2: "Weak"}),
    )
    squads = [{"id": 1, "name": "Strong"}, {"id": 2, "name": "Weak"}]
    rounds = [{"id": 1, "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 2, "homeScore": 2, "awayScore": 0,
         "status": "complete"} for _ in range(40)
    ]}]
    out = calibrate.fit_clean_sheet_spread(rounds, squads, base_rate=0.30)
    assert out is not None
    spread, n = out
    assert spread > 0 and n == 80  # two team-observations per match


def test_tunables_apply_calibration_and_consumer_pickup(tmp_path, monkeypatch):
    cal = tmp_path / "calibration.json"
    monkeypatch.setattr(tunables, "CALIBRATION_FILE", cal)

    tunables.apply_calibration()  # no file -> defaults
    assert tunables.tuned("clean_sheet_opponent_spread") == (
        tunables.CLEAN_SHEET_OPPONENT_SPREAD
    )

    cal.write_text(json.dumps(
        {"tunables": {"clean_sheet_opponent_spread": 0.0, "bogus": 9}}
    ))
    tunables.apply_calibration()
    assert tunables.tuned("clean_sheet_opponent_spread") == 0.0
    assert "bogus" not in tunables.TUNED

    # the consumer reads the live value: spread 0 -> the CS prob is flat (the
    # base rate) whatever the Elo gap, proving calibration reaches the model.
    players = pd.DataFrame([{"squadId": 1, "position": "DEF"},
                            {"squadId": 2, "position": "DEF"}])
    rounds = [{"id": 1, "tournaments": [{"homeSquadId": 1, "awaySquadId": 2}]}]
    p_cs, _ = player_points.team_clean_sheet(players, rounds, 1, {1: 1800.0, 2: 1400.0})
    assert (p_cs == player_points.WEIGHTS["flat_clean_sheet_rate"]).all()


def test_tunables_apply_calibration_survives_a_corrupt_file(tmp_path, monkeypatch):
    cal = tmp_path / "calibration.json"
    cal.write_text("{ not json")
    monkeypatch.setattr(tunables, "CALIBRATION_FILE", cal)
    tunables.apply_calibration()  # must not raise
    assert tunables.TUNED == tunables._CALIBRATABLE_DEFAULTS
