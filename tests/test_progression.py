"""Offline tests for the tournament progression model.

No network or API key: the heavy Monte Carlo is exercised with synthetic
squads/ratings, and the bracket invariants (how many teams can reach each
knockout round) pin the simulation's structure.
"""

import json

import pandas as pd
import pytest

from wcfantasy.models import elo, progression


# ----------------------------------------------------------- pure helpers

def test_team_strength_is_top_11_price_mean():
    players = pd.DataFrame([{"squadId": 1, "price": float(p)} for p in range(1, 16)])
    strength = progression.team_strength(players)
    # top 11 of prices 1..15 = 15..5, mean = 10
    assert strength.loc[1] == pytest.approx(sum(range(5, 16)) / 11)


def test_group_fixtures_keeps_only_group_rounds():
    rounds = [
        {"id": 1, "tournaments": [{"m": "g1"}]},
        {"id": 3, "tournaments": [{"m": "g3"}]},
        {"id": 4, "tournaments": [{"m": "ko"}]},  # R32, excluded
    ]
    assert progression._group_fixtures(rounds) == [{"m": "g1"}, {"m": "g3"}]


def _elo_history() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append(("A", "B", 2, 0) if i % 5 else ("A", "B", 1, 1))
        rows.append(("B", "C", 1, 0) if i % 4 else ("B", "C", 0, 1))
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "home_team": h, "away_team": a,
                "home_score": hs, "away_score": as_,
                "tournament": "FIFA World Cup qualification", "neutral": True,
            }
            for i, (h, a, hs, as_) in enumerate(rows)
        ]
    )


def test_outcome_prob_fn_normalised_and_monotonic():
    model = elo.fit(_elo_history())
    probs = progression._outcome_prob_fn(model)
    for diff in (-300, 0, 300):
        pw, pd_, pl = probs(diff)
        assert abs(pw + pd_ + pl - 1.0) < 1e-9
        assert pw >= 0 and pd_ >= 0 and pl >= 0
    # a bigger Elo edge means a higher win probability
    assert probs(300)[0] > probs(-300)[0]


# --------------------------------------------------------- fallback curve

def _fallback_inputs():
    squads = [
        {"id": 1, "name": "strong", "group": "a", "isEliminated": False},
        {"id": 2, "name": "weak", "group": "a", "isEliminated": False},
        {"id": 3, "name": "out", "group": "a", "isEliminated": True},
    ]
    players = pd.DataFrame(
        [{"squadId": 1, "price": 10.0}] * 11
        + [{"squadId": 2, "price": 4.0}] * 11
        + [{"squadId": 3, "price": 7.0}] * 11
    )
    return players, squads


def test_fallback_survival_curve_handles_elimination_and_strength():
    players, squads = _fallback_inputs()
    curve = progression._fallback_survival_curve(players, squads)
    # eliminated team: zero everywhere
    assert (curve.loc[3] == 0.0).all()
    # alive teams: certain through the group stage
    for r in progression.GROUP_ROUNDS:
        assert curve.loc[1, r] == 1.0 and curve.loc[2, r] == 1.0
    # knockout survival decays and the stronger team always ranks higher
    for r in progression.KO_ROUNDS:
        assert 0.0 <= curve.loc[1, r] <= 1.0
        assert curve.loc[1, r] >= curve.loc[2, r]


def test_survival_curve_falls_back_when_simulation_raises(monkeypatch):
    players, squads = _fallback_inputs()

    def boom(*args, **kwargs):
        raise RuntimeError("simulation unavailable")

    monkeypatch.setattr(progression, "_simulate_with_market", boom)
    curve = progression.survival_curve(players, squads, rounds=[])
    assert progression.last_run_info["model"].startswith("price heuristic")
    assert "ratings" not in progression.last_run_info  # no stale opponent factors
    assert (curve.loc[3] == 0.0).all()


def test_survival_curve_strict_reraises_instead_of_falling_back(monkeypatch):
    players, squads = _fallback_inputs()

    def boom(*args, **kwargs):
        raise RuntimeError("simulation unavailable")

    monkeypatch.setattr(progression, "_simulate_with_market", boom)
    monkeypatch.setenv("WCFANTASY_STRICT", "1")
    with pytest.raises(RuntimeError):
        progression.survival_curve(players, squads, rounds=[])


# ------------------------------------------------- end-to-end bracket sim

def _synthetic_world():
    """48 teams in 12 groups of 4 (a..l), matching the bracket template."""
    groups = "abcdefghijkl"
    squads, sid = [], 1
    for g in groups:
        for _ in range(4):
            squads.append(
                {"id": sid, "name": f"t{sid:02d}", "group": g, "isEliminated": False}
            )
            sid += 1
    group_matches = []
    for g in groups:
        members = [s["id"] for s in squads if s["group"] == g]
        for i in range(4):
            for j in range(i + 1, 4):  # round-robin: 6 per group
                group_matches.append(
                    {"homeSquadId": members[i], "awaySquadId": members[j], "date": None}
                )
    rounds = [
        {"id": 1, "status": "scheduled", "tournaments": group_matches[0::3]},
        {"id": 2, "status": "scheduled", "tournaments": group_matches[1::3]},
        {"id": 3, "status": "scheduled", "tournaments": group_matches[2::3]},
    ]
    rounds += [{"id": r, "status": "scheduled", "tournaments": []} for r in (4, 5, 6, 7, 8)]
    rating = {s["id"]: 1500.0 + s["id"] for s in squads}  # gentle strength gradient
    return squads, rounds, rating


def _simple_probs(diff):
    p = elo.expected_score(diff)
    draw = 0.24
    return p * (1 - draw), draw, (1 - p) * (1 - draw)


def test_sample_scoreline_signs_match_outcomes_with_real_margins():
    import numpy as np
    rng = np.random.default_rng(0)
    outcomes = np.array([1, -1, 0] * 200)
    gh, ga = progression._sample_scoreline(outcomes, 0.5, 0.25, rng)
    wins, losses, draws = outcomes == 1, outcomes == -1, outcomes == 0
    assert (gh[wins] > ga[wins]).all()      # a win really has the home edge
    assert (gh[losses] < ga[losses]).all()  # a loss the away edge
    assert (gh[draws] == ga[draws]).all()   # a draw is level
    # decisive games carry a >= 1 margin (no 0-0 "win")
    assert (np.abs(gh[wins] - ga[wins]) >= 1).all()


def test_simulate_respects_bracket_capacity():
    squads, rounds, rating = _synthetic_world()
    curve, champion = progression._simulate(
        squads, rounds, rating, _simple_probs, None, n_sims=200, seed=1
    )

    assert len(curve) == 48
    # everyone alive is certain through the group stage
    for r in progression.GROUP_ROUNDS:
        assert (curve[r] == 1.0).all()
    # exactly 32/16/8/4/2 team-slots exist at R32/R16/QF/SF/Final
    expected_slots = {4: 32, 5: 16, 6: 8, 7: 4, 8: 2}
    for r, slots in expected_slots.items():
        assert curve[r].sum() == pytest.approx(slots)
        assert ((curve[r] >= 0) & (curve[r] <= 1)).all()
    # exactly one champion per simulation
    assert sum(champion.values()) == pytest.approx(1.0)


def test_resolve_ko_pair_uses_real_result_by_pairing_not_position():
    # completed match stored as home=7, away=3, away wins 0-1
    actual_by_pair = {
        (7, 3): {"homeSquadId": 7, "awaySquadId": 3,
                 "homeScore": 0, "awayScore": 1, "status": "complete"}
    }
    sim = lambda h, a: -1  # sentinel: "this pairing was simulated"
    # real result honoured in either lookup orientation, winner = away (3)
    assert progression._resolve_ko_pair(7, 3, actual_by_pair, sim) == 3
    assert progression._resolve_ko_pair(3, 7, actual_by_pair, sim) == 3
    # a pairing with no completed feed match falls back to the simulator
    assert progression._resolve_ko_pair(1, 2, actual_by_pair, sim) == -1
    # a scheduled match (no score) is also simulated
    scheduled = {(5, 6): {"homeSquadId": 5, "awaySquadId": 6, "status": "scheduled"}}
    assert progression._resolve_ko_pair(5, 6, scheduled, sim) == -1


def test_resolve_ko_pair_breaks_a_draw_on_penalties():
    actual_by_pair = {
        (7, 3): {"homeSquadId": 7, "awaySquadId": 3, "homeScore": 1,
                 "awayScore": 1, "homePenaltyScore": 2, "awayPenaltyScore": 4,
                 "status": "complete"}
    }
    sim = lambda h, a: -1
    assert progression._resolve_ko_pair(7, 3, actual_by_pair, sim) == 3  # away wins shootout


def test_nudge_ratings_moves_toward_market_and_skips_unusable():
    rating = {1: 1500.0, 2: 1500.0, 3: 1500.0}
    champion = {1: 0.10, 2: 0.10, 3: 0.0}
    outright = {1: 0.20, 2: 0.05, 3: 0.30, 99: 0.10}  # 99 is not a squad
    nudged = progression.nudge_ratings(rating, outright, champion)
    assert nudged[1] > 1500.0   # market rates it above the sim -> rating up
    assert nudged[2] < 1500.0   # market below the sim -> rating down
    assert nudged[3] == 1500.0  # p_sim 0 -> no usable ratio, unchanged
    assert 99 not in nudged     # unknown team ignored


def test_nudge_ratings_is_clamped():
    big = progression.nudge_ratings({1: 1500.0}, {1: 0.99}, {1: 1e-3})
    assert big[1] - 1500.0 == pytest.approx(progression.ODDS_NUDGE_MAX)


def test_championship_probabilities_reads_the_stashed_champion(monkeypatch):
    def fake(squads, rounds, n_sims, seed):
        progression.last_run_info["champion"] = {1: 0.6, 2: 0.4}
        return "curve"  # the curve is ignored by championship_probabilities

    monkeypatch.setattr(progression, "_simulate_with_market", fake)
    assert progression.championship_probabilities([], []) == {1: 0.6, 2: 0.4}


def test_championship_probabilities_empty_when_simulation_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no history")

    monkeypatch.setattr(progression, "_simulate_with_market", boom)
    assert progression.championship_probabilities([], []) == {}


def test_bracket_file_is_well_formed():
    bracket = json.loads(progression._BRACKET_FILE.read_text(encoding="utf-8"))
    template = bracket["template"]
    assert [len(template[s]) for s in ("R32", "R16", "QF", "SF", "F")] == [16, 8, 4, 2, 1]
    # third-place allocation covers every way to pick 8 of the 12 groups
    from math import comb
    assert len(bracket["third_place_allocation"]) == comb(12, 8)
