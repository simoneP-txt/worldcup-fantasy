"""Offline tests for the predicted knockout bracket.

No network or API key: the modal-occupant extraction and the under-tie
advance split are pure, and the end-to-end schema is built from a synthetic
Monte Carlo run (`_simulate` on a synthetic world, the same trick the
progression tests use) so nothing downloads or simulates twice.
"""

import json

import pandas as pd
import pytest

from wcfantasy.models import elo, progression


# ---------------------------------------------------------- pure helpers

def test_modal_picks_argmax_and_none_on_empty():
    assert progression._modal({1: 5, 2: 9, 3: 2}) == 2
    assert progression._modal({}) is None


def test_match_probs_uses_h2h_in_both_orientations():
    h2h = {(1, 2): (0.6, 0.25, 0.15)}
    fifa_names = {1: "A", 2: "B"}
    probs_fn = lambda diff: (0.0, 0.0, 0.0)  # noqa: E731 - never reached
    assert progression._match_probs(1, 2, {}, fifa_names, probs_fn, h2h) == (0.6, 0.25, 0.15)
    # the reversed lookup flips home/away probabilities
    assert progression._match_probs(2, 1, {}, fifa_names, probs_fn, h2h) == (0.15, 0.25, 0.6)


def test_advance_probs_split_sums_to_one_and_collapses_draw_to_favourite():
    rating = {1: 1700.0, 2: 1500.0}  # team 1 the favourite
    fifa_names = {1: "A", 2: "B"}
    probs_fn = lambda diff: (0.5, 0.3, 0.2)  # noqa: E731
    ph, pa = progression._advance_probs(1, 2, rating, fifa_names, probs_fn, None)
    assert ph + pa == pytest.approx(1.0)
    assert 0.0 <= ph <= 1.0 and 0.0 <= pa <= 1.0
    # the 30% draw is split toward the higher-rated side, so it advances more
    # often than its 90-minute win probability alone.
    assert ph > 0.5


# --------------------------------------------------- assemble (feed-pinned)

def _two_team_squads():
    return [
        {"id": 1, "name": "Argentina", "group": "a"},
        {"id": 2, "name": "Cape Verde", "group": "b"},
    ]


def _flat_curve():
    return pd.DataFrame(
        {r: [1.0, 1.0] for r in range(1, 9)},
        index=pd.Index([1, 2], name="squadId"),
    )


def _template():
    return json.loads(progression._BRACKET_FILE.read_text(encoding="utf-8"))["template"]


def _find_round(schema, stage):
    return next(r for r in schema["rounds"] if r["stage"] == stage)


def test_assemble_bracket_pins_completed_feed_tie_to_100_0():
    rounds = [{"id": 4, "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 2, "homeScore": 2, "awayScore": 0,
         "status": "complete"},
    ]}]
    occupancy = {"R32": [[{1: 100}, {2: 100}]]}  # only slot 0 decided
    advance_fn = lambda h, a: (0.5, 0.5)  # noqa: E731 - unused for completed ties
    schema = progression._assemble_bracket(
        _template(), rounds, _two_team_squads(), occupancy, _flat_curve(),
        {1: 0.3, 2: 0.0}, advance_fn, "test-model",
    )
    tie = _find_round(schema, "R32")["ties"][0]
    assert tie["status"] == "completed"
    assert tie["home"]["team"] == "Argentina" and tie["home"]["p_advance"] == 1.0
    assert tie["away"]["team"] == "Cape Verde" and tie["away"]["p_advance"] == 0.0
    assert tie["home"]["locked"] and tie["away"]["locked"]
    assert tie["result"] == {"homeScore": 2, "awayScore": 0}
    # champion banner reads the stashed champion map
    assert schema["winner"]["team"] == "Argentina"


def test_assemble_bracket_scheduled_feed_tie_uses_model_head_to_head():
    rounds = [{"id": 4, "tournaments": [
        {"homeSquadId": 1, "awaySquadId": 2, "status": "scheduled"},
    ]}]
    occupancy = {"R32": [[{1: 100}, {2: 100}]]}
    advance_fn = lambda h, a: (0.7, 0.3)  # noqa: E731
    schema = progression._assemble_bracket(
        _template(), rounds, _two_team_squads(), occupancy, _flat_curve(),
        {1: 0.3}, advance_fn, "test-model",
    )
    tie = _find_round(schema, "R32")["ties"][0]
    assert tie["status"] == "scheduled"
    assert tie["home"]["p_advance"] == 0.7 and tie["away"]["p_advance"] == 0.3
    assert tie["result"] is None
    assert not tie["home"]["locked"]


def test_assemble_bracket_predicted_slot_shows_modal_occupants():
    occupancy = {"R32": [[{1: 80, 2: 20}, {2: 60, 1: 40}]]}
    advance_fn = lambda h, a: (0.55, 0.45)  # noqa: E731
    schema = progression._assemble_bracket(
        _template(), [], _two_team_squads(), occupancy, _flat_curve(),
        {1: 0.3}, advance_fn, "test-model",
    )
    tie = _find_round(schema, "R32")["ties"][0]
    assert tie["status"] == "predicted"
    assert tie["home"]["squadId"] == 1 and tie["away"]["squadId"] == 2
    assert tie["home"]["p_advance"] + tie["away"]["p_advance"] == pytest.approx(1.0)


# -------------------------------------------- end-to-end from a real sim

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
            for j in range(i + 1, 4):
                group_matches.append(
                    {"homeSquadId": members[i], "awaySquadId": members[j], "date": None}
                )
    rounds = [
        {"id": 1, "status": "scheduled", "tournaments": group_matches[0::3]},
        {"id": 2, "status": "scheduled", "tournaments": group_matches[1::3]},
        {"id": 3, "status": "scheduled", "tournaments": group_matches[2::3]},
    ]
    rounds += [{"id": r, "status": "scheduled", "tournaments": []} for r in (4, 5, 6, 7, 8)]
    rating = {s["id"]: 1500.0 + s["id"] for s in squads}
    return squads, rounds, rating


def _simple_probs(diff):
    p = elo.expected_score(diff)
    draw = 0.24
    return p * (1 - draw), draw, (1 - p) * (1 - draw)


def test_assemble_bracket_well_formed_from_synthetic_simulation():
    squads, rounds, rating = _synthetic_world()
    fifa_names = {s["id"]: s["name"] for s in squads}
    curve, champion = progression._simulate(
        squads, rounds, rating, _simple_probs, None, n_sims=200, seed=1
    )
    occupancy = progression.last_run_info["bracket"]["occupancy"]

    advance_fn = lambda h, a: progression._advance_probs(  # noqa: E731
        h, a, rating, fifa_names, _simple_probs, None
    )
    schema = progression._assemble_bracket(
        _template(), rounds, squads, occupancy, curve, champion, advance_fn, "test",
    )

    assert schema["available"] is True
    # right stages, in order, with the right number of ties
    assert [r["stage"] for r in schema["rounds"]] == ["R32", "R16", "QF", "SF", "F"]
    assert [len(r["ties"]) for r in schema["rounds"]] == [16, 8, 4, 2, 1]

    for rnd in schema["rounds"]:
        for tie in rnd["ties"]:
            for sd in (tie["home"], tie["away"]):
                assert 0.0 <= sd["p_advance"] <= 1.0
                assert sd["p_reach"] is None or 0.0 <= sd["p_reach"] <= 1.0
            # no feed KO matches here -> every tie is a model prediction
            assert tie["status"] == "predicted"
            assert tie["home"]["p_advance"] + tie["away"]["p_advance"] == pytest.approx(1.0, abs=1e-3)

    # champion list sorted descending, winner is its top entry
    ps = [c["p"] for c in schema["champion"]]
    assert ps == sorted(ps, reverse=True)
    assert schema["winner"]["team"] == schema["champion"][0]["team"]


def test_simulate_stashes_occupancy_summing_to_n_sims():
    squads, rounds, rating = _synthetic_world()
    progression._simulate(squads, rounds, rating, _simple_probs, None, n_sims=150, seed=2)
    bracket = progression.last_run_info["bracket"]
    assert bracket["n_sims"] == 150
    occ = bracket["occupancy"]
    assert [len(occ[s]) for s in ("R32", "R16", "QF", "SF", "F")] == [16, 8, 4, 2, 1]
    # every slot's home/away counter totals exactly n_sims occupants
    for stage in ("R32", "R16", "QF", "SF", "F"):
        for home_occ, away_occ in occ[stage]:
            assert sum(home_occ.values()) == 150
            assert sum(away_occ.values()) == 150
