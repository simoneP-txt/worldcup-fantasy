"""Offline test for the World Bank economic Elo adjustment.

No network: a synthetic cached indicators file is injected and the pure
z-score combination is checked.
"""

import json

import pytest

from wcfantasy.data_sources import economics


def test_elo_adjustments_zscores_and_sum_to_zero(monkeypatch, tmp_path):
    cache = tmp_path / "econ.json"
    cache.write_text(json.dumps({
        "BRA": {"gdp_pc": 9000.0, "population": 210_000_000.0},
        "QAT": {"gdp_pc": 60000.0, "population": 2_900_000.0},
        "USA": {"gdp_pc": 70000.0, "population": 330_000_000.0},
    }))
    monkeypatch.setattr(economics, "CACHE", cache)
    squads = [
        {"id": 1, "name": "Brazil"},
        {"id": 2, "name": "Qatar"},
        {"id": 3, "name": "USA"},
    ]
    adj = economics.elo_adjustments(squads)

    assert set(adj) == {1, 2, 3}
    # each indicator is z-scored, so the per-team adjustments sum to ~0
    assert sum(adj.values()) == pytest.approx(0.0, abs=1e-9)
    # USA has both the highest GDP per capita and the largest population,
    # so it must carry the largest (most positive) adjustment
    assert adj[3] == max(adj.values())
    # Brazil has the lowest GDP per capita -> a negative adjustment
    assert adj[1] < 0.0


def test_elo_adjustments_handles_unknown_team(monkeypatch, tmp_path):
    cache = tmp_path / "econ.json"
    cache.write_text(json.dumps({
        "BRA": {"gdp_pc": 9000.0, "population": 210_000_000.0},
        "USA": {"gdp_pc": 70000.0, "population": 330_000_000.0},
    }))
    monkeypatch.setattr(economics, "CACHE", cache)
    squads = [
        {"id": 1, "name": "Brazil"},
        {"id": 2, "name": "USA"},
        {"id": 9, "name": "Atlantis"},  # no ISO3 mapping -> excluded
    ]
    adj = economics.elo_adjustments(squads)
    assert 9 not in adj
    assert set(adj) == {1, 2}
