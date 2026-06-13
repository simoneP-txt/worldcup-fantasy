"""Shared test fixtures.

The model weights live in a module-global dict (player_points.WEIGHTS)
that calibration runs may mutate. Snapshot and restore it around every
test so a calibration-aware test cannot leak tuned weights into the
precise-value tests that assume the built-in defaults.
"""

import pytest

from wcfantasy.models import player_points


@pytest.fixture(autouse=True)
def _reset_model_weights():
    saved = dict(player_points.WEIGHTS)
    yield
    player_points.WEIGHTS.clear()
    player_points.WEIGHTS.update(saved)
