"""Shared test fixtures.

The model weights live in a module-global dict (player_points.WEIGHTS)
that calibration runs may mutate. Snapshot and restore it around every
test so a calibration-aware test cannot leak tuned weights into the
precise-value tests that assume the built-in defaults.

The per-run report channels (each module's `last_run_info`) are also
module-global dicts; clear them around every test so one run's diagnostics
(e.g. progression's Elo ratings) cannot leak into the next.
"""

import pytest

from wcfantasy import tunables
from wcfantasy.data_sources import availability
from wcfantasy.models import player_points, progression

_RUN_INFO_MODULES = (player_points, progression, availability)


@pytest.fixture(autouse=True)
def _reset_model_weights():
    saved = dict(player_points.WEIGHTS)
    saved_tuned = dict(tunables.TUNED)
    yield
    player_points.WEIGHTS.clear()
    player_points.WEIGHTS.update(saved)
    tunables.TUNED.clear()
    tunables.TUNED.update(saved_tuned)


@pytest.fixture(autouse=True)
def _reset_run_info():
    for mod in _RUN_INFO_MODULES:
        mod.last_run_info.clear()
    yield
    for mod in _RUN_INFO_MODULES:
        mod.last_run_info.clear()
