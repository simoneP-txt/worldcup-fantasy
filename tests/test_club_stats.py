"""Offline test for the FBref schema normalization.

Exercises _normalize with synthetic, already-flattened FBref tables - in
particular the double-count guard: a mid-season mover appears once per
club in the standard table and must collapse to one row BEFORE the
player-summed extra tables (shooting/misc/keeper) are merged in.
"""

import pandas as pd

from wcfantasy.data_sources.club_stats import CHANCES_PER_ASSIST, _normalize


def test_normalize_collapses_movers_and_maps_schema():
    standard = pd.DataFrame([
        # "Mover" played for two clubs this season -> two standard rows
        {"player": "Mover", "nation": "CIV", "pos": "FW",
         "Playing Time_MP": 5, "Playing Time_Min": 450,
         "Performance_Gls": 2, "Performance_Ast": 1,
         "Performance_CrdY": 1, "Performance_CrdR": 0},
        {"player": "Mover", "nation": "CIV", "pos": "FW",
         "Playing Time_MP": 5, "Playing Time_Min": 450,
         "Performance_Gls": 1, "Performance_Ast": 0,
         "Performance_CrdY": 0, "Performance_CrdR": 0},
        {"player": "Keeper", "nation": "BRA", "pos": "GK",
         "Playing Time_MP": 10, "Playing Time_Min": 900,
         "Performance_Gls": 0, "Performance_Ast": 0,
         "Performance_CrdY": 0, "Performance_CrdR": 0},
        {"player": "Bench", "nation": "ENG", "pos": "MF",
         "Playing Time_MP": 0, "Playing Time_Min": 0,
         "Performance_Gls": 0, "Performance_Ast": 0,
         "Performance_CrdY": 0, "Performance_CrdR": 0},
    ])
    # extra tables are already player-summed (one row per player)
    shooting = pd.DataFrame([
        {"player": "Mover", "Standard_SoT": 8},
        {"player": "Keeper", "Standard_SoT": 0},
    ])
    misc = pd.DataFrame([
        {"player": "Mover", "Performance_TklW": 3,
         "Performance_PKwon": 1, "Performance_PKcon": 0},
    ])
    keeper = pd.DataFrame([
        {"player": "Keeper", "Performance_Saves": 30,
         "Performance_CS": 4, "Performance_GA": 10},
    ])

    out = _normalize(standard, shooting, misc, keeper).set_index("name")

    # Bench dropped (0 minutes); the two Mover rows collapsed to one
    assert set(out.index) == {"Mover", "Keeper"}
    assert (out.index == "Mover").sum() == 1

    mover = out.loc["Mover"]
    assert mover["games"] == 10 and mover["minutes"] == 900
    assert mover["goals"] == 3.0 and mover["assists"] == 1.0
    assert mover["position"] == "FWD" and mover["nation"] == "CIV"
    assert mover["chances_created"] == CHANCES_PER_ASSIST * 1
    # the heart of the test: player-summed extras attach ONCE, after the
    # collapse - a regression would double them (16 SoT, 6 tackles, ...)
    assert mover["shots_on_target"] == 8.0
    assert mover["tackles"] == 3.0
    assert mover["penalties_won"] == 1.0

    keep = out.loc["Keeper"]
    assert keep["position"] == "GK"
    assert keep["saves"] == 30.0
    assert keep["clean_sheets"] == 4.0
    assert keep["goals_conceded"] == 10.0
