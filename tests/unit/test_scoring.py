from pathlib import Path

import pandas as pd

from app.ml.scoring import ScoringRules

FIX = Path(__file__).parents[1] / "fixtures"
POS = {"GK": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# The live game_config.scoring block (2026/27), abbreviated to the fields the engine reads.
GAME_CONFIG = {
    "long_play": 2, "short_play": 1, "goals_conceded": {"DEF": -1, "FWD": 0, "GKP": -1, "MID": 0}, "saves": 1,
    "goals_scored": {"DEF": 6, "FWD": 4, "GKP": 10, "MID": 5}, "assists": 3,
    "clean_sheets": {"DEF": 4, "FWD": 0, "GKP": 4, "MID": 1}, "penalties_saved": 5, "penalties_missed": -2,
    "yellow_cards": -1, "red_cards": -3, "own_goals": -2, "bonus": 1,
    "defensive_contribution": {"DEF": 2, "FWD": 2, "GKP": 0, "MID": 2},
}


def test_rules_parse_game_config():
    r = ScoringRules.from_game_config(GAME_CONFIG)
    assert r.goals_scored["GKP"] == 10 and r.clean_sheets["MID"] == 1 and r.defensive_contribution["GKP"] == 0


def test_rules_reproduce_real_fpl_points_exactly():
    """Every played match in the real 2025/26 sample must score exactly what FPL awarded."""
    rules = ScoringRules.from_game_config(GAME_CONFIG)
    m = pd.read_csv(FIX / "archive_2025-26_merged_gw.csv")
    m = m[m["minutes"] > 0]
    assert len(m) > 200
    mismatches = [
        (r["name"], r["total_points"], rules.points(POS[r["position"]], r))
        for r in m.to_dict("records")
        if rules.points(POS[r["position"]], r) != r["total_points"]
    ]
    assert mismatches == []


def test_defensive_contribution_threshold_is_position_specific():
    rules = ScoringRules()
    base = dict(minutes=90, goals_scored=0, assists=0, clean_sheets=0, goals_conceded=1, saves=0, penalties_saved=0,
                penalties_missed=0, yellow_cards=0, red_cards=0, own_goals=0, bonus=0)
    assert rules.points("DEF", {**base, "defensive_contribution": 10}) == 4
    assert rules.points("DEF", {**base, "defensive_contribution": 9}) == 2
    assert rules.points("MID", {**base, "defensive_contribution": 11}) == 2
    assert rules.points("MID", {**base, "defensive_contribution": 12}) == 4
    assert rules.points("GKP", {**base, "defensive_contribution": 30}) == 2
