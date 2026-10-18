"""
Phase 2 tests: cleaning + feature engineering.

Fixtures are tiny inline DataFrames, not real project data.

Run with:
    python -m pytest tests/ -v
or:
    python -m tests.test_preprocessing
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.preprocessing.clean import clean_data, MIN_MINUTES_FOR_RELIABLE_PER90
from src.preprocessing.features import engineer_features, POSITIONS


def _base_row(**overrides):
    row = {
        "match_id": "M001", "player_id": 9, "player_name": "Test Player",
        "team": "Ajax", "position": "FW", "minutes": 90, "rating": 7.0,
        "passes_completed_pct": 80.0, "xa": 0.2, "assists": 0,
        "progressive_passes": 2, "key_passes": 1, "xg": 0.5, "goals": 1,
        "shots_on_target_pct": 50.0, "dribbles_completed": 2,
        "tackles_completed_pct": 0.0, "headers_completed_pct": 0.0, "fouls": 1,
        "yellow_cards": 0, "red_cards": 0, "interceptions": 0, "clearances": 0,
        "key_tackles": 0, "saves_held": 0, "saves_tipped": 0, "saves_parried": 0,
        "shots_faced": 0,
    }
    row.update(overrides)
    return row


def test_unused_substitutes_are_dropped():
    df = pd.DataFrame([
        _base_row(player_id=1, minutes=90, rating=7.0),
        _base_row(player_id=2, minutes=0, rating=0.0),
    ])
    cleaned, report = clean_data(df)
    assert report["dropped_unused_substitutes"] == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["player_id"] == 1


def test_small_sample_flagged_not_dropped():
    df = pd.DataFrame([
        _base_row(player_id=1, minutes=5),
        _base_row(player_id=2, minutes=90),
    ])
    cleaned, report = clean_data(df)
    assert len(cleaned) == 2  # neither dropped
    assert report["small_sample_rows"] == 1
    small = cleaned[cleaned["player_id"] == 1].iloc[0]
    assert small["small_sample"] == True
    full = cleaned[cleaned["player_id"] == 2].iloc[0]
    assert full["small_sample"] == False


def test_duplicate_player_match_rows_dropped():
    df = pd.DataFrame([
        _base_row(player_id=1, match_id="M001"),
        _base_row(player_id=1, match_id="M001"),  # exact duplicate
    ])
    cleaned, report = clean_data(df)
    assert len(cleaned) == 1
    assert report["dropped_duplicate_player_match_rows"] == 1


def test_per90_rate_correct_for_normal_minutes():
    df = pd.DataFrame([_base_row(minutes=45, goals=1)])
    cleaned, _ = clean_data(df)
    featured = engineer_features(cleaned)
    assert featured.iloc[0]["goals_per90"] == 2.0  # 1 goal in 45 min -> 2/90


def test_per90_is_nan_for_small_sample():
    df = pd.DataFrame([_base_row(minutes=3, goals=1)])
    cleaned, _ = clean_data(df)
    featured = engineer_features(cleaned)
    assert pd.isna(featured.iloc[0]["goals_per90"])


def test_position_dummies_sum_to_one():
    df = pd.DataFrame([
        _base_row(player_id=1, position="gk"),
        _base_row(player_id=2, position="MF"),
    ])
    cleaned, _ = clean_data(df)
    featured = engineer_features(cleaned)
    dummy_cols = [f"position_{p}" for p in POSITIONS]
    assert (featured[dummy_cols].sum(axis=1) == 1).all()
    assert featured.iloc[0]["position_GK"] == 1


def test_discipline_score_weights_red_card_higher():
    df = pd.DataFrame([
        _base_row(player_id=1, yellow_cards=1, red_cards=0),
        _base_row(player_id=2, yellow_cards=0, red_cards=1),
    ])
    cleaned, _ = clean_data(df)
    featured = engineer_features(cleaned)
    yellow_score = featured[featured["player_id"] == 1].iloc[0]["discipline_score"]
    red_score = featured[featured["player_id"] == 2].iloc[0]["discipline_score"]
    assert yellow_score == 1
    assert red_score == 2
    assert red_score > yellow_score


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
