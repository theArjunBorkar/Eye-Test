"""
Phase 1 tests: per-match CSV combining + schema validation.

Fixtures here are tiny inline CSVs used only to test that the loader and
validator behave correctly - they are NOT the project's real dataset and
are never written anywhere near data/raw/matches/.

Run with:
    python -m pytest tests/ -v
or, without pytest installed:
    python -m tests.test_data_pipeline
"""

import os
import shutil
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_pipeline.schema import (
    validate_schema, MATCH_FILE_SCHEMA, VALID_POSITIONS, BACKFILL_ZERO_COLUMNS,
)
from src.data_pipeline.load_match_files import combine_match_files

HEADER = list(MATCH_FILE_SCHEMA.keys())

# Fixtures as dicts (not positional lists) so adding/reordering schema
# columns doesn't silently misalign values - each row is explicit.
GK_ROW = {
    "player_id": 7, "player_name": "T. de Vries", "team": "Ajax", "position": "gk",
    "minutes": 90, "rating": 7.1,
    "passes_completed_pct": 42.0, "xa": 0.0, "assists": 0, "progressive_passes": 1, "key_passes": 0,
    "xg": 0.0, "goals": 0, "shots_on_target_pct": 0.0, "dribbles_completed": 0,
    "tackles_completed_pct": 0.0, "headers_completed_pct": 0.0, "fouls": 0,
    "yellow_cards": 0, "red_cards": 0,
    "interceptions": 1, "clearances": 0, "key_tackles": 0,
    "saves_held": 3, "saves_tipped": 1, "saves_parried": 1, "shots_faced": 5,
}

FW_ROW = {
    "player_id": 9, "player_name": "B. Jansen", "team": "Ajax", "position": "FW",
    "minutes": 85, "rating": 8.2,
    "passes_completed_pct": 78.5, "xa": 0.4, "assists": 1, "progressive_passes": 3, "key_passes": 4,
    "xg": 0.9, "goals": 2, "shots_on_target_pct": 66.7, "dribbles_completed": 5,
    "tackles_completed_pct": 0.0, "headers_completed_pct": 50.0, "fouls": 1,
    "yellow_cards": 1, "red_cards": 0,
    "interceptions": 0, "clearances": 0, "key_tackles": 0,
    "saves_held": 0, "saves_tipped": 0, "saves_parried": 0, "shots_faced": 0,
}

OPPONENT_ROW = {
    "player_id": 4, "player_name": "X. Rival", "team": "PSV", "position": "df",
    "minutes": 90, "rating": 6.5,
    "passes_completed_pct": 80.0, "xa": 0.1, "assists": 0, "progressive_passes": 2, "key_passes": 1,
    "xg": 0.0, "goals": 0, "shots_on_target_pct": 0.0, "dribbles_completed": 1,
    "tackles_completed_pct": 60.0, "headers_completed_pct": 70.0, "fouls": 1,
    "yellow_cards": 0, "red_cards": 0,
    "interceptions": 3, "clearances": 4, "key_tackles": 1,
    "saves_held": 0, "saves_tipped": 0, "saves_parried": 0, "shots_faced": 0,
}


def _write_match_csv(dir_path, match_id, rows, drop_columns=None):
    df = pd.DataFrame(rows, columns=HEADER)
    if drop_columns:
        df = df.drop(columns=list(drop_columns))
    df.to_csv(os.path.join(dir_path, f"{match_id}.csv"), index=False)


def test_valid_match_file_passes_validation():
    df = pd.DataFrame([GK_ROW, FW_ROW], columns=HEADER)
    is_valid, problems = validate_schema(df, schema=MATCH_FILE_SCHEMA)
    blocking = [p for p in problems if "team != " not in p]  # both rows are Ajax here anyway
    assert is_valid, f"Unexpected problems: {blocking}"


def test_non_ajax_row_is_flagged():
    df = pd.DataFrame([FW_ROW, OPPONENT_ROW], columns=HEADER)
    is_valid, problems = validate_schema(df, schema=MATCH_FILE_SCHEMA)
    assert not is_valid
    assert any("Ajax" in p for p in problems)


def test_positions_normalize_case_insensitively():
    df = pd.DataFrame([GK_ROW, FW_ROW], columns=HEADER)
    is_valid, problems = validate_schema(df, schema=MATCH_FILE_SCHEMA)
    # "gk" lowercase in GK_ROW should not trigger an unknown-position problem
    assert not any("Unknown position" in p for p in problems)


def test_combine_match_files_drops_non_ajax_and_tags_match_id():
    tmp_dir = tempfile.mkdtemp()
    try:
        _write_match_csv(tmp_dir, "M001", [GK_ROW, FW_ROW, OPPONENT_ROW])
        _write_match_csv(tmp_dir, "M002", [FW_ROW])

        combined = combine_match_files(matches_dir=tmp_dir)

        assert set(combined["match_id"].unique()) == {"M001", "M002"}
        assert (combined["team"].str.lower() == "ajax").all()
        assert len(combined) == 3  # 2 Ajax rows from M001 + 1 from M002
        assert set(combined["position"].unique()).issubset(VALID_POSITIONS)
    finally:
        shutil.rmtree(tmp_dir)


def test_combine_match_files_raises_on_missing_dir():
    try:
        combine_match_files(matches_dir="/tmp/definitely_does_not_exist_xyz")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_legacy_file_missing_card_columns_gets_backfilled():
    tmp_dir = tempfile.mkdtemp()
    try:
        # Simulate an older match file recorded before card columns existed
        _write_match_csv(tmp_dir, "M000", [FW_ROW], drop_columns=BACKFILL_ZERO_COLUMNS)

        combined = combine_match_files(matches_dir=tmp_dir)

        for col in BACKFILL_ZERO_COLUMNS:
            assert col in combined.columns
            assert (combined[col] == 0).all()
    finally:
        shutil.rmtree(tmp_dir)


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
