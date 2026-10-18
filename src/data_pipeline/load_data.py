"""
Loads the COMBINED player-match dataset (built by load_match_files.py) and
validates it against the finalized schema.

This is intentionally minimal for Phase 1 - just load + validate + report.
Cleaning and feature engineering (handling missing values, outliers,
derived features) come in Phase 2 and will build on top of load_data().

Run directly:
    python -m src.data_pipeline.load_data
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import PLAYER_MATCH_DATA_PATH
from src.data_pipeline.schema import validate_schema, COMBINED_SCHEMA, COMBINED_COLUMNS


def load_data(path=PLAYER_MATCH_DATA_PATH):
    """
    Loads the combined player-match CSV into a DataFrame and validates it.
    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the schema is invalid.
    Returns the DataFrame if valid (warnings are printed but don't block).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No combined dataset found at {path}.\n"
            f"Add your per-match CSVs to data/raw/matches/ (one file per match, "
            f"filename = match_id) then run:\n"
            f"    python -m src.data_pipeline.load_match_files"
        )

    df = pd.read_csv(path)
    is_valid, problems = validate_schema(df, schema=COMBINED_SCHEMA)

    for p in problems:
        prefix = "ERROR" if _is_blocking(p) else "NOTE"
        print(f"[{prefix}] {p}")

    if not is_valid:
        raise ValueError("Dataset failed schema validation - see errors above.")

    return df


def _is_blocking(problem_str):
    return ("Missing required columns" in problem_str
            or "Unknown position" in problem_str
            or "outside expected range" in problem_str
            or "missing player_id" in problem_str
            or "team != " in problem_str)


def summarize(df):
    """Quick sanity-check summary, useful when eyeballing a new dataset."""
    print(f"\nMatches: {df['match_id'].nunique()}   Rows: {len(df)}   "
          f"Players: {df['player_id'].nunique()}")
    print("\nPosition breakdown:")
    print(df["position"].value_counts())
    numeric_cols = [c for c in COMBINED_COLUMNS if c in df.columns
                    and df[c].dtype.kind in "if"]
    print("\nNumeric column summary:")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    data = load_data()
    print("\nDataset loaded and passed schema validation.\n")
    summarize(data)
