"""
Combines the per-match CSVs in data/raw/matches/ into a single validated
player-match dataset at PLAYER_MATCH_DATA_PATH.

Each file in data/raw/matches/ is one match: filename (minus .csv) becomes
match_id, and every row in it is an Ajax player's stats for that match.

Enforces "Ajax players only" in code, not just as a convention: any row
where `team` isn't TEAM_FILTER is dropped, with a warning printed so you
know if something unexpected slipped into a file.

Run directly:
    python -m src.data_pipeline.load_match_files
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import MATCHES_DIR, PLAYER_MATCH_DATA_PATH, TEAM_FILTER
from src.data_pipeline.schema import (
    validate_schema, MATCH_FILE_SCHEMA, MATCH_FILE_COLUMNS, COMBINED_COLUMNS,
    BACKFILL_ZERO_COLUMNS,
)


def _read_match_csv(path):
    """
    Reads a per-match CSV, tolerating a leading title row (e.g. Numbers'
    CSV export adds a "Table 1" row before the real header). Tries a
    normal read first; if the expected columns aren't there, retries
    skipping leading rows until they are (or gives up after a few tries).
    """
    for skiprows in range(0, 4):
        df = pd.read_csv(path, skiprows=skiprows)
        if "player_id" in df.columns and "position" in df.columns:
            if skiprows > 0:
                print(f"[{os.path.basename(path)}] NOTE: skipped {skiprows} leading "
                      f"row(s) before the header (looks like an export title row)")
            return df
    raise ValueError(
        f"Could not find a 'player_id'/'position' header row in {path} "
        f"within the first few lines - check the file wasn't exported oddly."
    )


PCT_COLUMNS = [c for c in MATCH_FILE_SCHEMA if c.endswith("_pct")]


def _fix_percentage_scale(df, match_id):
    """
    Some spreadsheet tools export percentages as 0-1 fractions (0.76)
    instead of 0-100 (76.0). If every value in a _pct column is <= 1,
    it's almost certainly fraction form, so convert it and say so.
    """
    for col in PCT_COLUMNS:
        if col in df.columns and df[col].notna().any() and df[col].max() <= 1.0:
            df[col] = df[col] * 100
            print(f"[{match_id}] NOTE: '{col}' looked like 0-1 fractions "
                  f"(max value <= 1) - converted to 0-100 scale")
    return df


def _backfill_missing_columns(df, match_id):
    """
    Older match files won't have columns added after they were recorded
    (e.g. yellow_cards/red_cards). Fill those with 0 rather than treating
    the file as broken - see BACKFILL_ZERO_COLUMNS in schema.py.
    """
    for col in BACKFILL_ZERO_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            print(f"[{match_id}] NOTE: '{col}' column missing (older file format) "
                  f"- filled with 0 for all rows")
    return df


def _load_one_match_file(path):
    match_id = os.path.splitext(os.path.basename(path))[0]
    df = _read_match_csv(path)
    df = _fix_percentage_scale(df, match_id)
    df = _backfill_missing_columns(df, match_id)

    # Filter to Ajax rows FIRST (a match file may legitimately include the
    # opponent's players if you export a full team sheet) - dropping them
    # here means the schema check below never has to treat this as fatal.
    df["position"] = df["position"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df[df["team"].astype(str).str.strip().str.lower() == TEAM_FILTER.lower()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[{match_id}] WARNING: dropped {dropped} non-{TEAM_FILTER} row(s)")

    is_valid, problems = validate_schema(df, schema=MATCH_FILE_SCHEMA)
    for p in problems:
        print(f"[{match_id}] {p}")
    if not is_valid:
        raise ValueError(f"{match_id}.csv failed schema validation - see messages above.")

    df.insert(0, "match_id", match_id)
    return df


def combine_match_files(matches_dir=MATCHES_DIR):
    if not os.path.isdir(matches_dir):
        raise FileNotFoundError(
            f"No matches folder at {matches_dir}. Create it and add one CSV per "
            f"match (filename = match_id, e.g. M001.csv)."
        )

    files = sorted(f for f in os.listdir(matches_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(
            f"{matches_dir} exists but has no .csv files in it yet."
        )

    all_frames = []
    for fname in files:
        path = os.path.join(matches_dir, fname)
        all_frames.append(_load_one_match_file(path))

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined[COMBINED_COLUMNS]
    return combined


def main():
    combined = combine_match_files()
    os.makedirs(os.path.dirname(PLAYER_MATCH_DATA_PATH), exist_ok=True)
    combined.to_csv(PLAYER_MATCH_DATA_PATH, index=False)
    print(f"\nCombined {combined['match_id'].nunique()} match(es), "
          f"{len(combined)} player-match rows, "
          f"{combined['player_id'].nunique()} unique players "
          f"-> {PLAYER_MATCH_DATA_PATH}")


if __name__ == "__main__":
    main()
