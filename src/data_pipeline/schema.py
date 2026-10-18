"""
Defines the expected structure of the Ajax player-match data.

FINALIZED against the real column list (FM26 doesn't export CSVs directly,
so these are hand-recorded per match). Two schemas are defined:

- MATCH_FILE_SCHEMA: what each per-match CSV in data/raw/matches/ contains.
  No match_id column - the match is identified by the filename.
- COMBINED_SCHEMA: MATCH_FILE_SCHEMA + match_id, once files are combined
  by load_match_files.py. Everything downstream (cleaning, ML, factor
  analysis) works off the combined dataset.
"""

# column_name -> dtype family ("int", "float", "str")
MATCH_FILE_SCHEMA = {
    # general
    "player_id": "int",          # jersey number
    "player_name": "str",
    "team": "str",
    "position": "str",           # GK, DF, MF, FW (case-insensitive on input)
    "minutes": "int",
    "rating": "float",           # FM26's own match rating - a reference signal,
                                  # NOT a real-world expert rating. Document this
                                  # distinction wherever `rating` is used later.

    # passing
    "passes_completed_pct": "float",   # % of passes completed
    "xa": "float",
    "assists": "int",
    "progressive_passes": "int",
    "key_passes": "int",

    # attacking
    "xg": "float",
    "goals": "int",
    "shots_on_target_pct": "float",    # % of shots on target
    "dribbles_completed": "int",

    # defending
    "tackles_completed_pct": "float",  # % of tackles won
    "headers_completed_pct": "float",  # % of headers won
    "fouls": "int",
    "yellow_cards": "int",
    "red_cards": "int",
    "interceptions": "int",
    "clearances": "int",
    "key_tackles": "int",

    # goalkeeping (blank/0 for outfield players)
    "saves_held": "int",
    "saves_tipped": "int",
    "saves_parried": "int",
    "shots_faced": "int",
}

MATCH_FILE_COLUMNS = list(MATCH_FILE_SCHEMA.keys())

COMBINED_SCHEMA = {"match_id": "str", **MATCH_FILE_SCHEMA}
COMBINED_COLUMNS = list(COMBINED_SCHEMA.keys())

VALID_POSITIONS = {"GK", "DF", "MF", "FW"}
# Position convention: attacking midfielders (AM) are recorded as FW, not
# MF - there's no separate AM bucket. Apply this consistently across all
# match files so position-based analysis (feature engineering, factor
# analysis) isn't skewed by inconsistent classification.

# Columns that only make sense for outfield players vs. only for the keeper.
# Used to skip GK-only range checks on outfield rows and vice versa, since
# 0/blank is expected and not an error there.
GK_ONLY_COLUMNS = {"saves_held", "saves_tipped", "saves_parried", "shots_faced"}

# Columns older match files (recorded before these were added) may lack.
# The loader backfills these with 0 rather than rejecting the file, since
# it's a safe default (no cards recorded) and re-exporting old files is a
# waste of effort. New fields should only go here if 0 is a genuinely safe
# assumption for pre-existing data.
BACKFILL_ZERO_COLUMNS = {"yellow_cards", "red_cards"}

# (column, min, max) sanity bounds.
RANGE_CHECKS = [
    ("minutes", 0, 120),
    ("rating", 0, 10),
    ("passes_completed_pct", 0, 100),
    ("shots_on_target_pct", 0, 100),
    ("tackles_completed_pct", 0, 100),
    ("headers_completed_pct", 0, 100),
    ("goals", 0, 15),
    ("assists", 0, 15),
    ("yellow_cards", 0, 2),
    ("red_cards", 0, 1),
]


def _normalize_positions(df):
    if "position" in df.columns:
        df["position"] = df["position"].astype(str).str.strip().str.upper()
    return df


def validate_schema(df, schema=MATCH_FILE_SCHEMA):
    """
    Checks a DataFrame against the given schema and RANGE_CHECKS.
    Returns (is_valid: bool, problems: list[str]).
    Does not raise, and does not mutate df - callers decide what to do.
    """
    problems = []
    required_columns = list(schema.keys())

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in required_columns]
    if extra:
        problems.append(f"Unexpected extra columns (not an error, just noting): {extra}")

    if "position" in df.columns:
        positions_seen = set(df["position"].astype(str).str.strip().str.upper().unique())
        bad_positions = positions_seen - VALID_POSITIONS
        if bad_positions:
            problems.append(f"Unknown position values: {bad_positions} "
                             f"(expected one of {sorted(VALID_POSITIONS)}, case-insensitive)")

    for col, lo, hi in RANGE_CHECKS:
        if col in df.columns:
            out_of_range = df[(df[col] < lo) | (df[col] > hi)]
            if len(out_of_range) > 0:
                problems.append(
                    f"{len(out_of_range)} row(s) have '{col}' outside expected range [{lo}, {hi}]"
                )

    if "player_id" in df.columns and df["player_id"].isna().any():
        problems.append("Some rows have missing player_id")

    if "team" in df.columns:
        non_ajax = df[df["team"].astype(str).str.strip().str.lower() != "ajax"]
        if len(non_ajax) > 0:
            problems.append(
                f"{len(non_ajax)} row(s) have team != 'Ajax' - this project only "
                f"analyzes Ajax players, these rows should be removed"
            )

    blocking_markers = ("Missing required columns", "Unknown position",
                         "outside expected range", "missing player_id", "team != 'Ajax'")
    is_valid = not any(any(m in p for m in blocking_markers) for p in problems)
    return is_valid, problems
