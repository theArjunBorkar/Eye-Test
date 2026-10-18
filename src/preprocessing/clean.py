"""
Phase 2 - Data Cleaning

Takes the combined, schema-valid dataset from load_data() and removes/fixes
things that would corrupt training later, while being explicit about every
row that gets touched (nothing silently disappears).
"""

import pandas as pd

# Minutes below this are kept (real playing time) but flagged as "small
# sample" - per-90 rates computed from a handful of minutes are misleading
# (1 tackle in 3 minutes is not "30 tackles per 90") so features.py treats
# these specially rather than dropping them outright.
MIN_MINUTES_FOR_RELIABLE_PER90 = 10

# Count-type stat columns that should never be negative or NaN after
# cleaning - if a value is missing, 0 is the correct default (the event
# didn't happen), not "unknown".
COUNT_COLUMNS_FILL_ZERO = [
    "assists", "progressive_passes", "key_passes", "goals", "dribbles_completed",
    "fouls", "yellow_cards", "red_cards", "interceptions", "clearances",
    "key_tackles", "saves_held", "saves_tipped", "saves_parried", "shots_faced",
]


def clean_data(df):
    """
    Returns (cleaned_df, report) where report is a dict describing exactly
    what was changed, so cleaning decisions are auditable rather than
    silent.
    """
    report = {}
    df = df.copy()

    # 1. Drop unused substitutes: 0 minutes means the player didn't take
    #    part, so `rating` is a meaningless placeholder (FM records it as 0),
    #    not a real low performance. Keeping these would teach a model that
    #    "playing 0 minutes = worst possible rating", which is wrong.
    unused_subs = df[df["minutes"] == 0]
    report["dropped_unused_substitutes"] = len(unused_subs)
    df = df[df["minutes"] > 0].copy()

    # 2. Flag (don't drop) small-sample appearances.
    df["small_sample"] = df["minutes"] < MIN_MINUTES_FOR_RELIABLE_PER90
    report["small_sample_rows"] = int(df["small_sample"].sum())

    # 3. Fill any missing count stats with 0 (event didn't happen).
    filled_counts = {}
    for col in COUNT_COLUMNS_FILL_ZERO:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                filled_counts[col] = int(n_missing)
    report["filled_missing_counts"] = filled_counts

    # 4. Duplicate row check - same player_id + match_id twice would double
    #    count them in aggregates.
    dupes = df.duplicated(subset=["match_id", "player_id"], keep="first")
    report["dropped_duplicate_player_match_rows"] = int(dupes.sum())
    df = df[~dupes].copy()

    df = df.reset_index(drop=True)
    return df, report


def print_report(report):
    print("Cleaning report:")
    print(f"  Dropped unused substitutes (0 minutes): {report['dropped_unused_substitutes']}")
    print(f"  Small-sample appearances flagged (<{MIN_MINUTES_FOR_RELIABLE_PER90} min): "
          f"{report['small_sample_rows']}")
    if report["filled_missing_counts"]:
        print(f"  Filled missing count stats with 0: {report['filled_missing_counts']}")
    print(f"  Dropped duplicate player-match rows: {report['dropped_duplicate_player_match_rows']}")
