"""
Phase 2 - Feature Engineering

Builds on clean_data()'s output. Two kinds of new features:
1. Per-90 normalized rates for count stats (so a player who played 30
   minutes isn't directly compared to one who played 90).
2. A few composite features that combine related raw stats into signals
   that are more directly about performance (goal involvement, defensive
   workload, discipline).

Per-90 rates are left as NaN for small-sample rows (see clean.py) instead
of computing a misleading extrapolated rate - Phase 3's model training
needs to decide explicitly how to handle those NaNs (impute, drop, or use
raw counts instead), not have this module quietly paper over it.
"""

import pandas as pd

# Count stats that get a per-90 version. Percentage columns (pass_pct etc.)
# and the target (`rating`) are intentionally excluded - they're already
# rate-like or are the prediction target, not raw counts to normalize.
PER90_SOURCE_COLUMNS = [
    "assists", "progressive_passes", "key_passes", "goals", "dribbles_completed",
    "xa", "xg", "fouls", "interceptions", "clearances", "key_tackles",
    "saves_held", "saves_tipped", "saves_parried", "shots_faced",
]

POSITIONS = ["GK", "DF", "MF", "FW"]


def add_per90_features(df):
    df = df.copy()
    for col in PER90_SOURCE_COLUMNS:
        if col not in df.columns:
            continue
        per90_col = f"{col}_per90"
        rate = df[col] / df["minutes"] * 90
        # small-sample rows get NaN instead of an inflated/misleading rate
        df[per90_col] = rate.where(~df["small_sample"], other=pd.NA)
    return df


def add_composite_features(df):
    df = df.copy()

    # Goal involvement: raw counts (not per90) since these are usually
    # discussed as match totals, and per90 versions already exist above
    # if needed (goals_per90 + assists_per90).
    df["goal_contributions"] = df["goals"] + df["assists"]

    # Defensive workload - raw counts. tackles/headers are already
    # percentages (success rate) so they're not summed in here; this is
    # about volume of defensive involvement, not success rate.
    df["defensive_actions"] = df["interceptions"] + df["clearances"] + df["key_tackles"]

    # Discipline - red card weighted heavier since it ends the player's
    # match and often costs the team a player disadvantage.
    df["discipline_score"] = df["yellow_cards"] + 2 * df["red_cards"]

    # Shot quality proxy: actual goals vs. expected goals for the match.
    # Positive = outperformed xG (clinical), negative = underperformed.
    df["goals_minus_xg"] = df["goals"] - df["xg"]

    return df


def add_position_dummies(df):
    df = df.copy()
    df["position"] = df["position"].str.upper()
    for pos in POSITIONS:
        df[f"position_{pos}"] = (df["position"] == pos).astype(int)
    return df


def engineer_features(df):
    df = add_per90_features(df)
    df = add_composite_features(df)
    df = add_position_dummies(df)
    return df
