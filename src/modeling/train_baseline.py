"""
Phase 3 - Random Forest Baseline

Trains a RandomForestRegressor to predict `rating` (FM26's own match
rating - a reference signal, not a real-world expert rating).

UPDATED after Phase 8/9 work: now trains on the MERGED dataset
(player_match_with_video.csv), and every run compares two feature sets
using identical CV folds so "did video features actually help" has a
direct answer rather than an assumption:
  - stats_only:       the original Phase 2 engineered features alone
  - stats_plus_video: stats_only + the 7 features confirmed RELIABLE in
                       week 3/the team-classification extension
                       (avg/max players detected per frame, ball detection
                       rate, and the Ajax-specific versions of each)

The 4 features confirmed UNRELIABLE (num_tracks_detected,
avg_track_distance_px, total_sprint_events, avg_active_frames_ratio) and
diagnostic-only columns (track_fragmentation_warning, sampled_frames,
unclassifiable_crops) are never used as model inputs, in either feature
set - that was the whole point of week 3's fragmentation finding.

WHY K-FOLD CV INSTEAD OF A SINGLE TRAIN/TEST SPLIT:
After Phase 2 cleaning we have 78 rows. A single 80/20 split leaves ~16
test rows - too few to trust a single MAE/RMSE/R² number (one unlucky or
lucky split could swing the result a lot). 5-fold cross-validation trains
and evaluates 5 times on different splits and reports the average, which
is more honest about what a model this size can actually tell us.

WHY COMPARE AGAINST A DUMMY BASELINE:
With so little data, a model can look "accurate" purely by predicting
something close to the mean rating every time. Comparing the Random
Forest's CV error against a DummyRegressor(strategy="mean") shows whether
it's actually learned something from the features, not just the average.

KNOWN LIMITATION - NOT YET ADDRESSED:
Rows from the same player across different matches aren't independent
(a player who's "just good" appears multiple times), so a plain KFold
split can let information about a player leak between train and test
folds, inflating apparent performance. With only 5 matches, grouping folds
by player would leave very few groups per fold, so this isn't fixed yet -
revisit with GroupKFold once more matches are added.

ALSO WORTH READING BEFORE INTERPRETING RESULTS: video features are
MATCH-LEVEL - every player in the same match shares identical values for
them (Phase 8). This means they can only ever explain BETWEEN-match
variance (e.g. "this was a scrappier match overall"), never WHY one
player outscored a teammate in the same match. A modest improvement from
adding them is plausible; a dramatic one would be a signal to
double-check for a bug, not evidence the model got much smarter.

Run directly:
    python -m src.modeling.train_baseline
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_validate

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED

FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_with_video.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_baseline.joblib")
METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "phase3_baseline_metrics.json")

TARGET_COLUMN = "rating"

# Identifier / non-feature columns explicitly excluded, so it's clear
# nothing here is accidentally leaking match/player identity into the model.
NON_FEATURE_COLUMNS = {
    "match_id", "player_id", "player_name", "team", "position", TARGET_COLUMN,
}

# Confirmed RELIABLE in week 3 / the team-classification extension - safe
# to use as model inputs. See src/integration/merge_video_features.py for
# the same lists, kept in sync manually since importing across
# src.integration <-> src.modeling isn't worth the coupling for 2 lists.
RELIABLE_VIDEO_FEATURES = [
    "avg_players_detected_per_frame", "max_players_detected_per_frame", "ball_detection_rate",
    "avg_ajax_players_detected_per_frame", "max_ajax_players_detected_per_frame",
    "avg_not_ajax_players_detected_per_frame", "max_not_ajax_players_detected_per_frame",
]

# Confirmed UNRELIABLE (measure scene cuts/camera motion, not player
# activity - week 3 finding) or purely diagnostic - never used as features,
# regardless of which feature set is requested.
EXCLUDED_VIDEO_COLUMNS = [
    "num_tracks_detected", "avg_track_distance_px", "total_sprint_events",
    "avg_active_frames_ratio", "track_fragmentation_warning", "sampled_frames",
    "unclassifiable_crops",
]


def get_feature_columns(df, include_video_features=True):
    excluded = set(NON_FEATURE_COLUMNS) | set(EXCLUDED_VIDEO_COLUMNS)
    if not include_video_features:
        excluded |= set(RELIABLE_VIDEO_FEATURES)
    return [c for c in df.columns if c in df.columns and c not in excluded]


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"No merged dataset found at {FEATURES_PATH}. "
            f"Run: python -m src.integration.merge_video_features"
        )
    return pd.read_csv(FEATURES_PATH)


def _cv_metrics(model, X, y, cv, groups=None):
    """Runs cross_validate for a model and returns mean MAE/RMSE/R2 across folds.
    `groups` is only needed for group-aware splitters (e.g. LeaveOneGroupOut)."""
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, groups=groups)
    return {
        "mae": round(float(-results["test_mae"].mean()), 3),
        "mae_std": round(float(results["test_mae"].std()), 3),
        "rmse": round(float(-results["test_rmse"].mean()), 3),
        "r2": round(float(results["test_r2"].mean()), 3),
    }


def _build_X(df, feature_cols):
    """Median imputation for the per-90 NaNs on small-sample rows (Phase 2) -
    documented choice, not a silent default."""
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df[feature_cols]), columns=feature_cols)
    return X, imputer


CV_SCHEMES = ("kfold", "leave_one_match_out")


def make_cv(df, cv_scheme="kfold"):
    """
    Returns (cv_splitter, groups, n_splits) for the requested scheme.

    "kfold": random 5-fold. Rows from the SAME match land in both train and
        test folds, so a match-level feature (like every video feature) can
        act as a match fingerprint: the model learns "ratings run high in
        match X" from training rows and gets credit for it on test rows
        from the same match. That doesn't generalize to an unseen match.
    "leave_one_match_out": each fold holds out one entire match, so the
        test match's match-level values were never seen in training. This
        is the honest test of whether video features carry signal that
        transfers to a new match.
    """
    if cv_scheme == "kfold":
        n_splits = min(5, len(df))
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED), None, n_splits
    if cv_scheme == "leave_one_match_out":
        groups = df["match_id"].values
        return LeaveOneGroupOut(), groups, len(set(groups))
    raise ValueError(f"Unknown cv_scheme '{cv_scheme}', expected one of {CV_SCHEMES}")


def train_and_evaluate(include_video_features=True, cv_scheme="kfold"):
    df = load_features()
    feature_cols = get_feature_columns(df, include_video_features)

    X, imputer = _build_X(df, feature_cols)
    y = df[TARGET_COLUMN]

    cv, groups, n_splits = make_cv(df, cv_scheme)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=RANDOM_SEED
    )
    dummy = DummyRegressor(strategy="mean")

    rf_metrics = _cv_metrics(rf, X, y, cv, groups)
    dummy_metrics = _cv_metrics(dummy, X, y, cv, groups)

    # Fit the final model on ALL data for saving/inspection (feature
    # importances, later predictions) - the CV metrics above, not this
    # fit, are what tell us how well it generalizes.
    rf.fit(X, y)
    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]
    )

    report = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "include_video_features": include_video_features,
        "cv_scheme": cv_scheme,
        "cv_folds": n_splits,
        "random_forest": rf_metrics,
        "dummy_mean_baseline": dummy_metrics,
        "beats_dummy_baseline_on_mae": rf_metrics["mae"] < dummy_metrics["mae"],
        "top_10_feature_importances": [
            {"feature": f, "importance": round(float(v), 4)} for f, v in importances[:10]
        ],
    }
    return rf, imputer, feature_cols, report


def _print_comparison_table(title, reports):
    print(f"\n{title}")
    print(f"{'Feature set':<22}{'n_features':<12}{'MAE':<10}{'RMSE':<10}{'R2':<10}")
    print("-" * 64)
    for label, rep in reports:
        m = rep["random_forest"]
        print(f"{label:<22}{rep['n_features']:<12}{m['mae']:<10}{m['rmse']:<10}{m['r2']:<10}")
    d = reports[-1][1]["dummy_mean_baseline"]
    print(f"{'dummy (mean)':<22}{'-':<12}{d['mae']:<10}{d['rmse']:<10}{d['r2']:<10}")


def main():
    print("=" * 66)
    print("Random Forest: stats-only vs. stats + video features")
    print("Two CV schemes - the second is the honest test for match-level features")
    print("=" * 66)

    results = {}
    for scheme in CV_SCHEMES:
        _, _, _, rep_s = train_and_evaluate(include_video_features=False, cv_scheme=scheme)
        rf_v, imp_v, cols_v, rep_v = train_and_evaluate(include_video_features=True, cv_scheme=scheme)
        results[scheme] = {"stats_only": rep_s, "stats_plus_video": rep_v,
                           "mae_delta_from_adding_video": round(
                               rep_v["random_forest"]["mae"] - rep_s["random_forest"]["mae"], 3)}
        if scheme == "kfold":
            final_model = (rf_v, imp_v, cols_v)

    _print_comparison_table("Random 5-fold CV (rows from the same match can be in train AND test):",
                            [("stats_only", results["kfold"]["stats_only"]),
                             ("stats_plus_video", results["kfold"]["stats_plus_video"])])
    _print_comparison_table("Leave-one-match-out CV (test match is never seen in training):",
                            [("stats_only", results["leave_one_match_out"]["stats_only"]),
                             ("stats_plus_video", results["leave_one_match_out"]["stats_plus_video"])])

    d_kf = results["kfold"]["mae_delta_from_adding_video"]
    d_lo = results["leave_one_match_out"]["mae_delta_from_adding_video"]
    print(f"\nMAE change from adding video features:  random K-fold {d_kf:+.3f}   "
          f"leave-one-match-out {d_lo:+.3f}")
    if d_kf < 0 and d_lo >= -0.005:
        print("The K-fold improvement does NOT survive holding out whole matches. Video features are "
              "match-level (one value per match), so under random K-fold they can act as a match "
              "fingerprint - the model learns 'ratings run high in match X' from same-match training "
              "rows. Treat the K-fold gain as leakage, not as evidence that video features predict "
              "player ratings.")
    print("Note: R2 under leave-one-match-out is computed within each held-out match, where rating "
          "variance is small - it goes negative easily and is not very interpretable. Compare MAE.")

    print("\nTop 10 feature importances (stats + video, fit on all data):")
    rf_v, imp_v, cols_v = final_model
    imps = sorted(zip(cols_v, rf_v.feature_importances_), key=lambda x: -x[1])[:10]
    for f, v in imps:
        marker = " <- video feature" if f in RELIABLE_VIDEO_FEATURES else ""
        print(f"  {f:<38} {round(float(v), 4)}{marker}")

    print("\nKnown limitations:")
    for limitation in [
        "Only 78 training rows across 5 matches - results are indicative, not reliable.",
        "Random K-fold lets a player's other rows leak into training (players repeat across matches).",
        "Video features are match-level (5 distinct values in total) - see the leave-one-match-out result.",
        "Target is FM26's own rating, not a real-world expert rating.",
    ]:
        print(f"  - {limitation}")

    # Model saved = stats+video fit on all data (kept for continuity); the
    # report makes clear whether that choice is actually justified.
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": rf_v, "imputer": imp_v, "feature_columns": cols_v}, MODEL_PATH)

    # Backward-compatible top-level keys (random K-fold) so existing readers,
    # e.g. the dashboard, keep working; new key holds the stricter scheme.
    combined = dict(results["kfold"])
    combined["leave_one_match_out"] = results["leave_one_match_out"]
    with open(METRICS_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved comparison report -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
