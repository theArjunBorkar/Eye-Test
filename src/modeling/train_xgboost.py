"""
Phase 4 - XGBoost vs. Random Forest Comparison

Trains an XGBRegressor on the same features, target, and CV folds as
Phase 3's Random Forest, so the two are compared fairly rather than on
different splits. Reuses train_baseline.py's feature-selection and
CV-metrics logic directly instead of duplicating it.

UPDATED alongside Phase 3: now also compares stats_only vs.
stats_plus_video for XGBoost specifically, using identical CV folds -
Phase 3 already established a modest MAE improvement from video features
for Random Forest; this checks whether XGBoost sees the same effect.

Same caveats as Phase 3 apply here (see train_baseline.py's docstring):
small dataset (78 rows), CV results are indicative not reliable, KFold
(not GroupKFold) may leak player identity between folds, and video
features are match-level so can only explain between-match variance.

Run directly:
    python -m src.modeling.train_xgboost
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Run: pip install xgboost"
    ) from e

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from config.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
from src.modeling.train_baseline import (
    TARGET_COLUMN, RELIABLE_VIDEO_FEATURES, CV_SCHEMES, get_feature_columns, load_features,
    _cv_metrics, _build_X, make_cv,
)

MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_baseline.joblib")
METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "phase4_comparison_metrics.json")


def _make_models():
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=RANDOM_SEED
    )
    # Conservative depth/estimators given only 78 rows - a deep, high-capacity
    # XGBoost model would likely just memorize this small a dataset.
    xgb = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
    )
    dummy = DummyRegressor(strategy="mean")
    return rf, xgb, dummy


def compare_models(include_video_features=True, cv_scheme="kfold"):
    df = load_features()
    feature_cols = get_feature_columns(df, include_video_features)
    X, imputer = _build_X(df, feature_cols)
    y = df[TARGET_COLUMN]

    # Same splitter as Phase 3 (same random_state / same match groups) ->
    # identical folds, so RF, XGBoost, and the dummy baseline are evaluated
    # on exactly the same train/test splits, not just "similar" ones.
    cv, groups, n_splits = make_cv(df, cv_scheme)

    rf, xgb, dummy = _make_models()

    rf_metrics = _cv_metrics(rf, X, y, cv, groups)
    xgb_metrics = _cv_metrics(xgb, X, y, cv, groups)
    dummy_metrics = _cv_metrics(dummy, X, y, cv, groups)

    # Fit both on all data for saving + feature importance inspection.
    rf.fit(X, y)
    xgb.fit(X, y)

    xgb_importances = sorted(
        zip(feature_cols, xgb.feature_importances_), key=lambda x: -x[1]
    )

    report = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "include_video_features": include_video_features,
        "cv_scheme": cv_scheme,
        "cv_folds": n_splits,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "dummy_mean_baseline": dummy_metrics,
        "xgboost_beats_random_forest_on_mae": xgb_metrics["mae"] < rf_metrics["mae"],
        "xgboost_beats_dummy_on_mae": xgb_metrics["mae"] < dummy_metrics["mae"],
        "xgboost_top_10_feature_importances": [
            {"feature": f, "importance": round(float(v), 4)} for f, v in xgb_importances[:10]
        ],
    }
    return rf, xgb, imputer, feature_cols, report


def _print_table(title, rep_s, rep_v):
    print(f"\n{title}")
    print(f"{'Feature set':<20}{'Model':<16}{'MAE':<10}{'RMSE':<10}{'R2':<10}")
    print("-" * 66)
    for label, rep in [("stats_only", rep_s), ("stats_plus_video", rep_v)]:
        print(f"{label:<20}{'Random Forest':<16}{rep['random_forest']['mae']:<10}"
              f"{rep['random_forest']['rmse']:<10}{rep['random_forest']['r2']:<10}")
        print(f"{'':<20}{'XGBoost':<16}{rep['xgboost']['mae']:<10}"
              f"{rep['xgboost']['rmse']:<10}{rep['xgboost']['r2']:<10}")
    d = rep_v["dummy_mean_baseline"]
    print(f"{'':<20}{'Dummy (mean)':<16}{d['mae']:<10}{d['rmse']:<10}{d['r2']:<10}")


def main():
    print("=" * 66)
    print("XGBoost + Random Forest: stats-only vs. stats + video features")
    print("Two CV schemes - leave-one-match-out is the honest test for match-level features")
    print("=" * 66)

    results, final = {}, None
    for scheme in CV_SCHEMES:
        _, _, _, _, rep_s = compare_models(include_video_features=False, cv_scheme=scheme)
        rf_v, xgb_v, imp_v, cols_v, rep_v = compare_models(include_video_features=True, cv_scheme=scheme)
        results[scheme] = {
            "stats_only": rep_s, "stats_plus_video": rep_v,
            "xgboost_mae_delta_from_adding_video": round(
                rep_v["xgboost"]["mae"] - rep_s["xgboost"]["mae"], 3),
            "random_forest_mae_delta_from_adding_video": round(
                rep_v["random_forest"]["mae"] - rep_s["random_forest"]["mae"], 3),
        }
        if scheme == "kfold":
            final = (xgb_v, imp_v, cols_v, rep_v)

    _print_table("Random 5-fold CV (rows from the same match can be in train AND test):",
                 results["kfold"]["stats_only"], results["kfold"]["stats_plus_video"])
    _print_table("Leave-one-match-out CV (test match is never seen in training):",
                 results["leave_one_match_out"]["stats_only"], results["leave_one_match_out"]["stats_plus_video"])

    d_kf = results["kfold"]["xgboost_mae_delta_from_adding_video"]
    d_lo = results["leave_one_match_out"]["xgboost_mae_delta_from_adding_video"]
    print(f"\nXGBoost MAE change from adding video features:  random K-fold {d_kf:+.3f}   "
          f"leave-one-match-out {d_lo:+.3f}")
    print("If the K-fold gain shrinks or vanishes under leave-one-match-out, it was match "
          "fingerprinting (video features have one value per match), not transferable signal. "
          "Compare MAE; R2 within a single held-out match is not very interpretable.")

    xgb_v, imp_v, cols_v, rep_v = final
    print("\nXGBoost top 10 feature importances (stats + video, fit on all data):")
    for item in rep_v["xgboost_top_10_feature_importances"]:
        marker = " <- video feature" if item["feature"] in RELIABLE_VIDEO_FEATURES else ""
        print(f"  {item['feature']:<38} {item['importance']}{marker}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": xgb_v, "imputer": imp_v, "feature_columns": cols_v}, MODEL_PATH)
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
