"""
Phase 4 tests: XGBoost vs Random Forest comparison pipeline, including the
stats-only vs. stats+video toggle added alongside Phase 3's update.

Skips (doesn't fail) if xgboost isn't installed in the current environment,
since that's an environment issue, not a code bug - install it with
`pip install xgboost` to actually run these.

Run with:
    python -m tests.test_xgboost_comparison
"""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import numpy as np
import pandas as pd


def _fake_features_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "match_id": [f"M{i % 5:03d}" for i in range(n)],
        "player_id": rng.integers(1, 20, n),
        "player_name": [f"Player {i}" for i in range(n)],
        "team": "Ajax",
        "position": rng.choice(["GK", "DF", "MF", "FW"], n),
        "minutes": rng.integers(10, 90, n),
        "goals": rng.integers(0, 3, n),
        "goal_contributions": rng.integers(0, 4, n),
        "defensive_actions": rng.integers(0, 6, n),
        "position_GK": 0, "position_DF": 0, "position_MF": 0, "position_FW": 1,
        "avg_players_detected_per_frame": rng.uniform(10, 15, n),
        "num_tracks_detected": rng.integers(1000, 5000, n),  # unreliable - must never be a feature
        "rating": rng.uniform(5, 9, n),
    })


def test_comparison_pipeline_runs_end_to_end_both_feature_sets():
    import src.modeling.train_baseline as tb
    import src.modeling.train_xgboost as txgb

    df = _fake_features_df(n=40)
    original_path = tb.FEATURES_PATH
    try:
        with tempfile.TemporaryDirectory() as d:
            fake_path = os.path.join(d, "features.csv")
            df.to_csv(fake_path, index=False)
            tb.FEATURES_PATH = fake_path  # train_xgboost.py calls tb.load_features(), so this is the one that matters

            for include_video, scheme in [(False, "kfold"), (True, "kfold"), (True, "leave_one_match_out")]:
                rf, xgb, imputer, feature_cols, report = txgb.compare_models(include_video, scheme)
                assert report["cv_scheme"] == scheme

                assert report["n_rows"] == 40
                assert report["include_video_features"] == include_video
                for key in ("random_forest", "xgboost", "dummy_mean_baseline"):
                    assert "mae" in report[key]
                assert "xgboost_beats_random_forest_on_mae" in report
                assert len(report["xgboost_top_10_feature_importances"]) <= 10
                if not include_video:
                    feature_names = [item["feature"] for item in report["xgboost_top_10_feature_importances"]]
                    assert "avg_players_detected_per_frame" not in feature_names
    finally:
        tb.FEATURES_PATH = original_path


if __name__ == "__main__":
    if not XGBOOST_AVAILABLE:
        print("SKIPPED  xgboost not installed in this environment - "
              "run `pip install xgboost` to run these tests")
    else:
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
            except Exception as e:
                print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
