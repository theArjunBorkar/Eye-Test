"""
Phase 3 tests: training pipeline sanity checks, including the
stats-only vs. stats+video feature-set toggle added when Phase 3 was
updated to train on the merged (Phase 8) dataset.

Uses a small in-memory fixture DataFrame (not real project data) purely to
check train_and_evaluate() runs correctly and returns the expected
shapes/keys - not to judge model quality, which needs real data.

Run with:
    python -m tests.test_modeling
"""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import src.modeling.train_baseline as tb
from src.modeling.train_baseline import get_feature_columns, NON_FEATURE_COLUMNS, RELIABLE_VIDEO_FEATURES


def _fake_features_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
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
        "track_fragmentation_warning": True,                 # diagnostic - must never be a feature
        "rating": rng.uniform(5, 9, n),
    })
    return df


def test_feature_columns_exclude_identifiers_and_target():
    df = _fake_features_df()
    cols = get_feature_columns(df)
    for excluded in NON_FEATURE_COLUMNS:
        assert excluded not in cols


def test_feature_columns_always_exclude_unreliable_video_columns():
    df = _fake_features_df()
    cols_with_video = get_feature_columns(df, include_video_features=True)
    cols_without_video = get_feature_columns(df, include_video_features=False)
    # unreliable/diagnostic columns must never appear, regardless of the flag
    for col in ["num_tracks_detected", "track_fragmentation_warning"]:
        assert col not in cols_with_video
        assert col not in cols_without_video


def test_include_video_features_flag_controls_reliable_video_columns():
    df = _fake_features_df()
    cols_with_video = get_feature_columns(df, include_video_features=True)
    cols_without_video = get_feature_columns(df, include_video_features=False)
    assert "avg_players_detected_per_frame" in cols_with_video
    assert "avg_players_detected_per_frame" not in cols_without_video


def test_training_pipeline_runs_end_to_end_both_feature_sets():
    df = _fake_features_df(n=40)
    original_path = tb.FEATURES_PATH
    try:
        with tempfile.TemporaryDirectory() as d:
            fake_path = os.path.join(d, "features.csv")
            df.to_csv(fake_path, index=False)
            tb.FEATURES_PATH = fake_path

            for include_video in (False, True):
                rf, imputer, feature_cols, report = tb.train_and_evaluate(include_video)

                assert report["n_rows"] == 40
                assert report["include_video_features"] == include_video
                assert "mae" in report["random_forest"]
                assert "mae" in report["dummy_mean_baseline"]
                assert len(report["top_10_feature_importances"]) <= 10

                preds = rf.predict(pd.DataFrame(
                    imputer.transform(df[feature_cols]), columns=feature_cols
                ))
                assert len(preds) == 40
    finally:
        tb.FEATURES_PATH = original_path


def test_make_cv_leave_one_match_out_holds_out_whole_matches():
    df = _fake_features_df(n=40)
    cv, groups, n_splits = tb.make_cv(df, "leave_one_match_out")
    assert n_splits == df["match_id"].nunique()
    for train_idx, test_idx in cv.split(df, groups=groups):
        train_matches = set(df["match_id"].iloc[train_idx])
        test_matches = set(df["match_id"].iloc[test_idx])
        assert len(test_matches) == 1
        assert train_matches.isdisjoint(test_matches)  # the test match is never seen in training


def test_make_cv_rejects_unknown_scheme():
    df = _fake_features_df(n=40)
    try:
        tb.make_cv(df, "nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def _match_offset_df(seed=0):
    """Rating is driven ONLY by a per-match offset; the 'video' feature is just
    one distinct value per match (i.e. a match ID in disguise); noise_stat is unrelated."""
    rng = np.random.default_rng(seed)
    offsets = [-0.8, -0.3, 0.1, 0.5, 0.9]
    feature_vals = [12.4, 13.1, 13.3, 12.9, 13.6]
    rows = []
    for i, (off, fv) in enumerate(zip(offsets, feature_vals)):
        for _ in range(20):
            rows.append({"match_id": f"M{i}", "player_id": int(rng.integers(1, 20)),
                         "player_name": "p", "team": "Ajax", "position": "MF",
                         "noise_stat": rng.normal(), "avg_players_detected_per_frame": fv,
                         "rating": 6.5 + off + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def test_match_level_feature_gain_is_leakage_under_kfold_but_not_leave_one_match_out():
    """
    Regression test for the real finding on the project data: a match-level
    feature (one value per match) looks helpful under random K-fold because
    it lets the model recognise WHICH match a row came from, but gives no
    benefit when a whole match is held out. Reproduced here on synthetic
    data where the feature is provably only a match label.
    """
    df = _match_offset_df()
    original_path = tb.FEATURES_PATH
    try:
        with tempfile.TemporaryDirectory() as d:
            fake_path = os.path.join(d, "f.csv")
            df.to_csv(fake_path, index=False)
            tb.FEATURES_PATH = fake_path

            gain = {}
            for scheme in tb.CV_SCHEMES:
                mae_without = tb.train_and_evaluate(False, scheme)[3]["random_forest"]["mae"]
                mae_with = tb.train_and_evaluate(True, scheme)[3]["random_forest"]["mae"]
                gain[scheme] = mae_without - mae_with  # positive = video feature helped
    finally:
        tb.FEATURES_PATH = original_path

    assert gain["kfold"] > 0.15, f"expected a large apparent gain under K-fold, got {gain['kfold']:.3f}"
    assert gain["leave_one_match_out"] < 0.05, (
        f"expected no real gain when whole matches are held out, got {gain['leave_one_match_out']:.3f}")


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
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
