"""
Phase 8 tests: merging video features into the player-match dataset.

Fixtures are tiny inline CSVs, not real project data. Video feature rows
are built from mvf.VIDEO_FEATURE_COLUMNS rather than hardcoded field lists,
so adding a new video feature column doesn't silently break these fixtures
the way it did when avg_ajax_players_detected_per_frame etc. were added.

Run with:
    python -m tests.test_integration
"""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import src.integration.merge_video_features as mvf


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _video_row(match_id, avg_players=13.1, max_players=21, **overrides):
    """Builds a complete, valid video_features.csv row for tests, with
    sensible defaults for every column mvf.VIDEO_FEATURE_COLUMNS expects,
    so a fixture never silently drifts out of sync with the real schema."""
    row = {"match_id": match_id}
    defaults = {
        "avg_players_detected_per_frame": avg_players,
        "max_players_detected_per_frame": max_players,
        "ball_detection_rate": 0.028,
        "avg_ajax_players_detected_per_frame": 11.0,
        "max_ajax_players_detected_per_frame": 12,
        "avg_not_ajax_players_detected_per_frame": 9.0,
        "max_not_ajax_players_detected_per_frame": 10,
        "num_tracks_detected": 4229,
        "avg_track_distance_px": 142.4,
        "total_sprint_events": 10738,
        "avg_active_frames_ratio": 0.003,
        "track_fragmentation_warning": True,
        "sampled_frames": 1688,
        "unclassifiable_crops": 5,
    }
    for col in mvf.VIDEO_FEATURE_COLUMNS:
        row[col] = overrides.get(col, defaults[col])
    return row


def test_merge_broadcasts_match_level_features_to_every_player():
    with tempfile.TemporaryDirectory() as d:
        stats_path = os.path.join(d, "stats.csv")
        video_path = os.path.join(d, "video.csv")
        _write_csv(stats_path, [
            {"match_id": "M001", "player_id": 1, "rating": 7.0},
            {"match_id": "M001", "player_id": 2, "rating": 6.5},
            {"match_id": "M002", "player_id": 1, "rating": 8.0},
        ])
        _write_csv(video_path, [
            _video_row("M001", avg_players=13.1, max_players=21),
            _video_row("M002", avg_players=13.36, max_players=23),
        ])

        original_stats, original_video = mvf.STATS_PATH, mvf.VIDEO_PATH
        try:
            mvf.STATS_PATH, mvf.VIDEO_PATH = stats_path, video_path
            merged, report = mvf.merge_video_features()
        finally:
            mvf.STATS_PATH, mvf.VIDEO_PATH = original_stats, original_video

        assert len(merged) == 3
        m001_rows = merged[merged["match_id"] == "M001"]
        assert (m001_rows["avg_players_detected_per_frame"] == 13.1).all()
        assert (m001_rows["avg_ajax_players_detected_per_frame"] == 11.0).all()
        assert report["rows_with_reliable_video_features"] == 3


def test_matches_missing_video_get_nan_not_dropped():
    with tempfile.TemporaryDirectory() as d:
        stats_path = os.path.join(d, "stats.csv")
        video_path = os.path.join(d, "video.csv")
        _write_csv(stats_path, [
            {"match_id": "M001", "player_id": 1, "rating": 7.0},
            {"match_id": "M999", "player_id": 1, "rating": 6.0},  # no video for this match
        ])
        _write_csv(video_path, [_video_row("M001")])

        original_stats, original_video = mvf.STATS_PATH, mvf.VIDEO_PATH
        try:
            mvf.STATS_PATH, mvf.VIDEO_PATH = stats_path, video_path
            merged, report = mvf.merge_video_features()
        finally:
            mvf.STATS_PATH, mvf.VIDEO_PATH = original_stats, original_video

        assert len(merged) == 2  # M999 row kept, not dropped
        assert report["matches_without_video"] == ["M999"]
        m999_row = merged[merged["match_id"] == "M999"]
        assert m999_row["avg_players_detected_per_frame"].isna().all()


def test_reliable_and_unreliable_lists_dont_overlap():
    assert set(mvf.RELIABLE_VIDEO_FEATURES).isdisjoint(set(mvf.UNRELIABLE_VIDEO_FEATURES))


def test_ajax_specific_columns_are_reliable():
    ajax_cols = [c for c in mvf.RELIABLE_VIDEO_FEATURES if "ajax" in c]
    assert len(ajax_cols) == 4  # avg/max x ajax/not_ajax


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
