"""
Tests for the tracking + feature-extraction logic that doesn't depend on
YOLO/ultralytics - these run in any environment, including this sandbox.

YOLO detection itself (src/cv_pipeline/detect.py) and the full video
pipeline (run_cv_pipeline.py) are NOT tested here - they need ultralytics
+ network + real video, none of which are available in this sandbox. Run
those locally.

Run with:
    python -m tests.test_cv_pipeline
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_pipeline.track import CentroidTracker
from src.cv_pipeline.extract_features import (
    extract_match_features, _track_distance_and_sprints, summarize_frame_counts,
    summarize_frame_counts_by_team,
)


def test_tracker_assigns_consistent_id_to_slow_moving_point():
    tracker = CentroidTracker(max_distance_px=50)
    # a single point drifting slightly frame to frame - should stay one track
    for i in range(10):
        tracker.update(i, [(100 + i * 2, 100)])
    assert len(tracker.tracks) == 1
    tid = list(tracker.tracks.keys())[0]
    assert len(tracker.history[tid]) == 10


def test_tracker_splits_into_new_track_beyond_max_distance():
    tracker = CentroidTracker(max_distance_px=10)
    tracker.update(0, [(100, 100)])
    tracker.update(1, [(500, 500)])  # way beyond max_distance_px - can't be the same track
    assert len(tracker.tracks) == 2


def test_tracker_drops_track_after_too_many_missed_frames():
    tracker = CentroidTracker(max_distance_px=50, max_missed_frames=2)
    tracker.update(0, [(100, 100)])
    tid = list(tracker.tracks.keys())[0]
    tracker.update(1, [])  # missed=1
    tracker.update(2, [])  # missed=2
    tracker.update(3, [])  # missed=3 > max_missed_frames -> dropped
    assert tid not in tracker.tracks


def test_tracker_handles_multiple_simultaneous_points():
    tracker = CentroidTracker(max_distance_px=30)
    tracker.update(0, [(100, 100), (300, 300)])
    tracker.update(1, [(110, 100), (310, 300)])  # both move slightly
    assert len(tracker.tracks) == 2


def test_distance_and_sprints_basic_math():
    # 3 points, moving 10px then 40px (sprint if threshold=25)
    positions = [(0, 0, 0), (1, 10, 0), (2, 50, 0)]
    dist, sprints, n = _track_distance_and_sprints(positions, sprint_threshold_px=25)
    assert dist == 50.0  # 10 + 40
    assert sprints == 1  # only the 40px step exceeds 25
    assert n == 3


def test_extract_match_features_empty_history_returns_zeros():
    features = extract_match_features({}, total_sampled_frames=0)
    assert features["num_tracks_detected"] == 0
    assert features["avg_track_distance_px"] == 0.0


def test_extract_match_features_aggregates_across_tracks():
    track_history = {
        0: [(0, 0, 0), (1, 10, 0), (2, 20, 0)],   # 20px over 3 frames
        1: [(0, 0, 0), (1, 0, 30)],                # 30px over 2 frames
    }
    features = extract_match_features(track_history, total_sampled_frames=3)
    assert features["num_tracks_detected"] == 2
    assert features["avg_track_distance_px"] == 25.0  # (20 + 30) / 2
    assert 0 < features["avg_active_frames_ratio"] <= 1


def test_fragmentation_warning_flags_short_lived_tracks():
    # 50 tracks each lasting exactly 1 frame out of 1000 sampled - classic
    # fragmentation signature (mirrors the real M001-M005 output pattern)
    track_history = {i: [(i, 0, 0)] for i in range(50)}
    features = extract_match_features(track_history, total_sampled_frames=1000)
    assert features["track_fragmentation_warning"] is True


def test_no_fragmentation_warning_for_stable_tracks():
    # a handful of tracks that each persist for most of the video
    track_history = {
        0: [(i, i, 0) for i in range(90)],
        1: [(i, 0, i) for i in range(85)],
    }
    features = extract_match_features(track_history, total_sampled_frames=100)
    assert features["track_fragmentation_warning"] is False


def test_summarize_frame_counts_basic():
    counts = [20, 22, 21, 0, 23]  # a 0 mid-video, e.g. a scene cut with no detections
    result = summarize_frame_counts(counts)
    assert result["avg_players_detected_per_frame"] == 17.2
    assert result["max_players_detected_per_frame"] == 23


def test_summarize_frame_counts_empty():
    result = summarize_frame_counts([])
    assert result["avg_players_detected_per_frame"] == 0.0
    assert result["max_players_detected_per_frame"] == 0


def test_summarize_frame_counts_by_team_separates_correctly():
    ajax_counts = [11, 10, 12]       # roughly a full Ajax side per frame
    not_ajax_counts = [9, 10, 8]     # opponent + referee etc.
    result = summarize_frame_counts_by_team(ajax_counts, not_ajax_counts)
    assert result["avg_ajax_players_detected_per_frame"] == 11.0
    assert result["max_ajax_players_detected_per_frame"] == 12
    assert result["avg_not_ajax_players_detected_per_frame"] == 9.0
    assert result["max_not_ajax_players_detected_per_frame"] == 10


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
