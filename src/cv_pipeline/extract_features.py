"""
Feature extraction from tracked centroid histories.

IMPORTANT LIMITATION - read before interpreting any output:
Without camera calibration (knowing the real-world size of the pitch as it
appears in each video), movement can only be measured in PIXELS per
sampled frame, not meters or real speed. "Distance" and "sprint" features
here are RELATIVE ACTIVITY PROXIES, comparable within the same video, but
NOT directly comparable across videos shot from different camera
distances/angles, and NOT real-world distances. This must be stated
explicitly wherever these features are used or reported - do not present
them as if they were meters covered or km/h.

These are match-level aggregates across all tracks (players), not
attributed to individual named players - see the CV pipeline scoping note
in the project README.

Pure Python/numpy - testable without YOLO or video I/O.
"""

import numpy as np

from config.config import SPRINT_SPEED_THRESHOLD_PX


def _track_distance_and_sprints(positions, sprint_threshold_px):
    """
    positions: list of (frame_idx, x, y) for ONE track, in frame order.
    Returns (total_distance_px, sprint_count, n_points).
    """
    if len(positions) < 2:
        return 0.0, 0, len(positions)

    pts = np.array([(x, y) for _, x, y in positions])
    deltas = np.diff(pts, axis=0)
    step_distances = np.linalg.norm(deltas, axis=1)

    total_distance = float(step_distances.sum())
    sprint_count = int((step_distances > sprint_threshold_px).sum())
    return total_distance, sprint_count, len(positions)


def summarize_frame_counts(frame_player_counts):
    """
    Simple per-frame detection counts (no tracking required) - how many
    people were detected in each sampled frame, averaged. Unlike distance/
    sprint features, this is immune to scene cuts and camera panning: it
    doesn't depend on matching anyone between frames, so a cut or a fast
    pan doesn't corrupt it the way tracking-based features can.
    """
    if not frame_player_counts:
        return {"avg_players_detected_per_frame": 0.0, "max_players_detected_per_frame": 0}
    return {
        "avg_players_detected_per_frame": round(float(np.mean(frame_player_counts)), 2),
        "max_players_detected_per_frame": int(np.max(frame_player_counts)),
    }


def summarize_frame_counts_by_team(ajax_frame_counts, not_ajax_frame_counts):
    """
    Same idea as summarize_frame_counts(), split by team using confirmed
    team labels (see classify_teams.py / apply_team_labels.py). Only
    meaningful for matches that have a confirmed labels file - callers
    should leave these fields blank/None for matches without one, rather
    than silently reporting zero (which would look like "no Ajax players
    detected" instead of "not classified").
    """
    ajax_stats = summarize_frame_counts(ajax_frame_counts)
    not_ajax_stats = summarize_frame_counts(not_ajax_frame_counts)
    return {
        "avg_ajax_players_detected_per_frame": ajax_stats["avg_players_detected_per_frame"],
        "max_ajax_players_detected_per_frame": ajax_stats["max_players_detected_per_frame"],
        "avg_not_ajax_players_detected_per_frame": not_ajax_stats["avg_players_detected_per_frame"],
        "max_not_ajax_players_detected_per_frame": not_ajax_stats["max_players_detected_per_frame"],
    }


FRAGMENTATION_WARNING_RATIO = 0.05  # avg_active_frames_ratio below this suggests tracks are
                                     # too short-lived to trust distance/sprint numbers


def extract_match_features(track_history, total_sampled_frames,
                            sprint_threshold_px=SPRINT_SPEED_THRESHOLD_PX):
    """
    track_history: dict {track_id: [(frame_idx, x, y), ...]} as produced by
        CentroidTracker.history after processing a video.
    total_sampled_frames: number of frames actually processed (after
        FRAME_SAMPLE_RATE downsampling), used to compute activity ratios.

    Returns a flat dict of match-level aggregate features - NOT per-player.
    Includes a `track_fragmentation_warning` flag: if the average track
    survives fewer than FRAGMENTATION_WARNING_RATIO of all sampled frames,
    distance/sprint numbers are likely dominated by scene cuts or camera
    motion rather than real movement (see module docstring) and should not
    be trusted as activity signals.
    """
    if not track_history or total_sampled_frames == 0:
        return {
            "num_tracks_detected": 0,
            "avg_track_distance_px": 0.0,
            "total_sprint_events": 0,
            "avg_active_frames_ratio": 0.0,
            "track_fragmentation_warning": False,
        }

    per_track_distances = []
    per_track_sprints = []
    per_track_active_ratios = []

    for tid, positions in track_history.items():
        dist, sprints, n_points = _track_distance_and_sprints(positions, sprint_threshold_px)
        per_track_distances.append(dist)
        per_track_sprints.append(sprints)
        per_track_active_ratios.append(n_points / total_sampled_frames)

    avg_ratio = float(np.mean(per_track_active_ratios))

    return {
        "num_tracks_detected": len(track_history),
        "avg_track_distance_px": round(float(np.mean(per_track_distances)), 1),
        "total_sprint_events": int(sum(per_track_sprints)),
        "avg_active_frames_ratio": round(avg_ratio, 3),
        "track_fragmentation_warning": avg_ratio < FRAGMENTATION_WARNING_RATIO,
    }
