"""
Week 3 CV pipeline entry point.

video -> sample frames (FRAME_SAMPLE_RATE) -> YOLO player detection ->
CentroidTracker -> extract_match_features() -> one row per match in
data/processed/video_features.csv

Also computes Ajax-specific per-frame counts (avg_ajax_players_detected_per_frame
etc.) when a confirmed team_labels_{match_id}.json exists for that match
(see classify_teams.py / apply_team_labels.py) - matches without confirmed
labels still get the original all-persons counts, just not the Ajax-split
versions, with a clear warning rather than a silent gap.

Requires: pip install ultralytics opencv-python
Requires each match video at data/videos/raw/{match_id}.mp4 - same
match_id convention as the CSV match files, so features can be joined to
the stats data later (Phase 8).

Run:
    python -m src.cv_pipeline.run_cv_pipeline
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is not installed. Run: pip install opencv-python") from e

import pandas as pd

from config.config import VIDEO_RAW_DIR, PROCESSED_DATA_DIR, FRAME_SAMPLE_RATE
from src.cv_pipeline.detect import load_model, detect_frame_full
from src.cv_pipeline.track import CentroidTracker
from src.cv_pipeline.extract_features import (
    extract_match_features, summarize_frame_counts, summarize_frame_counts_by_team,
)
from src.cv_pipeline.team_classify import extract_shirt_color, assign_to_nearest_cluster
from src.cv_pipeline.apply_team_labels import load_team_centroids

OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "video_features.csv")


def process_video(video_path, model, team_labels=None):
    """
    team_labels: optional (team_names, centroids_bgr) tuple from
    apply_team_labels.load_team_centroids() - pooled centroids by final
    team label, not raw per-cluster centroids (see that function's
    docstring for why the distinction matters - it fixes a real measured
    classification bias). If provided, every detected player also gets
    classified per frame using nearest nearest-team-centroid. If None,
    only the original all-persons counts are computed (no team split).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    tracker = CentroidTracker()
    frame_idx = 0
    sampled_frames = 0
    ball_detections = 0
    frame_player_counts = []
    ajax_frame_counts = []
    not_ajax_frame_counts = []
    unclassifiable_crops = 0

    team_names, team_centroids = team_labels if team_labels else (None, None)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_SAMPLE_RATE == 0:
            player_dets, ball_dets = detect_frame_full(model, frame)
            centroids = [d["centroid"] for d in player_dets]
            tracker.update(sampled_frames, centroids)
            frame_player_counts.append(len(player_dets))

            if team_labels is not None:
                ajax_count, not_ajax_count = 0, 0
                for det in player_dets:
                    color = extract_shirt_color(frame, det["bbox"])
                    if color is None:
                        unclassifiable_crops += 1
                        continue
                    team_idx = assign_to_nearest_cluster(color, team_centroids)
                    team = team_names[team_idx]
                    if team == "ajax":
                        ajax_count += 1
                    else:
                        not_ajax_count += 1
                ajax_frame_counts.append(ajax_count)
                not_ajax_frame_counts.append(not_ajax_count)

            if ball_dets:
                ball_detections += 1
            sampled_frames += 1
        frame_idx += 1

    cap.release()

    features = extract_match_features(tracker.history, sampled_frames)
    features.update(summarize_frame_counts(frame_player_counts))
    features["ball_detection_rate"] = (
        round(ball_detections / sampled_frames, 3) if sampled_frames else 0.0
    )
    features["sampled_frames"] = sampled_frames

    if team_labels is not None:
        features.update(summarize_frame_counts_by_team(ajax_frame_counts, not_ajax_frame_counts))
        features["unclassifiable_crops"] = unclassifiable_crops
    else:
        features.update({
            "avg_ajax_players_detected_per_frame": None,
            "max_ajax_players_detected_per_frame": None,
            "avg_not_ajax_players_detected_per_frame": None,
            "max_not_ajax_players_detected_per_frame": None,
            "unclassifiable_crops": None,
        })

    if features["track_fragmentation_warning"]:
        print(f"  WARNING: track_fragmentation_warning=True - avg_active_frames_ratio="
              f"{features['avg_active_frames_ratio']} means tracks survive only a "
              f"fraction of the video before being lost. This is very likely scene cuts "
              f"and/or camera panning (common in highlight footage), not player movement. "
              f"'avg_track_distance_px' and 'total_sprint_events' should NOT be trusted "
              f"as activity signals for this match - use 'avg_players_detected_per_frame' "
              f"and 'ball_detection_rate' instead, which don't require tracking continuity.")

    return features


def run():
    if not os.path.isdir(VIDEO_RAW_DIR):
        raise FileNotFoundError(f"No videos folder at {VIDEO_RAW_DIR}")

    video_files = sorted(
        f for f in os.listdir(VIDEO_RAW_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v"))
    )
    if not video_files:
        raise FileNotFoundError(
            f"No video files found in {VIDEO_RAW_DIR}. Expected e.g. M001.mp4, matching "
            f"the match_id used in data/raw/matches/."
        )

    print("Loading YOLO model (downloads weights on first run if not cached)...")
    model = load_model()

    rows = []
    matches_missing_labels = []
    for fname in video_files:
        match_id = os.path.splitext(fname)[0]
        print(f"\nProcessing {fname} (match_id={match_id})...")

        try:
            team_labels = load_team_centroids(match_id)
        except (FileNotFoundError, ValueError) as e:
            print(f"  NOTE: no confirmed team labels for {match_id} ({e}) - "
                  f"Ajax-specific counts will be blank for this match.")
            team_labels = None
            matches_missing_labels.append(match_id)

        features = process_video(os.path.join(VIDEO_RAW_DIR, fname), model, team_labels)
        features["match_id"] = match_id
        rows.append(features)
        print(f"  {features}")

    df = pd.DataFrame(rows)
    cols = ["match_id"] + [c for c in df.columns if c != "match_id"]
    df = df[cols]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} match(es) of video features -> {OUTPUT_PATH}")

    n_flagged = int(df["track_fragmentation_warning"].sum()) if "track_fragmentation_warning" in df else 0
    if n_flagged:
        print(f"\n{n_flagged}/{len(df)} match(es) flagged with track_fragmentation_warning=True. "
              f"For those, treat 'avg_track_distance_px'/'total_sprint_events' as unreliable "
              f"(see warnings above) - 'avg_players_detected_per_frame' and "
              f"'ball_detection_rate' are the features not affected by this issue.")
    if matches_missing_labels:
        print(f"\nMatches without confirmed team labels (Ajax-specific counts blank): "
              f"{matches_missing_labels}")
    print("\nReminder: all features are match-level aggregates (not per-player), and pixel "
          "distances are relative activity proxies, not real-world units. See "
          "src/cv_pipeline docstrings for full details.")


if __name__ == "__main__":
    run()
