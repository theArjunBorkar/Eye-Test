"""
Phase 8 - Integrate video-derived features with the player-match dataset.

Video features (week 3's CV pipeline) are MATCH-LEVEL, not per-player (see
the scoping decision documented in the CV pipeline README section and in
src/cv_pipeline/extract_features.py). So this merge gives every Ajax
player in a given match the SAME video-derived values for that match -
it's not personalized per player, and that's a real limitation to carry
forward into anything built on top of this merged dataset, not a subtle
detail to gloss over.

Also separates video columns into RELIABLE_VIDEO_FEATURES and
UNRELIABLE_VIDEO_FEATURES (per the track_fragmentation_warning finding in
week 3) so anything downstream (Phase 9 factor analysis, any further
modeling) can deliberately choose to include or exclude the unreliable
ones, rather than them quietly ending up in an analysis.

Run directly:
    python -m src.integration.merge_video_features
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from config.config import PROCESSED_DATA_DIR

STATS_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_features.csv")
VIDEO_PATH = os.path.join(PROCESSED_DATA_DIR, "video_features.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_with_video.csv")

# Confirmed reliable on real data (week 3): don't require tracking
# continuity, so scene cuts / camera panning can't corrupt them.
RELIABLE_VIDEO_FEATURES = [
    "avg_players_detected_per_frame",
    "max_players_detected_per_frame",
    "ball_detection_rate",
    "avg_ajax_players_detected_per_frame",
    "max_ajax_players_detected_per_frame",
    "avg_not_ajax_players_detected_per_frame",
    "max_not_ajax_players_detected_per_frame",
]

# Confirmed UNRELIABLE on real data (week 3): all 5 real matches triggered
# track_fragmentation_warning=True, meaning these measure scene cuts/camera
# motion, not player activity. Kept in the merged dataset for transparency
# and traceability, but should not be used as model inputs without first
# addressing the fragmentation issue (e.g. shot-cut detection).
UNRELIABLE_VIDEO_FEATURES = [
    "num_tracks_detected",
    "avg_track_distance_px",
    "total_sprint_events",
    "avg_active_frames_ratio",
]

VIDEO_FEATURE_COLUMNS = RELIABLE_VIDEO_FEATURES + UNRELIABLE_VIDEO_FEATURES + [
    "track_fragmentation_warning", "sampled_frames", "unclassifiable_crops",
]


def merge_video_features():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"No processed stats found at {STATS_PATH}. Run: python -m src.preprocessing.run_phase2"
        )
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"No video features found at {VIDEO_PATH}. Run: python -m src.cv_pipeline.run_cv_pipeline"
        )

    stats = pd.read_csv(STATS_PATH)
    video = pd.read_csv(VIDEO_PATH)

    missing_video_cols = [c for c in VIDEO_FEATURE_COLUMNS if c not in video.columns]
    if missing_video_cols:
        raise ValueError(f"video_features.csv is missing expected columns: {missing_video_cols}")

    video_subset = video[["match_id"] + VIDEO_FEATURE_COLUMNS]

    merged = stats.merge(video_subset, on="match_id", how="left")

    stats_matches = set(stats["match_id"].unique())
    video_matches = set(video["match_id"].unique())
    matches_without_video = stats_matches - video_matches

    report = {
        "n_rows": len(merged),
        "n_matches_in_stats": len(stats_matches),
        "n_matches_with_video": len(video_matches & stats_matches),
        "matches_without_video": sorted(matches_without_video),
        "rows_with_reliable_video_features": int(
            merged[RELIABLE_VIDEO_FEATURES].notna().all(axis=1).sum()
        ),
        "reliable_video_features": RELIABLE_VIDEO_FEATURES,
        "unreliable_video_features": UNRELIABLE_VIDEO_FEATURES,
    }
    return merged, report


def print_report(report):
    print(f"Merged {report['n_rows']} player-match rows.")
    print(f"Matches in stats data: {report['n_matches_in_stats']}, "
          f"with video features: {report['n_matches_with_video']}")
    if report["matches_without_video"]:
        print(f"NOTE: matches with no video data (video columns will be NaN for these): "
              f"{report['matches_without_video']}")
    print(f"Rows with all reliable video features present: "
          f"{report['rows_with_reliable_video_features']}/{report['n_rows']}")
    print(f"\nReliable video features (safe to use as model inputs): {report['reliable_video_features']}")
    print(f"UNRELIABLE video features (kept for transparency only, not "
          f"recommended as model inputs - see week 3 findings): {report['unreliable_video_features']}")
    print("\nReminder: video features are MATCH-LEVEL - every Ajax player in the same match "
          "shares identical video-derived values, they are not personalized per player.")


def main():
    merged, report = merge_video_features()
    print_report(report)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
