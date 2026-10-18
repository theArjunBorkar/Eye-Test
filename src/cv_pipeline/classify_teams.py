"""
Team classification pipeline.

Samples shirt colors from detected players across a match video, clusters
them into teams (default: 3 clusters = two teams + referee/officials), and
prints a sample crop + color swatch per cluster so YOU can label which
cluster is Ajax - this script does not know what Ajax's kit looks like in
this footage (FM26's rendering may not match real-world kit colors, and
kits vary by home/away/season).

Output: data/processed/team_labels_{match_id}.json with cluster color
info; you edit in the "team" field (e.g. "ajax", "opponent", "referee")
after looking at the sample crops saved alongside it.

Run:
    python -m src.cv_pipeline.classify_teams --match_id M001
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is not installed. Run: pip install opencv-python") from e

import numpy as np

from config.config import VIDEO_RAW_DIR, PROCESSED_DATA_DIR, FRAME_SAMPLE_RATE, KIT_COLOR_REFERENCES
from src.cv_pipeline.detect import load_model, detect_frame_full
from src.cv_pipeline.team_classify import (
    extract_shirt_color, cluster_teams, cluster_summary, suggest_team_labels, cluster_tightness,
    compute_calibration_stride,
)

SAMPLE_CROPS_DIR_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, "team_sample_crops_{match_id}")
LABELS_PATH_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, "team_labels_{match_id}.json")
CALIBRATION_DATA_PATH_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, "team_calibration_data_{match_id}.npz")

MAX_FRAMES_TO_SAMPLE = 150  # cap how many sampled frames we pull shirt colors from - this is a
                             # calibration step, not the full match, so it doesn't need every frame


def collect_shirt_colors(video_path, model, max_frames=MAX_FRAMES_TO_SAMPLE):
    """
    Samples shirt colors spread evenly across the ENTIRE video, not just
    the opening portion.

    WHY THIS MATTERS: an earlier version of this function read frames
    sequentially from the start and stopped once max_frames was hit. For
    a long match, that meant calibration only ever saw the first several
    percent of the video (confirmed on real project data: as little as
    ~5% for M002, ~23% even in the best case, M003). If lighting or scene
    composition differs later in the match - which is likely, given this
    footage is a montage of cuts from different camera angles - a cluster
    center learned only from the opening frames won't represent scenes
    calibration never saw, and full-match classification will be
    systematically biased. This was caught from a real, consistent effect:
    Ajax was undercounted 1.5-3.3x relative to not-Ajax across all 5
    matches when calibration only covered the start of each video.

    Frames are read sequentially (not seeked to, which is unreliable with
    some video codecs) but only kept for calibration every `stride` frames,
    where `stride` is computed from the video's total frame count so that
    max_frames samples end up spread across the full video regardless of
    its length.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = compute_calibration_stride(total_frames, max_frames, FRAME_SAMPLE_RATE)

    colors = []
    sample_crops = []  # (color, crop_image) kept for a few examples per rough color bucket
    frame_idx = 0
    frames_sampled = 0

    while frames_sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            player_dets, _ = detect_frame_full(model, frame)
            for det in player_dets:
                color = extract_shirt_color(frame, det["bbox"])
                if color is not None:
                    colors.append(color)
                    if len(sample_crops) < 300:
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        sample_crops.append((color, frame[max(y1, 0):y2, max(x1, 0):x2].copy()))
            frames_sampled += 1
        frame_idx += 1

    cap.release()
    return np.array(colors), sample_crops


def save_sample_crops(labels, sample_crops, match_id, n_per_cluster=3):
    out_dir = SAMPLE_CROPS_DIR_TEMPLATE.format(match_id=match_id)
    os.makedirs(out_dir, exist_ok=True)

    for cluster_id in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == cluster_id][:n_per_cluster]
        for j, i in enumerate(idxs):
            _, crop = sample_crops[i]
            if crop.size == 0:
                continue
            cv2.imwrite(os.path.join(out_dir, f"cluster{cluster_id}_sample{j}.png"), crop)

    return out_dir


def run(match_id, model, n_clusters=3, n_per_cluster=8):
    video_path = None
    for ext in (".mp4", ".mov", ".m4v"):
        candidate = os.path.join(VIDEO_RAW_DIR, match_id + ext)
        if os.path.exists(candidate):
            video_path = candidate
            break
    if video_path is None:
        raise FileNotFoundError(f"No video found for match_id={match_id} in {VIDEO_RAW_DIR}")

    print(f"Sampling shirt colors from {video_path} (up to {MAX_FRAMES_TO_SAMPLE} sampled frames)...")
    colors, sample_crops = collect_shirt_colors(video_path, model)
    print(f"Collected {len(colors)} shirt color samples.")

    if len(colors) < n_clusters:
        raise ValueError(
            f"Only {len(colors)} usable shirt-color samples collected - not enough to cluster "
            f"into {n_clusters} groups. Try a video with clearer player visibility, or reduce "
            f"n_clusters."
        )

    labels, centers = cluster_teams(colors, n_clusters=n_clusters)
    summary = cluster_summary(labels, centers)
    summary = suggest_team_labels(summary, KIT_COLOR_REFERENCES)
    tightness, flagged = cluster_tightness(colors, labels, centers)

    crops_dir = save_sample_crops(labels, sample_crops, match_id, n_per_cluster=n_per_cluster)

    print("Cluster summary (confirm against the saved sample crops before trusting a suggestion):")
    for s in summary:
        suggestion_note = (f" -> suggested: {s['suggested_team']} (distance {s['suggestion_distance']})"
                            if s["suggested_team"] else " -> no confident suggestion")
        spread_note = " *** HIGH SPREAD - check crops carefully, likely mixed ***" if s["cluster_id"] in flagged else ""
        print(f"  cluster {s['cluster_id']}: {s['n_samples']} samples, "
              f"representative color {s['color_hex']}, spread={tightness[s['cluster_id']]}"
              f"{suggestion_note}{spread_note}")
    print(f"Sample crops saved to: {crops_dir}")

    labels_path = LABELS_PATH_TEMPLATE.format(match_id=match_id)
    output = {
        "match_id": match_id,
        "n_clusters": n_clusters,
        "clusters": [{**s, "team": s["suggested_team"] or "UNLABELED - edit this field"} for s in summary],
    }
    with open(labels_path, "w") as f:
        json.dump(output, f, indent=2)

    # Raw calibration samples + their cluster assignments, saved so that
    # once labels are confirmed, centroids can be POOLED BY FINAL TEAM
    # LABEL rather than reused as the original raw per-cluster centroids -
    # see pool_centroids_by_team() in team_classify.py for why this matters.
    calibration_path = CALIBRATION_DATA_PATH_TEMPLATE.format(match_id=match_id)
    np.savez(calibration_path, colors=colors, cluster_labels=labels)

    print(f"Labels file (suggestions pre-filled where confident - verify before trusting) -> {labels_path}")
    print(f"Raw calibration data (for pooled-centroid classification) -> {calibration_path}\n")


def run_all(n_clusters=3, n_per_cluster=8):
    """
    Runs classify_teams for every match video found in VIDEO_RAW_DIR,
    loading the YOLO model once and reusing it across matches rather than
    reloading per match. Each match still gets its own labels file and
    sample crops - kit colors, lighting, and which side is "away" can
    differ per match, so nothing here assumes matches share a labeling.
    """
    video_files = sorted(
        f for f in os.listdir(VIDEO_RAW_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v"))
    )
    if not video_files:
        raise FileNotFoundError(f"No video files found in {VIDEO_RAW_DIR}")

    print("Loading YOLO model (once, reused across all matches)...")
    model = load_model()

    results = {}
    for fname in video_files:
        match_id = os.path.splitext(fname)[0]
        print(f"\n=== {match_id} ===")
        try:
            run(match_id, model, n_clusters, n_per_cluster)
            results[match_id] = "ok"
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results[match_id] = f"failed: {e}"

    print("Summary:")
    for match_id, status in results.items():
        print(f"  {match_id}: {status}")
    print("\nEach match's labels file needs its own manual review - kit colors, lighting, and "
          "home/away sides can differ per match. Don't assume M001's cluster mapping carries "
          "over to M002-M005.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id", help="Run for a single match, e.g. M001")
    parser.add_argument("--all", action="store_true", help="Run for every video in data/videos/raw/")
    parser.add_argument("--n_clusters", type=int, default=3,
                         help="Try 5 if teams are getting mixed - separates GK kits from outfield teams")
    parser.add_argument("--n_crops", type=int, default=8,
                         help="Sample crops saved per cluster, for manual review")
    args = parser.parse_args()

    if not args.match_id and not args.all:
        parser.error("Provide --match_id M001 (single match) or --all (every match video found)")

    if args.all:
        run_all(args.n_clusters, args.n_crops)
    else:
        print("Loading YOLO model...")
        model = load_model()
        run(args.match_id, model, args.n_clusters, args.n_crops)
