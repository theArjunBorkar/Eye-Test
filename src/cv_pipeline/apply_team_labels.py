"""
Loads a team_labels_{match_id}.json file (after you've confirmed/edited
the 'team' field per cluster) and provides a function to classify any
shirt color from the full match against those confirmed clusters.

Deliberately refuses to load a labels file that still has an unconfirmed
'suggested_team' auto-fill sitting in the 'team' field without you having
looked at it - see REQUIRE_EXPLICIT_CONFIRMATION below. A suggestion is a
starting point, not a substitute for checking the sample crops.

Run as a script to sanity-check a labels file:
    python -m src.cv_pipeline.apply_team_labels --match_id M001
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from config.config import PROCESSED_DATA_DIR
from src.cv_pipeline.team_classify import assign_to_nearest_cluster, pool_centroids_by_team

LABELS_PATH_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, "team_labels_{match_id}.json")
CALIBRATION_DATA_PATH_TEMPLATE = os.path.join(PROCESSED_DATA_DIR, "team_calibration_data_{match_id}.npz")

UNCONFIRMED_MARKER = "UNLABELED - edit this field"


def load_confirmed_labels(match_id, allow_unreviewed_suggestions=False):
    """
    Returns (cluster_centers_bgr: np.ndarray, cluster_id_to_team: dict).

    Raises ValueError if any cluster is still the raw "UNLABELED" marker -
    forces every cluster to be explicitly decided (even if the decision is
    just "yes, the suggestion is right") before this can be used.

    allow_unreviewed_suggestions=True skips that check - only use this if
    you've already visually confirmed every suggested_team yourself and
    don't want to bother re-typing it into 'team'. Off by default so a
    suggestion can't silently become "confirmed" just by existing.
    """
    path = LABELS_PATH_TEMPLATE.format(match_id=match_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No labels file at {path}. Run: python -m src.cv_pipeline.classify_teams "
            f"--match_id {match_id}"
        )

    with open(path) as f:
        data = json.load(f)

    unconfirmed = [c for c in data["clusters"] if c["team"] == UNCONFIRMED_MARKER]
    if unconfirmed and not allow_unreviewed_suggestions:
        ids = [c["cluster_id"] for c in unconfirmed]
        raise ValueError(
            f"Cluster(s) {ids} in {path} are still unlabeled. Open the sample crops in "
            f"data/processed/team_sample_crops_{match_id}/ and fill in the 'team' field "
            f"for every cluster before this can be used."
        )

    # BGR centers reconstructed from the stored hex (RGB) color
    centers_bgr = []
    cluster_id_to_team = {}
    for c in sorted(data["clusters"], key=lambda x: x["cluster_id"]):
        hex_color = c["color_hex"]
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        centers_bgr.append([b, g, r])
        team = c["team"] if c["team"] != UNCONFIRMED_MARKER else (c["suggested_team"] or "unknown")
        cluster_id_to_team[c["cluster_id"]] = team

    return np.array(centers_bgr), cluster_id_to_team


def classify_shirt_color(color_bgr, cluster_centers_bgr, cluster_id_to_team):
    cluster_id = assign_to_nearest_cluster(color_bgr, cluster_centers_bgr)
    return cluster_id_to_team[cluster_id]


def load_team_centroids(match_id, allow_unreviewed_suggestions=False):
    """
    Returns (team_names: list[str], centroids_bgr: np.ndarray) for direct
    nearest-centroid classification - PREFER THIS over load_confirmed_labels()
    for actually classifying detections.

    Uses pool_centroids_by_team() on the raw calibration samples (saved
    alongside the labels file by classify_teams.py) rather than the
    original raw per-cluster centroids. This fixes a real, measured bias:
    nearest-centroid classification against N raw clusters collapsed into
    fewer final teams structurally favors whichever team absorbed more raw
    clusters - confirmed on real project data as a persistent 1.6-2.5x
    undercount of Ajax even after a separate calibration-coverage bug was
    fixed. See pool_centroids_by_team()'s docstring for the full reasoning.

    Falls back to the old raw-cluster-centroid behavior (with a clear
    warning) if the companion calibration data file doesn't exist - e.g.
    for a labels file generated before this fix. That fallback carries the
    same bias this function exists to avoid, so re-running
    classify_teams.py for that match is the real fix, not a long-term
    substitute for it.
    """
    centers_bgr, cluster_id_to_team = load_confirmed_labels(match_id, allow_unreviewed_suggestions)

    calibration_path = CALIBRATION_DATA_PATH_TEMPLATE.format(match_id=match_id)
    if not os.path.exists(calibration_path):
        print(f"WARNING: no calibration data at {calibration_path} - falling back to raw "
              f"per-cluster centroids, which carry a known bias (see load_team_centroids() "
              f"docstring). Re-run classify_teams.py for {match_id} to fix this properly.")
        team_names = sorted(set(cluster_id_to_team.values()))
        # crude fallback: just use the raw cluster centroid closest to each team's own clusters'
        # mean position, since we don't have the raw per-sample data to pool properly
        fallback_centroids = []
        for team in team_names:
            cluster_ids = [cid for cid, t in cluster_id_to_team.items() if t == team]
            fallback_centroids.append(centers_bgr[cluster_ids].mean(axis=0))
        return team_names, np.array(fallback_centroids)

    data = np.load(calibration_path)
    return pool_centroids_by_team(data["colors"], data["cluster_labels"], cluster_id_to_team)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id", required=True)
    parser.add_argument("--allow_unreviewed_suggestions", action="store_true")
    args = parser.parse_args()

    centers, mapping = load_confirmed_labels(args.match_id, args.allow_unreviewed_suggestions)
    print(f"Loaded confirmed labels for {args.match_id}:")
    for cid, team in mapping.items():
        print(f"  cluster {cid} -> {team}")

    team_names, pooled_centroids = load_team_centroids(args.match_id, args.allow_unreviewed_suggestions)
    print(f"\nPooled team centroids (used for actual classification):")
    for name, centroid in zip(team_names, pooled_centroids):
        b, g, r = centroid
        print(f"  {name}: BGR=({b:.1f}, {g:.1f}, {r:.1f})")
