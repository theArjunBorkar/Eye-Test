"""
Team classification via shirt color clustering.

WHAT THIS DOES AND DOESN'T DO - read before using:
- Groups detected players into teams by shirt color (e.g. "these 14 crops
  are one team, these 11 are another, these 2 are probably the referee").
- Does NOT identify which specific named player anyone is. "Cluster 0" is
  not "Kasper Dolberg" - it's just "the group wearing this color." Mapping
  a cluster to "Ajax" requires a human to look at one sample crop per
  cluster and say which one it is (see `label_clusters_interactively` /
  the sample-crop output) - this module does not know what Ajax's kit
  looks like in this footage, since FM26's in-game rendering may not match
  real-world kit colors and kits vary between home/away/third and season.

METHOD:
1. For each detected player bounding box, crop a torso-region patch and
   extract its dominant color, after masking out likely pitch-green pixels
   (a simple heuristic, not true background subtraction).
2. K-means cluster all shirt colors across the match into `n_clusters`
   groups (default 3: two teams + referee/officials).

Pure numpy/opencv-free color math for the clustering step (uses sklearn's
KMeans, already a project dependency) - the only new dependency is reading
pixel data from crops, which reuses opencv already required by the CV
pipeline.
"""

import colorsys

import numpy as np
from sklearn.cluster import KMeans


def _bgr_to_hsv_degrees(bgr):
    """BGR (0-255) -> (hue_degrees 0-360, saturation_pct 0-100, value_pct 0-100)."""
    b, g, r = bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s * 100, v * 100


def perceptual_color_distance(bgr1, bgr2):
    """
    Perceptually-motivated distance between two BGR colors, using hue as
    the primary signal rather than raw RGB Euclidean distance.

    WHY: raw RGB Euclidean distance produced a confident-looking but
    visually wrong match in practice (a muted gray-green cluster in real
    M005 data matched "blue" at distance 47.5 out of a 0-441 possible
    range - close enough to pass a naive threshold despite not looking
    blue at all). Hue captures "is this actually the same color family"
    far better: true blue-vs-blue comparisons in real data landed at
    6-23 degrees apart, while the wrong green-vs-blue match was 75
    degrees apart and an unrelated olive color was 129 degrees apart -
    a clean separation raw RGB distance did not give us.

    An earlier version of this function also hard-penalized any
    comparison where either color had low saturation (near-gray), meant
    to catch unreliable/noisy hue readings. That penalty turned out to be
    unnecessary for every real case observed (hue alone already separates
    them by 2x+ the threshold - see the project README) AND actively
    broke a legitimate case: Ajax's home kit (white with a red band) is
    itself inherently low-saturation, so the guard made that reference
    fail to match even itself. Removed - hue-led distance alone has
    proven sufficient on real project data so far.
    """
    h1, s1, v1 = _bgr_to_hsv_degrees(bgr1)
    h2, s2, v2 = _bgr_to_hsv_degrees(bgr2)

    hue_diff = abs(h1 - h2)
    hue_diff = min(hue_diff, 360 - hue_diff)  # circular

    sat_diff = abs(s1 - s2)
    val_diff = abs(v1 - v2)
    return hue_diff + 0.3 * sat_diff + 0.1 * val_diff


def extract_shirt_color(frame, bbox, grass_hue_range=(35, 85)):
    """
    frame: full video frame, BGR (as read by cv2), shape (H, W, 3).
    bbox: (x1, y1, x2, y2) in pixel coordinates.
    Returns the median BGR color of the torso region of the crop, with
    likely pitch-green pixels masked out first.

    Torso region = middle third vertically (avoids head/hair and
    legs/socks, which are less reliably the team's primary shirt color),
    inner 60% horizontally (avoids background bleeding in at the edges of
    a loose bounding box).
    """
    import cv2

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None

    ty1, ty2 = y1 + h // 3, y1 + 2 * h // 3
    tx1, tx2 = x1 + int(w * 0.2), x1 + int(w * 0.8)
    crop = frame[max(ty1, 0):ty2, max(tx1, 0):tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    # OpenCV hue range is 0-179; scale the 0-360 grass range down accordingly
    lo, hi = grass_hue_range[0] / 2, grass_hue_range[1] / 2
    is_grass = (hue >= lo) & (hue <= hi)

    pixels = crop.reshape(-1, 3)
    mask = ~is_grass.reshape(-1)
    non_grass_pixels = pixels[mask]

    if len(non_grass_pixels) < 5:  # almost entirely grass-colored, unreliable crop
        return None

    return np.median(non_grass_pixels, axis=0)  # BGR


def cluster_teams(shirt_colors, n_clusters=3, random_state=42):
    """
    shirt_colors: (n_samples, 3) array of BGR colors (e.g. from
        extract_shirt_color, with None entries already filtered out).
    Returns (labels, cluster_centers_bgr).
    """
    shirt_colors = np.asarray(shirt_colors)
    if len(shirt_colors) < n_clusters:
        raise ValueError(
            f"Need at least n_clusters={n_clusters} color samples, got {len(shirt_colors)}"
        )

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(shirt_colors)
    return labels, km.cluster_centers_


def cluster_summary(labels, cluster_centers_bgr):
    """
    Human-readable summary for labeling clusters by hand: how many samples
    per cluster and the cluster's representative color (as RGB hex, more
    familiar to read than BGR floats). Whichever cluster's color matches
    Ajax's kit in your footage is the one to treat as "Ajax" downstream -
    this function doesn't decide that for you.
    """
    summary = []
    for cluster_id in sorted(set(labels)):
        count = int(np.sum(labels == cluster_id))
        b, g, r = cluster_centers_bgr[cluster_id]
        hex_color = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
        summary.append({"cluster_id": int(cluster_id), "n_samples": count, "color_hex": hex_color})
    return sorted(summary, key=lambda s: -s["n_samples"])


def cluster_tightness(shirt_colors, labels, cluster_centers_bgr):
    """
    Mean distance of each cluster's own members to its center - a rough,
    no-manual-review proxy for how likely a cluster is contaminated
    (mixing two different kit colors). A cluster with much higher spread
    than the others is a signal to check its sample crops carefully before
    trusting its label, even before opening any images.

    Not a substitute for actually looking at crops - two different kits
    that happen to be similarly-colored (or the same kit shown under very
    different lighting from a scene cut) can still produce a deceptively
    tight-looking but wrongly-mixed cluster. This is a triage signal, not
    a guarantee.
    """
    shirt_colors = np.asarray(shirt_colors)
    tightness = {}
    all_dists = []
    for cluster_id in sorted(set(labels)):
        members = shirt_colors[labels == cluster_id]
        center = cluster_centers_bgr[cluster_id]
        dists = np.linalg.norm(members - center, axis=1)
        tightness[int(cluster_id)] = round(float(dists.mean()), 1)
        all_dists.extend(dists.tolist())

    overall_mean = float(np.mean(all_dists)) if all_dists else 0.0
    flagged = {cid: d for cid, d in tightness.items() if d > 1.3 * overall_mean}
    return tightness, flagged


def suggest_team_labels(summary, reference_colors, max_distance=40):
    """
    summary: output of cluster_summary().
    reference_colors: dict {name: (r, g, b)} of known kit colors.
    max_distance: max perceptual (hue-led) distance to accept a suggestion.
        Calibrated against real project data: confirmed-correct blue
        matches landed at 11-31 on this scale, while a confirmed-wrong
        match landed at 175 and an unrelated color at 129+ - 40 sits
        clearly between those groups. See perceptual_color_distance() for
        why this replaced raw RGB Euclidean distance.

    Adds "suggested_team" and "suggestion_distance" fields to each entry.
    This is a hint to speed up manual labeling, not a substitute for it -
    always confirm against the saved sample crops before trusting it.
    """
    annotated = []
    for cluster in summary:
        hex_color = cluster["color_hex"]
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        cluster_bgr = (b, g, r)

        best_name, best_dist = None, float("inf")
        for name, ref_rgb in reference_colors.items():
            ref_r, ref_g, ref_b = ref_rgb
            dist = perceptual_color_distance(cluster_bgr, (ref_b, ref_g, ref_r))
            if dist < best_dist:
                best_name, best_dist = name, dist

        suggestion = best_name if best_dist <= max_distance else None
        annotated.append({**cluster, "suggested_team": suggestion,
                           "suggestion_distance": round(best_dist, 1) if suggestion else None})
    return annotated


def compute_calibration_stride(total_frames, max_frames, min_stride):
    """
    Frame stride so that ~max_frames calibration samples spread evenly
    across a video of `total_frames` length, regardless of how long it
    is. Never finer than min_stride (no point calibrating faster than the
    main pipeline itself samples). Falls back to min_stride if
    total_frames is unknown/zero (some video codecs don't report it
    reliably).

    WHY THIS MATTERS: an earlier version of the calibration step read
    frames sequentially from the start of the video and stopped once it
    had enough samples - for a long match, that meant calibration only
    ever saw the opening several percent of the footage (confirmed on
    real project data: as little as ~5% for one match). If lighting or
    scene composition differs later in the match - likely, given this
    footage is a montage of cuts from different camera angles - a cluster
    center learned only from the opening frames won't represent scenes
    calibration never saw. This was caught from a real, consistent
    effect: Ajax was undercounted 1.5-3.3x relative to not-Ajax across
    all 5 matches when calibration only covered the start of each video.
    """
    if total_frames <= 0:
        return min_stride
    return max(min_stride, total_frames // max_frames)


def pool_centroids_by_team(colors, cluster_labels, cluster_id_to_team):
    """
    Recomputes centroids grouped by FINAL team label, not by the original
    raw K-means cluster ID. This fixes a real, measured bias: with 3 raw
    clusters (1 Ajax + 2 non-Ajax merged into "not_ajax"), nearest-centroid
    classification against all 3 raw centroids structurally favors
    not_ajax - an ambiguous color has two centroids pulling for "not_ajax"
    and only one for "ajax", regardless of the true population balance on
    the pitch. Confirmed on real project data: even after fixing a
    separate calibration-coverage bug, Ajax was still undercounted
    1.6-2.5x relative to not-Ajax across all 5 matches, consistently.

    Pooling every calibration sample by its final team label (not its
    original cluster) into exactly N_teams centroids removes this
    structural imbalance - it's now a clean N-way decision, not a
    K-raw-clusters-collapsed-into-N-teams one.

    colors: (n_samples, 3) BGR array from calibration.
    cluster_labels: (n_samples,) raw cluster ID per sample, same order.
    cluster_id_to_team: {cluster_id: team_name} from the confirmed labels file.

    Returns (team_names: list[str], centroids_bgr: np.ndarray of shape
    (n_teams, 3)), ready for nearest-centroid classification.
    """
    colors = np.asarray(colors)
    cluster_labels = np.asarray(cluster_labels)

    team_names = sorted(set(cluster_id_to_team.values()))
    centroids = []
    for team in team_names:
        cluster_ids_for_team = [cid for cid, t in cluster_id_to_team.items() if t == team]
        mask = np.isin(cluster_labels, cluster_ids_for_team)
        if not mask.any():
            raise ValueError(f"No calibration samples found for team '{team}' - cannot pool a centroid.")
        centroids.append(colors[mask].mean(axis=0))

    return team_names, np.array(centroids)


def assign_to_nearest_cluster(color_bgr, cluster_centers_bgr):
    """
    Classifies a single new shirt color (e.g. from a detection in the full
    match, outside the ~150-frame calibration sample) by nearest cluster
    center. This is what lets team labels learned from calibration apply
    to every detection in the full pipeline run, not just the sample.
    """
    color_bgr = np.asarray(color_bgr)
    dists = np.linalg.norm(cluster_centers_bgr - color_bgr, axis=1)
    return int(np.argmin(dists))
