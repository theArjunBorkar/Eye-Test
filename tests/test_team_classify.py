"""
Tests for team classification (shirt color extraction + clustering).
Fully synthetic - no real video needed, runs in this sandbox.

Run with:
    python -m tests.test_team_classify
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.cv_pipeline.team_classify import (
    extract_shirt_color, cluster_teams, cluster_summary, suggest_team_labels, cluster_tightness,
    perceptual_color_distance, pool_centroids_by_team, assign_to_nearest_cluster,
)


def _solid_frame(color_bgr, size=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


def test_extract_shirt_color_returns_dominant_color():
    red_bgr = (30, 30, 200)  # BGR: strong red
    frame = _solid_frame(red_bgr)
    bbox = (50, 50, 150, 150)
    color = extract_shirt_color(frame, bbox)
    assert color is not None
    assert abs(color[2] - 200) < 10  # red channel dominant


def test_extract_shirt_color_masks_grass_green():
    size = 200
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 200)          # red shirt background
    frame[:60, :] = (40, 160, 40)         # top strip: grass-like green (BGR)
    bbox = (0, 0, size, size)
    color = extract_shirt_color(frame, bbox)
    assert color is not None
    # median should reflect the red majority, not be dragged toward green
    assert color[2] > color[1]  # red channel > green channel


def test_extract_shirt_color_handles_degenerate_bbox():
    frame = _solid_frame((30, 30, 200))
    assert extract_shirt_color(frame, (10, 10, 10, 10)) is None  # zero area


def test_cluster_teams_separates_distinct_colors():
    rng = np.random.default_rng(0)
    red_team = rng.normal([20, 20, 200], 5, size=(15, 3))     # BGR reds
    blue_team = rng.normal([200, 20, 20], 5, size=(15, 3))    # BGR blues
    ref_black = rng.normal([10, 10, 10], 3, size=(4, 3))      # near-black

    all_colors = np.vstack([red_team, blue_team, ref_black])
    labels, centers = cluster_teams(all_colors, n_clusters=3)

    # everyone within the same synthetic group should share a cluster label
    red_labels = set(labels[:15])
    blue_labels = set(labels[15:30])
    ref_labels = set(labels[30:])
    assert len(red_labels) == 1
    assert len(blue_labels) == 1
    assert len(ref_labels) == 1
    assert red_labels != blue_labels != ref_labels


def test_cluster_teams_raises_if_too_few_samples():
    try:
        cluster_teams(np.array([[1, 2, 3], [4, 5, 6]]), n_clusters=3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cluster_summary_sorted_by_size_with_hex_colors():
    labels = np.array([0, 0, 0, 1, 1, 2])
    centers = np.array([[0, 0, 200], [200, 0, 0], [10, 10, 10]])  # BGR
    summary = cluster_summary(labels, centers)
    assert summary[0]["n_samples"] == 3  # cluster 0 is largest
    assert summary[0]["color_hex"].startswith("#")
    assert summary[0]["color_hex"] == "#C80000"  # BGR(0,0,200) -> RGB hex


def test_suggest_team_labels_matches_close_reference():
    summary = [
        {"cluster_id": 0, "n_samples": 10, "color_hex": "#335678"},  # near the Ajax away blue ref
        {"cluster_id": 1, "n_samples": 8, "color_hex": "#FFFFFF"},   # far from any reference
    ]
    refs = {"ajax_away_blue": (51, 86, 120)}
    annotated = suggest_team_labels(summary, refs, max_distance=20)

    assert annotated[0]["suggested_team"] == "ajax_away_blue"
    assert annotated[1]["suggested_team"] is None  # too far, no forced guess


def _hex_to_bgr(hexc):
    hexc = hexc.lstrip("#")
    r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
    return (b, g, r)


def test_perceptual_distance_regression_against_real_project_data():
    """
    Locked-in regression test using real cluster colors observed against
    the actual project videos (M001-M005), where visual ground truth is
    already known: M001/M003/M004's suggested Ajax clusters were visually
    confirmed correct; M005's was visually confirmed WRONG (a muted
    green, not blue) despite passing the old raw-RGB-distance threshold;
    M002's had no real Ajax-blue cluster at all. If this test ever fails,
    the distance function has regressed on a case we already know the
    right answer to.
    """
    ref_bgr = _hex_to_bgr("#335678")  # ajax_away_blue reference

    should_match = ["#334C4F", "#4159A6", "#263C4A"]     # M001, M004, M003 - real, confirmed/plausible
    should_not_match = ["#526256", "#6D7D4E"]             # M005 (visually wrong), M002 (no real match)

    for hexc in should_match:
        d = perceptual_color_distance(_hex_to_bgr(hexc), ref_bgr)
        assert d <= 40, f"{hexc} should match (expected <=40, got {d})"

    for hexc in should_not_match:
        d = perceptual_color_distance(_hex_to_bgr(hexc), ref_bgr)
        assert d > 40, f"{hexc} should NOT match (expected >40, got {d})"


def test_perceptual_distance_rejects_unrelated_gray_via_hue_alone():
    # true gray has an arbitrary/unstable hue, but it still happens to land
    # far from blue's hue in practice - no special-case penalty needed,
    # see team_classify.py's docstring for why that guard was removed
    gray_bgr = _hex_to_bgr("#808080")
    blue_bgr = _hex_to_bgr("#335678")
    d = perceptual_color_distance(gray_bgr, blue_bgr)
    assert d > 40


def test_perceptual_distance_matches_low_saturation_reference_to_itself():
    # regression test for a real bug: Ajax's home kit (white + red band)
    # is itself inherently low-saturation, so a naive "low saturation ->
    # penalize" rule broke matching it even to itself. This must be ~0.
    home_kit_bgr = _hex_to_bgr("#CAB7C0")
    d = perceptual_color_distance(home_kit_bgr, home_kit_bgr)
    assert d < 1


def test_cluster_tightness_flags_mixed_cluster():
    rng = np.random.default_rng(0)
    tight_cluster = rng.normal([20, 20, 200], 3, size=(20, 3))       # tight, single real kit
    mixed_cluster = np.vstack([
        rng.normal([20, 20, 200], 3, size=(10, 3)),                  # half one kit color
        rng.normal([200, 20, 20], 3, size=(10, 3)),                  # half a totally different kit color
    ])
    all_colors = np.vstack([tight_cluster, mixed_cluster])
    labels = np.array([0] * 20 + [1] * 20)
    centers = np.array([tight_cluster.mean(axis=0), mixed_cluster.mean(axis=0)])

    tightness, flagged = cluster_tightness(all_colors, labels, centers)
    assert tightness[1] > tightness[0]  # mixed cluster has higher spread
    assert 1 in flagged
    assert 0 not in flagged


def test_pool_centroids_by_team_basic():
    colors = np.array([[100, 0, 0], [110, 0, 0], [0, 100, 0], [0, 110, 0], [0, 0, 100]])
    cluster_labels = np.array([0, 0, 1, 1, 2])
    cluster_id_to_team = {0: "ajax", 1: "not_ajax", 2: "not_ajax"}

    team_names, centroids = pool_centroids_by_team(colors, cluster_labels, cluster_id_to_team)

    assert set(team_names) == {"ajax", "not_ajax"}
    ajax_idx = team_names.index("ajax")
    not_ajax_idx = team_names.index("not_ajax")
    assert np.allclose(centroids[ajax_idx], [105, 0, 0])
    # not_ajax pools clusters 1 AND 2 together: mean of (0,100,0),(0,110,0),(0,0,100)
    assert np.allclose(centroids[not_ajax_idx], [0, 70, 100 / 3])


def test_pool_centroids_by_team_raises_if_team_has_no_samples():
    colors = np.array([[100, 0, 0], [0, 100, 0]])
    cluster_labels = np.array([0, 1])
    cluster_id_to_team = {0: "ajax", 1: "not_ajax", 2: "referee"}  # cluster 2 never appears in labels
    try:
        pool_centroids_by_team(colors, cluster_labels, cluster_id_to_team)
        assert False, "expected ValueError for a team with no calibration samples"
    except ValueError:
        pass


def test_pooling_corrects_geometric_bias_when_extra_cluster_near_boundary():
    """
    Regression test reproducing the real bias found in this project:
    nearest-centroid classification against raw per-cluster centroids
    (1 ajax cluster vs 2 not-ajax clusters, e.g. opponent + referee) is
    structurally biased toward not-ajax when the extra cluster sits near
    the real decision boundary - confirmed with a severe 6.4:1 skew in
    testing, corrected to ~1.3:1 by pooling. This locks that in.
    """
    rng = np.random.default_rng(1)
    ajax_samples = rng.normal([120, 80, 50], 10, size=(100, 3))
    opp_samples = rng.normal([50, 120, 50], 10, size=(100, 3))
    ref_samples = rng.normal([90, 95, 50], 10, size=(20, 3))  # near the ajax/opp boundary

    colors = np.vstack([ajax_samples, opp_samples, ref_samples])
    cluster_labels = np.array([0] * 100 + [1] * 100 + [2] * 20)
    cluster_id_to_team = {0: "ajax", 1: "not_ajax", 2: "not_ajax"}
    raw_centers = np.array([ajax_samples.mean(0), opp_samples.mean(0), ref_samples.mean(0)])

    test_points = rng.normal([85, 100, 50], 20, size=(1000, 3))

    old_ajax = sum(1 for p in test_points
                   if cluster_id_to_team[assign_to_nearest_cluster(p, raw_centers)] == "ajax")

    team_names, pooled = pool_centroids_by_team(colors, cluster_labels, cluster_id_to_team)
    new_ajax = sum(1 for p in test_points
                   if team_names[assign_to_nearest_cluster(p, pooled)] == "ajax")

    # pooling should substantially increase how often ajax is correctly
    # chosen for these boundary-region points, not just marginally
    assert new_ajax > old_ajax * 1.5


def test_assign_to_nearest_cluster_picks_closest():
    centers = np.array([[200, 20, 20], [20, 20, 200], [10, 10, 10]])  # BGR: blue, red, black
    assert assign_to_nearest_cluster([190, 25, 15], centers) == 0   # close to blue center
    assert assign_to_nearest_cluster([15, 20, 195], centers) == 1   # close to red center
    assert assign_to_nearest_cluster([5, 5, 5], centers) == 2       # close to black center


def test_load_confirmed_labels_rejects_unconfirmed():
    import tempfile
    import json as json_module
    import src.cv_pipeline.apply_team_labels as atl

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "team_labels_MTEST.json")
        with open(path, "w") as f:
            json_module.dump({
                "match_id": "MTEST", "n_clusters": 2,
                "clusters": [
                    {"cluster_id": 0, "n_samples": 5, "color_hex": "#334C4F",
                     "suggested_team": "ajax_away_blue", "suggestion_distance": 10.0,
                     "team": "UNLABELED - edit this field"},
                ],
            }, f)

        original_template = atl.LABELS_PATH_TEMPLATE
        try:
            atl.LABELS_PATH_TEMPLATE = os.path.join(d, "team_labels_{match_id}.json")
            try:
                atl.load_confirmed_labels("MTEST")
                assert False, "expected ValueError for unconfirmed cluster"
            except ValueError:
                pass
        finally:
            atl.LABELS_PATH_TEMPLATE = original_template


def test_compute_calibration_stride_spans_full_video():
    from src.cv_pipeline.team_classify import compute_calibration_stride
    # regression test for the real bug: calibration used to only cover the
    # first ~5-23% of a match. Using M002's real numbers (total ~14665
    # frames), stride must be large enough that 150 samples spans ~100%.
    total_frames = 14665
    max_frames = 150
    stride = compute_calibration_stride(total_frames, max_frames, min_stride=5)
    coverage = min(max_frames * stride, total_frames) / total_frames
    assert coverage > 0.9, f"expected calibration to cover >90% of the video, got {coverage:.1%}"


def test_compute_calibration_stride_never_below_min_stride():
    from src.cv_pipeline.team_classify import compute_calibration_stride
    # a very short video shouldn't force a stride finer than min_stride
    stride = compute_calibration_stride(total_frames=50, max_frames=150, min_stride=5)
    assert stride == 5


def test_compute_calibration_stride_falls_back_when_frame_count_unknown():
    from src.cv_pipeline.team_classify import compute_calibration_stride
    stride = compute_calibration_stride(total_frames=0, max_frames=150, min_stride=5)
    assert stride == 5


def test_load_confirmed_labels_succeeds_when_confirmed():
    import tempfile
    import json as json_module
    import src.cv_pipeline.apply_team_labels as atl

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "team_labels_MTEST.json")
        with open(path, "w") as f:
            json_module.dump({
                "match_id": "MTEST", "n_clusters": 2,
                "clusters": [
                    {"cluster_id": 0, "n_samples": 5, "color_hex": "#334C4F",
                     "suggested_team": "ajax_away_blue", "suggestion_distance": 10.0,
                     "team": "ajax"},
                    {"cluster_id": 1, "n_samples": 5, "color_hex": "#682B26",
                     "suggested_team": None, "suggestion_distance": None,
                     "team": "not_ajax"},
                ],
            }, f)

        original_template = atl.LABELS_PATH_TEMPLATE
        try:
            atl.LABELS_PATH_TEMPLATE = os.path.join(d, "team_labels_{match_id}.json")
            centers, mapping = atl.load_confirmed_labels("MTEST")
            assert mapping[0] == "ajax"
            assert mapping[1] == "not_ajax"
            assert centers.shape == (2, 3)
        finally:
            atl.LABELS_PATH_TEMPLATE = original_template


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
