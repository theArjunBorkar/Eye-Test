"""
Tests for the hand-implemented KMO, Bartlett's test, varimax rotation, and
parallel analysis building blocks used by Phase 9.

Run with:
    python -m tests.test_factor_analysis
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.factor_analysis.stats_tests import kmo_test, bartlett_sphericity
from src.factor_analysis.rotation import varimax
from src.factor_analysis.parallel_analysis import parallel_analysis, suggest_n_factors
import src.factor_analysis.run_efa as efa_module


def _correlated_data(n=300, seed=0):
    """6 variables with real 2-factor structure + noise - should give good KMO."""
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    X = np.column_stack([
        0.8 * f1 + 0.2 * rng.standard_normal(n),
        0.75 * f1 + 0.25 * rng.standard_normal(n),
        0.85 * f1 + 0.15 * rng.standard_normal(n),
        0.8 * f2 + 0.2 * rng.standard_normal(n),
        0.75 * f2 + 0.25 * rng.standard_normal(n),
        0.85 * f2 + 0.15 * rng.standard_normal(n),
    ])
    return X


def _uncorrelated_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 6))


def test_kmo_high_for_genuinely_correlated_data():
    X = _correlated_data()
    _, kmo_overall = kmo_test(X)
    assert kmo_overall > 0.6, f"expected adequate KMO for correlated data, got {kmo_overall}"


def test_kmo_low_for_random_uncorrelated_data():
    X = _uncorrelated_data()
    _, kmo_overall = kmo_test(X)
    assert kmo_overall < 0.6, f"expected inadequate KMO for random data, got {kmo_overall}"


def test_bartlett_significant_for_correlated_data():
    X = _correlated_data()
    chi2, dof, p = bartlett_sphericity(X)
    assert p < 0.05, f"expected significant Bartlett's test for correlated data, got p={p}"


def test_bartlett_not_significant_for_uncorrelated_data():
    X = _uncorrelated_data()
    chi2, dof, p = bartlett_sphericity(X)
    assert p > 0.05, f"expected non-significant Bartlett's test for random data, got p={p}"


def test_varimax_recovers_simple_structure():
    true_loadings = np.array([
        [0.8, 0.1], [0.75, 0.15], [0.85, 0.05],
        [0.1, 0.8], [0.15, 0.75], [0.05, 0.85],
    ])
    theta = np.radians(30)
    mix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mixed = true_loadings @ mix

    rotated, R = varimax(mixed)
    # rotation should be orthogonal
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-6)
    # rotated loadings should be closer to the clean block structure than the mixed input
    off_block_mixed = np.abs(mixed[:3, 1]).sum() + np.abs(mixed[3:, 0]).sum()
    off_block_rotated = np.abs(rotated[:3, 1]).sum() + np.abs(rotated[3:, 0]).sum()
    assert off_block_rotated < off_block_mixed


def test_parallel_analysis_thresholds_decrease():
    thresholds = parallel_analysis(n_obs=200, n_vars=8, n_iter=50, random_state=1)
    assert len(thresholds) == 8
    assert all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1))


def test_suggest_n_factors_stops_at_first_below_threshold():
    real = np.array([3.0, 2.0, 1.5, 0.9, 0.5])
    threshold = np.array([1.2, 1.1, 1.6, 1.0, 0.9])
    # factor 3 (1.5) is below its threshold (1.6) -> should stop there, not count factor 4 or 5
    # even though factor 4's real (0.9) happens to be below its own threshold too
    n = suggest_n_factors(real, threshold)
    assert n == 2


def test_load_efa_data_excludes_gk_and_small_sample():
    import tempfile
    import pandas as pd
    import src.factor_analysis.run_efa as efa

    with tempfile.TemporaryDirectory() as d:
        fake_path = os.path.join(d, "merged.csv")
        rows = []
        for i in range(20):
            row = {v: float(rng_val(i, v)) for v in efa.EFA_VARIABLES}
            row["position"] = "GK" if i < 3 else "MF"
            row["small_sample"] = i < 6  # first 6 rows are small_sample (overlaps GK rows 0-2)
            rows.append(row)
        pd.DataFrame(rows).to_csv(fake_path, index=False)

        original_path = efa.INPUT_PATH
        try:
            efa.INPUT_PATH = fake_path
            df_vars, report = efa.load_efa_data()
        finally:
            efa.INPUT_PATH = original_path

        assert report["n_total_rows"] == 20
        assert report["n_after_excluding_gk"] == 17    # 20 - 3 GK rows
        assert report["n_after_excluding_small_sample"] == 14  # 17 - (6-3 remaining small_sample rows)
        assert len(df_vars) == report["n_final"]


def rng_val(i, varname):
    # deterministic pseudo-random-ish value, just needs to be numeric and vary
    return (hash((i, varname)) % 1000) / 100.0


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
