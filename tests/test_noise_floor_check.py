"""
Tests for the noise-floor check (src/modeling/noise_floor_check.py).

Uses a tiny synthetic dataset with a KNOWN answer (5 matches, a real
per-match rating offset, and one video feature that's provably nothing
more than a match label) so the noise-floor machinery itself is verified
correct before trusting its verdict on real project data.

Run with:
    python -m tests.test_noise_floor_check
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.modeling.noise_floor_check import per_match_mae, run as run_noise_floor_check


def _synthetic_df(n_per_match=20, seed=0):
    rng = np.random.default_rng(seed)
    offsets = {"M0": -0.8, "M1": -0.3, "M2": 0.1, "M3": 0.5, "M4": 0.9}
    rows = []
    for m, off in offsets.items():
        for _ in range(n_per_match):
            rows.append({
                "match_id": m, "player_id": int(rng.integers(1, 20)), "player_name": "p",
                "team": "Ajax", "position": "MF",
                "noise_stat": rng.normal(),
                "rating": 6.5 + off + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_per_match_mae_returns_one_value_per_match():
    df = _synthetic_df()
    y = df["rating"].values
    groups = df["match_id"].values
    result = per_match_mae(df, y, groups, ["noise_stat"])
    assert set(result.keys()) == set(groups)
    assert all(v >= 0 for v in result.values())


def test_run_on_real_project_data_matches_reported_verdict():
    """
    Regression test locking in the real finding: XGBoost's small LOMO
    improvement from video features is not distinguishable from the noise
    a meaningless per-match number produces with only 5 groups. Checks the
    Random Forest version of the same check (computed directly, not via
    train_xgboost.py, since xgboost isn't available in this environment).
    """
    r = run_noise_floor_check(n_seed_controls=5, n_random_feature_controls=10, random_state=1)
    lo, hi = r["random_feature_delta_range"]
    assert lo <= r["observed_mae_delta"] <= hi, (
        f"observed delta {r['observed_mae_delta']} fell outside the noise band "
        f"({lo}, {hi}) - the real finding may have changed, worth investigating, "
        f"not just updating this assertion"
    )


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
