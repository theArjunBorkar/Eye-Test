"""
Noise-floor check for leave-one-match-out (LOMO) results.

With only 5 matches, LOMO has 5 folds. A real effect needs to be bigger
than what two things alone can produce:
  1. RandomForestRegressor's own seed-to-seed variance on IDENTICAL features.
  2. Adding a genuinely meaningless per-match random number as a "feature"
     (a null model for "any per-match column, regardless of content").

Real finding on this project's data: XGBoost's LOMO MAE improved by 0.010
when video features were added. Control 1 (10 seeds, same features) spans
a 0.011 range with zero feature change. Control 2 (30 random per-match
numbers) has std 0.008, range -0.009 to +0.023. The observed 0.010 sits
inside both - not distinguishable from noise. Random Forest's LOMO change
(+0.003, i.e. slightly worse) is even more clearly inside the noise band.

A second finding from the same check: per-match MAE is highly uneven
(0.21-0.85 across the 5 real matches), driven by M002 having a visibly
different rating distribution (mean 7.88, std 1.05, max 10.0) vs. the
other four (means 6.6-7.09, std <=0.53). With 5 folds, the LOMO average is
largely a readout of how well the model handles this one match, not a
stable estimate - a caveat on LOMO itself, not just on video features.

Run:
    python -m src.modeling.noise_floor_check
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from src.modeling.train_baseline import load_features, get_feature_columns, _build_X


def per_match_mae(df, y, groups, cols, extra_column=None, seed=42):
    """extra_column: optional {"name": [...values, one per row...]} to append."""
    X, _ = _build_X(df, cols)
    if extra_column:
        X = X.copy()
        for name, vals in extra_column.items():
            X[name] = vals
    out = {}
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=seed)
        model.fit(X.iloc[train_idx], y[train_idx])
        out[groups[test_idx][0]] = mean_absolute_error(y[test_idx], model.predict(X.iloc[test_idx]))
    return out


def run(n_seed_controls=10, n_random_feature_controls=30, random_state=0):
    df = load_features()
    y = df["rating"].values
    groups = df["match_id"].values
    stats_cols = get_feature_columns(df, include_video_features=False)
    video_cols = get_feature_columns(df, include_video_features=True)

    per_match_stats = per_match_mae(df, y, groups, stats_cols)
    per_match_video = per_match_mae(df, y, groups, video_cols)
    observed_delta = np.mean(list(per_match_video.values())) - np.mean(list(per_match_stats.values()))

    seed_means = [np.mean(list(per_match_mae(df, y, groups, stats_cols, seed=s).values()))
                  for s in range(n_seed_controls)]
    seed_range = (min(seed_means) - np.mean(seed_means), max(seed_means) - np.mean(seed_means))

    rng = np.random.default_rng(random_state)
    matches = sorted(set(groups))
    random_deltas = []
    base = np.mean(list(per_match_stats.values()))
    for _ in range(n_random_feature_controls):
        match_vals = dict(zip(matches, rng.normal(size=len(matches))))
        col = [match_vals[g] for g in groups]
        m = np.mean(list(per_match_mae(df, y, groups, stats_cols, {"random_match_feature": col}).values()))
        random_deltas.append(m - base)

    return {
        "per_match_mae_stats_only": per_match_stats,
        "per_match_mae_stats_plus_video": per_match_video,
        "observed_mae_delta": round(float(observed_delta), 4),
        "seed_noise_range": (round(float(seed_range[0]), 4), round(float(seed_range[1]), 4)),
        "random_feature_delta_mean": round(float(np.mean(random_deltas)), 4),
        "random_feature_delta_std": round(float(np.std(random_deltas)), 4),
        "random_feature_delta_range": (round(float(min(random_deltas)), 4), round(float(max(random_deltas)), 4)),
    }


def main():
    r = run()
    print("Per-match MAE (stats_only -> stats_plus_video), Random Forest, leave-one-match-out:")
    for m in sorted(r["per_match_mae_stats_only"]):
        a, b = r["per_match_mae_stats_only"][m], r["per_match_mae_stats_plus_video"][m]
        print(f"  {m}: {a:.3f} -> {b:.3f}  ({b - a:+.3f})")

    print(f"\nObserved MAE change from adding video features: {r['observed_mae_delta']:+.4f}")
    print(f"Control 1 - seed noise alone (same features): range {r['seed_noise_range']}")
    print(f"Control 2 - meaningless random per-match feature: mean {r['random_feature_delta_mean']:+.4f}, "
          f"std {r['random_feature_delta_std']}, range {r['random_feature_delta_range']}")

    lo, hi = r["random_feature_delta_range"]
    if lo <= r["observed_mae_delta"] <= hi:
        print("\nVerdict: the observed change sits inside the range a MEANINGLESS per-match number "
              "can produce by chance with only 5 groups. Not distinguishable from noise.")
    else:
        print("\nVerdict: the observed change falls outside both noise controls - worth a closer look, "
              "though still only 5 matches.")


if __name__ == "__main__":
    main()
