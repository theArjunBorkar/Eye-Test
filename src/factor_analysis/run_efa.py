"""
Phase 9 - Exploratory Factor Analysis.

Identifies latent performance dimensions from correlated match-stat
variables, WITHOUT assuming the factor structure in advance - factors are
determined by parallel analysis on the real data, not picked to match a
preconceived "attacking/defensive/etc." story.

SCOPE DECISIONS (documented, not silent):
- Outfield players only (GK excluded). Goalkeeping stats (saves, shots
  faced) are structurally zero for outfield players and vice versa -
  mixing them in would create spurious correlation patterns driven by
  position membership rather than genuine performance dimensions.
- Small-sample rows excluded (see Phase 2's clean.py) - their per-90 rates
  are NaN by design, and there aren't enough of them to justify imputing
  rather than excluding.
- Per-90 rates used instead of raw counts, and composite features
  (goal_contributions, defensive_actions, discipline_score, goals_minus_xg)
  excluded, because they're deterministic sums of other included variables
  - including both would create perfect/near-perfect collinearity.
- Match-level video features excluded from this analysis - they're
  constant within a match (shared by every player in it, per Phase 8),
  which would create block-correlation artifacts unrelated to individual
  performance.
- `rating` (the target) is excluded from the factor analysis itself -
  EFA describes the STRUCTURE of the performance variables, not their
  relationship to the target.

Run:
    python -m src.factor_analysis.run_efa
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from config.config import PROCESSED_DATA_DIR
from src.factor_analysis.stats_tests import kmo_test, bartlett_sphericity
from src.factor_analysis.rotation import varimax
from src.factor_analysis.parallel_analysis import parallel_analysis, suggest_n_factors

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_with_video.csv")
LOADINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_factor_loadings.csv")
SCORES_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_factor_scores.csv")
SCREE_PLOT_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_scree_plot.png")
ADEQUACY_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_adequacy.json")

EFA_VARIABLES = [
    "passes_completed_pct",
    "xa_per90",
    "assists_per90",
    "progressive_passes_per90",
    "key_passes_per90",
    "xg_per90",
    "goals_per90",
    "shots_on_target_pct",
    "dribbles_completed_per90",
    "tackles_completed_pct",
    "headers_completed_pct",
    "fouls_per90",
    "interceptions_per90",
    "clearances_per90",
    "key_tackles_per90",
]


def load_efa_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"No merged dataset at {INPUT_PATH}. Run: python -m src.integration.merge_video_features"
        )
    df = pd.read_csv(INPUT_PATH)

    n_total = len(df)
    df = df[df["position"] != "GK"].copy()
    n_after_gk = len(df)
    df = df[df["small_sample"] == False].copy()
    n_after_small_sample = len(df)

    missing = [c for c in EFA_VARIABLES if c not in df.columns]
    if missing:
        raise ValueError(f"Expected EFA variables missing from dataset: {missing}")

    df_vars = df[EFA_VARIABLES].copy()
    n_before_dropna = len(df_vars)
    df_vars = df_vars.dropna()
    n_after_dropna = len(df_vars)

    report = {
        "n_total_rows": n_total,
        "n_after_excluding_gk": n_after_gk,
        "n_after_excluding_small_sample": n_after_small_sample,
        "n_dropped_for_remaining_nan": n_before_dropna - n_after_dropna,
        "n_final": n_after_dropna,
        "n_variables": len(EFA_VARIABLES),
    }
    return df_vars.reset_index(drop=True), report


def run_efa():
    df_vars, load_report = load_efa_data()

    X = StandardScaler().fit_transform(df_vars.values)
    n_obs, n_vars = X.shape

    kmo_per_var, kmo_overall = kmo_test(X)
    chi2, dof, p_value = bartlett_sphericity(X)

    corr = np.corrcoef(X, rowvar=False)
    real_eigenvalues = np.linalg.eigvalsh(corr)[::-1]
    pa_thresholds = parallel_analysis(n_obs, n_vars, n_iter=200, random_state=42)
    n_factors_suggested = suggest_n_factors(real_eigenvalues, pa_thresholds)
    n_factors_kaiser = int(np.sum(real_eigenvalues > 1))

    obs_per_var = n_obs / n_vars

    adequacy_report = {
        "n_obs": n_obs,
        "n_vars": n_vars,
        "obs_per_variable": round(obs_per_var, 1),
        "kmo_overall": round(kmo_overall, 3),
        "kmo_adequate": kmo_overall >= 0.6,
        "bartlett_chi2": round(chi2, 2),
        "bartlett_dof": dof,
        "bartlett_p_value": round(p_value, 5),
        "bartlett_significant": p_value < 0.05,
        "n_factors_parallel_analysis": n_factors_suggested,
        "n_factors_kaiser_criterion": n_factors_kaiser,
    }

    result = {
        "load_report": load_report,
        "adequacy_report": adequacy_report,
        "real_eigenvalues": real_eigenvalues,
        "pa_thresholds": pa_thresholds,
        "kmo_per_variable": dict(zip(EFA_VARIABLES, kmo_per_var.round(3))),
    }

    n_factors = max(n_factors_suggested, 1)
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    unrotated_scores = fa.fit_transform(X)
    unrotated_loadings = fa.components_.T  # (n_vars, n_factors)

    rotated_loadings, R = varimax(unrotated_loadings)
    rotated_scores = unrotated_scores @ R

    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=EFA_VARIABLES,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )
    scores_df = pd.DataFrame(
        rotated_scores, columns=[f"Factor{i+1}_score" for i in range(n_factors)]
    )

    result["n_factors_used"] = n_factors
    result["loadings_df"] = loadings_df
    result["scores_df"] = scores_df
    return result


def make_scree_plot(real_eigenvalues, pa_thresholds, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, len(real_eigenvalues) + 1)
    ax.plot(x, real_eigenvalues, "o-", color="#D2122E", linewidth=2, label="Actual data eigenvalues")
    ax.plot(x, pa_thresholds, "s--", color="#3A2E30", linewidth=1.5,
            label="Parallel analysis (95th pct of random data)")
    ax.axhline(1, color="#9A8E90", linestyle=":", label="Kaiser criterion (eigenvalue=1)")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(result):
    lr = result["load_report"]
    ar = result["adequacy_report"]

    print("Data used for EFA:")
    print(f"  Total player-match rows: {lr['n_total_rows']}")
    print(f"  After excluding GK: {lr['n_after_excluding_gk']}")
    print(f"  After excluding small-sample appearances: {lr['n_after_excluding_small_sample']}")
    print(f"  Final rows used: {lr['n_final']}  |  Variables: {lr['n_variables']}")
    print(f"  Observations per variable: {ar['obs_per_variable']} "
          f"(rule of thumb wants 5-10+, ideally more)\n")

    print("Sampling adequacy:")
    print(f"  KMO overall: {ar['kmo_overall']}  ({'ADEQUATE' if ar['kmo_adequate'] else 'INADEQUATE'} - threshold 0.6)")
    print(f"  Bartlett's test: chi2={ar['bartlett_chi2']}, dof={ar['bartlett_dof']}, "
          f"p={ar['bartlett_p_value']}  ({'significant' if ar['bartlett_significant'] else 'NOT significant'})\n")

    print(f"Factors suggested by parallel analysis: {ar['n_factors_parallel_analysis']}")
    print(f"Factors suggested by Kaiser criterion (eigenvalue>1): {ar['n_factors_kaiser_criterion']}")
    print(f"Factors actually extracted: {result['n_factors_used']}\n")

    if not ar["kmo_adequate"] or not ar["bartlett_significant"]:
        print("*** WARNING: sampling adequacy tests indicate factor analysis is NOT well "
              "justified on this data yet. Results below are shown for pipeline verification "
              "and transparency, not as a valid factor structure to interpret or report as "
              "a real finding. Revisit once significantly more matches are added. ***\n")

    print("Rotated factor loadings:")
    print(result["loadings_df"].round(2).to_string())


def main():
    result = run_efa()
    print_report(result)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(ADEQUACY_PATH, "w") as f:
        json.dump({"adequacy": result["adequacy_report"], "load": result["load_report"],
                   "n_factors_used": result["n_factors_used"]}, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    result["loadings_df"].to_csv(LOADINGS_PATH)
    result["scores_df"].to_csv(SCORES_PATH, index=False)
    make_scree_plot(result["real_eigenvalues"], result["pa_thresholds"], SCREE_PLOT_PATH)

    print(f"\nSaved loadings -> {LOADINGS_PATH}")
    print(f"Saved factor scores -> {SCORES_PATH}")
    print(f"Saved scree plot -> {SCREE_PLOT_PATH}")


if __name__ == "__main__":
    main()
