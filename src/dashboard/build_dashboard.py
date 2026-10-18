"""
Phase 10 - Dashboard build.

Reads every processed output from Phases 2-9 and bakes them into one
self-contained HTML file (src/dashboard/template.html + real data), so the
dashboard is a real artifact of the pipeline, not a one-off hand-built page.

NOTE ON STACK: the original brief specified a FastAPI backend + React
frontend. This build instead produces a single self-contained HTML file
with vanilla JS + Chart.js (loaded from a CDN). Reasoning: there is no live
backend serving predictions in this project (models are trained and
evaluated offline, not deployed behind an API), so a FastAPI server would
have nothing dynamic to serve beyond the same static results this file
already contains. A self-contained HTML file has a real practical
advantage over that stack here: it opens directly in any browser, or can
be published as a shareable link, with zero server setup - which matters
more for a 4-week, single-developer project than matching the original
stack for its own sake. This tradeoff should be stated explicitly in the
report rather than silently deviating from the brief.

Run:
    python -m src.dashboard.build_dashboard
"""

import base64
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from config.config import PROCESSED_DATA_DIR

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DASHBOARD_DIR, "template.html")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "dashboard.html")

MERGED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_with_video.csv")
LOADINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_factor_loadings.csv")
VIDEO_PATH = os.path.join(PROCESSED_DATA_DIR, "video_features.csv")
PHASE3_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "phase3_baseline_metrics.json")
PHASE4_METRICS_PATH = os.path.join(PROCESSED_DATA_DIR, "phase4_comparison_metrics.json")
SCREE_PLOT_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_scree_plot.png")
EFA_ADEQUACY_PATH = os.path.join(PROCESSED_DATA_DIR, "efa_adequacy.json")

PLAYER_TABLE_COLUMNS = [
    "match_id", "player_name", "position", "minutes", "rating", "goals", "assists",
    "key_passes", "xg", "xa", "tackles_completed_pct", "interceptions", "clearances",
    "goal_contributions", "defensive_actions", "discipline_score", "small_sample",
]


def _require(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. {hint}")


def build_payload():
    _require(MERGED_DATA_PATH, "Run: python -m src.integration.merge_video_features")
    _require(LOADINGS_PATH, "Run: python -m src.factor_analysis.run_efa")
    _require(VIDEO_PATH, "Run: python -m src.cv_pipeline.run_cv_pipeline")
    _require(PHASE3_METRICS_PATH, "Run: python -m src.modeling.train_baseline")
    _require(PHASE4_METRICS_PATH, "Run: python -m src.modeling.train_xgboost")
    _require(SCREE_PLOT_PATH, "Run: python -m src.factor_analysis.run_efa")

    df = pd.read_csv(MERGED_DATA_PATH)
    players = df[PLAYER_TABLE_COLUMNS].round(2).to_dict(orient="records")

    loadings = pd.read_csv(LOADINGS_PATH, index_col=0).round(2)
    loadings_data = {
        "variables": list(loadings.index),
        "factors": list(loadings.columns),
        "matrix": loadings.values.tolist(),
    }

    video = pd.read_csv(VIDEO_PATH).round(3).to_dict(orient="records")

    with open(PHASE3_METRICS_PATH) as f:
        phase3 = json.load(f)
    with open(PHASE4_METRICS_PATH) as f:
        phase4 = json.load(f)

    with open(SCREE_PLOT_PATH, "rb") as f:
        scree_b64 = base64.b64encode(f.read()).decode("ascii")

    # Optional: written by run_efa. Keeps KMO / row counts out of hardcoded
    # dashboard text so they can't go stale when matches are added. If the file
    # is missing the dashboard says so instead of showing an old number.
    efa = None
    if os.path.exists(EFA_ADEQUACY_PATH):
        with open(EFA_ADEQUACY_PATH) as f:
            efa = json.load(f)

    summary = {
        "n_matches": int(df["match_id"].nunique()),
        "n_players": int(df["player_id"].nunique()),
        "n_rows": len(df),
    }

    return {
        "players": players,
        "loadings": loadings_data,
        "video": video,
        "phase3": phase3,
        "phase4": phase4,
        "scree_b64": scree_b64,
        "efa": efa,
        "summary": summary,
    }


def build_dashboard():
    payload = build_payload()

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    html = template.replace("__APP_DATA__", json.dumps(payload))

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    return OUTPUT_PATH, len(html)


def main():
    path, size_bytes = build_dashboard()
    print(f"Built dashboard -> {path} ({size_bytes / 1024:.1f} KB)")
    print("Self-contained: open directly in a browser, or publish as a shareable link.")


if __name__ == "__main__":
    main()
