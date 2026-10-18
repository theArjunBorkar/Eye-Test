"""
Phase 2 pipeline: load_data() -> clean_data() -> engineer_features() ->
writes data/processed/player_match_features.csv

Run directly:
    python -m src.preprocessing.run_phase2
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import PROCESSED_DATA_DIR
from src.data_pipeline.load_data import load_data
from src.preprocessing.clean import clean_data, print_report
from src.preprocessing.features import engineer_features

OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "player_match_features.csv")


def run():
    raw = load_data()
    print(f"Loaded {len(raw)} raw rows across {raw['match_id'].nunique()} match(es).\n")

    cleaned, report = clean_data(raw)
    print_report(report)
    print(f"\n{len(cleaned)} rows remain after cleaning.\n")

    featured = engineer_features(cleaned)
    new_cols = [c for c in featured.columns if c not in cleaned.columns]
    print(f"Added {len(new_cols)} engineered features:")
    print(f"  {new_cols}\n")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    featured.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")

    return featured


if __name__ == "__main__":
    run()
