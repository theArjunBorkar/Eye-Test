"""
Central configuration for the project.
Every script imports paths from here instead of hardcoding strings, so when
we add the database / video paths in later phases, this is the only file
that needs to grow.
"""

import os

# Project root = the folder that contains this "config" folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# --- Real player-match data ---
# One CSV per match goes in MATCHES_DIR, named "{match_id}.csv" (e.g. "M001.csv").
# Each file holds one row per Ajax player who played that match, using the
# exact columns in schema.py. load_match_files.py combines them all into a
# single validated dataset written to PLAYER_MATCH_DATA_PATH.
MATCHES_DIR = os.path.join(RAW_DATA_DIR, "matches")
PLAYER_MATCH_DATA_PATH = os.path.join(RAW_DATA_DIR, "player_match_data.csv")

# This project analyzes Ajax players only. Any row where `team` isn't this
# value gets dropped (with a warning) rather than silently included.
TEAM_FILTER = "Ajax"

# --- Real match metadata (links stats rows to highlight videos) ---
MATCHES_METADATA_PATH = os.path.join(RAW_DATA_DIR, "matches.csv")

# --- Models ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- CV pipeline (week 3) ---
YOLO_MODEL_NAME = "yolov8n.pt"       # nano model - fastest/smallest, appropriate for a scoped-down pipeline
FRAME_SAMPLE_RATE = 5                # process every 5th frame (reduces compute, ~6fps effective on 30fps source)
DETECTION_CONFIDENCE_THRESHOLD = 0.35
TRACK_MAX_DISTANCE_PX = 80           # max pixel distance to associate a detection with an existing track between sampled frames
TRACK_MAX_MISSED_FRAMES = 5          # frames a track can go undetected before being dropped
SPRINT_SPEED_THRESHOLD_PX = 25       # pixel displacement per sampled frame above which movement counts as a "sprint" - NOT calibrated to real speed, see cv_pipeline docs

# --- Known kit color references (RGB), for suggesting team labels ---
# Sampled directly from real evidence, not assumed. Used only as a
# SUGGESTION in classify_teams.py - the user still confirms by eye via the
# saved sample crops, since lighting/rendering varies per match.
KIT_COLOR_REFERENCES = {
    "ajax_away_blue": (51, 86, 120),  # sampled from a user-provided screenshot, see project README
    "ajax_home_white_red": (202, 183, 192),  # sampled from M002 cluster0, visually confirmed
    #   against the real sample crops (white shirt, thick red vertical band). Inherently a
    #   low-saturation color (white+red blends toward pale pink) - see team_classify.py's
    #   perceptual_color_distance() docstring for why the distance metric had to be fixed to
    #   handle this correctly.
}

# --- Highlight videos ---
# Upload short per-match highlight clips here once we reach the CV phase.
# Filename convention: {match_id}.mp4 (match_id must match the matches.csv
# row and the player_match_data.csv match_id column), e.g. "M001.mp4".
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
VIDEO_RAW_DIR = os.path.join(VIDEO_DIR, "raw")
VIDEO_PROCESSED_DIR = os.path.join(VIDEO_DIR, "processed")  # extracted frames / tracking output, later phases

# Random seed used anywhere randomness is involved (train/test splits,
# any dev-only synthetic data), so results are reproducible for grading.
RANDOM_SEED = 42
