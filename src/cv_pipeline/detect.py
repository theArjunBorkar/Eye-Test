"""
YOLO player/ball detection wrapper.

Uses an off-the-shelf pretrained YOLOv8 model (COCO classes) - NOT
fine-tuned on football footage. This means:
- "person" (COCO class 0) is used as a stand-in for "player" - it will
  also pick up the referee, and won't distinguish teams.
- "sports ball" (COCO class 32) is used for the ball - a small, fast-moving
  ball in broadcast/highlight footage is a genuinely hard detection case
  for a general-purpose model; expect it to be missed in a lot of frames.
Both are accepted limitations of the scoped-down CV pipeline, not bugs.

Requires: pip install ultralytics opencv-python
Requires network on first run (downloads model weights) - this couldn't be
verified in the development sandbox, see README.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is not installed. Run: pip install ultralytics"
    ) from e

from config.config import YOLO_MODEL_NAME, DETECTION_CONFIDENCE_THRESHOLD

COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32

_model_cache = {}


def load_model():
    if YOLO_MODEL_NAME not in _model_cache:
        _model_cache[YOLO_MODEL_NAME] = YOLO(YOLO_MODEL_NAME)
    return _model_cache[YOLO_MODEL_NAME]


def detect_frame_full(model, frame, confidence=DETECTION_CONFIDENCE_THRESHOLD):
    """
    Runs detection on one frame (numpy array, BGR - as read by cv2).
    Returns (player_detections, ball_detections), each a list of dicts:
        {"bbox": (x1, y1, x2, y2), "centroid": (x, y), "confidence": float}
    This is the full-detail version - detect_frame() below is a thin
    backward-compatible wrapper that only returns centroids, for callers
    (like the tracker) that don't need the bounding box.
    """
    results = model.predict(frame, conf=confidence, verbose=False)[0]

    player_detections = []
    ball_detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        conf = float(box.conf[0])
        det = {"bbox": (x1, y1, x2, y2), "centroid": centroid, "confidence": conf}

        if cls == COCO_PERSON_CLASS:
            player_detections.append(det)
        elif cls == COCO_SPORTS_BALL_CLASS:
            ball_detections.append(det)

    return player_detections, ball_detections


def detect_frame(model, frame, confidence=DETECTION_CONFIDENCE_THRESHOLD):
    """
    Backward-compatible wrapper: returns just (player_centroids,
    ball_centroids), each a list of (x, y) tuples. Use detect_frame_full()
    when bounding boxes are needed (e.g. for shirt-color extraction).
    """
    player_dets, ball_dets = detect_frame_full(model, frame, confidence)
    return [d["centroid"] for d in player_dets], [d["centroid"] for d in ball_dets]
