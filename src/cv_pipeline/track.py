"""
Simple centroid tracker.

Deliberately basic per the scoped-down CV decision: assigns consistent
track IDs to detections across frames by nearest-centroid matching within
a distance threshold. No appearance model, no re-identification, no
recovery after a long occlusion - if a track goes missing for more than
TRACK_MAX_MISSED_FRAMES frames, it's dropped and a new detection nearby
starts a new track ID rather than being recognized as "the same player
again." This means one real player can accumulate multiple track IDs
across a video (e.g. after being temporarily hidden behind another
player) - a known limitation of this scope, not a bug.

Pure Python/numpy - no dependency on YOLO or video I/O, so this is fully
unit-testable with synthetic detections.
"""

import numpy as np

from config.config import TRACK_MAX_DISTANCE_PX, TRACK_MAX_MISSED_FRAMES


class CentroidTracker:
    def __init__(self, max_distance_px=TRACK_MAX_DISTANCE_PX,
                 max_missed_frames=TRACK_MAX_MISSED_FRAMES):
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self._next_id = 0
        self.tracks = {}   # track_id -> {"centroid": (x, y), "missed": int}
        self.history = {}  # track_id -> list of (frame_idx, x, y)

    def update(self, frame_idx, centroids):
        """
        centroids: list of (x, y) detected centroids for this frame.
        Returns: dict {track_id: (x, y)} of tracks active (matched) this frame.
        """
        track_ids = list(self.tracks.keys())
        matched_track_idx = set()
        matched_det_idx = set()

        if track_ids and centroids:
            track_positions = np.array([self.tracks[tid]["centroid"] for tid in track_ids])
            det_positions = np.array(centroids)
            dists = np.linalg.norm(
                track_positions[:, None, :] - det_positions[None, :, :], axis=2
            )
            # greedy nearest-neighbor matching, closest pairs first
            flat_order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
            for t_idx, d_idx in flat_order:
                t_idx, d_idx = int(t_idx), int(d_idx)
                if t_idx in matched_track_idx or d_idx in matched_det_idx:
                    continue
                if dists[t_idx, d_idx] > self.max_distance_px:
                    continue
                tid = track_ids[t_idx]
                self.tracks[tid]["centroid"] = centroids[d_idx]
                self.tracks[tid]["missed"] = 0
                self.history[tid].append((frame_idx, *centroids[d_idx]))
                matched_track_idx.add(t_idx)
                matched_det_idx.add(d_idx)

        # unmatched detections start new tracks
        for d_idx in range(len(centroids)):
            if d_idx not in matched_det_idx:
                self._start_track(frame_idx, centroids[d_idx])

        # tracks not matched this frame: increment missed count, drop if too stale
        active_this_frame = {}
        for t_idx, tid in enumerate(track_ids):
            if tid not in self.tracks:
                continue  # could've been removed already in a pathological case
            if t_idx in matched_track_idx:
                active_this_frame[tid] = self.tracks[tid]["centroid"]
            else:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed_frames:
                    del self.tracks[tid]

        # include newly-started tracks from this frame as active too
        for tid, t in self.tracks.items():
            if tid not in active_this_frame and self.history[tid][-1][0] == frame_idx:
                active_this_frame[tid] = t["centroid"]

        return active_this_frame

    def _start_track(self, frame_idx, centroid):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {"centroid": centroid, "missed": 0}
        self.history[tid] = [(frame_idx, *centroid)]
        return tid
