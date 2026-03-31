"""
Multi-object centroid tracker with IoU + distance matching.
Each track owns a predictor instance created by the predictor_factory callable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Detection = Tuple[int, int, int, int, float, str]  # x, y, w, h, conf, label
Point = Tuple[float, float]

# Palette of distinct BGR colours for tracks
_PALETTE: List[Tuple[int, int, int]] = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 220, 50), (255, 80, 220), (80, 220, 255),
    (255, 140, 0), (0, 200, 120), (180, 60, 255),
    (30, 144, 255), (0, 230, 160), (240, 128, 128),
    (100, 200, 200), (200, 100, 200), (200, 200, 100),
]


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Point               # raw centroid this frame
    smooth_center: Point        # EMA-smoothed centroid fed to predictor
    label: str
    color: Tuple[int, int, int]
    history: List[Point] = field(default_factory=list)
    age: int = 0                # frames alive
    missed: int = 0             # consecutive missed detections
    predictor: object = None    # BasePredictor or DeepPredictor instance
    predicted_traj: List[Point] = field(default_factory=list)  # blended output


class CentroidTracker:
    """
    IoU-first, then distance-fallback matching.
    Assigns a fresh predictor instance to each new track.
    """

    def __init__(self, max_missed: int = 12, max_distance: float = 120.0,
                 iou_threshold: float = 0.10, max_history: int = 60,
                 input_smooth_alpha: float = 0.4):
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.max_history = max_history
        # EMA alpha for centroid smoothing: 0 = frozen, 1 = no smoothing
        self.input_smooth_alpha = input_smooth_alpha

    # ------------------------------------------------------------------
    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Detection],
        predictor_factory: Optional[Callable] = None,
    ) -> List[Track]:

        if not detections:
            for t in self._tracks.values():
                t.missed += 1
            self._prune()
            return self.tracks

        if not self._tracks:
            for det in detections:
                self._create(det, predictor_factory)
            self._prune()
            return self.tracks

        track_ids = list(self._tracks.keys())
        matched_t: set = set()
        matched_d: set = set()

        # --- Phase 1: IoU matching ---
        iou_scores = []
        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou = _iou(self._tracks[tid].bbox, det[:4])
                if iou >= self.iou_threshold:
                    iou_scores.append((iou, ti, di))
        iou_scores.sort(key=lambda x: -x[0])
        for _, ti, di in iou_scores:
            if ti not in matched_t and di not in matched_d:
                self._update_track(track_ids[ti], detections[di])
                matched_t.add(ti)
                matched_d.add(di)

        # --- Phase 2: Distance matching for remaining ---
        unmatched_t = [i for i in range(len(track_ids)) if i not in matched_t]
        unmatched_d = [j for j in range(len(detections)) if j not in matched_d]

        for ti in unmatched_t:
            tid = track_ids[ti]
            tc = self._tracks[tid].center
            best_j, best_dist = None, self.max_distance
            for dj in unmatched_d:
                dc = _centroid(detections[dj][:4])
                dist = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))
                if dist < best_dist:
                    best_dist, best_j = dist, dj
            if best_j is not None:
                self._update_track(tid, detections[best_j])
                unmatched_d.remove(best_j)
            else:
                self._tracks[tid].missed += 1

        # --- Phase 3: New tracks for unmatched detections ---
        for dj in unmatched_d:
            self._create(detections[dj], predictor_factory)

        self._prune()
        return self.tracks

    # ------------------------------------------------------------------
    def _create(self, det: Detection, predictor_factory: Optional[Callable]) -> None:
        tid = self._next_id
        self._next_id += 1
        x, y, w, h, _, label = det
        center = _centroid((x, y, w, h))
        color = _PALETTE[tid % len(_PALETTE)]
        predictor = predictor_factory() if predictor_factory else None
        track = Track(
            track_id=tid,
            bbox=(x, y, w, h),
            center=center,
            smooth_center=center,   # first frame: smooth == raw
            label=label,
            color=color,
            history=[center],
            predictor=predictor,
        )
        if predictor:
            predictor.update(center)
        self._tracks[tid] = track

    def _update_track(self, tid: int, det: Detection) -> None:
        if tid not in self._tracks:
            return  # track was pruned or reset between match and update
        x, y, w, h, _, label = det
        raw = _centroid((x, y, w, h))
        t = self._tracks[tid]

        # EMA-smooth the centroid to dampen detector jitter
        a = self.input_smooth_alpha
        sc = (
            a * raw[0] + (1 - a) * t.smooth_center[0],
            a * raw[1] + (1 - a) * t.smooth_center[1],
        )

        t.bbox = (x, y, w, h)
        t.center = raw           # keep raw for display (bbox anchor)
        t.smooth_center = sc     # smoothed → fed to predictor & history
        t.label = label
        t.age += 1
        t.missed = 0
        t.history.append(sc)     # history uses smoothed positions
        if len(t.history) > self.max_history:
            t.history = t.history[-self.max_history:]
        if t.predictor:
            t.predictor.update(sc)

    def _prune(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in dead:
            del self._tracks[tid]

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _centroid(bbox: Tuple[int, int, int, int]) -> Point:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _iou(b1: Tuple, b2: Tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)
