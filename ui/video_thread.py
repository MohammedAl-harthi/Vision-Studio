"""
VideoThread: runs video capture + detection + tracking + prediction
in a background QThread, emitting processed frames and stats signals.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from models.detectors import DETECTOR_REGISTRY, BaseDetector
from models.predictors import (
    KalmanPredictor, ConstantVelocityPredictor,
    ConstantAccelerationPredictor, SmoothedVelocityPredictor,
    SocialForcePredictor,
)
from models.deep_models import (
    LSTMPredictor, GRUPredictor, TransformerPredictor,
    TrajectronPredictor, LaneGCNPredictor,
)
from utils.tracker import CentroidTracker
from utils.visualization import draw_tracks, draw_hud

# ---------------------------------------------------------------------------
# Predictor registry
# ---------------------------------------------------------------------------
PREDICTOR_REGISTRY: Dict[str, Any] = {
    "Kalman Filter": KalmanPredictor,
    "Constant Velocity": ConstantVelocityPredictor,
    "Constant Acceleration": ConstantAccelerationPredictor,
    "Smoothed Velocity (EMA)": SmoothedVelocityPredictor,
    "Social Force (Damped)": SocialForcePredictor,
    "LSTM": LSTMPredictor,
    "GRU": GRUPredictor,
    "Transformer (mmTransformer)": TransformerPredictor,
    "Trajectron++": TrajectronPredictor,
    "LaneGCN": LaneGCNPredictor,
}

PREDICTOR_GROUPS = {
    "Classical": [
        "Kalman Filter",
        "Constant Velocity",
        "Constant Acceleration",
        "Smoothed Velocity (EMA)",
        "Social Force (Damped)",
    ],
    "Deep Learning": [
        "LSTM",
        "GRU",
        "Transformer (mmTransformer)",
        "Trajectron++",
        "LaneGCN",
    ],
}


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()

        # Video source
        self._source: Union[int, str] = 0

        # Algorithm names
        self._detector_name = "Background Subtraction (MOG2)"
        self._predictor_name = "Kalman Filter"

        # Instances (re-created on change)
        self._detector: Optional[BaseDetector] = None
        self._tracker = CentroidTracker()

        # Visualisation options
        self.show_history: bool = True
        self.show_predictions: bool = True
        self.pred_steps: int = 15
        self.show_labels: bool = True

        # Tracker params (read live each frame)
        self.max_missed: int = 12
        self.max_distance: float = 120.0

        # Smoothing — readable/writable from UI thread (atomic float writes are safe)
        # input_smooth_alpha: EMA weight on raw centroid  (0=frozen … 1=no smoothing)
        self.input_smooth_alpha: float = 0.4
        # pred_blend_alpha: EMA weight on new prediction  (0=frozen … 1=no blending)
        self.pred_blend_alpha: float = 0.25

        # Object size filter (pixels) — applied after detection, before tracking
        self.min_width: int = 10
        self.min_height: int = 10
        self.max_width: int = 99999
        self.max_height: int = 99999

        # Runtime flags
        self._running = False
        self._paused = False

        # Pending tracker reset — set by UI thread, consumed by video thread
        self._tracker_reset_pending = False

        # DL training status (for HUD)
        self._dl_trained = False

    # ------------------------------------------------------------------
    # Public setters (safe to call from UI thread while running)
    # ------------------------------------------------------------------

    def set_source(self, source: Union[int, str]) -> None:
        with self._lock:
            self._source = source

    def set_detector(self, name: str) -> None:
        with self._lock:
            self._detector_name = name
            factory = DETECTOR_REGISTRY.get(name)
            self._detector = factory() if factory else None
            self._tracker_reset_pending = True   # consumed safely by video thread

    def set_predictor(self, name: str) -> None:
        with self._lock:
            self._predictor_name = name
            self._dl_trained = False
            self._tracker_reset_pending = True   # consumed safely by video thread

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True

        with self._lock:
            source = self._source
            det_name = self._detector_name
            pred_name = self._predictor_name

        # Initialise detector
        with self._lock:
            if self._detector is None:
                factory = DETECTOR_REGISTRY.get(det_name)
                self._detector = factory() if factory else None

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open source: {source}")
            return

        fps_timer = time.perf_counter()
        frame_count = 0
        display_fps = 0.0

        while self._running:
            if self._paused:
                time.sleep(0.03)
                continue

            ret, frame = cap.read()
            if not ret:
                # Loop video files; stop webcam
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # FPS calculation
            frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = now

            # Apply pending tracker reset (requested by UI thread)
            with self._lock:
                if self._tracker_reset_pending:
                    self._tracker.reset()
                    self._tracker_reset_pending = False
                detector = self._detector
                det_name = self._detector_name
                pred_name = self._predictor_name

            # Update tracker settings
            self._tracker.max_missed = self.max_missed
            self._tracker.max_distance = self.max_distance

            # --- Detect ---

            # --- Detect + size-filter ---
            detections = []
            n_raw = 0
            if detector is not None:
                try:
                    raw = detector.detect(frame)
                    n_raw = len(raw)
                    mn_w, mn_h = self.min_width, self.min_height
                    mx_w, mx_h = self.max_width, self.max_height
                    detections = [
                        d for d in raw
                        if mn_w <= d[2] <= mx_w and mn_h <= d[3] <= mx_h
                    ]
                except Exception as e:
                    self.error_occurred.emit(f"Detection error: {e}")

            # --- Track (with predictor factory) ---
            self._tracker.input_smooth_alpha = self.input_smooth_alpha
            pred_cls = PREDICTOR_REGISTRY.get(pred_name)
            factory = pred_cls if pred_cls else None
            tracks = self._tracker.update(detections, factory)

            # --- Compute & temporally blend predictions ---
            blend_a = self.pred_blend_alpha
            for t in tracks:
                if t.predictor is None:
                    continue
                try:
                    new_pts = t.predictor.predict(self.pred_steps)
                except Exception:
                    new_pts = []
                if not new_pts:
                    t.predicted_traj = []
                    continue
                prev = t.predicted_traj
                if len(prev) == len(new_pts):
                    # Blend each point with the previous prediction
                    t.predicted_traj = [
                        (blend_a * nx + (1 - blend_a) * px,
                         blend_a * ny + (1 - blend_a) * py)
                        for (px, py), (nx, ny) in zip(prev, new_pts)
                    ]
                else:
                    t.predicted_traj = new_pts  # first frame or step count changed

            # Check if any DL predictor is trained
            dl_trained = False
            for t in tracks:
                if t.predictor and hasattr(t.predictor, "is_trained") and t.predictor.is_trained:
                    dl_trained = True
                    break
            self._dl_trained = dl_trained

            # --- Visualise ---
            vis = draw_tracks(
                frame, tracks,
                show_history=self.show_history,
                show_predictions=self.show_predictions,
                pred_steps=self.pred_steps,
                show_labels=self.show_labels,
            )
            vis = draw_hud(vis, display_fps, len(tracks), len(detections),
                           det_name, pred_name, dl_trained)

            # --- Emit ---
            self.frame_ready.emit(vis)

            track_details = [
                {
                    "id": t.track_id,
                    "label": t.label,
                    "age": t.age,
                    "history_len": len(t.history),
                    "trained": getattr(t.predictor, "is_trained", None),
                }
                for t in tracks
            ]
            self.stats_updated.emit({
                "fps": display_fps,
                "tracks": len(tracks),
                "detections": len(detections),
                "detections_raw": n_raw,
                "filtered_out": n_raw - len(detections),
                "detector": det_name,
                "predictor": pred_name,
                "dl_trained": dl_trained,
                "track_details": track_details,
            })

        cap.release()
        self._running = False
