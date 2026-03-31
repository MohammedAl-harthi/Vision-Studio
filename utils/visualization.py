"""
Visualization helpers: draws bounding boxes, track histories,
predicted trajectories, and HUD overlay onto OpenCV frames.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from utils.tracker import Track


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    show_history: bool = True,
    show_predictions: bool = True,
    pred_steps: int = 15,
    show_labels: bool = True,
    overlay_alpha: float = 0.75,
) -> np.ndarray:
    overlay = frame.copy()
    out = frame.copy()

    for track in tracks:
        color = track.color
        x, y, w, h = track.bbox
        cx, cy = int(track.smooth_center[0]), int(track.smooth_center[1])

        # Bounding box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # Filled corner markers (aesthetic touch)
        arm = min(w, h) // 5
        _corner_markers(overlay, x, y, w, h, color, arm)

        # Center dot
        cv2.circle(overlay, (cx, cy), 4, color, -1)

        # Label badge
        if show_labels:
            label = f"#{track.track_id} {track.label}  age:{track.age}"
            _draw_label(overlay, label, (x, y - 4), color)

        # History trail (fades from dim → bright)
        if show_history and len(track.history) > 1:
            _draw_trail(overlay, track.history, color)

        # Predicted trajectory (pre-computed + blended by video thread)
        if show_predictions and track.predicted_traj:
            _draw_prediction(overlay, track.smooth_center, track.predicted_traj, color)

    cv2.addWeighted(overlay, overlay_alpha, out, 1 - overlay_alpha, 0, out)
    return out


# ---------------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------------

def _corner_markers(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int],
    arm: int,
) -> None:
    corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (cx, cy), (dx, dy) in zip(corners, dirs):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, 2)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, 2)


def _draw_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.42,
    thickness: int = 1,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    by = max(y - th - baseline - 2, 0)
    # semi-transparent background
    bg = img[by:by + th + baseline + 4, x:x + tw + 6].copy()
    dark = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    cv2.rectangle(img, (x, by), (x + tw + 6, by + th + baseline + 4), dark, -1)
    cv2.putText(img, text, (x + 3, by + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _draw_trail(
    img: np.ndarray,
    history: list,
    color: Tuple[int, int, int],
    max_len: int = 40,
) -> None:
    pts = history[-max_len:]
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(v * alpha) for v in color)
        thickness = max(1, int(3 * alpha))
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, p1, p2, c, thickness, cv2.LINE_AA)


def _draw_prediction(
    img: np.ndarray,
    origin: tuple,
    preds: list,
    color: Tuple[int, int, int],
) -> None:
    n = len(preds)
    prev = (int(origin[0]), int(origin[1]))
    for i, pt in enumerate(preds):
        alpha = 1.0 - (i / n) * 0.7
        c = tuple(int(v * alpha) for v in color)
        cur = (int(pt[0]), int(pt[1]))
        # dashed line effect: draw only even segments
        if i % 2 == 0:
            cv2.line(img, prev, cur, c, 1, cv2.LINE_AA)
        radius = max(1, 3 - i * 3 // n)
        cv2.circle(img, cur, radius, c, -1, cv2.LINE_AA)
        prev = cur


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    n_dets: int,
    detector_name: str,
    predictor_name: str,
    dl_trained: bool = False,
) -> np.ndarray:
    """Draws a semi-transparent HUD panel in the top-left corner."""
    lines = [
        f"FPS: {fps:5.1f}",
        f"Tracks: {n_tracks}",
        f"Dets:   {n_dets}",
        f"Det:  {detector_name}",
        f"Pred: {predictor_name}",
    ]
    if dl_trained:
        lines.append("DL: TRAINED")
    elif predictor_name in ("LSTM", "GRU", "Transformer (mmTransformer)",
                            "Trajectron++", "LaneGCN"):
        lines.append("DL: collecting...")

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.45, 1
    line_h = 18
    pad = 6
    panel_w = 250
    panel_h = len(lines) * line_h + pad * 2

    # Transparent panel
    roi = frame[4:4 + panel_h, 4:4 + panel_w]
    dark = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.3, dark, 0.7, 0, roi)
    frame[4:4 + panel_h, 4:4 + panel_w] = roi

    for i, line in enumerate(lines):
        color = (0, 255, 180) if i == 0 else (200, 200, 200)
        y = 4 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (10, y), font, fs, color, th, cv2.LINE_AA)

    return frame
