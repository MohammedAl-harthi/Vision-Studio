"""
Main application window.
"""

from __future__ import annotations

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QCheckBox, QGroupBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QSizePolicy,
    QTabWidget, QTextEdit, QSlider, QStatusBar, QApplication,
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

from ui.video_thread import VideoThread, DETECTOR_REGISTRY, PREDICTOR_REGISTRY, PREDICTOR_GROUPS


# ---------------------------------------------------------------------------
# Algorithm descriptions
# ---------------------------------------------------------------------------
def _enumerate_cameras(max_test: int = 8) -> list[dict]:
    """
    Probe camera indices 0..max_test-1 and return a list of
    {'index': int, 'label': str} dicts for those that open successfully.
    """
    found = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            label = f"Camera {idx}  ({w}×{h}"
            if fps and fps > 0:
                label += f" @ {fps:.0f} fps"
            label += ")"
            found.append({"index": idx, "label": label})
        else:
            cap.release()
    return found


DETECTOR_DESCRIPTIONS = {
    "Background Subtraction (MOG2)": (
        "Gaussian Mixture Model background subtraction. "
        "Learns per-pixel background distributions and flags moving pixels. "
        "Good general-purpose motion detector."
    ),
    "Background Subtraction (KNN)": (
        "K-Nearest Neighbours background model. "
        "More robust to noise than MOG2, slightly slower."
    ),
    "Optical Flow (Farneback)": (
        "Dense optical flow computes per-pixel motion vectors. "
        "Regions with flow magnitude above threshold are clustered into detections."
    ),
    "HOG + SVM (Pedestrian)": (
        "Histogram of Oriented Gradients + SVM. "
        "OpenCV's pre-trained pedestrian detector. "
        "Classic feature-engineering approach."
    ),
    "Haar Cascade (Face)": "Viola-Jones Haar-feature cascade for frontal faces.",
    "Haar Cascade (Body)": "Viola-Jones Haar-feature cascade for full body.",
    "Haar Cascade (Upper Body)": "Viola-Jones Haar-feature cascade for upper body.",
    "YOLOv8-Nano": (
        "YOLOv8 Nano — smallest/fastest YOLO variant. "
        "Detects 80 COCO classes. Requires ultralytics."
    ),
    "YOLOv8-Small": "YOLOv8 Small — balanced speed/accuracy. Requires ultralytics.",
    "YOLOv8-Medium": "YOLOv8 Medium — higher accuracy, more GPU intensive. Requires ultralytics.",
}

PREDICTOR_DESCRIPTIONS = {
    "Kalman Filter": (
        "Linear Bayesian state estimator with constant-velocity motion model. "
        "Optimal for linear Gaussian systems. Handles missed detections well."
    ),
    "Constant Velocity": (
        "Averages velocity over a sliding window of recent positions "
        "and linearly extrapolates. Simple baseline."
    ),
    "Constant Acceleration": (
        "Fits a quadratic polynomial to recent trajectory history via "
        "least-squares. Captures moderate acceleration."
    ),
    "Smoothed Velocity (EMA)": (
        "Exponential moving average of per-frame velocity. "
        "Smoother than raw CV with configurable memory (alpha)."
    ),
    "Social Force (Damped)": (
        "Velocity-decaying Social Force model. Velocity shrinks each step, "
        "modelling deceleration towards a goal."
    ),
    "LSTM": (
        "Long Short-Term Memory network.\n"
        "Architecture: LSTM encoder (2 layers, 128 hidden) → MLP decoder.\n"
        "Trains online on accumulated trajectory windows. "
        "Falls back to constant-velocity until enough data is collected.\n"
        "Inspired by: Social-LSTM, pedestrian trajectory forecasting."
    ),
    "GRU": (
        "Gated Recurrent Unit network.\n"
        "Architecture: bidirectional GRU encoder → MLP decoder.\n"
        "Fewer parameters than LSTM, faster online training.\n"
        "Inspired by: Trajectron++ node-level GRU."
    ),
    "Transformer (mmTransformer)": (
        "Positional-encoded Transformer encoder + MLP decoder.\n"
        "Self-attention captures long-range temporal dependencies.\n"
        "Architecture: Linear input projection → sinusoidal PE → "
        "3× TransformerEncoderLayer (d=64, heads=4) → MLP.\n"
        "Inspired by: mmTransformer (Liu et al., CVPR 2021)."
    ),
    "Trajectron++": (
        "Bidirectional GRU node model with social-context pooling.\n"
        "In this standalone version social context is implicit (per-track).\n"
        "Trains online on observed trajectories.\n"
        "Inspired by: Trajectron++ (Salzmann et al., ECCV 2020)."
    ),
    "LaneGCN": (
        "Graph Convolutional Network over trajectory nodes.\n"
        "Treats each time step as a node; edges connect adjacent steps.\n"
        "3 graph-conv layers aggregate neighbourhood features.\n"
        "Inspired by: LaneGCN (Liang et al., ECCV 2020) — used in AV forecasting."
    ),
}


# ---------------------------------------------------------------------------
# Video display widget
# ---------------------------------------------------------------------------

class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;"
        )
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("No video source\n\nSelect Webcam or open a video file,\nthen press  ▶  Start")
        self.setFont(QFont("Segoe UI", 13))
        self.setStyleSheet(
            "background-color: #0d1117; color: #484f58; "
            "border: 1px solid #30363d; border-radius: 6px;"
        )

    def update_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qi).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)
        self.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px;"
        )


# ---------------------------------------------------------------------------
# Reusable separator
# ---------------------------------------------------------------------------

def _separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #30363d;")
    return f


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection & Motion Prediction Studio")
        self.setMinimumSize(1300, 760)
        self._source = 0
        self._thread: VideoThread = VideoThread()
        self._setup_ui()
        self._apply_theme()
        self._connect_thread(self._thread)

    # ======================================================================
    # UI Construction
    # ======================================================================

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---- Left column: video + source controls ----
        left = QVBoxLayout()
        left.setSpacing(6)

        self._video = VideoWidget()
        left.addWidget(self._video)
        left.addLayout(self._build_source_bar())  # returns QVBoxLayout

        left_w = QWidget()
        left_w.setLayout(left)

        # ---- Right column: controls ----
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(340)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_inner = QWidget()
        right_layout = QVBoxLayout(right_inner)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        right_layout.addWidget(self._build_detection_group())
        right_layout.addWidget(self._build_prediction_group())
        right_layout.addWidget(self._build_viz_group())
        right_layout.addWidget(self._build_smoothing_group())
        right_layout.addWidget(self._build_tracker_group())
        right_layout.addWidget(self._build_size_filter_group())
        right_layout.addWidget(self._build_stats_group())
        right_layout.addStretch()

        right_scroll.setWidget(right_inner)

        # ---- Splitter ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        root.addWidget(splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        # Probe cameras now that _status exists
        self._refresh_cameras()

    # ------------------------------------------------------------------
    def _build_source_bar(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(5)

        # ---- Row 1: webcam selector ----
        cam_row = QHBoxLayout()
        cam_row.setSpacing(6)

        cam_row.addWidget(QLabel("Camera:"))

        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(220)
        self._cam_combo.setToolTip("Select which webcam to use")
        cam_row.addWidget(self._cam_combo, 1)

        self._btn_refresh_cams = QPushButton("⟳ Refresh")
        self._btn_refresh_cams.setToolTip("Re-scan for connected cameras")
        self._btn_refresh_cams.setFixedWidth(90)
        self._btn_refresh_cams.clicked.connect(self._refresh_cameras)
        cam_row.addWidget(self._btn_refresh_cams)

        self._btn_file = QPushButton("Open File…")
        self._btn_file.clicked.connect(self._on_select_file)
        cam_row.addWidget(self._btn_file)

        col.addLayout(cam_row)

        # ---- Row 2: playback controls ----
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        self._btn_start = QPushButton("▶  Start")
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_stop = QPushButton("■  Stop")

        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)

        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        for btn in (self._btn_start, self._btn_pause, self._btn_stop):
            ctrl_row.addWidget(btn)

        col.addLayout(ctrl_row)

        # _refresh_cameras() is called later in _setup_ui, after _status exists
        self._cameras: list[dict] = []

        # Selecting from combo updates source live
        self._cam_combo.currentIndexChanged.connect(self._on_cam_combo_changed)

        return col

    # ------------------------------------------------------------------
    def _build_detection_group(self) -> QGroupBox:
        g = QGroupBox("Detection Algorithm")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)

        self._det_combo = QComboBox()
        self._det_combo.addItems(list(DETECTOR_REGISTRY.keys()))
        self._det_combo.setCurrentText("Background Subtraction (MOG2)")
        lay.addWidget(self._det_combo)

        self._det_desc = QLabel()
        self._det_desc.setWordWrap(True)
        self._det_desc.setObjectName("desc")
        lay.addWidget(self._det_desc)

        self._det_combo.currentTextChanged.connect(self._on_det_changed)
        self._update_det_desc(self._det_combo.currentText())
        return g

    # ------------------------------------------------------------------
    def _build_prediction_group(self) -> QGroupBox:
        g = QGroupBox("Motion Prediction Algorithm")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)

        # Tab: Classical / Deep Learning
        tabs = QTabWidget()
        tabs.setFixedHeight(160)

        # Classical tab
        cl_w = QWidget()
        cl_lay = QVBoxLayout(cl_w)
        self._cl_combo = QComboBox()
        self._cl_combo.addItems(PREDICTOR_GROUPS["Classical"])
        cl_lay.addWidget(self._cl_combo)
        cl_lay.addStretch()
        tabs.addTab(cl_w, "Classical")

        # Deep Learning tab
        dl_w = QWidget()
        dl_lay = QVBoxLayout(dl_w)
        self._dl_combo = QComboBox()
        self._dl_combo.addItems(PREDICTOR_GROUPS["Deep Learning"])
        self._dl_status = QLabel("Status: waiting for data…")
        self._dl_status.setObjectName("desc")
        dl_lay.addWidget(self._dl_combo)
        dl_lay.addWidget(self._dl_status)
        dl_lay.addStretch()
        tabs.addTab(dl_w, "Deep Learning")

        self._pred_tabs = tabs
        lay.addWidget(tabs)

        self._pred_desc = QLabel()
        self._pred_desc.setWordWrap(True)
        self._pred_desc.setObjectName("desc")
        lay.addWidget(self._pred_desc)

        self._apply_pred_btn = QPushButton("Apply Predictor")
        self._apply_pred_btn.clicked.connect(self._on_apply_predictor)
        lay.addWidget(self._apply_pred_btn)

        self._cl_combo.currentTextChanged.connect(lambda _: self._update_pred_desc())
        self._dl_combo.currentTextChanged.connect(lambda _: self._update_pred_desc())
        tabs.currentChanged.connect(lambda _: self._update_pred_desc())
        self._update_pred_desc()

        return g

    # ------------------------------------------------------------------
    def _build_viz_group(self) -> QGroupBox:
        g = QGroupBox("Visualisation")
        lay = QVBoxLayout(g)
        lay.setSpacing(4)

        self._cb_history = QCheckBox("Show trajectory history")
        self._cb_history.setChecked(True)
        self._cb_preds = QCheckBox("Show prediction trajectory")
        self._cb_preds.setChecked(True)
        self._cb_labels = QCheckBox("Show track labels")
        self._cb_labels.setChecked(True)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Prediction steps:"))
        self._spin_steps = QSpinBox()
        self._spin_steps.setRange(1, 60)
        self._spin_steps.setValue(15)
        step_row.addWidget(self._spin_steps)

        for w in (self._cb_history, self._cb_preds, self._cb_labels):
            lay.addWidget(w)
        lay.addLayout(step_row)

        self._cb_history.stateChanged.connect(self._sync_viz)
        self._cb_preds.stateChanged.connect(self._sync_viz)
        self._cb_labels.stateChanged.connect(self._sync_viz)
        self._spin_steps.valueChanged.connect(self._sync_viz)

        return g

    # ------------------------------------------------------------------
    def _build_smoothing_group(self) -> QGroupBox:
        g = QGroupBox("Prediction Smoothing")
        lay = QVBoxLayout(g)
        lay.setSpacing(6)

        def _slider_row(label_text, attr, default_pct):
            """Returns (layout, value_label). Slider 0–100 maps to 0.0–1.0."""
            row = QVBoxLayout()
            top = QHBoxLayout()
            top.addWidget(QLabel(label_text))
            val_lbl = QLabel(f"{default_pct/100:.2f}")
            val_lbl.setObjectName("desc")
            val_lbl.setFixedWidth(34)
            top.addStretch()
            top.addWidget(val_lbl)
            row.addLayout(top)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_pct)
            slider.setTickInterval(10)
            row.addWidget(slider)
            return row, slider, val_lbl

        # Input smoothing (centroid EMA)
        in_row, self._sl_input_smooth, self._lbl_input_smooth = _slider_row(
            "Input centroid smoothing", "input_smooth_alpha", 40
        )
        hint1 = QLabel("← less smooth  |  more smooth →")
        hint1.setObjectName("desc")
        hint1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Output prediction blending
        out_row, self._sl_pred_blend, self._lbl_pred_blend = _slider_row(
            "Prediction temporal blend", "pred_blend_alpha", 25
        )
        hint2 = QLabel("← smoother/slower  |  faster/jittery →")
        hint2.setObjectName("desc")
        hint2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addLayout(in_row)
        lay.addWidget(hint1)
        lay.addLayout(out_row)
        lay.addWidget(hint2)

        self._sl_input_smooth.valueChanged.connect(self._sync_smoothing)
        self._sl_pred_blend.valueChanged.connect(self._sync_smoothing)

        return g

    def _build_tracker_group(self) -> QGroupBox:
        g = QGroupBox("Tracker Settings")
        lay = QVBoxLayout(g)
        lay.setSpacing(4)

        def _row(label, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addStretch()
            r.addWidget(widget)
            return r

        self._spin_max_miss = QSpinBox()
        self._spin_max_miss.setRange(1, 60)
        self._spin_max_miss.setValue(12)

        self._spin_max_dist = QSpinBox()
        self._spin_max_dist.setRange(10, 600)
        self._spin_max_dist.setValue(120)

        lay.addLayout(_row("Max missed frames:", self._spin_max_miss))
        lay.addLayout(_row("Max match distance (px):", self._spin_max_dist))

        apply_btn = QPushButton("Apply Tracker Settings")
        apply_btn.clicked.connect(self._sync_tracker)
        lay.addWidget(apply_btn)

        return g

    # ------------------------------------------------------------------
    def _build_size_filter_group(self) -> QGroupBox:
        g = QGroupBox("Object Size Filter")
        lay = QVBoxLayout(g)
        lay.setSpacing(6)

        def _row(label_text, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label_text))
            r.addStretch()
            r.addWidget(widget)
            return r

        # Min / Max Width
        self._spin_min_w = QSpinBox()
        self._spin_min_w.setRange(1, 4000)
        self._spin_min_w.setValue(10)
        self._spin_min_w.setSuffix(" px")

        self._spin_max_w = QSpinBox()
        self._spin_max_w.setRange(1, 4000)
        self._spin_max_w.setValue(2000)
        self._spin_max_w.setSuffix(" px")

        # Min / Max Height
        self._spin_min_h = QSpinBox()
        self._spin_min_h.setRange(1, 4000)
        self._spin_min_h.setValue(10)
        self._spin_min_h.setSuffix(" px")

        self._spin_max_h = QSpinBox()
        self._spin_max_h.setRange(1, 4000)
        self._spin_max_h.setValue(2000)
        self._spin_max_h.setSuffix(" px")

        lay.addLayout(_row("Min width:", self._spin_min_w))
        lay.addLayout(_row("Max width:", self._spin_max_w))
        lay.addLayout(_row("Min height:", self._spin_min_h))
        lay.addLayout(_row("Max height:", self._spin_max_h))

        # Live preview label
        self._size_preview = QLabel()
        self._size_preview.setObjectName("desc")
        self._size_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._size_preview)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_size_filter)
        lay.addWidget(reset_btn)

        # Wire all spinboxes to live-apply
        for spin in (self._spin_min_w, self._spin_max_w,
                     self._spin_min_h, self._spin_max_h):
            spin.valueChanged.connect(self._sync_size_filter)

        self._update_size_preview()
        return g

    # ------------------------------------------------------------------
    def _build_stats_group(self) -> QGroupBox:
        g = QGroupBox("Live Statistics")
        lay = QVBoxLayout(g)
        lay.setSpacing(4)

        self._lbl_fps = QLabel("FPS: —")
        self._lbl_tracks = QLabel("Tracks: —")
        self._lbl_dets = QLabel("Detections: —")
        self._lbl_filtered = QLabel("Filtered out: —")
        self._lbl_dl = QLabel("")

        mono = QFont("Courier New", 10)
        for lbl in (self._lbl_fps, self._lbl_tracks, self._lbl_dets,
                    self._lbl_filtered, self._lbl_dl):
            lbl.setFont(mono)
            lay.addWidget(lbl)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["ID", "Class", "Age", "History"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(160)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table)

        return g

    # ======================================================================
    # Theme
    # ======================================================================

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #21262d;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                font-weight: 600;
                color: #58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 4px 8px;
                color: #c9d1d9;
                min-height: 26px;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #161b22;
                color: #c9d1d9;
                selection-background-color: #1f3a5f;
                border: 1px solid #30363d;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 6px 14px;
                color: #c9d1d9;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #58a6ff;
                color: #ffffff;
            }
            QPushButton:pressed { background-color: #1f6feb; }
            QPushButton:disabled {
                background-color: #161b22;
                color: #484f58;
                border-color: #21262d;
            }
            QPushButton#start {
                background-color: #1a4f21;
                border-color: #2ea043;
                color: #7ee787;
            }
            QPushButton#start:hover { background-color: #2ea043; color: #fff; }
            QPushButton#stop {
                background-color: #4a1a1a;
                border-color: #da3633;
                color: #ff7b72;
            }
            QPushButton#stop:hover { background-color: #da3633; color: #fff; }
            QCheckBox { color: #c9d1d9; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #484f58;
                border-radius: 3px;
                background-color: #161b22;
            }
            QCheckBox::indicator:checked {
                background-color: #1f6feb;
                border-color: #58a6ff;
            }
            QLabel#desc { color: #8b949e; font-size: 11px; }
            QTableWidget {
                background-color: #0d1117;
                border: 1px solid #21262d;
                gridline-color: #21262d;
                color: #c9d1d9;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #58a6ff;
                border: 1px solid #21262d;
                padding: 3px;
                font-size: 11px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 2px 6px;
                color: #c9d1d9;
                min-width: 60px;
            }
            QScrollArea { border: none; }
            QScrollBar:vertical {
                background: #0d1117; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #30363d; border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #58a6ff; }
            QTabWidget::pane {
                border: 1px solid #21262d;
                border-radius: 4px;
                background: #0d1117;
            }
            QTabBar::tab {
                background: #161b22;
                border: 1px solid #21262d;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 5px 14px;
                color: #8b949e;
            }
            QTabBar::tab:selected {
                background: #0d1117;
                color: #58a6ff;
                border-bottom: 2px solid #1f6feb;
            }
            QStatusBar { background-color: #161b22; color: #8b949e; font-size: 11px; }
        """)
        self._btn_start.setObjectName("start")
        self._btn_stop.setObjectName("stop")
        self._btn_start.setStyleSheet(
            "QPushButton { background:#1a4f21; border-color:#2ea043; color:#7ee787; "
            "border-radius:5px; padding:6px 14px; font-weight:600; }"
            "QPushButton:hover { background:#2ea043; color:#fff; }"
        )
        self._btn_stop.setStyleSheet(
            "QPushButton { background:#4a1a1a; border-color:#da3633; color:#ff7b72; "
            "border-radius:5px; padding:6px 14px; font-weight:600; }"
            "QPushButton:hover { background:#da3633; color:#fff; }"
            "QPushButton:disabled { background:#161b22; color:#484f58; border-color:#21262d; }"
        )

    # ======================================================================
    # Description helpers
    # ======================================================================

    def _update_det_desc(self, name: str) -> None:
        self._det_desc.setText(DETECTOR_DESCRIPTIONS.get(name, ""))

    def _update_pred_desc(self) -> None:
        name = self._current_predictor_name()
        self._pred_desc.setText(PREDICTOR_DESCRIPTIONS.get(name, ""))

    def _current_predictor_name(self) -> str:
        if self._pred_tabs.currentIndex() == 0:
            return self._cl_combo.currentText()
        return self._dl_combo.currentText()

    # ======================================================================
    # Thread wiring
    # ======================================================================

    def _connect_thread(self, thread: VideoThread) -> None:
        thread.frame_ready.connect(self._on_frame)
        thread.stats_updated.connect(self._on_stats)
        thread.error_occurred.connect(self._on_error)

    # ======================================================================
    # Slots — source controls
    # ======================================================================

    def _refresh_cameras(self) -> None:
        """Enumerate connected cameras and repopulate the combo box."""
        self._btn_refresh_cams.setEnabled(False)
        self._btn_refresh_cams.setText("Scanning…")
        QApplication.processEvents()

        self._cameras = _enumerate_cameras()

        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        if self._cameras:
            for cam in self._cameras:
                self._cam_combo.addItem(cam["label"], userData=cam["index"])
            self._cam_combo.setCurrentIndex(0)
            self._source = self._cameras[0]["index"]
            self._status.showMessage(
                f"Found {len(self._cameras)} camera(s). Selected: {self._cameras[0]['label']}"
            )
        else:
            self._cam_combo.addItem("No cameras found")
            self._status.showMessage("No cameras detected. Open a video file instead.")
        self._cam_combo.blockSignals(False)

        self._btn_refresh_cams.setEnabled(True)
        self._btn_refresh_cams.setText("⟳ Refresh")

    def _on_cam_combo_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._cameras):
            return
        cam = self._cameras[index]
        self._source = cam["index"]
        self._status.showMessage(f"Source: {cam['label']}")

    def _on_select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All Files (*)"
        )
        if path:
            self._source = path
            self._status.showMessage(f"Source: {path}")

    def _on_start(self) -> None:
        if self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(2000)

        self._thread = VideoThread()
        self._connect_thread(self._thread)

        self._thread.set_source(self._source)
        self._thread.set_detector(self._det_combo.currentText())
        self._thread.set_predictor(self._current_predictor_name())
        self._sync_viz()
        self._sync_tracker()
        self._sync_smoothing()
        self._sync_size_filter()

        self._thread.start()

        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._status.showMessage("Running…")

    def _on_pause(self) -> None:
        if self._thread.isRunning():
            if not self._thread._paused:
                self._thread.pause()
                self._btn_pause.setText("▶  Resume")
                self._status.showMessage("Paused")
            else:
                self._thread.resume()
                self._btn_pause.setText("⏸  Pause")
                self._status.showMessage("Running…")

    def _on_stop(self) -> None:
        self._thread.stop()
        self._thread.wait(3000)
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_pause.setText("⏸  Pause")
        self._video._show_placeholder()
        self._status.showMessage("Stopped")

    # ======================================================================
    # Slots — algorithm changes
    # ======================================================================

    def _on_det_changed(self, name: str) -> None:
        self._update_det_desc(name)
        if self._thread.isRunning():
            self._thread.set_detector(name)

    def _on_apply_predictor(self) -> None:
        name = self._current_predictor_name()
        self._update_pred_desc()
        if self._thread.isRunning():
            self._thread.set_predictor(name)
        self._status.showMessage(f"Predictor switched to: {name}")

    # ======================================================================
    # Slots — viz / tracker settings
    # ======================================================================

    def _sync_viz(self) -> None:
        t = self._thread
        t.show_history = self._cb_history.isChecked()
        t.show_predictions = self._cb_preds.isChecked()
        t.show_labels = self._cb_labels.isChecked()
        t.pred_steps = self._spin_steps.value()

    def _sync_tracker(self) -> None:
        t = self._thread
        t.max_missed = self._spin_max_miss.value()
        t.max_distance = float(self._spin_max_dist.value())

    def _sync_smoothing(self) -> None:
        v_in = self._sl_input_smooth.value() / 100.0
        v_out = self._sl_pred_blend.value() / 100.0
        self._lbl_input_smooth.setText(f"{v_in:.2f}")
        self._lbl_pred_blend.setText(f"{v_out:.2f}")
        self._thread.input_smooth_alpha = v_in
        self._thread.pred_blend_alpha = v_out

    def _sync_size_filter(self) -> None:
        mn_w = self._spin_min_w.value()
        mx_w = self._spin_max_w.value()
        mn_h = self._spin_min_h.value()
        mx_h = self._spin_max_h.value()
        # Clamp: min never exceeds max
        if mn_w > mx_w:
            self._spin_max_w.blockSignals(True)
            self._spin_max_w.setValue(mn_w)
            self._spin_max_w.blockSignals(False)
            mx_w = mn_w
        if mn_h > mx_h:
            self._spin_max_h.blockSignals(True)
            self._spin_max_h.setValue(mn_h)
            self._spin_max_h.blockSignals(False)
            mx_h = mn_h
        self._thread.min_width = mn_w
        self._thread.max_width = mx_w
        self._thread.min_height = mn_h
        self._thread.max_height = mx_h
        self._update_size_preview()

    def _update_size_preview(self) -> None:
        mn_w = self._spin_min_w.value()
        mx_w = self._spin_max_w.value()
        mn_h = self._spin_min_h.value()
        mx_h = self._spin_max_h.value()
        self._size_preview.setText(
            f"Allowed: W [{mn_w}–{mx_w}] px  ×  H [{mn_h}–{mx_h}] px"
        )

    def _reset_size_filter(self) -> None:
        for spin in (self._spin_min_w, self._spin_min_h):
            spin.setValue(10)
        for spin in (self._spin_max_w, self._spin_max_h):
            spin.setValue(2000)

    # ======================================================================
    # Slots — incoming data
    # ======================================================================

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._video.update_frame(frame)

    @pyqtSlot(dict)
    def _on_stats(self, stats: dict) -> None:
        self._lbl_fps.setText(f"FPS:        {stats['fps']:5.1f}")
        self._lbl_tracks.setText(f"Tracks:     {stats['tracks']}")
        self._lbl_dets.setText(f"Detections: {stats['detections']}")
        n_filt = stats.get("filtered_out", 0)
        filt_color = "#ff7b72" if n_filt > 0 else "#8b949e"
        self._lbl_filtered.setText(f"Filtered:   {n_filt} (size)")
        self._lbl_filtered.setStyleSheet(f"color: {filt_color}; font-family: 'Courier New'; font-size: 10px;")

        if stats.get("dl_trained"):
            self._lbl_dl.setText("DL model:   ✓ TRAINED")
            self._lbl_dl.setStyleSheet("color: #7ee787; font-family: 'Courier New'; font-size: 10px;")
            self._dl_status.setText("Status: model trained ✓")
        elif stats["predictor"] in ("LSTM", "GRU", "Transformer (mmTransformer)",
                                    "Trajectron++", "LaneGCN"):
            n_buf = sum(t.get("history_len", 0) for t in stats.get("track_details", []))
            self._lbl_dl.setText(f"DL model:   collecting ({n_buf} pts)")
            self._lbl_dl.setStyleSheet("color: #d29922; font-family: 'Courier New'; font-size: 10px;")
            self._dl_status.setText(f"Status: collecting data ({n_buf} points)…")
        else:
            self._lbl_dl.setText("")
            self._dl_status.setText("Status: N/A (classical predictor active)")

        details = stats.get("track_details", [])
        self._table.setRowCount(len(details))
        for i, t in enumerate(details):
            trained_str = ""
            if t.get("trained") is True:
                trained_str = " ✓"
            elif t.get("trained") is False:
                trained_str = " …"
            self._table.setItem(i, 0, QTableWidgetItem(str(t["id"])))
            self._table.setItem(i, 1, QTableWidgetItem(t["label"] + trained_str))
            self._table.setItem(i, 2, QTableWidgetItem(str(t["age"])))
            self._table.setItem(i, 3, QTableWidgetItem(str(t["history_len"])))

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._status.showMessage(f"Error: {msg}")

    # ======================================================================
    # Cleanup
    # ======================================================================

    def closeEvent(self, event) -> None:
        self._thread.stop()
        self._thread.wait(3000)
        event.accept()
