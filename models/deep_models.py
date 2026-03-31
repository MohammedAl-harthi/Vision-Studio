"""
Deep Learning Trajectory Prediction Models.

Architectures:
  - LSTM      : Vanilla LSTM encoder → MLP decoder
  - GRU       : GRU encoder → MLP decoder  (lighter, faster)
  - mmTransformer-style : Positional-encoded Transformer encoder → MLP decoder
  - Trajectron++ style  : Social/context-aware GRU with social pooling stub
  - LaneGCN-style       : Graph convolution on trajectory graph + MLP head

All models train **online** on accumulated trajectory data from the running
video and fall back to constant-velocity extrapolation until enough data
has been collected.

Each model is wrapped in a DeepPredictor that:
  1. Normalises the trajectory (zero-mean, unit-std per axis)
  2. Maintains a replay buffer of (input_seq, target_seq) pairs
  3. Trains one mini-batch every TRAIN_EVERY updates
  4. At inference time runs the neural net (or falls back to CV)
"""

from __future__ import annotations

import math
import threading
from typing import List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

Point = Tuple[float, float]

OBS_LEN = 20       # frames of history fed to the model
PRED_LEN = 15      # frames to predict
TRAIN_EVERY = 8    # train a mini-batch every N update() calls
BATCH_SIZE = 32    # number of samples per training step
MIN_SAMPLES = 10   # minimum samples in buffer before first train step


# ===========================================================================
# Neural Network Architectures
# ===========================================================================

if TORCH_AVAILABLE:

    # -----------------------------------------------------------------------
    # LSTM Trajectory Predictor
    # -----------------------------------------------------------------------
    class LSTMNet(nn.Module):
        """
        Seq2Seq LSTM: encodes OBS_LEN positions, decodes PRED_LEN positions.
        Input:  (B, OBS_LEN, 2)
        Output: (B, PRED_LEN, 2)
        """

        def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.encoder = nn.LSTM(2, hidden, layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, PRED_LEN * 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, (h, _) = self.encoder(x)
            out = self.decoder(h[-1])
            return out.view(-1, PRED_LEN, 2)

    # -----------------------------------------------------------------------
    # GRU Trajectory Predictor
    # -----------------------------------------------------------------------
    class GRUNet(nn.Module):
        """
        GRU encoder → MLP decoder.
        Inspired by Trajectron++ social-force variant (without scene context).
        Input:  (B, OBS_LEN, 2)
        Output: (B, PRED_LEN, 2)
        """

        def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.encoder = nn.GRU(2, hidden, layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, PRED_LEN * 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, h = self.encoder(x)
            out = self.decoder(h[-1])
            return out.view(-1, PRED_LEN, 2)

    # -----------------------------------------------------------------------
    # Transformer Trajectory Predictor (mmTransformer style)
    # -----------------------------------------------------------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class TransformerNet(nn.Module):
        """
        mmTransformer-inspired: position-encoded Transformer encoder + MLP decoder.
        Captures long-range temporal dependencies via self-attention.
        Input:  (B, OBS_LEN, 2)
        Output: (B, PRED_LEN, 2)
        """

        def __init__(self, d_model: int = 64, nhead: int = 4,
                     num_layers: int = 3, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(2, d_model)
            self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, PRED_LEN * 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.pos_enc(self.input_proj(x))
            enc = self.transformer(x)
            out = self.decoder(enc[:, -1])
            return out.view(-1, PRED_LEN, 2)

    # -----------------------------------------------------------------------
    # Trajectron++ inspired: Social GRU with pooling
    # -----------------------------------------------------------------------
    class TrajectronNet(nn.Module):
        """
        Simplified Trajectron++ node model.
        Uses bi-directional GRU + pooled social context (identity when solo).
        Input:  (B, OBS_LEN, 2)
        Output: (B, PRED_LEN, 2)
        """

        def __init__(self, hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            self.node_enc = nn.GRU(2, hidden, 2, batch_first=True,
                                   bidirectional=True,
                                   dropout=dropout)
            # social context: average pool from all tracks (identity here)
            self.context_proj = nn.Linear(hidden * 2, hidden)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ELU(),
                nn.Linear(hidden, PRED_LEN * 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, h = self.node_enc(x)
            # h: (2*layers, B, hidden) -> concat last fwd + bwd
            fwd = h[-2]
            bwd = h[-1]
            ctx = self.context_proj(torch.cat([fwd, bwd], dim=-1))
            out = self.decoder(ctx)
            return out.view(-1, PRED_LEN, 2)

    # -----------------------------------------------------------------------
    # LaneGCN inspired: Graph Conv on trajectory
    # -----------------------------------------------------------------------
    class TrajectoryGraphConv(nn.Module):
        """Single graph conv layer over trajectory nodes."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.W = nn.Linear(in_ch, out_ch, bias=False)
            self.W_self = nn.Linear(in_ch, out_ch)
            self.act = nn.ReLU()

        def forward(self, x: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, C),  A: (T, T) normalised adjacency
            agg = torch.matmul(A, x)
            return self.act(self.W_self(x) + self.W(agg))

    class LaneGCNNet(nn.Module):
        """
        LaneGCN-inspired trajectory graph network.
        Treats each time step as a graph node; edges connect consecutive steps.
        Input:  (B, OBS_LEN, 2)
        Output: (B, PRED_LEN, 2)
        """

        def __init__(self, hidden: int = 64, gc_layers: int = 3):
            super().__init__()
            self.input_proj = nn.Linear(2, hidden)
            self.gc_layers = nn.ModuleList([
                TrajectoryGraphConv(hidden, hidden) for _ in range(gc_layers)
            ])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.decoder = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, PRED_LEN * 2),
            )
            # Build a fixed normalised adjacency (chain graph + self-loops)
            A = torch.zeros(OBS_LEN, OBS_LEN)
            for i in range(OBS_LEN - 1):
                A[i, i + 1] = 1.0
                A[i + 1, i] = 1.0
            A = A + torch.eye(OBS_LEN)
            D_inv = torch.diag(1.0 / A.sum(dim=1))
            self.register_buffer("A_norm", D_inv @ A)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.input_proj(x)
            for gc in self.gc_layers:
                h = gc(h, self.A_norm)
            pooled = self.pool(h.transpose(1, 2)).squeeze(-1)
            out = self.decoder(pooled)
            return out.view(-1, PRED_LEN, 2)


# ===========================================================================
# Online-learning wrapper
# ===========================================================================

class DeepPredictor:
    """
    Wraps a PyTorch nn.Module with:
      - trajectory history collection
      - online mini-batch training
      - normalisation
      - CV fallback when not enough data
    """

    def __init__(self, model_fn, lr: float = 3e-3, name: str = "Deep"):
        self._name = name
        self._lock = threading.Lock()
        self._history: List[Point] = []
        self._buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._frame_count = 0
        self._trained = False
        self._mean = np.zeros(2)
        self._std = np.ones(2)

        if not TORCH_AVAILABLE:
            self._available = False
            self._model = None
            self._opt = None
            return

        self._available = True
        self._model = model_fn()
        self._opt = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        self._sched = optim.lr_scheduler.StepLR(self._opt, step_size=200, gamma=0.5)
        self._criterion = nn.HuberLoss()

    # ------------------------------------------------------------------
    def _normalise(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self._mean) / (self._std + 1e-8)

    def _denormalise(self, pts: np.ndarray) -> np.ndarray:
        return pts * (self._std + 1e-8) + self._mean

    # ------------------------------------------------------------------
    def update(self, position: Point) -> None:
        with self._lock:
            self._history.append(position)
            self._frame_count += 1

            if len(self._history) >= 10:
                arr = np.array(self._history, dtype=float)
                self._mean = arr.mean(0)
                self._std = arr.std(0) + 1e-6

            required = OBS_LEN + PRED_LEN
            if len(self._history) >= required:
                start = len(self._history) - required
                seq = np.array(self._history[start:start + OBS_LEN], dtype=float)
                tgt = np.array(self._history[start + OBS_LEN:start + required], dtype=float)
                self._buffer.append((seq, tgt))
                if len(self._buffer) > 500:
                    self._buffer = self._buffer[-500:]

            if (self._frame_count % TRAIN_EVERY == 0
                    and len(self._buffer) >= MIN_SAMPLES
                    and self._available):
                self._train_step()

    # ------------------------------------------------------------------
    def _train_step(self) -> None:
        import torch
        idx = np.random.choice(len(self._buffer),
                               min(BATCH_SIZE, len(self._buffer)),
                               replace=False)
        batch_x, batch_y = [], []
        for i in idx:
            seq, tgt = self._buffer[i]
            batch_x.append(self._normalise(seq))
            batch_y.append(self._normalise(tgt))

        bx = torch.FloatTensor(np.stack(batch_x))
        by = torch.FloatTensor(np.stack(batch_y))

        self._model.train()
        self._opt.zero_grad()
        pred = self._model(bx)
        loss = self._criterion(pred, by)
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._opt.step()
        self._sched.step()
        self._trained = True

    # ------------------------------------------------------------------
    def predict(self, n_steps: int = 15) -> List[Point]:
        with self._lock:
            hist = list(self._history)

        if len(hist) < 2:
            return []

        # Constant-velocity fallback
        if not self._available or not self._trained or len(hist) < OBS_LEN:
            vx = hist[-1][0] - hist[-2][0]
            vy = hist[-1][1] - hist[-2][1]
            lx, ly = hist[-1]
            return [(lx + vx * i, ly + vy * i) for i in range(1, n_steps + 1)]

        import torch
        seq = np.array(hist[-OBS_LEN:], dtype=float)
        x_norm = self._normalise(seq)
        x_t = torch.FloatTensor(x_norm).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            raw = self._model(x_t).squeeze(0).numpy()

        pts = self._denormalise(raw)
        return [(float(p[0]), float(p[1])) for p in pts[:n_steps]]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._history = []
            self._buffer = []
            self._frame_count = 0
            self._trained = False

    def get_name(self) -> str:
        return self._name

    @property
    def is_trained(self) -> bool:
        return self._trained


# ===========================================================================
# Public factory classes (used by VideoThread registry)
# ===========================================================================

class LSTMPredictor(DeepPredictor):
    def __init__(self):
        super().__init__(
            lambda: LSTMNet(hidden=128, layers=2) if TORCH_AVAILABLE else None,
            lr=3e-3,
            name="LSTM",
        )


class GRUPredictor(DeepPredictor):
    def __init__(self):
        super().__init__(
            lambda: GRUNet(hidden=128, layers=2) if TORCH_AVAILABLE else None,
            lr=3e-3,
            name="GRU",
        )


class TransformerPredictor(DeepPredictor):
    def __init__(self):
        super().__init__(
            lambda: TransformerNet(d_model=64, nhead=4, num_layers=3) if TORCH_AVAILABLE else None,
            lr=2e-3,
            name="Transformer (mmTransformer)",
        )


class TrajectronPredictor(DeepPredictor):
    def __init__(self):
        super().__init__(
            lambda: TrajectronNet(hidden=64) if TORCH_AVAILABLE else None,
            lr=3e-3,
            name="Trajectron++",
        )


class LaneGCNPredictor(DeepPredictor):
    def __init__(self):
        super().__init__(
            lambda: LaneGCNNet(hidden=64) if TORCH_AVAILABLE else None,
            lr=3e-3,
            name="LaneGCN",
        )
