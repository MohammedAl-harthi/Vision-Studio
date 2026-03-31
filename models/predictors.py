"""
Classical motion prediction algorithms.
Each predictor maintains state per-track instance.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

Point = Tuple[float, float]


class BasePredictor(ABC):
    @abstractmethod
    def update(self, position: Point) -> None:
        """Feed new observed position."""

    @abstractmethod
    def predict(self, n_steps: int = 15) -> List[Point]:
        """Return predicted future positions."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state."""

    @abstractmethod
    def get_name(self) -> str:
        pass


# ---------------------------------------------------------------------------
# Minimal standalone Kalman filter (no filterpy dependency)
# ---------------------------------------------------------------------------

class KalmanFilter2D:
    """4-state constant velocity Kalman filter: [x, y, vx, vy]."""

    def __init__(self):
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 0.01     # process noise
        self.R = np.eye(2) * 5.0      # measurement noise
        self.P = np.eye(4) * 100.0
        self.x = np.zeros(4)
        self.initialized = False

    def init(self, position: Point):
        self.x = np.array([position[0], position[1], 0.0, 0.0])
        self.initialized = True

    def update(self, z: np.ndarray):
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def project(self, n_steps: int) -> List[Point]:
        state = self.x.copy()
        results = []
        for _ in range(n_steps):
            state = self.F @ state
            results.append((float(state[0]), float(state[1])))
        return results


class KalmanPredictor(BasePredictor):
    def __init__(self):
        self._kf = KalmanFilter2D()

    def update(self, position: Point) -> None:
        if not self._kf.initialized:
            self._kf.init(position)
        else:
            self._kf.update(np.array(position, dtype=float))

    def predict(self, n_steps: int = 15) -> List[Point]:
        if not self._kf.initialized:
            return []
        return self._kf.project(n_steps)

    def reset(self) -> None:
        self._kf = KalmanFilter2D()

    def get_name(self) -> str:
        return "Kalman Filter"


# ---------------------------------------------------------------------------
# Constant Velocity
# ---------------------------------------------------------------------------

class ConstantVelocityPredictor(BasePredictor):
    def __init__(self, window: int = 5):
        self._history: List[Point] = []
        self._window = window

    def update(self, position: Point) -> None:
        self._history.append(position)

    def _velocity(self) -> Optional[Point]:
        hist = self._history[-self._window:]
        if len(hist) < 2:
            return None
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        vx = (xs[-1] - xs[0]) / (len(xs) - 1)
        vy = (ys[-1] - ys[0]) / (len(ys) - 1)
        return (vx, vy)

    def predict(self, n_steps: int = 15) -> List[Point]:
        if not self._history:
            return []
        vel = self._velocity()
        if vel is None:
            return []
        last = self._history[-1]
        return [(last[0] + vel[0] * i, last[1] + vel[1] * i) for i in range(1, n_steps + 1)]

    def reset(self) -> None:
        self._history = []

    def get_name(self) -> str:
        return "Constant Velocity"


# ---------------------------------------------------------------------------
# Constant Acceleration (least-squares quadratic fit)
# ---------------------------------------------------------------------------

class ConstantAccelerationPredictor(BasePredictor):
    def __init__(self, window: int = 20):
        self._history: List[Point] = []
        self._window = window

    def update(self, position: Point) -> None:
        self._history.append(position)

    def predict(self, n_steps: int = 15) -> List[Point]:
        hist = self._history[-self._window:]
        if len(hist) < 3:
            return []
        n = len(hist)
        t = np.arange(n, dtype=float)
        xs = np.array([p[0] for p in hist])
        ys = np.array([p[1] for p in hist])
        px = np.polyfit(t, xs, 2)
        py = np.polyfit(t, ys, 2)
        results = []
        for i in range(1, n_steps + 1):
            ti = n - 1 + i
            results.append((float(np.polyval(px, ti)), float(np.polyval(py, ti))))
        return results

    def reset(self) -> None:
        self._history = []

    def get_name(self) -> str:
        return "Constant Acceleration"


# ---------------------------------------------------------------------------
# Exponentially Smoothed Velocity
# ---------------------------------------------------------------------------

class SmoothedVelocityPredictor(BasePredictor):
    def __init__(self, alpha: float = 0.4):
        self._history: List[Point] = []
        self._alpha = alpha
        self._smooth_vel: Optional[Point] = None

    def update(self, position: Point) -> None:
        self._history.append(position)
        if len(self._history) >= 2:
            raw = (
                self._history[-1][0] - self._history[-2][0],
                self._history[-1][1] - self._history[-2][1],
            )
            if self._smooth_vel is None:
                self._smooth_vel = raw
            else:
                a = self._alpha
                self._smooth_vel = (
                    a * raw[0] + (1 - a) * self._smooth_vel[0],
                    a * raw[1] + (1 - a) * self._smooth_vel[1],
                )

    def predict(self, n_steps: int = 15) -> List[Point]:
        if not self._history or self._smooth_vel is None:
            return []
        last = self._history[-1]
        vx, vy = self._smooth_vel
        return [(last[0] + vx * i, last[1] + vy * i) for i in range(1, n_steps + 1)]

    def reset(self) -> None:
        self._history = []
        self._smooth_vel = None

    def get_name(self) -> str:
        return "Smoothed Velocity (EMA)"


# ---------------------------------------------------------------------------
# Social Force inspired: repulsion/attraction extrapolation
# ---------------------------------------------------------------------------

class SocialForcePredictor(BasePredictor):
    """Simplified social force: uses velocity + position-decay damping."""

    def __init__(self, damping: float = 0.95):
        self._history: List[Point] = []
        self._damping = damping

    def update(self, position: Point) -> None:
        self._history.append(position)

    def predict(self, n_steps: int = 15) -> List[Point]:
        if len(self._history) < 2:
            return []
        vx = self._history[-1][0] - self._history[-2][0]
        vy = self._history[-1][1] - self._history[-2][1]
        px, py = self._history[-1]
        results = []
        for _ in range(n_steps):
            vx *= self._damping
            vy *= self._damping
            px += vx
            py += vy
            results.append((px, py))
        return results

    def reset(self) -> None:
        self._history = []

    def get_name(self) -> str:
        return "Social Force (Damped)"
