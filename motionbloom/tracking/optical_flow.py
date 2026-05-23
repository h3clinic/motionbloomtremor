"""Sparse Lucas-Kanade optical flow for ROI-local motion measurement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None


@dataclass(frozen=True)
class SparseLKConfig:
    max_corners: int = 80
    quality_level: float = 0.01
    min_distance: int = 5
    block_size: int = 7
    lk_window_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_criteria_count: int = 20
    lk_criteria_eps: float = 0.03
    min_valid_points: int = 6
    refresh_below_points: int = 12
    max_point_mad_px: float = 6.0


@dataclass
class OpticalFlowMotion:
    dx: float = 0.0
    dy: float = 0.0
    mad_dx: float = 0.0
    mad_dy: float = 0.0
    valid_points: int = 0
    total_points: int = 0
    track_survival_rate: float = 0.0
    usable: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def robust_mad(self) -> float:
        return float(np.hypot(self.mad_dx, self.mad_dy))


class SparseLKTracker:
    """Stateful sparse LK tracker that summarizes point motion by median/MAD."""

    def __init__(self, config: SparseLKConfig | None = None) -> None:
        self.config = config or SparseLKConfig()
        self.previous_gray: np.ndarray | None = None
        self.previous_points: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_points = None

    def initialize(self, gray: np.ndarray, mask: np.ndarray | None = None) -> OpticalFlowMotion:
        self._require_cv2()
        prepared = self._prepare_gray(gray)
        self.previous_gray = prepared
        self.previous_points = self._detect_points(prepared, mask)
        total_points = 0 if self.previous_points is None else int(len(self.previous_points))
        usable = total_points >= self.config.min_valid_points
        reasons = ["initialized sparse optical-flow tracks"] if usable else ["not enough features to initialize tracks"]
        return OpticalFlowMotion(total_points=total_points, valid_points=total_points, track_survival_rate=1.0 if usable else 0.0, usable=usable, reasons=reasons)

    def update(self, gray: np.ndarray, mask: np.ndarray | None = None) -> OpticalFlowMotion:
        self._require_cv2()
        prepared = self._prepare_gray(gray)
        if self.previous_gray is None or self.previous_points is None or len(self.previous_points) < self.config.min_valid_points:
            return self.initialize(prepared, mask)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            prepared,
            self.previous_points,
            None,
            winSize=self.config.lk_window_size,
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.config.lk_criteria_count, self.config.lk_criteria_eps),
        )
        if next_points is None or status is None:
            self.previous_gray = prepared
            self.previous_points = self._detect_points(prepared, mask)
            return OpticalFlowMotion(reasons=["optical-flow update failed; tracks refreshed"])

        status_mask = status.reshape(-1).astype(bool)
        old_points = self.previous_points.reshape(-1, 2)[status_mask]
        new_points = next_points.reshape(-1, 2)[status_mask]
        total_points = int(len(self.previous_points))
        valid_points = int(len(new_points))
        if valid_points == 0:
            self.previous_gray = prepared
            self.previous_points = self._detect_points(prepared, mask)
            return OpticalFlowMotion(total_points=total_points, reasons=["no optical-flow tracks survived"])

        deltas = new_points - old_points
        dx_values = deltas[:, 0]
        dy_values = deltas[:, 1]
        median_dx = float(np.median(dx_values))
        median_dy = float(np.median(dy_values))
        mad_dx = float(np.median(np.abs(dx_values - median_dx)))
        mad_dy = float(np.median(np.abs(dy_values - median_dy)))
        stable = (np.abs(dx_values - median_dx) <= self.config.max_point_mad_px) & (np.abs(dy_values - median_dy) <= self.config.max_point_mad_px)
        if np.any(stable):
            stable_deltas = deltas[stable]
            median_dx = float(np.median(stable_deltas[:, 0]))
            median_dy = float(np.median(stable_deltas[:, 1]))
            mad_dx = float(np.median(np.abs(stable_deltas[:, 0] - median_dx)))
            mad_dy = float(np.median(np.abs(stable_deltas[:, 1] - median_dy)))
            new_points = new_points[stable]
            valid_points = int(len(new_points))

        survival_rate = valid_points / max(total_points, 1)
        usable = valid_points >= self.config.min_valid_points
        reasons = ["sparse optical-flow tracks updated"] if usable else ["too few stable optical-flow tracks"]

        self.previous_gray = prepared
        if valid_points < self.config.refresh_below_points:
            refreshed = self._detect_points(prepared, mask)
            self.previous_points = refreshed if refreshed is not None else new_points.reshape(-1, 1, 2).astype(np.float32)
            reasons.append("tracks refreshed after low survival")
        else:
            self.previous_points = new_points.reshape(-1, 1, 2).astype(np.float32)

        return OpticalFlowMotion(
            dx=median_dx,
            dy=median_dy,
            mad_dx=mad_dx,
            mad_dy=mad_dy,
            valid_points=valid_points,
            total_points=total_points,
            track_survival_rate=survival_rate,
            usable=usable,
            reasons=reasons,
        )

    def _detect_points(self, gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
            mask=mask,
        )

    @staticmethod
    def _prepare_gray(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim == 3:
            if cv2 is None:
                raise RuntimeError("OpenCV is required to convert color frames to grayscale")
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @staticmethod
    def _require_cv2() -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for sparse LK optical flow")
