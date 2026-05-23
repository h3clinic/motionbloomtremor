"""ROI-level motion tracker that keeps tremor measurement separate from anchoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .optical_flow import OpticalFlowMotion, SparseLKConfig, SparseLKTracker


@dataclass(frozen=True)
class HandROIAnchor:
    """Pixel-space hand ROI anchor from any coarse detector, including MediaPipe."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    def padded_bounds(self, frame_shape: tuple[int, ...], padding: float = 0.25) -> tuple[int, int, int, int]:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        pad_x = int(round(self.width * padding))
        pad_y = int(round(self.height * padding))
        left = max(0, self.x - pad_x)
        top = max(0, self.y - pad_y)
        right = min(frame_w, self.x + self.width + pad_x)
        bottom = min(frame_h, self.y + self.height + pad_y)
        return left, top, right, bottom

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class ROIFlowSample:
    timestamp: float
    hand_dx: float
    hand_dy: float
    relative_dx: float
    relative_dy: float
    valid_points: int
    track_survival_rate: float
    roi_visibility: float
    usable: bool
    reasons: list[str] = field(default_factory=list)
    background_dx: float = 0.0
    background_dy: float = 0.0
    hand_mad: float = 0.0
    background_mad: float = 0.0


class ROIFlowTracker:
    """Tracks hand ROI optical flow and optional background flow for camera-jitter subtraction."""

    def __init__(
        self,
        *,
        flow_config: SparseLKConfig | None = None,
        roi_padding: float = 0.25,
        min_anchor_confidence: float = 0.4,
    ) -> None:
        self.hand_tracker = SparseLKTracker(flow_config)
        self.background_tracker = SparseLKTracker(flow_config)
        self.roi_padding = roi_padding
        self.min_anchor_confidence = min_anchor_confidence

    def reset(self) -> None:
        self.hand_tracker.reset()
        self.background_tracker.reset()

    def update(self, timestamp: float, frame: np.ndarray, anchor: HandROIAnchor | None) -> ROIFlowSample:
        if anchor is None or anchor.area <= 0 or anchor.confidence < self.min_anchor_confidence:
            self.reset()
            return ROIFlowSample(
                timestamp=timestamp,
                hand_dx=0.0,
                hand_dy=0.0,
                relative_dx=0.0,
                relative_dy=0.0,
                valid_points=0,
                track_survival_rate=0.0,
                roi_visibility=0.0,
                usable=False,
                reasons=["hand ROI anchor missing or low confidence"],
            )

        left, top, right, bottom = anchor.padded_bounds(frame.shape, self.roi_padding)
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            self.reset()
            return ROIFlowSample(
                timestamp=timestamp,
                hand_dx=0.0,
                hand_dy=0.0,
                relative_dx=0.0,
                relative_dy=0.0,
                valid_points=0,
                track_survival_rate=0.0,
                roi_visibility=0.0,
                usable=False,
                reasons=["hand ROI crop is empty"],
            )

        hand_motion = self.hand_tracker.update(roi)
        background_motion = self.background_tracker.update(frame, mask=self._background_mask(frame.shape, (left, top, right, bottom)))
        relative_dx = hand_motion.dx - background_motion.dx
        relative_dy = hand_motion.dy - background_motion.dy
        usable = hand_motion.usable and anchor.confidence >= self.min_anchor_confidence
        reasons = list(hand_motion.reasons)
        if background_motion.usable:
            reasons.append("background/global motion available for subtraction")
        else:
            reasons.append("background/global motion unavailable or weak")

        return ROIFlowSample(
            timestamp=timestamp,
            hand_dx=hand_motion.dx,
            hand_dy=hand_motion.dy,
            relative_dx=relative_dx,
            relative_dy=relative_dy,
            valid_points=hand_motion.valid_points,
            track_survival_rate=hand_motion.track_survival_rate,
            roi_visibility=float(anchor.confidence),
            usable=usable,
            reasons=reasons,
            background_dx=background_motion.dx,
            background_dy=background_motion.dy,
            hand_mad=hand_motion.robust_mad,
            background_mad=background_motion.robust_mad,
        )

    @staticmethod
    def _background_mask(frame_shape: tuple[int, ...], roi_bounds: tuple[int, int, int, int]) -> np.ndarray:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        left, top, right, bottom = roi_bounds
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[top:bottom, left:right] = 0
        return mask


def samples_to_motion_arrays(samples: list[ROIFlowSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.array([sample.timestamp for sample in samples], dtype=np.float64)
    motion_x = np.array([sample.relative_dx for sample in samples], dtype=np.float64)
    motion_y = np.array([sample.relative_dy for sample in samples], dtype=np.float64)
    return timestamps, motion_x, motion_y


def summarize_sample_quality(samples: list[ROIFlowSample]) -> tuple[int, float, float]:
    if not samples:
        return 0, 0.0, 0.0
    valid_points = int(np.median([sample.valid_points for sample in samples]))
    track_survival_rate = float(np.median([sample.track_survival_rate for sample in samples]))
    roi_visibility = float(np.median([sample.roi_visibility for sample in samples]))
    return valid_points, track_survival_rate, roi_visibility
