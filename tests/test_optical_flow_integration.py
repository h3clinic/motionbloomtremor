"""Synthetic-frame tests for HandROIFlowSampler.

These tests drive the LK sampler with hand-crafted grayscale frames so we
can verify:
  1. Pure ROI translation registers as flow.
  2. A textured hand interior with 5 Hz oscillation produces a 5 Hz
     tremor signal when fed into analyze_tremor_motion (the same
     analyzer used by the live pipeline).
  3. Untextured / empty ROIs degrade gracefully (no crash, low quality).
"""

from __future__ import annotations

import unittest

import numpy as np

from motionbloom.analysis.tremor_signal import TremorAnalysisConfig, analyze_tremor_motion
from motionbloom.tracker import HandROIFlowSampler


FRAME_W, FRAME_H = 320, 240
HAND_BOX = (110, 80, 210, 180)  # 100x100 ROI


def _make_textured_frame(offset_xy: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Build a deterministic textured grayscale frame.

    A high-frequency speckle pattern fills the whole frame so LK has
    corners to lock onto both inside the hand ROI and in the background.
    The hand patch is shifted by ``offset_xy`` pixels relative to the
    background to simulate isolated hand motion.
    """
    rng = np.random.default_rng(42)
    base = rng.integers(40, 215, size=(FRAME_H, FRAME_W), dtype=np.uint8)
    # Hand patch is a brighter speckle layer placed inside HAND_BOX.
    hand_rng = np.random.default_rng(7)
    hand_patch = hand_rng.integers(
        50, 230, size=(HAND_BOX[3] - HAND_BOX[1], HAND_BOX[2] - HAND_BOX[0]), dtype=np.uint8
    )
    frame = base.copy()
    # Subpixel offset via cv2.warpAffine for the hand patch only.
    import cv2

    H = np.array([[1.0, 0.0, offset_xy[0]], [0.0, 1.0, offset_xy[1]]], dtype=np.float32)
    shifted = cv2.warpAffine(
        hand_patch, H, (hand_patch.shape[1], hand_patch.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    frame[HAND_BOX[1]:HAND_BOX[3], HAND_BOX[0]:HAND_BOX[2]] = shifted
    return frame


# Palm-body axes for the synthetic frames: aligned with the image axes,
# so right_axis = (1,0), up_axis = (0,1). Body scale ~= 1/3 frame width
# (a plausible palm-length-in-pixels for a 320-wide capture).
RIGHT_AXIS = np.array([1.0, 0.0])
UP_AXIS = np.array([0.0, 1.0])
BODY_SCALE_PX = float(FRAME_W) / 3.0


class TestHandROIFlowSampler(unittest.TestCase):
    def test_pure_static_scene_yields_near_zero_motion(self):
        sampler = HandROIFlowSampler()
        gray = _make_textured_frame((0.0, 0.0))
        # Prime
        sampler.update(gray, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                       up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        # Second identical frame
        result = sampler.update(gray, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                                up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        self.assertIsNotNone(result)
        self.assertLess(abs(result["body_dx"]), 1e-2)
        self.assertLess(abs(result["body_dy"]), 1e-2)
        self.assertGreaterEqual(result["valid_points"], 6)

    def test_subpixel_hand_translation_is_detected(self):
        sampler = HandROIFlowSampler()
        # Prime on neutral frame
        sampler.update(_make_textured_frame((0.0, 0.0)),
                       hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                       up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        # Shift hand patch +2 px in x
        result = sampler.update(_make_textured_frame((2.0, 0.0)),
                                hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                                up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        self.assertIsNotNone(result)
        # body_dx is in palm-body units (px / body_scale_px). +2 px on a
        # ~107 px palm == ~0.0187.
        expected = 2.0 / BODY_SCALE_PX
        self.assertAlmostEqual(result["body_dx"], expected, delta=expected * 0.5)
        self.assertLess(abs(result["body_dy"]), expected)

    def test_five_hz_oscillation_inside_roi_is_recoverable_as_tremor(self):
        sampler = HandROIFlowSampler()
        fs = 60.0
        duration = 4.0
        t = np.arange(0.0, duration, 1.0 / fs)
        # Smooth 5 Hz oscillation in x, sub-pixel amplitude (1.0 px).
        amp_px = 1.0
        offsets = amp_px * np.sin(2.0 * np.pi * 5.0 * t)

        # Prime
        sampler.update(_make_textured_frame((float(offsets[0]), 0.0)),
                       hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                       up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        body_dx = []
        body_dy = []
        timestamps = []
        for i, off in enumerate(offsets[1:], start=1):
            frame = _make_textured_frame((float(off), 0.0))
            res = sampler.update(frame, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                                 up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
            if res is None:
                continue
            body_dx.append(res["body_dx"])
            body_dy.append(res["body_dy"])
            timestamps.append(t[i])

        ts = np.asarray(timestamps)
        dx = np.asarray(body_dx)
        dy = np.asarray(body_dy)
        # The flow signal is per-frame displacement (a derivative of
        # position); integrate to recover position-like trajectory the
        # tremor analyzer expects.
        pos_x = np.cumsum(dx)
        pos_y = np.cumsum(dy)

        result = analyze_tremor_motion(
            ts, pos_x, pos_y,
            config=TremorAnalysisConfig(),
            valid_points=20,
            track_survival_rate=0.9,
            roi_visibility=1.0,
        )
        # We accept either label as long as the peak frequency lands at 5 Hz.
        self.assertIn(result.label, {"Likely rhythmic tremor", "Possible rhythmic tremor"})
        self.assertAlmostEqual(result.peak_frequency_hz, 5.0, delta=0.75)

    def test_empty_roi_returns_none_or_low_quality(self):
        sampler = HandROIFlowSampler()
        flat = np.full((FRAME_H, FRAME_W), 128, dtype=np.uint8)
        sampler.update(flat, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                       up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        result = sampler.update(flat, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                                up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        # Either nothing usable came back, or quality is very low.
        if result is not None:
            self.assertLess(result["flow_quality"], 0.4)

    def test_missing_axes_resets_cleanly(self):
        sampler = HandROIFlowSampler()
        gray = _make_textured_frame((0.0, 0.0))
        sampler.update(gray, hand_box_px=HAND_BOX, right_axis=RIGHT_AXIS,
                       up_axis=UP_AXIS, body_scale_px=BODY_SCALE_PX)
        # Now hand is lost -> sampler should reset, return None.
        out = sampler.update(gray, hand_box_px=None, right_axis=None,
                             up_axis=None, body_scale_px=BODY_SCALE_PX)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
