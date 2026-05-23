from __future__ import annotations

import unittest

import numpy as np

from motionbloom.signal import (
    MotionClassification,
    analyze_palm_relative_fingertips,
    compute_metrics,
    compute_palm_center_motion_gate,
)
from motionbloom.tracker import compute_palm_relative_landmarks


def _synthetic_open_hand() -> np.ndarray:
    xy = np.zeros((21, 2), dtype=np.float64)
    xy[0] = (0.50, 0.70)
    xy[5] = (0.43, 0.55)
    xy[9] = (0.50, 0.50)
    xy[13] = (0.57, 0.55)
    xy[17] = (0.63, 0.62)
    xy[4] = (0.34, 0.47)
    xy[8] = (0.40, 0.30)
    xy[12] = (0.50, 0.26)
    xy[16] = (0.60, 0.31)
    xy[20] = (0.70, 0.42)
    for idx in range(21):
        if not np.any(xy[idx]):
            xy[idx] = xy[0]
    return xy


def _relative_signal_from_frames(frames: list[np.ndarray], name: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for frame in frames:
        result = compute_palm_relative_landmarks(frame)
        assert result is not None
        rel_x, rel_y = result["relative"][name]
        xs.append(rel_x)
        ys.append(rel_y)
    return np.asarray(xs), np.asarray(ys)


class TestPalmMotionGate(unittest.TestCase):
    def test_steady_palm_allows_tremor_analysis(self):
        t = np.linspace(0.0, 1.5, 45)
        palm_x = 0.5 + 0.002 * np.sin(2.0 * np.pi * 5.0 * t)
        palm_y = 0.5 + 0.002 * np.cos(2.0 * np.pi * 5.0 * t)
        box_size = np.full_like(t, 0.22)

        gate = compute_palm_center_motion_gate(palm_x, palm_y, box_size)

        self.assertEqual(gate["state"], "steady")
        self.assertTrue(gate["allow_tremor"])
        self.assertFalse(gate["pause_tremor"])
        self.assertLess(gate["gross_motion_ratio"], 0.15)

    def test_moderate_palm_motion_is_reported_as_diagnostic(self):
        palm_x = np.linspace(0.5, 0.56, 45)
        palm_y = np.full_like(palm_x, 0.5)
        box_size = np.full_like(palm_x, 0.22)

        gate = compute_palm_center_motion_gate(palm_x, palm_y, box_size)

        self.assertEqual(gate["state"], MotionClassification.MOTION_TOO_HIGH_FOR_TREMOR)
        self.assertFalse(gate["allow_tremor"])
        self.assertFalse(gate["pause_tremor"])
        self.assertGreaterEqual(gate["gross_motion_ratio"], 0.15)
        self.assertLessEqual(gate["gross_motion_ratio"], 0.35)

    def test_wide_range_palm_motion_is_reported_as_diagnostic(self):
        palm_x = np.linspace(0.25, 0.45, 45)
        palm_y = np.full_like(palm_x, 0.5)
        box_size = np.full_like(palm_x, 0.22)

        gate = compute_palm_center_motion_gate(palm_x, palm_y, box_size)

        self.assertEqual(gate["state"], MotionClassification.WIDE_RANGE_HAND_MOVEMENT)
        self.assertFalse(gate["allow_tremor"])
        self.assertTrue(gate["pause_tremor"])
        self.assertGreater(gate["gross_motion_ratio"], 0.35)

    def test_cross_screen_palm_motion_pauses_tremor_analysis(self):
        palm_x = np.linspace(0.20, 0.50, 45)
        palm_y = np.full_like(palm_x, 0.5)
        box_size = np.full_like(palm_x, 0.22)

        gate = compute_palm_center_motion_gate(palm_x, palm_y, box_size)

        self.assertEqual(gate["state"], MotionClassification.WIDE_RANGE_HAND_MOVEMENT)
        self.assertTrue(gate["pause_tremor"])
        self.assertGreater(gate["net_screen_displacement_ratio"], 0.05)
        self.assertIn("Too much hand movement", gate["reason"])

    def test_more_than_half_hand_diagonal_pauses_tremor_analysis(self):
        palm_x = np.linspace(0.50, 0.60, 45)
        palm_y = np.full_like(palm_x, 0.5)
        box_size = np.full_like(palm_x, 0.15)

        gate = compute_palm_center_motion_gate(palm_x, palm_y, box_size)

        self.assertEqual(gate["state"], MotionClassification.WIDE_RANGE_HAND_MOVEMENT)
        self.assertTrue(gate["pause_tremor"])
        self.assertGreater(gate["hand_relative_travel"], 0.50)

    def test_wide_moving_palm_vetoes_palm_relative_tremor_score(self):
        fs = 60.0
        t = np.arange(0.0, 1.5, 1.0 / fs)
        tremor = 0.012 * np.sin(2.0 * np.pi * 5.0 * t)
        y = np.zeros_like(tremor)
        moving_palm_x = np.linspace(0.25, 0.55, t.size)
        moving_palm_y = np.full_like(t, 0.5)
        hand_box_size = np.full_like(t, 0.22)
        relative_signals = {
            "index_tip": (tremor.copy(), y.copy()),
            "middle_tip": (tremor.copy(), y.copy()),
            "ring_tip": (tremor.copy(), y.copy()),
        }

        metrics = compute_metrics(
            tremor,
            y,
            fs,
            local_xu=tremor,
            local_yu=y,
            global_xu=moving_palm_x,
            global_yu=moving_palm_y,
            palm_center_x=moving_palm_x,
            palm_center_y=moving_palm_y,
            hand_box_size=hand_box_size,
            relative_fingertip_signals=relative_signals,
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.motion_classification, MotionClassification.WIDE_RANGE_HAND_MOVEMENT)
        self.assertIsNone(metrics.final_tremor_score)
        self.assertFalse(metrics.research_valid)
        self.assertTrue(metrics.tremor_analysis_paused)
        self.assertEqual(metrics.palm_motion_state, MotionClassification.WIDE_RANGE_HAND_MOVEMENT)

    def test_large_rhythmic_waving_does_not_return_valid_tremor(self):
        fs = 60.0
        t = np.arange(0.0, 1.5, 1.0 / fs)
        waving = 0.16 * np.sin(2.0 * np.pi * 4.0 * t)
        y = np.full_like(t, 0.5)
        palm_x = 0.5 + waving
        hand_box_size = np.full_like(t, 0.22)

        metrics = compute_metrics(
            palm_x,
            y,
            fs,
            local_xu=palm_x,
            local_yu=y,
            global_xu=palm_x,
            global_yu=y,
            palm_center_x=palm_x,
            palm_center_y=y,
            hand_box_size=hand_box_size,
        )

        self.assertIsNotNone(metrics)
        self.assertNotEqual(metrics.motion_classification, MotionClassification.VALID_TREMOR)
        self.assertIsNone(metrics.final_tremor_score)
        self.assertTrue(metrics.tremor_analysis_paused)

    def test_small_steady_fingertip_tremor_can_return_valid_tremor(self):
        fs = 60.0
        t = np.arange(0.0, 1.5, 1.0 / fs)
        tremor = 0.010 * np.sin(2.0 * np.pi * 5.0 * t)
        y = np.zeros_like(tremor)
        palm_x = 0.5 + 0.001 * np.sin(2.0 * np.pi * 1.0 * t)
        palm_y = np.full_like(t, 0.5)
        hand_box_size = np.full_like(t, 0.22)
        relative_signals = {
            "index_tip": (tremor.copy(), y.copy()),
            "middle_tip": (tremor.copy(), y.copy()),
            "ring_tip": (tremor.copy(), y.copy()),
        }

        metrics = compute_metrics(
            tremor,
            y,
            fs,
            local_xu=tremor,
            local_yu=y,
            global_xu=palm_x,
            global_yu=palm_y,
            palm_center_x=palm_x,
            palm_center_y=palm_y,
            hand_box_size=hand_box_size,
            relative_fingertip_signals=relative_signals,
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.motion_classification, MotionClassification.VALID_TREMOR)
        self.assertIsNotNone(metrics.final_tremor_score)
        self.assertFalse(metrics.tremor_analysis_paused)
        self.assertLess(metrics.tremor_amp_ratio, 0.12)

    def test_whole_hand_translation_cancels_in_palm_relative_coordinates(self):
        base = _synthetic_open_hand()
        translated = base + np.array([0.08, -0.04])

        rel_base = compute_palm_relative_landmarks(base)
        rel_translated = compute_palm_relative_landmarks(translated)

        self.assertIsNotNone(rel_base)
        self.assertIsNotNone(rel_translated)
        for name in ("index_tip", "middle_tip", "ring_tip"):
            np.testing.assert_allclose(
                rel_base["relative"][name],
                rel_translated["relative"][name],
                atol=1e-12,
            )

    def test_fingertip_tremor_relative_to_palm_is_preserved(self):
        fs = 60.0
        t = np.arange(0.0, 1.5, 1.0 / fs)
        base = _synthetic_open_hand()
        frames = []
        for ti in t:
            frame = base.copy()
            oscillation = 0.012 * np.sin(2.0 * np.pi * 5.0 * ti)
            for idx in (8, 12, 16):
                frame[idx, 0] += oscillation
            frames.append(frame)

        signals = {
            "index_tip": _relative_signal_from_frames(frames, "index_tip"),
            "middle_tip": _relative_signal_from_frames(frames, "middle_tip"),
            "ring_tip": _relative_signal_from_frames(frames, "ring_tip"),
        }
        result = analyze_palm_relative_fingertips(signals, fs)

        self.assertEqual(result["classification"], "likely")
        self.assertGreaterEqual(result["agreement_count"], 2)
        self.assertAlmostEqual(result["median_peak_hz"], 5.0, delta=1.0)

    def test_slow_hand_rotation_does_not_become_likely_tremor(self):
        fs = 60.0
        t = np.arange(0.0, 1.5, 1.0 / fs)
        base = _synthetic_open_hand()
        palm = base[[0, 5, 9, 13, 17]].mean(axis=0)
        frames = []
        for ti in t:
            angle = 0.20 * np.sin(2.0 * np.pi * 0.7 * ti)
            rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            frames.append((base - palm) @ rot.T + palm)

        signals = {
            "index_tip": _relative_signal_from_frames(frames, "index_tip"),
            "middle_tip": _relative_signal_from_frames(frames, "middle_tip"),
            "ring_tip": _relative_signal_from_frames(frames, "ring_tip"),
        }
        result = analyze_palm_relative_fingertips(signals, fs)

        self.assertNotEqual(result["classification"], "likely")


if __name__ == "__main__":
    unittest.main(verbosity=2)
