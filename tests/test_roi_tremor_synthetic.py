from __future__ import annotations

import json
import unittest

import numpy as np

from motionbloom.analysis.tremor_signal import TremorAnalysisConfig, analyze_tremor_motion


CONFIG = TremorAnalysisConfig(
    likely_snr=3.0,
    possible_snr=1.8,
    min_band_power=1e-8,
    max_gross_to_tremor_ratio=20.0,
)


def make_timestamps(fs: float = 60.0, duration: float = 4.0, drop_every: int | None = None) -> np.ndarray:
    t = np.arange(0.0, duration, 1.0 / fs)
    if drop_every and drop_every > 1:
        keep = np.ones_like(t, dtype=bool)
        keep[::drop_every] = False
        t = t[keep]
    return t


def synthetic_motion(
    t: np.ndarray,
    *,
    tremor_hz: float = 0.0,
    tremor_amp: float = 0.0,
    gross_hz: float = 0.0,
    gross_amp: float = 0.0,
    noise_amp: float = 0.0,
    seed: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t, dtype=np.float64)
    y = np.zeros_like(t, dtype=np.float64)
    if gross_hz > 0 and gross_amp > 0:
        x += gross_amp * np.sin(2.0 * np.pi * gross_hz * t)
        y += 0.6 * gross_amp * np.cos(2.0 * np.pi * gross_hz * t)
    if tremor_hz > 0 and tremor_amp > 0:
        x += tremor_amp * np.sin(2.0 * np.pi * tremor_hz * t)
        y += 0.75 * tremor_amp * np.cos(2.0 * np.pi * tremor_hz * t)
    if noise_amp > 0:
        x += noise_amp * rng.normal(size=t.size)
        y += noise_amp * rng.normal(size=t.size)
    return x, y


def analyze_case(name: str, t: np.ndarray, x: np.ndarray, y: np.ndarray, **kwargs):
    result = analyze_tremor_motion(
        t,
        x,
        y,
        config=CONFIG,
        valid_points=kwargs.pop("valid_points", 32),
        track_survival_rate=kwargs.pop("track_survival_rate", 0.88),
        roi_visibility=kwargs.pop("roi_visibility", 1.0),
        **kwargs,
    )
    print(f"\n[SYNTHETIC] {name}")
    print(json.dumps(result.as_diagnostic_dict(), indent=2))
    return result


class TestROITremorSynthetic(unittest.TestCase):
    def test_gross_motion_without_tremor_is_not_tremor(self):
        t = make_timestamps()
        x, y = synthetic_motion(t, gross_hz=1.0, gross_amp=0.03)
        result = analyze_case("0Hz tremor + 1Hz gross motion", t, x, y)
        self.assertEqual(result.label, "No tremor detected")

    def test_five_hz_tremor_without_gross_motion_is_likely(self):
        t = make_timestamps()
        x, y = synthetic_motion(t, tremor_hz=5.0, tremor_amp=0.04)
        result = analyze_case("5Hz tremor + no gross motion", t, x, y)
        self.assertEqual(result.label, "Likely rhythmic tremor")
        self.assertAlmostEqual(result.peak_frequency_hz, 5.0, delta=0.35)

    def test_five_hz_tremor_with_one_hz_gross_motion_is_likely_if_snr_holds(self):
        t = make_timestamps()
        x, y = synthetic_motion(t, tremor_hz=5.0, tremor_amp=0.05, gross_hz=1.0, gross_amp=0.012)
        result = analyze_case("5Hz tremor + 1Hz gross motion", t, x, y)
        self.assertEqual(result.label, "Likely rhythmic tremor")
        self.assertAlmostEqual(result.peak_frequency_hz, 5.0, delta=0.35)

    def test_random_jitter_noise_is_unreliable_or_no_tremor(self):
        t = make_timestamps()
        x, y = synthetic_motion(t, noise_amp=0.02)
        result = analyze_case("random jitter/noise", t, x, y, track_survival_rate=0.42)
        self.assertIn(result.label, {"No reliable tremor signal", "No tremor detected"})

    def test_low_fps_is_unusable(self):
        t = make_timestamps(fs=20.0)
        x, y = synthetic_motion(t, tremor_hz=5.0, tremor_amp=0.04)
        result = analyze_case("low FPS", t, x, y)
        self.assertEqual(result.label, "Unusable recording")

    def test_global_camera_jitter_removed_by_background_signal_is_not_likely(self):
        t = make_timestamps()
        global_x, global_y = synthetic_motion(t, tremor_hz=5.0, tremor_amp=0.04)
        hand_x = global_x.copy()
        hand_y = global_y.copy()
        result = analyze_case(
            "5Hz global/camera jitter equally on hand and background",
            t,
            hand_x,
            hand_y,
            background_motion_x=global_x,
            background_motion_y=global_y,
        )
        self.assertNotEqual(result.label, "Likely rhythmic tremor")
        self.assertIn(result.label, {"No tremor detected", "No reliable tremor signal"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
