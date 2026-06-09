"""Tests for the optical-flow → tremor-scoring integration in app.py.

These tests exercise:
  1. The pure `select_tracking_source` decision logic.
  2. UI publication of the tracking source / flow status StringVars.
  3. End-to-end: feed a 5 Hz flow trajectory + matching palm-relative
     samples into TremorTracker's buffers, run App._refresh_analysis,
     and confirm the optical-flow branch is taken.
  4. Graceful fallback when flow_status is None.
"""

from __future__ import annotations

import os
import time
import unittest
from collections import deque

import numpy as np

# Run Tk headlessly on macOS by ensuring DISPLAY var is fine; Tk on
# macOS uses the native windowing system, so withdraw() is the trick.
os.environ.setdefault("MOTIONBLOOM_QUIET", "1")

from motionbloom.app import (  # noqa: E402
    FLOW_MIN_QUALITY,
    FLOW_MIN_VALID_POINTS,
    FLOW_MAX_PER_FRAME_DISPLACEMENT,
    FLOW_GROSS_FRAME_FRACTION_MAX,
    gate_flow_for_microtremor,
    select_tracking_source,
)


class TestSelectTrackingSource(unittest.TestCase):
    def test_high_quality_flow_wins(self):
        snap = (np.zeros(20),) * 8  # tuple of 8 arrays, mocked
        status = {"usable": True, "flow_quality": 0.7, "valid_points": 25}
        src, reason = select_tracking_source(snap, status, palm_relative_available=True)
        self.assertEqual(src, "optical_flow")
        self.assertIn("Optical flow", reason)

    def test_low_quality_flow_falls_back_to_landmark(self):
        snap = (np.zeros(20),) * 8
        status = {"usable": True, "flow_quality": 0.25, "valid_points": 25}
        src, reason = select_tracking_source(snap, status, palm_relative_available=True)
        self.assertEqual(src, "mediapipe_palm_body")
        self.assertIn("fallback", reason.lower())

    def test_few_points_falls_back(self):
        snap = (np.zeros(20),) * 8
        status = {"usable": True, "flow_quality": 0.9, "valid_points": 4}
        src, reason = select_tracking_source(snap, status, palm_relative_available=True)
        self.assertEqual(src, "mediapipe_palm_body")

    def test_no_flow_snapshot_uses_landmark(self):
        src, reason = select_tracking_source(None, None, palm_relative_available=True)
        self.assertEqual(src, "mediapipe_palm_body")

    def test_no_flow_and_no_palm_returns_fallback(self):
        src, reason = select_tracking_source(None, None, palm_relative_available=False)
        self.assertEqual(src, "fallback")

    def test_thresholds_match_constants(self):
        self.assertEqual(FLOW_MIN_QUALITY, 0.40)
        self.assertEqual(FLOW_MIN_VALID_POINTS, 10)


def _make_app():
    """Build a withdrawn App for headless UI tests."""
    from tkinter import Tk
    from motionbloom.app import App
    root = Tk()
    root.withdraw()
    app = App(root)
    app._stop_startup_intro()
    # Don't actually capture from the camera during tests.
    if app.tracker.thread:
        app.tracker.stop()
    return root, app


def _seed_palm_relative(tracker, n: int, fs: float = 30.0):
    """Inject n synthetic palm-relative samples ending now."""
    t0 = time.time() - n / fs
    for i in range(n):
        t = t0 + i / fs
        tracker.palm_relative_samples.append((
            t,
            0.01, 0.02,   # index_tip
            0.00, 0.03,   # middle_tip
            -0.01, 0.02,  # ring_tip
            -0.02, 0.00,  # thumb_tip
            0.02, -0.01,  # pinky_tip
            -0.005, 0.005,  # index_mcp
            0.005, 0.005,   # middle_mcp
            0.5, 0.5,    # palm xy (normalised image coords)
            0.20,        # hand_size
            0.95,        # tracking_quality
        ))
    # palm_center stream so the gate doesn't pause us
    for i in range(n):
        t = t0 + i / fs
        tracker.palm_center_samples.append((
            t, 0.5, 0.5, 0.10, 0.12, 0.156, 0.95,
        ))


def _seed_flow(tracker, n: int, *, hz: float, amp: float, fs: float = 30.0,
               quality: float = 0.85, points: int = 30):
    """Inject n synthetic optical-flow samples with a `hz`-Hz oscillation."""
    t0 = time.time() - n / fs
    ts = np.array([t0 + i / fs for i in range(n)], dtype=np.float64)
    # Frame-to-frame displacement of a sinusoid at hz with amplitude amp.
    pos = amp * np.sin(2.0 * np.pi * hz * ts)
    dx = np.diff(pos, prepend=pos[0])
    for i in range(n):
        tracker.flow_samples.append((
            float(ts[i]), float(dx[i]), float(dx[i]) * 0.5,
            float(points), 0.9, float(quality),
            0.0, 0.0,
        ))
    tracker._flow_status = {
        "usable": True,
        "flow_quality": quality,
        "valid_points": points,
        "survival_rate": 0.9,
        "bg_subtracted": True,
        "body_dx": float(dx[-1]),
        "body_dy": float(dx[-1] * 0.5),
        "body_dx_px": 0.0,
        "body_dy_px": 0.0,
        "hand_dx_px": 0.0,
        "hand_dy_px": 0.0,
        "bg_dx_px": 0.0,
        "bg_dy_px": 0.0,
        "total_points": points,
        "mad_px": 0.5,
        "body_scale_px": 100.0,
    }


class TestAppFlowIntegration(unittest.TestCase):
    def test_ui_publishes_optical_flow_when_quality_high(self):
        root, app = _make_app()
        try:
            _seed_palm_relative(app.tracker, n=60)
            _seed_flow(app.tracker, n=60, hz=5.0, amp=0.02, quality=0.85, points=30)
            # Run the analysis tick directly (don't reschedule).
            app._refresh_analysis()
            src = app.tracking_source_var.get()
            flow_line = app.flow_status_var.get()
            self.assertIn("optical_flow", src)
            self.assertIn("Flow:", flow_line)
            self.assertIn("pts=30", flow_line)
        finally:
            root.destroy()

    def test_ui_publishes_landmark_fallback_when_quality_low(self):
        root, app = _make_app()
        try:
            _seed_palm_relative(app.tracker, n=60)
            _seed_flow(app.tracker, n=60, hz=5.0, amp=0.02, quality=0.20, points=30)
            app._refresh_analysis()
            src = app.tracking_source_var.get()
            self.assertIn("mediapipe_palm_body", src)
        finally:
            root.destroy()

    def test_ui_handles_missing_flow_status_without_crash(self):
        root, app = _make_app()
        try:
            _seed_palm_relative(app.tracker, n=60)
            # No flow seeded -- _flow_status stays None
            app._refresh_analysis()
            self.assertIn("mediapipe_palm_body", app.tracking_source_var.get())
            self.assertIn("warming up", app.flow_status_var.get().lower())
        finally:
            root.destroy()


class TestPhysiologicalFlowGate(unittest.TestCase):
    def test_micro_amplitude_5hz_allowed(self):
        fs = 30.0
        t = np.arange(0, 1.5, 1.0 / fs)
        # 0.01 palm-length amplitude oscillation, peak per-frame
        # displacement at 5 Hz is ~0.01 * 2*pi*5/30 ~= 0.01 -> well
        # below the 0.08 gate.
        pos = 0.01 * np.sin(2 * np.pi * 5.0 * t)
        dx = np.diff(pos, prepend=pos[0])
        dy = np.zeros_like(dx)
        allow, reason, gross, cdx, cdy = gate_flow_for_microtremor(dx, dy)
        self.assertTrue(allow, reason)
        self.assertLess(gross, 0.05)

    def test_whole_arm_wave_rejected(self):
        fs = 30.0
        t = np.arange(0, 1.5, 1.0 / fs)
        # 0.3 palm-length amplitude at 2 Hz -> peak per-frame
        # displacement ~0.3 * 2*pi*2/30 ~= 0.126, well over 0.08.
        pos = 0.3 * np.sin(2 * np.pi * 2.0 * t)
        dx = np.diff(pos, prepend=pos[0])
        dy = np.zeros_like(dx)
        allow, reason, gross, _cdx, _cdy = gate_flow_for_microtremor(dx, dy)
        self.assertFalse(allow)
        self.assertIn("Too much hand movement", reason)
        self.assertGreater(gross, FLOW_GROSS_FRAME_FRACTION_MAX)

    def test_single_spike_clipped_but_window_still_allowed(self):
        dx = np.zeros(60)
        dy = np.zeros(60)
        dx[30] = 1.5  # one massive frame, e.g., a re-detect snap
        allow, reason, gross, cdx, cdy = gate_flow_for_microtremor(dx, dy)
        # 1/60 gross frames < 20%, so window is allowed but the spike
        # must be clipped to prevent poisoning the PSD.
        self.assertTrue(allow)
        self.assertLessEqual(float(np.abs(cdx).max()), FLOW_MAX_PER_FRAME_DISPLACEMENT + 1e-12)

    def test_empty_arrays_reject(self):
        allow, reason, gross, _cdx, _cdy = gate_flow_for_microtremor(
            np.array([]), np.array([])
        )
        self.assertFalse(allow)

    def test_gate_runs_inside_app_and_pauses_score_on_huge_flow(self):
        root, app = _make_app()
        try:
            _seed_palm_relative(app.tracker, n=60)
            # Seed a high-amplitude flow that fails the physiological gate.
            _seed_flow(app.tracker, n=60, hz=2.0, amp=0.3, quality=0.85, points=30)
            metrics = app._refresh_analysis()
            # Source was selected as optical_flow but gate vetoed scoring.
            self.assertIn("optical_flow", app.tracking_source_var.get())
            self.assertIsNone(metrics)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
