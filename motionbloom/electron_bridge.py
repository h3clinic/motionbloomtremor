"""Headless bridge for Electron UI.

Streams JSON lines to stdout with status and metric updates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Remove this script's own directory from sys.path[0] if present, so that
# `import signal` (and other stdlib names) never get shadowed by sibling
# modules such as motionbloom/signal.py when run as a script file.
_here = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] in (_here, ""):
    sys.path.pop(0)

import signal

# Add repo root to sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from motionbloom.analysis_engine import TaskMode, TremorAnalysisEngine
from motionbloom.tracker import CAM_HEIGHT, CAM_WIDTH, TremorTracker


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def format_metrics(result) -> dict:
    return {
        "live_score": int(result.live_score),
        "final_score": int(result.final_tremor_score)
        if result.final_tremor_score is not None
        else None,
        "confidence": str(result.confidence_level),
        "peak_hz": f"{result.peak_hz:.2f} Hz" if result.peak_hz else "—",
        "band_ratio": f"{result.band_ratio * 100:.0f}%" if result.band_ratio else "—",
        "amp_mm": f"{result.rms_amp_mm:.2f} mm" if result.rms_amp_mm else "—",
        "snr_db": f"{result.snr_db:.1f} dB" if result.snr_db else "—",
        "tracking_quality": f"{result.tracking_quality_pct}%",
    }


def run_once_dry() -> int:
    emit(
        {
            "type": "status",
            "running": False,
            "message": "Dry run ok: bridge executable and JSON stream healthy",
        }
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MotionBloom Electron bridge")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--analysis-interval", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        return run_once_dry()

    running = True

    def handle_signal(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        emit(
            {
                "type": "status",
                "running": False,
                "message": "Camera open failed",
            }
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    tracker = TremorTracker()
    engine = TremorAnalysisEngine(task_mode=TaskMode.POSTURAL_GENERAL)
    engine.reset()
    tracker.start(cap)

    emit({"type": "status", "running": True, "message": "Analysis started"})

    try:
        while running:
            result = engine.tick(tracker)
            emit(
                {
                    "type": "metrics",
                    "metrics": format_metrics(result),
                    "status_message": result.status_message,
                }
            )
            time.sleep(max(args.analysis_interval, 0.05))
    finally:
        try:
            tracker.stop()
        except Exception:
            pass
        emit({"type": "status", "running": False, "message": "Analysis stopped"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
