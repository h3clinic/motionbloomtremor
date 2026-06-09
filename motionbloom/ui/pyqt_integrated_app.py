"""
MotionBloom PyQt6 Application with FULL Tremor Backend Integration.

Uses the same TremorAnalysisEngine that mirrors the legacy Tkinter
`App._refresh_analysis` exactly:
  - TremorTracker (MediaPipe + optical flow, its own thread)
  - palm-center motion gate + state machine
  - optical-flow gross-motion gate
  - assess_trial_quality + resample_uniform
  - compute_metrics (same call, same params)
  - Adaptive baseline RMS

Light white/red MotionBloom theme preserved.
"""

import sys
import time
import platform
from pathlib import Path
from typing import Optional
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFrame, QLabel, QMessageBox, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage

import cv2
import numpy as np

from . import pyqt_theme as theme
from .pyqt_components import (
    DuoHeroCard, DuoMetricsCard, DuoVerdictCard, DuoButton,
)

# === SAME BACKEND AS LEGACY APP ===
from motionbloom._cv_lock import CV_LOCK  # noqa: F401
from motionbloom.tracker import (
    TremorTracker, CAM_WIDTH, CAM_HEIGHT, LANDMARK_CHOICES,
)
from motionbloom.signal import TaskMode
from motionbloom.analysis_engine import TremorAnalysisEngine, AnalysisResult


VIDEO_MS = 33      # ~30 fps display
ANALYSIS_MS = 200  # 5 Hz analysis (same as legacy)
HISTORY_MAX = 300


class MotionBloomMainWindow(QMainWindow):
    """Main window with full tremor backend via TremorAnalysisEngine."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MotionBloom — Hand check")
        self.setGeometry(100, 100, 1500, 950)
        self.setStyleSheet(theme.generate_qss())

        # === BACKEND ===
        self.tracker = TremorTracker()
        self.tracker.set_landmark(LANDMARK_CHOICES.get("Index fingertip", 8))
        self.engine = TremorAnalysisEngine(task_mode=TaskMode.POSTURAL_GENERAL)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._t0 = 0.0
        self.hist_score: deque = deque(maxlen=HISTORY_MAX)
        self.hist_peak: deque = deque(maxlen=HISTORY_MAX)
        self.hist_amp: deque = deque(maxlen=HISTORY_MAX)

        # === UI ===
        self._build_ui()

        # === TIMERS (mirror legacy root.after loops) ===
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._video_loop)
        self.video_timer.start(VIDEO_MS)

        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self._analysis_loop)
        self.analysis_timer.start(ANALYSIS_MS)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central.setLayout(layout)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_tabs(), 1)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setStyleSheet(
            f"QFrame {{ background-color: white; "
            f"border-bottom: 3px solid {theme.PRIMARY}; }}"
        )
        header.setFixedHeight(85)

        layout = QHBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_MD,
                                  theme.PADDING_LG, theme.PADDING_MD)
        layout.setSpacing(theme.PADDING_LG)

        # Try to load the bright red/white MotionBloom logo
        logo_path = Path(__file__).parent.parent / "assets" / "motionbloom_logo.png"
        if logo_path.exists():
            logo_pix = QPixmap(str(logo_path)).scaledToHeight(
                65, Qt.TransformationMode.SmoothTransformation
            )
            logo_label = QLabel()
            logo_label.setPixmap(logo_pix)
        else:
            logo_label = QLabel("🌸")
            logo_label.setStyleSheet(f"font-size: {theme.FONT_SIZE_2XL}px;")

        title = QLabel("MotionBloom")
        title.setStyleSheet(
            f"color: {theme.PRIMARY}; font-weight: bold; "
            f"font-size: {theme.FONT_SIZE_LG}px;"
        )
        subtitle = QLabel("Check")
        subtitle.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.FONT_SIZE_SM}px;"
        )

        brand_v = QVBoxLayout()
        brand_v.setContentsMargins(0, 0, 0, 0)
        brand_v.setSpacing(0)
        brand_v.addWidget(title)
        brand_v.addWidget(subtitle)

        brand_h = QHBoxLayout()
        brand_h.setSpacing(theme.PADDING_MD)
        brand_h.addWidget(logo_label)
        brand_h.addLayout(brand_v)
        layout.addLayout(brand_h)
        layout.addStretch()

        self.status_label = QLabel("✓ Ready")
        self.status_label.setStyleSheet(
            f"color: {theme.COLOR_SUCCESS}; font-weight: bold; "
            f"font-size: {theme.FONT_SIZE_SM}px;"
        )
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.FONT_SIZE_SM}px;"
        )
        layout.addWidget(self.fps_label)
        layout.addWidget(self.status_label)

        header.setLayout(layout)
        return header

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_main_tab(), "MAIN")
        tabs.addTab(self._build_reports_tab(), "Reports")
        return tabs

    def _build_main_tab(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                   theme.PADDING_LG, theme.PADDING_LG)
        layout.setSpacing(theme.PADDING_LG)

        # Left: camera
        left = QVBoxLayout()
        left.setSpacing(theme.PADDING_MD)

        cam_label = QLabel("CAMERA")
        cam_label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; "
            f"font-size: {theme.FONT_SIZE_SM}px; font-weight: bold;"
        )

        self.camera_frame = QFrame()
        self.camera_frame.setStyleSheet(
            f"QFrame {{ background-color: #000000; "
            f"border: 3px solid {theme.PRIMARY}; "
            f"border-radius: {theme.CORNER_RADIUS_LG}px; }}"
        )
        self.camera_frame.setMinimumHeight(560)
        cam_v = QVBoxLayout()
        cam_v.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: #000000; color: white; "
            f"font-size: {theme.FONT_SIZE_LG}px; font-weight: bold;"
        )
        cam_v.addWidget(self.video_label)
        self.camera_frame.setLayout(cam_v)

        # Source / flow labels below the camera
        self.source_label = QLabel("")
        self.source_label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; "
            f"font-size: {theme.FONT_SIZE_SM}px;"
        )
        self.flow_label = QLabel("")
        self.flow_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; "
            f"font-size: {theme.FONT_SIZE_SM}px;"
        )

        left.addWidget(cam_label)
        left.addWidget(self.camera_frame, 1)
        left.addWidget(self.source_label)
        left.addWidget(self.flow_label)

        # Right: cards
        right = QVBoxLayout()
        right.setSpacing(theme.PADDING_MD)

        self.hero_card = DuoHeroCard()
        self.hero_card.set_mascot_emoji("👋")
        self.hero_card.set_title("Score: —")
        self.hero_card.set_subtitle("Position hand")
        right.addWidget(self.hero_card)

        # Metrics card — scrollable with only key metrics
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background-color: white; }}"
        )
        
        metrics_container = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(0)
        
        self.metrics_card = DuoMetricsCard()
        self.metrics_card.add_metric("live", "Motion", "—")
        self.metrics_card.add_metric("final", "Score", "—")
        self.metrics_card.add_metric("confidence", "Conf", "—")
        self.metrics_card.add_metric("peak", "Peak", "— Hz")
        self.metrics_card.add_metric("band", "Band", "—")
        self.metrics_card.add_metric("amp_mm", "Amp", "— mm")
        self.metrics_card.add_metric("snr", "SNR", "— dB")
        self.metrics_card.add_metric("track_q", "Track", "—")
        
        metrics_layout.addWidget(self.metrics_card)
        metrics_layout.addStretch()
        metrics_container.setLayout(metrics_layout)
        metrics_scroll.setWidget(metrics_container)
        right.addWidget(metrics_scroll, 1)

        self.verdict_card = DuoVerdictCard()
        self.verdict_card.set_status(
            "Ready", status_type="info"
        )
        right.addWidget(self.verdict_card)

        self.check_button = DuoButton("✓ Check", variant="primary")
        self.check_button.setFixedHeight(theme.BUTTON_HEIGHT_LG)
        self.check_button.clicked.connect(self._on_check_hand_clicked)
        right.addWidget(self.check_button)

        layout.addLayout(left, 2)
        layout.addLayout(right, 1)
        container.setLayout(layout)
        return container

    def _build_reports_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                   theme.PADDING_LG, theme.PADDING_LG)

        title = QLabel("History")
        title.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-weight: bold; "
            f"font-size: {theme.FONT_SIZE_LG}px;"
        )
        self.history_label = QLabel("No checks")
        self.history_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.FONT_SIZE_BASE}px;"
        )
        self.history_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_label.setWordWrap(True)

        history_frame = QFrame()
        history_frame.setStyleSheet(
            f"QFrame {{ background-color: white; "
            f"border: 2px solid {theme.BORDER_COLOR}; "
            f"border-radius: {theme.CORNER_RADIUS_LG}px; }}"
        )
        hv = QVBoxLayout()
        hv.setContentsMargins(theme.PADDING_MD, theme.PADDING_MD,
                              theme.PADDING_MD, theme.PADDING_MD)
        hv.addWidget(self.history_label)
        hv.addStretch()
        history_frame.setLayout(hv)

        layout.addWidget(title)
        layout.addWidget(history_frame, 1)
        container.setLayout(layout)
        return container

    # ---------------------------------------------------------- camera control
    def _on_check_hand_clicked(self) -> None:
        if not self.is_running:
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self) -> None:
        print("[APP] Starting capture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._show_camera_error()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap = cap
        self.tracker.start(cap)

        self.engine.reset()
        self._t0 = time.time()
        self.hist_score.clear()
        self.hist_peak.clear()
        self.hist_amp.clear()
        self.is_running = True

        self.check_button.setText("⏸ Stop")
        self.status_label.setText("📊")
        self.status_label.setStyleSheet(
            f"color: {theme.PRIMARY}; font-weight: bold; "
            f"font-size: {theme.FONT_SIZE_SM}px;"
        )
        print("[APP] Capture started successfully")

    def _stop_capture(self) -> None:
        print("[APP] Stopping capture...")
        try:
            self.tracker.stop()
        except Exception as e:
            print(f"[APP] tracker.stop() error: {e}")
        self.cap = None
        self.is_running = False

        self.check_button.setText("✓ Check")
        self.status_label.setText("✓")
        self.status_label.setStyleSheet(
            f"color: {theme.COLOR_SUCCESS}; font-weight: bold; "
            f"font-size: {theme.FONT_SIZE_SM}px;"
        )
        self.video_label.clear()
        self.video_label.setText("Start")
        self.verdict_card.set_status("Stopped",
                                     status_type="info")

    def _show_camera_error(self) -> None:
        if platform.system() == "Darwin":
            msg = ("Camera not available.\n\nOn macOS, grant camera permission "
                   "in System Settings → Privacy & Security → Camera.")
        elif platform.system() == "Windows":
            msg = ("Camera not available.\n\nCheck Windows Settings → "
                   "Privacy & security → Camera.")
        else:
            msg = "Camera not available. Check connections and permissions."
        QMessageBox.critical(self, "Camera unavailable", msg)

    # ---------------------------------------------------------- timers
    @pyqtSlot()
    def _video_loop(self) -> None:
        if not self.is_running:
            return
        try:
            frame = self.tracker.get_frame()
            if frame is None:
                return
            self._render_video(frame)
            fps = getattr(self.tracker, "fps_meas", 0.0) or 0.0
            if fps > 0:
                self.fps_label.setText(f"{fps:.0f} fps")
        except Exception as e:
            print(f"[VIDEO] error: {e}")

    def _render_video(self, frame: np.ndarray) -> None:
        try:
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = frame
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            rgb_c = np.ascontiguousarray(rgb)
            qimg = QImage(rgb_c.data, w, h, w * 3,
                          QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            tw = max(self.video_label.width(), 320)
            th = max(self.video_label.height(), 240)
            scaled = pix.scaled(
                tw, th,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled)
        except Exception as e:
            print(f"[RENDER] error: {e}")

    @pyqtSlot()
    def _analysis_loop(self) -> None:
        if not self.is_running:
            return
        try:
            result = self.engine.tick(self.tracker)
            self._update_ui(result)
        except Exception as e:
            print(f"[ANALYSIS] error: {e}")
            import traceback
            traceback.print_exc()

    def _update_ui(self, r: AnalysisResult) -> None:
        # Hero
        if r.final_tremor_score is not None:
            self.hero_card.set_title(f"{r.final_tremor_score}%")
        else:
            self.hero_card.set_title(f"{r.live_score}%")
        self.hero_card.set_subtitle(
            f"{r.peak_hz:.2f} Hz · {r.mode_label}"
        )

        # Verdict
        self.verdict_card.set_status(
            r.verdict_message or r.status_message, status_type=r.verdict_type
        )

        # Status badge in header
        self.status_label.setText(f"📊")

        # Metrics - only essential ones
        m = self.metrics_card
        m.update_metric("live", f"{r.live_score}%")
        m.update_metric(
            "final",
            f"{r.final_tremor_score}%" if r.final_tremor_score is not None else "—"
        )
        m.update_metric("confidence", r.confidence_level[0].upper())
        m.update_metric("peak", f"{r.peak_hz:.2f} Hz")
        m.update_metric("band", f"{r.band_ratio * 100:.0f}%")
        m.update_metric(
            "amp_mm",
            f"{r.rms_amp_mm:.2f} mm" if r.rms_amp_mm > 0 else "—"
        )
        m.update_metric("snr", f"{r.snr_db:.1f} dB" if r.snr_db != 0 else "—")
        m.update_metric("track_q", f"{r.tracking_quality_pct}%")

        # History (only when certified)
        if r.final_tremor_score is not None:
            self.hist_score.append(r.final_tremor_score)
            self.hist_peak.append(r.peak_hz)
            self.hist_amp.append(r.rms_amp_mm)
            self.history_label.setText(self._format_history())

    def _format_history(self) -> str:
        if not self.hist_score:
            return "No checks"
        recent = list(self.hist_score)[-10:]
        avg = sum(recent) / len(recent)
        return f"Avg: {avg:.1f}% ({len(self.hist_score)} total)"

    # ---------------------------------------------------------- shutdown
    def closeEvent(self, event):
        try:
            self.video_timer.stop()
            self.analysis_timer.stop()
            if self.is_running:
                self.tracker.stop()
        except Exception:
            pass
        super().closeEvent(event)


def launch_pyqt6_app() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MotionBloomMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_pyqt6_app()
