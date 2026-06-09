"""
MotionBloom PyQt6 Application
Light white/red theme with premium card UI and animations
"""

import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFrame, QLabel, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QIcon
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from . import pyqt_theme as theme
from .pyqt_components import (
    DuoCard, DuoHeroCard, DuoMetricsCard, DuoVerdictCard, DuoButton
)


class MotionBloomMainWindow(QMainWindow):
    """Main application window with Duolingo-style UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MotionBloom — Simple hand check")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply global stylesheet
        self.setStyleSheet(theme.generate_qss())
        
        # Build UI
        self._build_ui()
        
        # State tracking
        self.is_checking = False
        self.current_score = 0
        self.streak_count = 7
        self.hand_present = False
    
    def _build_ui(self) -> None:
        """Build the complete UI structure."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Header bar
        header = self._build_header()
        main_layout.addWidget(header)
        
        # Content area with tabs
        tab_widget = self._build_tabs()
        main_layout.addWidget(tab_widget, 1)
    
    def _build_header(self) -> QFrame:
        """Build top header with MotionBloom logo and status."""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-bottom: 3px solid {theme.PRIMARY};
            }}
        """)
        header.setFixedHeight(85)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_MD, 
                                   theme.PADDING_LG, theme.PADDING_MD)
        layout.setSpacing(theme.PADDING_LG)
        
        # Logo/branding - Load actual MotionBloom logo
        logo_path = Path(__file__).parent.parent / "assets" / "motionbloom_logo.png"
        if logo_path.exists():
            logo_pixmap = QPixmap(str(logo_path))
            logo_pixmap = logo_pixmap.scaledToHeight(65, Qt.TransformationMode.SmoothTransformation)
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label = QLabel("🌸")
            logo_label.setStyleSheet(f"font-size: {theme.FONT_SIZE_2XL}px;")
        
        title_label = QLabel("MotionBloom")
        title_label.setStyleSheet(f"""
            color: {theme.PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_LG}px;
        """)
        
        subtitle_label = QLabel("Real-time Hand Check")
        subtitle_label.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-size: {theme.FONT_SIZE_SM}px;
        """)
        
        branding_layout = QVBoxLayout()
        branding_layout.setContentsMargins(0, 0, 0, 0)
        branding_layout.setSpacing(0)
        branding_layout.addWidget(title_label)
        branding_layout.addWidget(subtitle_label)
        
        logo_branding_layout = QHBoxLayout()
        logo_branding_layout.setContentsMargins(0, 0, 0, 0)
        logo_branding_layout.setSpacing(theme.PADDING_MD)
        logo_branding_layout.addWidget(logo_label)
        logo_branding_layout.addLayout(branding_layout)
        
        layout.addLayout(logo_branding_layout)
        layout.addStretch()
        
        # Right: Status + help
        status_label = QLabel("📱 Ready")
        status_label.setStyleSheet(f"""
            color: {theme.COLOR_SUCCESS};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_SM}px;
        """)
        
        help_button = DuoButton("? Help", variant="secondary")
        help_button.setFixedWidth(80)
        
        layout.addWidget(status_label)
        layout.addWidget(help_button)
        
        header.setLayout(layout)
        return header
    
    def _build_tabs(self) -> QTabWidget:
        """Build tab widget with MAIN, Video, Reports tabs."""
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(theme.generate_qss())
        
        # MAIN tab (live checking)
        main_tab = self._build_main_tab()
        tab_widget.addTab(main_tab, "MAIN")
        
        # Video tab (camera feed)
        video_tab = self._build_video_tab()
        tab_widget.addTab(video_tab, "Video")
        
        # Reports tab (historical data)
        reports_tab = self._build_reports_tab()
        tab_widget.addTab(reports_tab, "Reports")
        
        return tab_widget
    
    def _build_main_tab(self) -> QWidget:
        """Build the main checking tab with hero, metrics, and verdict cards."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                   theme.PADDING_LG, theme.PADDING_LG)
        layout.setSpacing(theme.PADDING_LG)
        
        # Left column: Camera feed
        left_column = QVBoxLayout()
        left_column.setSpacing(theme.PADDING_MD)
        
        camera_label = QLabel("LIVE CAMERA VIEW")
        camera_label.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-size: {theme.FONT_SIZE_SM}px;
            font-weight: bold;
        """)
        
        camera_frame = QFrame()
        camera_frame.setStyleSheet(f"""
            QFrame {{
                background-color: black;
                border: 3px solid {theme.PRIMARY};
                border-radius: {theme.CORNER_RADIUS_LG}px;
            }}
        """)
        camera_frame.setMinimumHeight(600)
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder for video widget
        video_placeholder = QLabel("Press 'Check My Hand' to start")
        video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_placeholder.setStyleSheet(f"""
            color: white;
            font-size: {theme.FONT_SIZE_LG}px;
            font-weight: bold;
            background-color: black;
        """)
        camera_layout.addWidget(video_placeholder)
        camera_frame.setLayout(camera_layout)
        
        left_column.addWidget(camera_label)
        left_column.addWidget(camera_frame, 1)
        
        # Right column: Cards (hero, metrics, verdict)
        right_column = QVBoxLayout()
        right_column.setSpacing(theme.PADDING_MD)
        
        # Hero card
        hero_card = DuoHeroCard()
        hero_card.set_mascot_emoji("🌸")
        hero_card.set_title("Place your hand in the camera view and relax.")
        hero_card.set_subtitle("The score updates live. The green dot is only a movement guide.")
        right_column.addWidget(hero_card)
        
        # Metrics card
        metrics_card = DuoMetricsCard()
        metrics_card.add_metric("check_type", "Check", "Regular check")
        metrics_card.add_metric("finger_speed", "Finger check", "-")
        metrics_card.add_metric("quality", "Reading quality", "Building")
        right_column.addWidget(metrics_card)
        
        # Verdict card
        verdict_card = DuoVerdictCard()
        verdict_card.set_status("Hold your hand still. Getting ready (9/37).", status_type="warning")
        right_column.addWidget(verdict_card)
        
        # Main action button
        check_button = DuoButton("✓ Check My Hand", variant="primary")
        check_button.setFixedHeight(theme.BUTTON_HEIGHT_LG)
        check_button.clicked.connect(self._on_check_hand_clicked)
        right_column.addWidget(check_button)
        
        # Assemble
        layout.addLayout(left_column, 2)
        layout.addLayout(right_column, 1)
        
        container.setLayout(layout)
        self._main_tab_container = container  # Store for updates
        self._hero_card = hero_card
        self._metrics_card = metrics_card
        self._verdict_card = verdict_card
        
        return container
    
    def _build_video_tab(self) -> QWidget:
        """Build video tab with full-screen camera feed."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                   theme.PADDING_LG, theme.PADDING_LG)
        
        video_frame = QFrame()
        video_frame.setStyleSheet(f"""
            QFrame {{
                background-color: black;
                border: 3px solid {theme.PRIMARY};
                border-radius: {theme.CORNER_RADIUS_LG}px;
            }}
        """)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder
        video_placeholder = QLabel("Press 'Check My Hand' to start camera")
        video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_placeholder.setStyleSheet(f"""
            color: white;
            font-size: {theme.FONT_SIZE_XL}px;
            font-weight: bold;
            background-color: black;
        """)
        video_layout.addWidget(video_placeholder)
        video_frame.setLayout(video_layout)
        
        layout.addWidget(video_frame, 1)
        container.setLayout(layout)
        
        return container
    
    def _build_reports_tab(self) -> QWidget:
        """Build reports/history tab."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                   theme.PADDING_LG, theme.PADDING_LG)
        layout.setSpacing(theme.PADDING_MD)
        
        title_label = QLabel("Check History")
        title_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_LG}px;
        """)
        
        # Placeholder for history
        history_frame = QFrame()
        history_frame.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 2px solid {theme.BORDER_COLOR};
                border-radius: {theme.CORNER_RADIUS_LG}px;
            }}
        """)
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(theme.PADDING_MD, theme.PADDING_MD,
                                          theme.PADDING_MD, theme.PADDING_MD)
        
        history_placeholder = QLabel("[Check history and stats here]")
        history_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_placeholder.setStyleSheet(f"""
            color: {theme.TEXT_MUTED};
            font-size: {theme.FONT_SIZE_BASE}px;
        """)
        history_layout.addWidget(history_placeholder)
        history_frame.setLayout(history_layout)
        
        layout.addWidget(title_label)
        layout.addWidget(history_frame, 1)
        
        container.setLayout(layout)
        return container
    
    def _on_check_hand_clicked(self) -> None:
        """Handle 'Check My Hand' button click."""
        self.is_checking = not self.is_checking
        
        if self.is_checking:
            self._hero_card.set_mascot_emoji("👀")
            self._hero_card.set_subtitle("Analyzing your hand position...")
            self._metrics_card.update_metric("check_type", "Analyzing")
            self._verdict_card.set_status("Analyzing hand position...", status_type="info")
        else:
            self._hero_card.set_mascot_emoji("✓")
            self._hero_card.set_subtitle("Check complete! Great job!")
            self._metrics_card.update_metric("check_type", "Complete")
            self._verdict_card.set_status("Hand check successful!", status_type="success")
            self.streak_count += 1
            self._hero_card.set_streak(self.streak_count)


def launch_pyqt6_app() -> None:
    """Launch the PyQt6 application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = MotionBloomMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_pyqt6_app()
