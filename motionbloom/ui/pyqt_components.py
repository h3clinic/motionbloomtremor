"""
PyQt6 Duolingo-style UI Components
Reusable, styled, animated card and widget components
"""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette
from typing import Optional, Dict, Callable
from pathlib import Path

from . import pyqt_theme as theme


class DuoCard(QFrame):
    """
    Base 3D card component with Duolingo styling.
    Features: Rounded corners, border, hover effects.
    """

    def __init__(self, parent: Optional[QWidget] = None, **kwargs):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setProperty("type", "card")
        
        # Animation for hover border
        self._border_animation = QPropertyAnimation(self, b"borderColor")
        self._border_animation.setDuration(theme.ANIMATION_BASE)
        self._border_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Apply basic styling
        self.setStyleSheet(f"""
            DuoCard {{
                background-color: {theme.BG_SECONDARY};
                border: {theme.CARD_BORDER_WIDTH}px solid {theme.BORDER_COLOR};
                border-radius: {theme.CORNER_RADIUS_LG}px;
            }}
        """)
        
        # Hover border animation (optional, can be enhanced)
        self._border_color = theme.BORDER_COLOR
        self._hover_enabled = kwargs.get("hover_enabled", True)
    
    def enterEvent(self, event) -> None:
        """Animate border on hover."""
        if self._hover_enabled:
            self.setStyleSheet(f"""
                DuoCard {{
                    background-color: {theme.BG_SECONDARY};
                    border: {theme.CARD_BORDER_WIDTH}px solid {theme.PRIMARY};
                    border-radius: {theme.CORNER_RADIUS_LG}px;
                }}
            """)
        super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        """Restore border on leave."""
        if self._hover_enabled:
            self.setStyleSheet(f"""
                DuoCard {{
                    background-color: {theme.BG_SECONDARY};
                    border: {theme.CARD_BORDER_WIDTH}px solid {theme.BORDER_COLOR};
                    border-radius: {theme.CORNER_RADIUS_LG}px;
                }}
            """)
        super().leaveEvent(event)


class DuoHeroCard(DuoCard):
    """
    Hero card with 3-column layout:
    - Left: Mascot emoji/image (80×80 fixed)
    - Center: Title + subtitle (flexible)
    - Right: Streak badge (fixed)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG, 
                                            theme.PADDING_LG, theme.PADDING_LG)
        self.main_layout.setSpacing(theme.PADDING_MD)
        
        # Left: Mascot frame
        self.mascot_frame = QFrame()
        self.mascot_frame.setFixedSize(theme.MASCOT_SIZE_MD, theme.MASCOT_SIZE_MD)
        self.mascot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.RED_LIGHT};
                border-radius: {theme.CORNER_RADIUS_MD}px;
                border: 2px solid {theme.PRIMARY};
            }}
        """)
        self.mascot_layout = QVBoxLayout()
        self.mascot_layout.setContentsMargins(0, 0, 0, 0)
        self.mascot_frame.setLayout(self.mascot_layout)
        
        self.mascot_label = QLabel("😊")
        self.mascot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mascot_label.setStyleSheet(f"font-size: {theme.FONT_SIZE_2XL}px;")
        self.mascot_layout.addWidget(self.mascot_label)
        
        # Mascot state tracking
        self._mascot_state = "idle"  # idle, analyzing, success, warning
        self._mascot_image_paths = {}  # State -> image path mapping
        self._setup_mascot_paths()
        
        # Center: Text content (flexible)
        self.text_layout = QVBoxLayout()
        self.text_layout.setSpacing(theme.PADDING_SM)
        
        self.title_label = QLabel("Welcome back!")
        self.title_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_XL}px;
        """)
        
        self.subtitle_label = QLabel("Keep practicing to improve your score.")
        self.subtitle_label.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-size: {theme.FONT_SIZE_BASE}px;
        """)
        self.subtitle_label.setWordWrap(True)
        
        self.text_layout.addWidget(self.title_label)
        self.text_layout.addWidget(self.subtitle_label)
        self.text_layout.addStretch()
        
        # Right: Streak badge (fixed)
        self.streak_frame = QFrame()
        self.streak_frame.setFixedSize(80, 80)
        self.streak_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.PRIMARY};
                border-radius: {theme.CORNER_RADIUS_MD}px;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }}
        """)
        self.streak_layout = QVBoxLayout()
        self.streak_layout.setContentsMargins(0, 0, 0, 0)
        self.streak_frame.setLayout(self.streak_layout)
        
        self.streak_icon = QLabel("🔥")
        self.streak_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.streak_icon.setStyleSheet(f"font-size: {theme.FONT_SIZE_LG}px;")
        
        self.streak_label = QLabel("7")
        self.streak_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.streak_label.setStyleSheet(f"""
            color: white;
            font-weight: bold;
            font-size: {theme.FONT_SIZE_XL}px;
        """)
        
        self.streak_layout.addWidget(self.streak_icon)
        self.streak_layout.addWidget(self.streak_label)
        
        # Pulse animation for streak updates
        self._pulse_animation = QPropertyAnimation(self.streak_frame, b"geometry")
        self._pulse_animation.setDuration(theme.ANIMATION_BASE)
        self._pulse_animation.setEasingCurve(QEasingCurve.Type.OutElastic)
        
        # Assemble layout
        self.main_layout.addWidget(self.mascot_frame, 0, Qt.AlignmentFlag.AlignTop)
        self.main_layout.addLayout(self.text_layout, 1)
        self.main_layout.addWidget(self.streak_frame, 0, Qt.AlignmentFlag.AlignTop)
        
        self.setLayout(self.main_layout)
    
    def _setup_mascot_paths(self) -> None:
        """Setup mascot image paths from assets directory."""
        assets_dir = Path(__file__).parent.parent / "assets" / "duolingo"
        
        self._mascot_image_paths = {
            "idle": assets_dir / "bloom_idle.png",
            "analyzing": assets_dir / "bloom_wave.png",
            "success": assets_dir / "bloom_cheer.png",
            "warning": assets_dir / "bloom_sad.png",
        }
    
    def set_mascot_emoji(self, emoji: str) -> None:
        """Set mascot emoji."""
        self.mascot_label.setText(emoji)
    
    def set_mascot_image(self, image_path: str) -> None:
        """Set mascot from image file."""
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaledToWidth(theme.MASCOT_SIZE_MD - 8, Qt.TransformationMode.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Clear previous
        while self.mascot_layout.count():
            self.mascot_layout.takeAt(0).widget().deleteLater()
        
        self.mascot_layout.addWidget(label)
    
    def set_mascot_state(self, state: str) -> None:
        """
        Set mascot state and load corresponding image.
        States: idle, analyzing, success, warning
        """
        if state not in self._mascot_image_paths:
            return
        
        self._mascot_state = state
        image_path = self._mascot_image_paths[state]
        
        if image_path.exists():
            self.set_mascot_image(str(image_path))
        else:
            # Fallback emoji if image not found
            emoji_map = {
                "idle": "😊",
                "analyzing": "👀",
                "success": "✓",
                "warning": "⚠",
            }
            self.set_mascot_emoji(emoji_map.get(state, "😊"))
    
    def set_title(self, text: str) -> None:
        """Set hero title."""
        self.title_label.setText(text)
    
    def set_subtitle(self, text: str) -> None:
        """Set hero subtitle."""
        self.subtitle_label.setText(text)
    
    def set_streak(self, count: int) -> None:
        """Set streak count and trigger pulse animation."""
        self.streak_label.setText(str(count))
        self._pulse_streak()
    
    def _pulse_streak(self) -> None:
        """Animate streak badge with pulse effect."""
        # Get current geometry
        current_geo = self.streak_frame.geometry()
        
        # Slight scale animation (simulate pulse via geometry expansion)
        start_geo = current_geo.adjusted(-2, -2, 2, 2)
        end_geo = current_geo
        
        self._pulse_animation.setStartValue(start_geo)
        self._pulse_animation.setEndValue(end_geo)
        self._pulse_animation.start()
    
    def set_streak_hidden(self, hidden: bool) -> None:
        """Hide/show streak badge."""
        self.streak_frame.setVisible(not hidden)


class DuoMetricsCard(DuoCard):
    """
    Metrics/diagnostics card with 2-column grid layout.
    Dynamically add rows for key-value pairs.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                       theme.PADDING_LG, theme.PADDING_LG)
        self.layout.setSpacing(theme.PADDING_MD)
        
        # Title
        self.title_label = QLabel("Diagnostic Metrics")
        self.title_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_LG}px;
        """)
        self.layout.addWidget(self.title_label)
        
        # Metrics grid
        self.metrics_grid = QGridLayout()
        self.metrics_grid.setSpacing(theme.PADDING_MD)
        self.layout.addLayout(self.metrics_grid)
        
        self.layout.addStretch()
        self.setLayout(self.layout)
        
        self._metrics: Dict[str, tuple] = {}  # Store metric widgets
    
    def add_metric(self, key: str, label: str, value: str) -> None:
        """Add a metric row (label + value)."""
        row = len(self._metrics)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-size: {theme.FONT_SIZE_SM}px;
        """)
        
        value_widget = QLabel(value)
        value_widget.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_BASE}px;
        """)
        
        self.metrics_grid.addWidget(label_widget, row, 0)
        self.metrics_grid.addWidget(value_widget, row, 1, Qt.AlignmentFlag.AlignRight)
        
        self._metrics[key] = (label_widget, value_widget)
    
    def update_metric(self, key: str, value: str) -> None:
        """Update metric value."""
        if key in self._metrics:
            _, value_widget = self._metrics[key]
            value_widget.setText(value)


class DuoVerdictCard(DuoCard):
    """
    Status/verdict card with color-coded badge and message.
    Shows health status (success, warning, error).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(theme.PADDING_LG, theme.PADDING_LG,
                                       theme.PADDING_LG, theme.PADDING_LG)
        self.layout.setSpacing(theme.PADDING_MD)
        
        # Badge container
        self.badge_frame = QFrame()
        self.badge_layout = QHBoxLayout()
        self.badge_layout.setContentsMargins(0, 0, 0, 0)
        self.badge_layout.setSpacing(theme.PADDING_SM)
        self.badge_frame.setLayout(self.badge_layout)
        
        self.badge_icon = QLabel("✓")
        self.badge_icon.setStyleSheet(f"""
            color: white;
            font-weight: bold;
            font-size: {theme.FONT_SIZE_LG}px;
        """)
        
        self.badge_label = QLabel("Status OK")
        self.badge_label.setStyleSheet(f"""
            color: {theme.TEXT_PRIMARY};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_BASE}px;
        """)
        
        self.badge_layout.addWidget(self.badge_icon)
        self.badge_layout.addWidget(self.badge_label)
        self.badge_layout.addStretch()
        
        # Message
        self.message_label = QLabel("All systems nominal.")
        self.message_label.setStyleSheet(f"""
            color: {theme.TEXT_SECONDARY};
            font-size: {theme.FONT_SIZE_SM}px;
        """)
        self.message_label.setWordWrap(True)
        
        self.layout.addWidget(self.badge_frame)
        self.layout.addWidget(self.message_label)
        self.layout.addStretch()
        
        self.setLayout(self.layout)
        
        self._status_color = theme.PRIMARY
    
    def set_status(self, message: str, status_type: str = "success") -> None:
        """
        Set status with message and color.
        status_type: "success" (green), "warning" (orange), "error" (red), "info" (blue)
        Smooth color transition animation.
        """
        color_map = {
            "success": (theme.COLOR_SUCCESS, "✓"),
            "warning": (theme.COLOR_WARNING, "⚠"),
            "error": (theme.COLOR_ERROR, "✕"),
            "info": (theme.ACCENT, "ℹ"),
        }
        
        color, icon = color_map.get(status_type, (theme.PRIMARY, "•"))
        
        # Animate icon color change
        self.badge_icon.setText(icon)
        self._animate_color_change(self.badge_icon, color)
        
        self.badge_label.setText(message)
        self._status_color = color
    
    def _animate_color_change(self, label: QLabel, target_color: str) -> None:
        """Animate label color smoothly to target."""
        # For simplicity, apply immediately with smooth visual feedback
        # A full implementation would use QPropertyAnimation with color interpolation
        label.setStyleSheet(f"""
            color: {target_color};
            font-weight: bold;
            font-size: {theme.FONT_SIZE_LG}px;
        """)


class DuoButton(QPushButton):
    """Duolingo-styled button with hover and press animations - Bright Red Theme."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, variant: str = "primary"):
        super().__init__(text, parent)
        self.variant = variant
        self._scale_animation = QPropertyAnimation(self, b"minimumHeight")
        self._scale_animation.setDuration(theme.ANIMATION_FAST)
        self._scale_animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        self._setup_styling()
    
    def _setup_styling(self) -> None:
        """Apply variant-specific styling with bright red theme."""
        if self.variant == "primary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme.PRIMARY};
                    color: white;
                    border: none;
                    border-radius: {theme.CORNER_RADIUS_MD}px;
                    padding: {theme.PADDING_SM}px {theme.PADDING_MD}px;
                    font-weight: bold;
                    font-size: {theme.FONT_SIZE_BASE}px;
                    min-height: {theme.BUTTON_HEIGHT}px;
                }}
                QPushButton:hover {{
                    background-color: {theme.RED_MEDIUM};
                }}
                QPushButton:pressed {{
                    background-color: {theme.RED_DARK};
                    padding-top: {theme.PADDING_SM + 2}px;
                    padding-bottom: {theme.PADDING_SM - 2}px;
                }}
            """)
        elif self.variant == "secondary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme.BG_SECONDARY};
                    color: {theme.PRIMARY};
                    border: 2px solid {theme.PRIMARY};
                    border-radius: {theme.CORNER_RADIUS_MD}px;
                    padding: {theme.PADDING_SM}px {theme.PADDING_MD}px;
                    font-weight: bold;
                    font-size: {theme.FONT_SIZE_BASE}px;
                    min-height: {theme.BUTTON_HEIGHT}px;
                }}
                QPushButton:hover {{
                    border-color: {theme.PRIMARY};
                    background-color: {theme.RED_LIGHT};
                }}
                QPushButton:pressed {{
                    background-color: {theme.BORDER_COLOR};
                }}
            """)
    
    def mousePressEvent(self, event) -> None:
        """Animate button press."""
        self._scale_animation.setStartValue(theme.BUTTON_HEIGHT)
        self._scale_animation.setEndValue(theme.BUTTON_HEIGHT - 4)
        self._scale_animation.start()
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event) -> None:
        """Animate button release."""
        self._scale_animation.setStartValue(theme.BUTTON_HEIGHT - 4)
        self._scale_animation.setEndValue(theme.BUTTON_HEIGHT)
        self._scale_animation.start()
        super().mouseReleaseEvent(event)
