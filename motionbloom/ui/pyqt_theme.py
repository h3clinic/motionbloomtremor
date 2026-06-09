"""
PyQt6 Theme System for MotionBloom
Vibrant bright red and white theme with premium cards and animations
"""

from typing import Dict

# ============================================================================
# VIBRANT RED & WHITE PALETTE
# ============================================================================

# Core Colors - Bright, Vibrant, Modern
BG_DARK = "#FFFFFF"          # Pure white background
BG_SECONDARY = "#FFFFFF"     # White surfaces
BORDER_COLOR = "#FECDD3"     # Light red border

# Accent Colors (Bright Red Theme)
PRIMARY = "#E63946"          # MotionBloom Vibrant Red - Primary accent (brighter)
ACCENT = "#FF6B6B"           # Bright red accent - Secondary emphasis
WARMTH = "#FF5252"           # Bright warm red - Motivation
CRITICAL = "#C1121F"         # Deep red - Errors

# Red gradient stops for premium effects
RED_LIGHT = "#FFE5E5"        # Light red for backgrounds
RED_MEDIUM = "#FF8C8C"       # Medium red for highlights
RED_DARK = "#A01818"         # Dark red for pressed states

# White accents and surfaces
WHITE_SURFACE = "#FFFFFF"    # Pure white surfaces
WHITE_SECONDARY = "#F9F9F9"  # Off-white background

# Text & UI
TEXT_PRIMARY = "#0F1117"     # Almost black for contrast
TEXT_SECONDARY = "#57606A"   # Muted text
TEXT_MUTED = "#8B949E"       # Disabled/tertiary text

# Semantic Colors
COLOR_SUCCESS = PRIMARY
COLOR_WARNING = WARMTH
COLOR_ERROR = CRITICAL
COLOR_INFO = ACCENT

# ============================================================================
# LAYOUT & SPACING
# ============================================================================

PADDING_XS = 4
PADDING_SM = 8
PADDING_MD = 16
PADDING_LG = 24
PADDING_XL = 32
PADDING_2XL = 48

CORNER_RADIUS_SM = 8
CORNER_RADIUS_MD = 12
CORNER_RADIUS_LG = 16
CORNER_RADIUS_XL = 24

# ============================================================================
# TYPOGRAPHY
# ============================================================================

FONT_FAMILY = "Segoe UI, -apple-system, BlinkMacSystemFont, Arial"

# Font Sizes
FONT_SIZE_TINY = 10
FONT_SIZE_SM = 12
FONT_SIZE_BASE = 14
FONT_SIZE_MD = 16
FONT_SIZE_LG = 18
FONT_SIZE_XL = 24
FONT_SIZE_2XL = 32

# Font Weights
FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_MEDIUM = 500
FONT_WEIGHT_BOLD = 700

# ============================================================================
# COMPONENT SIZING
# ============================================================================

BUTTON_HEIGHT = 40
BUTTON_HEIGHT_SM = 32
BUTTON_HEIGHT_LG = 48

INPUT_HEIGHT = 40

CARD_HEIGHT_MIN = 120
CARD_BORDER_WIDTH = 2

MASCOT_SIZE_SM = 60
MASCOT_SIZE_MD = 80
MASCOT_SIZE_LG = 120

# ============================================================================
# SHADOW & EFFECTS
# ============================================================================

SHADOW_SM = "0px 2px 4px rgba(230, 57, 70, 0.08)"
SHADOW_MD = "0px 4px 8px rgba(230, 57, 70, 0.12)"
SHADOW_LG = "0px 8px 16px rgba(230, 57, 70, 0.15)"

# ============================================================================
# ANIMATION TIMINGS (milliseconds)
# ============================================================================

ANIMATION_FAST = 150
ANIMATION_BASE = 300
ANIMATION_SLOW = 500

# ============================================================================
# QSS STYLESHEET GENERATOR
# ============================================================================

def generate_qss() -> str:
    """Generate complete QSS stylesheet for vibrant red/white theme."""
    qss = f"""
    /* ===== MAIN WINDOW & WIDGETS ===== */
    QMainWindow, QWidget {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BASE}px;
    }}

    /* ===== LABELS ===== */
    QLabel {{
        color: {TEXT_PRIMARY};
        background-color: transparent;
    }}

    QLabel[type="secondary"] {{
        color: {TEXT_SECONDARY};
    }}

    QLabel[type="muted"] {{
        color: {TEXT_MUTED};
    }}

    QLabel[type="success"] {{
        color: {COLOR_SUCCESS};
        font-weight: bold;
    }}

    QLabel[type="warning"] {{
        color: {COLOR_WARNING};
        font-weight: bold;
    }}

    QLabel[type="error"] {{
        color: {COLOR_ERROR};
        font-weight: bold;
    }}

    /* ===== PUSH BUTTONS ===== */
    QPushButton {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        border-radius: {CORNER_RADIUS_MD}px;
        padding: {PADDING_SM}px {PADDING_MD}px;
        font-weight: bold;
        font-size: {FONT_SIZE_BASE}px;
        height: {BUTTON_HEIGHT}px;
    }}

    QPushButton:hover {{
        background-color: {RED_MEDIUM};
        padding-top: {PADDING_SM}px;
    }}

    QPushButton:pressed {{
        background-color: {RED_DARK};
        padding-top: {PADDING_SM + 2}px;
        padding-bottom: {PADDING_SM - 2}px;
    }}

    QPushButton:disabled {{
        background-color: {TEXT_MUTED};
        color: white;
    }}

    /* ===== SECONDARY BUTTON ===== */
    QPushButton[type="secondary"] {{
        background-color: white;
        color: {PRIMARY};
        border: 2px solid {PRIMARY};
        font-weight: bold;
    }}

    QPushButton[type="secondary"]:hover {{
        background-color: {RED_LIGHT};
    }}

    QPushButton[type="secondary"]:pressed {{
        background-color: {BORDER_COLOR};
    }}

    /* ===== LINE EDITS / INPUT ===== */
    QLineEdit, QTextEdit {{
        background-color: white;
        color: {TEXT_PRIMARY};
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_SM}px;
        padding: {PADDING_SM}px;
        selection-background-color: {PRIMARY};
    }}

    QLineEdit:focus, QTextEdit:focus {{
        border: 2px solid {PRIMARY};
        background-color: {WHITE_SECONDARY};
    }}

    /* ===== COMBO BOXES ===== */
    QComboBox {{
        background-color: white;
        color: {TEXT_PRIMARY};
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_SM}px;
        padding: {PADDING_SM}px;
    }}

    QComboBox:hover {{
        border: 2px solid {PRIMARY};
    }}

    QComboBox:focus {{
        border: 2px solid {PRIMARY};
        background-color: {WHITE_SECONDARY};
    }}

    QComboBox::drop-down {{
        border: none;
        background-color: transparent;
    }}

    QComboBox QAbstractItemView {{
        background-color: white;
        color: {TEXT_PRIMARY};
        selection-background-color: {PRIMARY};
        border: 2px solid {BORDER_COLOR};
    }}

    /* ===== TABS ===== */
    QTabWidget::pane {{
        border: none;
        background-color: white;
    }}

    QTabBar::tab {{
        background-color: {WHITE_SECONDARY};
        color: {TEXT_SECONDARY};
        padding: {PADDING_SM}px {PADDING_MD}px;
        border: 2px solid {BORDER_COLOR};
        border-bottom: none;
        margin-right: 2px;
        font-weight: 500;
    }}

    QTabBar::tab:hover {{
        color: {PRIMARY};
        border-color: {PRIMARY};
        background-color: {RED_LIGHT};
    }}

    QTabBar::tab:selected {{
        background-color: white;
        color: {PRIMARY};
        border: 2px solid {PRIMARY};
        border-bottom: none;
        font-weight: bold;
    }}

    /* ===== SCROLL BARS ===== */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
    }}

    QScrollBar::handle:vertical {{
        background-color: {BORDER_COLOR};
        border-radius: 5px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {PRIMARY};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}

    QScrollBar:horizontal {{
        background-color: transparent;
        height: 10px;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {BORDER_COLOR};
        border-radius: 5px;
        min-width: 20px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {PRIMARY};
    }}

    /* ===== FRAMES & CONTAINERS ===== */
    QFrame {{
        background-color: transparent;
        border: none;
    }}

    QFrame[type="card"] {{
        background-color: white;
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_LG}px;
    }}

    QFrame[type="card"]:hover {{
        border-color: {PRIMARY};
    }}

    /* ===== PROGRESS BAR ===== */
    QProgressBar {{
        background-color: {RED_LIGHT};
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_SM}px;
        text-align: center;
        color: {TEXT_PRIMARY};
        font-weight: bold;
    }}

    QProgressBar::chunk {{
        background-color: {PRIMARY};
        border-radius: {CORNER_RADIUS_SM - 1}px;
    }}

    /* ===== SLIDERS ===== */
    QSlider::groove:horizontal {{
        background-color: {RED_LIGHT};
        border: none;
        height: 6px;
        border-radius: 3px;
    }}

    QSlider::handle:horizontal {{
        background-color: {PRIMARY};
        border: 2px solid white;
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {RED_MEDIUM};
    }}

    /* ===== DIALOGS ===== */
    QDialog {{
        background-color: white;
    }}

    /* ===== CUSTOM CLASS STYLES ===== */
    DuoCard {{
        background-color: white;
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_LG}px;
    }}

    DuoCard:hover {{
        border-color: {PRIMARY};
    }}

    DuoHeroCard {{
        background-color: white;
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_LG}px;
    }}

    DuoHeroCard:hover {{
        border-color: {PRIMARY};
    }}

    DuoMetricsCard {{
        background-color: white;
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_LG}px;
    }}

    DuoMetricsCard:hover {{
        border-color: {PRIMARY};
    }}

    DuoVerdictCard {{
        background-color: white;
        border: 2px solid {BORDER_COLOR};
        border-radius: {CORNER_RADIUS_LG}px;
    }}

    DuoVerdictCard:hover {{
        border-color: {PRIMARY};
    }}

    DuoButton {{
        font-weight: bold;
    }}
    """
    return qss


# ============================================================================
# COLOR CONSTANTS EXPORT
# ============================================================================

COLORS = {
    "bg_dark": BG_DARK,
    "bg_secondary": BG_SECONDARY,
    "border": BORDER_COLOR,
    "primary": PRIMARY,
    "accent": ACCENT,
    "warmth": WARMTH,
    "critical": CRITICAL,
    "text_primary": TEXT_PRIMARY,
    "text_secondary": TEXT_SECONDARY,
    "text_muted": TEXT_MUTED,
    "red_light": RED_LIGHT,
    "red_medium": RED_MEDIUM,
    "red_dark": RED_DARK,
}

SPACING = {
    "xs": PADDING_XS,
    "sm": PADDING_SM,
    "md": PADDING_MD,
    "lg": PADDING_LG,
    "xl": PADDING_XL,
    "2xl": PADDING_2XL,
}

RADIUS = {
    "sm": CORNER_RADIUS_SM,
    "md": CORNER_RADIUS_MD,
    "lg": CORNER_RADIUS_LG,
    "xl": CORNER_RADIUS_XL,
}

FONT_SIZES = {
    "tiny": FONT_SIZE_TINY,
    "sm": FONT_SIZE_SM,
    "base": FONT_SIZE_BASE,
    "md": FONT_SIZE_MD,
    "lg": FONT_SIZE_LG,
    "xl": FONT_SIZE_XL,
    "2xl": FONT_SIZE_2XL,
}
