"""MotionBloom theme and styling constants.

Centralized color palette, fonts, spacing, and appearance defaults
for both Tkinter and CustomTkinter.

Supports two themes:
1. CLASSIC: Light red/white (original)
2. DUOLINGO: Dark mode with 3D depth, bold accents, gamified feel
"""

from pathlib import Path

# ============================================================================
# THEME SELECTION (CLASSIC or DUOLINGO)
# ============================================================================

ACTIVE_THEME = "DUOLINGO"  # "CLASSIC" or "DUOLINGO"

# ============================================================================
# DUOLINGO DESIGN SYSTEM (3D, Playful, Gamified)
# ============================================================================

if ACTIVE_THEME == "DUOLINGO":
    # Dark mode base
    BG = "#0F1419"                # Deep background (almost black)
    SURFACE = "#1A2332"           # Card background (dark slate)
    SURFACE_ALT = "#253548"       # Elevated surface (lighter slate)
    SURFACE_PRESSED = "#141D28"   # Pressed state
    
    # 3D Border depth (thick shadow effect)
    BORDER = "#37464F"            # Card border (dark gray)
    BORDER_STRONG = "#4A5A68"     # Hover/active border
    
    # Primary: Duo Green (vibrant, high energy)
    PRIMARY = "#58CC02"           # Bright green (success, active)
    PRIMARY_DARK = "#46A302"      # Darker green (shadow/pressed)
    
    # Secondary: Bright Blue (info, interactive)
    ACCENT = "#58A7FF"            # Bright blue
    ACCENT_DARK = "#4180C4"       # Darker blue (shadow)
    
    # Warmth: Orange/Red (motivational, streak)
    WARMTH = "#FF9600"            # Streak orange
    WARMTH_DARK = "#CC7600"       # Darker orange
    
    # Text
    TEXT = "#FFFFFF"              # High contrast white
    TEXT_2 = "#E8EAED"            # Secondary white
    MUTED = "#AFBBBF"             # Muted gray
    MUTED_2 = "#8A9397"           # Darker muted
    
    # Sidebar (dark with green accents)
    SIDEBAR_BG = "#141D28"
    SIDEBAR_ALT = "#1F2D3D"
    SIDEBAR_TEXT = "#FFFFFF"
    SIDEBAR_MUTED = "#AFBBBF"
    SIDEBAR_ACTIVE = "#58CC02"    # Active = bright green
    
    # Status colors
    RED = "#FF6B6B"               # Error/warning red
    RED_SOFT = "#FF8B8B"
    OK_GREEN = "#58CC02"
    OK_DEEP = "#46A302"
    WARN = "#FFB74D"
    WARN_DEEP = "#CC7600"
    BAD = "#FF6B6B"
    
    # Chart
    GOLD = "#FFB74D"
    GOLD_TINT = "#FFE5B4"
    PLOT_BG = "#1A2332"
    PLOT_GRID = "#37464F"

# ============================================================================
# CLASSIC DESIGN SYSTEM (Light, Professional)
# ============================================================================

elif ACTIVE_THEME == "CLASSIC":
    # Light mode base
    BG = "#ffffff"
    SURFACE = "#ffffff"
    SURFACE_ALT = "#fff1f2"
    SURFACE_ELEV = "#ffffff"
    SURFACE_PRESSED = "#ffe4e6"
    
    # Light borders
    BORDER = "#fecdd3"
    BORDER_STRONG = "#fb7185"
    
    # Muted reds
    PRIMARY = "#dc2626"
    PRIMARY_DARK = "#991b1b"
    
    ACCENT = "#3b82f6"
    ACCENT_DARK = "#1e40af"
    
    WARMTH = "#f59e0b"
    WARMTH_DARK = "#d97706"
    
    # Text
    TEXT = "#1f1f1f"
    TEXT_2 = "#3f1d24"
    MUTED = "#667085"
    MUTED_2 = "#9f6b75"
    
    # Sidebar
    SIDEBAR_BG = "#ffffff"
    SIDEBAR_ALT = "#fff1f2"
    SIDEBAR_TEXT = "#1f1f1f"
    SIDEBAR_MUTED = "#b91c1c"
    SIDEBAR_ACTIVE = "#ffe4e6"
    
    # Status
    RED = "#dc2626"
    RED_DEEP = "#991b1b"
    RED_SOFT = "#fee2e2"
    RED_TINT = "#fecaca"
    OK_GREEN = "#16a34a"
    OK_DEEP = "#065f46"
    OK_TINT = "#ecfdf5"
    WARN = "#f59e0b"
    WARN_TINT = "#fffbeb"
    WARN_DEEP = "#92400e"
    BAD = "#dc2626"
    
    # Chart
    GOLD = "#ef4444"
    GOLD_TINT = "#fff1f2"
    PLOT_BG = "#ffffff"
    PLOT_GRID = "#ffe4e6"

# Keep old names for backward compatibility (use primary theme)
CANVAS = BG
SURFACE_ELEV = SURFACE if ACTIVE_THEME == "DUOLINGO" else "#ffffff"
RED_SOFT = "#fee2e2" if ACTIVE_THEME == "CLASSIC" else "#FF8B8B"
RED_TINT = "#fecaca" if ACTIVE_THEME == "CLASSIC" else "#FF6B6B"

# ============================================================================
# BRAND PALETTE (white and red customer theme)
# ============================================================================


# ============================================================================
# FONTS
# ============================================================================

FONT_FAMILY = "Helvetica"

# Font sizes (in points)
FONT_XS = 9
FONT_SM = 11
FONT_BASE = 13
FONT_MD = 14
FONT_LG = 16
FONT_XL = 18
FONT_2XL = 20
FONT_3XL = 24
FONT_4XL = 32
FONT_5XL = 48
FONT_6XL = 64
FONT_7XL = 72

# Common font tuples for Tk
FONT_BODY = (FONT_FAMILY, FONT_BASE)
FONT_BODY_SM = (FONT_FAMILY, FONT_SM)
FONT_BODY_LG = (FONT_FAMILY, FONT_LG)
FONT_LABEL = (FONT_FAMILY, FONT_MD)
FONT_LABEL_BOLD = (FONT_FAMILY, FONT_MD, "bold")
FONT_HEADING = (FONT_FAMILY, FONT_LG, "bold")
FONT_TITLE = (FONT_FAMILY, FONT_3XL, "bold")
FONT_SCORE_DISPLAY = (FONT_FAMILY, FONT_7XL, "bold")

# ============================================================================
# SPACING AND PADDING
# ============================================================================

PADDING_XS = 4
PADDING_SM = 8
PADDING_BASE = 12
PADDING_MD = 16
PADDING_LG = 24
PADDING_XL = 32
PADDING_2XL = 48

CORNER_RADIUS_SM = 4
CORNER_RADIUS_MD = 8
CORNER_RADIUS_LG = 12
CORNER_RADIUS_XL = 16

# ============================================================================
# CUSTOMTKINTER APPEARANCE
# ============================================================================

# Set CTk appearance mode and colors (to be called at app init)
if ACTIVE_THEME == "DUOLINGO":
    CTK_APPEARANCE_MODE = "dark"
    CTK_COLOR_THEME = "dark-blue"  # Dark theme with blue accents
else:
    CTK_APPEARANCE_MODE = "light"
    CTK_COLOR_THEME = "blue"

# CTk button/widget defaults
CTK_BUTTON_HEIGHT = 40
CTK_BUTTON_CORNER_RADIUS = 12
CTK_FRAME_CORNER_RADIUS = 16  # Duolingo-style larger radius
CTK_ENTRY_CORNER_RADIUS = 8

# ============================================================================
# CUSTOMTKINTER INITIALIZATION
# ============================================================================


def init_customtkinter_appearance() -> None:
    """Initialize CustomTkinter appearance settings with theme-specific colors.

    Call this once at app startup if using CustomTkinter.
    """
    try:
        import customtkinter as ctk
        ctk.set_appearance_mode(CTK_APPEARANCE_MODE)
        ctk.set_default_color_theme(CTK_COLOR_THEME)
    except ImportError:
        pass  # CustomTkinter not installed; app will use plain Tk


# ============================================================================
# TIMING CONSTANTS
# ============================================================================

WINDOW_SECONDS = 1.5  # Optimal window per research (was 4.0)
VIDEO_MS = 33
ANALYSIS_MS = 200
HISTORY_MAX = 600
HAND_SETTLING_SECONDS = 1.5
HAND_MOVING_WARNING_SECONDS = 3.0

# ============================================================================
# HAND STATE MACHINE
# ============================================================================

HAND_STATE_NO_HAND = "NO_HAND"
HAND_STATE_MOVING = "HAND_MOVING"
HAND_STATE_MOVING_WIDE = "HAND_MOVING_WIDE_RANGE"
HAND_STATE_DRIFTING = "HAND_DRIFTING"
HAND_STATE_SETTLING = "HAND_SETTLING"
HAND_STATE_STEADY = "HAND_STEADY"
HAND_STATE_ACTIVE = "TREMOR_ANALYSIS_ACTIVE"
HAND_MOVING_STATES = {
    HAND_STATE_MOVING,
    HAND_STATE_MOVING_WIDE,
    HAND_STATE_DRIFTING,
}

# ============================================================================
# APP METADATA AND ASSETS
# ============================================================================

APP_NAME = "MotionBloom"
LOGO_ASSET = Path(__file__).parent.parent / "assets" / "motionbloom_logo.png"
STARTUP_VIDEO_ASSET = Path(__file__).parent.parent / "assets" / "startup_intro.mp4"
STARTUP_INTRO_SECONDS = 7.0
APP_TAGLINE = "A simple hand-movement check with your camera."

# ============================================================================
# OPTICAL FLOW THRESHOLDS
# ============================================================================

# Optical-flow source selection thresholds. Kept at module scope so
# tests can reach the same logic the app uses without spinning up Tk.
FLOW_MIN_QUALITY = 0.40
FLOW_MIN_VALID_POINTS = 10

# Microtremor is, anatomically, sub-millimetre to a few millimetres of
# tip motion at 3-12 Hz. On a webcam, the palm spans ~100-200 px, so
# even a generous 5 mm tip excursion is ~0.05 palm-lengths peak-to-peak.
# Per-frame **velocity** at 30 fps is bounded by the same physics:
# anything above ~0.08 palm-lengths/frame (~2.4 palm/sec) is gross
# voluntary motion, not tremor. We use this as a hard reject on the
# optical-flow signal and as a soft per-window gross-fraction gate.
FLOW_MAX_PER_FRAME_DISPLACEMENT = 0.08   # palm-lengths/frame, hard clip
FLOW_GROSS_FRAME_FRACTION_MAX = 0.20     # if >20% of window frames are gross, pause
