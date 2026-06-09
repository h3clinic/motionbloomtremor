"""MotionBloom - consumer-grade tremor-sensing desktop app.

Senior-friendly card layout, large type, and plain-language guidance.
"""

from __future__ import annotations

import csv
import os
import random
import re
import sys
import time
import webbrowser
from collections import deque
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse
from tkinter import (
    Tk, Toplevel, Canvas, Frame, Label, Button, Entry, ttk,
    StringVar, BooleanVar, DoubleVar, filedialog, messagebox,
)

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from scipy.signal import welch

try:
    import customtkinter as ctk
    HAS_CTK = True
except ImportError:
    HAS_CTK = False

from motionbloom._cv_lock import CV_LOCK
from motionbloom.ui import theme, assets

from .exercises import EXERCISES, Exercise, ExerciseSession, Stage
from .exercises import verify_hold_object, verify_touch_nose
from .signal import (
    PALM_DRIFT_RATIO_THRESHOLD,
    PALM_STEADY_GROSS_RATIO,
    PALM_STEADY_FRAME_MOTION,
    PALM_UNCERTAIN_GROSS_RATIO,
    PRIMARY_TREMOR_BAND,
    TREMOR_BAND,
    AdaptiveBaseline,
    TaskMode,
    bandpass,
    compute_metrics,
    compute_palm_center_motion_gate,
    highpass,
    movement_residual_features,
    resample_uniform,
    rolling_spectrogram,
)
from .tracker import CAM_HEIGHT, CAM_WIDTH, LANDMARK_CHOICES, TremorTracker
from .video_gate import VideoGate
from .video_player import LocalVideoPlayer, create_video_player
from .reports import SessionReportStore

# ---------- UI Framework configuration ----
# Set MOTIONBLOOM_UI_EXPERIMENTAL=1 to enable CustomTkinter experimental shell
MOTIONBLOOM_UI_EXPERIMENTAL = os.getenv("MOTIONBLOOM_UI_EXPERIMENTAL", "0") == "1"

# Initialize CustomTkinter appearance if available
if HAS_CTK:
    theme.init_customtkinter_appearance()

# ---------- Brand palette (white and red customer theme) -------------------
BG = "#ffffff"               # white background
CANVAS = "#ffffff"            # main content panel
SURFACE = "#ffffff"           # cards
SURFACE_ALT = "#fff1f2"       # soft red alt surface / chips
SURFACE_ELEV = "#ffffff"
BORDER = "#fecdd3"
BORDER_STRONG = "#fb7185"
TEXT = "#1f1f1f"              # high-contrast black
TEXT_2 = "#3f1d24"
MUTED = "#667085"
MUTED_2 = "#9f6b75"

# Sidebar (white with red accents)
SIDEBAR_BG = "#ffffff"
SIDEBAR_ALT = "#fff1f2"
SIDEBAR_TEXT = "#1f1f1f"
SIDEBAR_MUTED = "#b91c1c"
SIDEBAR_ACTIVE = "#ffe4e6"

RED = "#dc2626"               # primary accent red
RED_DEEP = "#991b1b"
RED_SOFT = "#fee2e2"
RED_TINT = "#fecaca"
GOLD = "#ef4444"
GOLD_TINT = "#fff1f2"
PLOT_BG = "#ffffff"
PLOT_GRID = "#ffe4e6"

OK_GREEN = "#16a34a"
OK_TINT = "#ecfdf5"
OK_DEEP = "#065f46"
WARN = "#f59e0b"
WARN_TINT = "#fffbeb"
WARN_DEEP = "#92400e"
BAD = "#dc2626"

FONT_FAMILY = "Helvetica"

# SOTA Update (Sun et al. 2023): Reduced from 4.0s to 1.5s for better action tremor isolation
# Shorter windows prevent macro voluntary movements from contaminating tremor metrics
WINDOW_SECONDS = 1.5  # Optimal window per research (was 4.0)
VIDEO_MS = 33
ANALYSIS_MS = 200
HISTORY_MAX = 600
HAND_SETTLING_SECONDS = 1.5
HAND_MOVING_WARNING_SECONDS = 3.0

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

APP_NAME = "MotionBloom"
LOGO_ASSET = Path(__file__).with_name("assets") / "motionbloom_logo.png"
STARTUP_VIDEO_ASSET = Path(__file__).with_name("assets") / "startup_intro.mp4"
STARTUP_INTRO_SECONDS = 7.0
APP_TAGLINE = "A simple hand-movement check with your camera."


# Optical-flow source selection thresholds. Kept at module scope so
# tests can reach the same logic the app uses without spinning up Tk.
FLOW_MIN_QUALITY = 0.40
FLOW_MIN_VALID_POINTS = 10

# --- Physiological gates for the optical-flow channel ------------------
# Microtremor is, anatomically, sub-millimetre to a few millimetres of
# tip motion at 3-12 Hz. On a webcam, the palm spans ~100-200 px, so
# even a generous 5 mm tip excursion is ~0.05 palm-lengths peak-to-peak.
# Per-frame **velocity** at 30 fps is bounded by the same physics:
# anything above ~0.08 palm-lengths/frame (~2.4 palm/sec) is gross
# voluntary motion, not tremor. We use this as a hard reject on the
# optical-flow signal and as a soft per-window gross-fraction gate.
FLOW_MAX_PER_FRAME_DISPLACEMENT = 0.08   # palm-lengths/frame, hard clip
FLOW_GROSS_FRAME_FRACTION_MAX = 0.20     # if >20% of window frames are gross, pause


def gate_flow_for_microtremor(
    body_dx,
    body_dy,
    *,
    max_per_frame: float = FLOW_MAX_PER_FRAME_DISPLACEMENT,
    gross_fraction_max: float = FLOW_GROSS_FRAME_FRACTION_MAX,
):
    """Reject flow windows that contain physiologically impossible motion.

    Returns ``(allow, reason, gross_fraction, clipped_dx, clipped_dy)``.

    ``allow=False`` means the window contains too much non-tremor motion
    and the live score should be paused. Clipped arrays are always
    returned so callers can still draw a chart even when scoring is
    paused.
    """
    import numpy as _np

    dx = _np.asarray(body_dx, dtype=_np.float64)
    dy = _np.asarray(body_dy, dtype=_np.float64)
    if dx.size == 0 or dy.size == 0:
        return False, "no flow samples", 0.0, dx, dy

    mag = _np.hypot(dx, dy)
    gross_mask = mag > max_per_frame
    gross_fraction = float(gross_mask.mean())

    # Always clip so the analyzer can't be poisoned by one massive frame
    clip = float(max_per_frame)
    cdx = _np.clip(dx, -clip, clip)
    cdy = _np.clip(dy, -clip, clip)

    if gross_fraction > gross_fraction_max:
        return (False,
                f"Too much hand movement for tremor ({int(gross_fraction*100)}% of frames gross)",
                gross_fraction, cdx, cdy)
    return True, "flow within physiological range", gross_fraction, cdx, cdy


def select_tracking_source(
    flow_snapshot,
    flow_status: dict | None,
    palm_relative_available: bool,
    *,
    min_quality: float = FLOW_MIN_QUALITY,
    min_valid_points: int = FLOW_MIN_VALID_POINTS,
) -> tuple[str, str]:
    """Pick the live tremor signal source.

    Returns (source, reason) where source is one of
    ``"optical_flow"``, ``"mediapipe_palm_body"``, ``"fallback"`` and
    reason is a short human string for the UI.
    """
    if flow_snapshot is None or flow_status is None:
        if palm_relative_available:
            return "mediapipe_palm_body", "Optical flow warming up; using landmark"
        return "fallback", "Waiting for tracking"

    usable = bool(flow_status.get("usable", False))
    quality = float(flow_status.get("flow_quality", 0.0))
    pts = int(flow_status.get("valid_points", 0))

    if usable and quality >= min_quality and pts >= min_valid_points:
        return "optical_flow", f"Optical flow ({pts} pts, q={quality:.2f})"
    if palm_relative_available:
        return "mediapipe_palm_body", f"Optical flow weak (q={quality:.2f}); using landmark fallback"
    return "fallback", "No usable tracking"


def classify_hand_motion_state(
    palm_gate: dict | None,
    now: float,
    previous_state: str,
    previous_state_since: float,
    steady_since: float | None,
    moving_since: float | None,
) -> dict:
    """Advance the palm-center motion state machine.

    This gate runs before tremor scoring. It treats palm-center movement as
    whole-hand motion, not tremor, and only allows scoring after a settling
    interval with a steady palm center.
    """
    if palm_gate is None:
        new_state = HAND_STATE_NO_HAND
        state_since = previous_state_since if previous_state == new_state else now
        return {
            "state": new_state,
            "state_since": state_since,
            "steady_since": None,
            "moving_since": None,
            "analysis_active": False,
            "message": "No hand detected.",
        }

    gross_ratio = float(palm_gate.get("gross_motion_ratio", 0.0))
    velocity_p95 = float(palm_gate.get("per_frame_velocity_p95", 0.0))
    drift_ratio = float(palm_gate.get("drift_ratio", 0.0))
    gate_state = str(palm_gate.get("state", "unknown"))

    if gate_state == "unknown":
        new_state = HAND_STATE_SETTLING
        new_steady_since = steady_since if steady_since is not None else now
        message = "Hold your hand still. Getting ready."
    elif gross_ratio > PALM_UNCERTAIN_GROSS_RATIO:
        new_state = HAND_STATE_MOVING_WIDE
        new_steady_since = None
        message = "Too much hand movement. Keep your hand in one place."
    elif velocity_p95 > PALM_STEADY_FRAME_MOTION or gross_ratio >= PALM_STEADY_GROSS_RATIO:
        new_state = HAND_STATE_MOVING
        new_steady_since = None
        message = "Your hand is moving. Try to keep it still."
    elif drift_ratio > PALM_DRIFT_RATIO_THRESHOLD:
        new_state = HAND_STATE_DRIFTING
        new_steady_since = None
        message = "Your hand is slowly moving. Hold it still."
    else:
        if previous_state in HAND_MOVING_STATES or previous_state == HAND_STATE_NO_HAND or steady_since is None:
            new_steady_since = now
        else:
            new_steady_since = steady_since
        steady_duration = now - new_steady_since
        if steady_duration < HAND_SETTLING_SECONDS:
            new_state = HAND_STATE_SETTLING
            remaining = max(0.0, HAND_SETTLING_SECONDS - steady_duration)
            message = f"Hold steady. Clearer reading in {remaining:.1f}s."
        else:
            new_state = HAND_STATE_ACTIVE
            message = "Hand is still. Reading is ready."

    state_since = previous_state_since if previous_state == new_state else now
    if new_state in HAND_MOVING_STATES:
        if previous_state in HAND_MOVING_STATES and moving_since is not None:
            new_moving_since = moving_since
        else:
            new_moving_since = now
        if now - new_moving_since >= HAND_MOVING_WARNING_SECONDS:
            message = "Too much movement. Hold your hand still for a clearer reading."
    else:
        new_moving_since = None

    return {
        "state": new_state,
        "state_since": state_since,
        "steady_since": new_steady_since,
        "moving_since": new_moving_since,
        "analysis_active": new_state == HAND_STATE_ACTIVE,
        "message": message,
    }


# ---------- Tiny UI helpers -------------------------------------------------
def _style_axes(ax, xlabel="", ylabel=""):
    ax.set_facecolor(PLOT_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=PLOT_GRID, linewidth=0.8)


class HoverButton(Label):
    """Pseudo-button built on Label so macOS honors bg/fg colors.

    Native tk.Button on macOS ignores background/foreground and renders
    using the system theme, making custom-styled buttons look disabled.
    This Label-based implementation renders correctly everywhere.
    """

    def __init__(self, master, *, variant: str = "primary",
                 small: bool = False, command=None, text: str = "",
                 **kwargs):
        self.variant = variant
        if variant == "primary":
            bg, fg, hover = RED, "#ffffff", RED_DEEP
        elif variant == "ghost":
            bg, fg, hover = SURFACE, RED, RED_SOFT
        elif variant == "neutral":
            bg, fg, hover = SURFACE_ALT, TEXT, "#e5e7eb"
        elif variant == "dark":
            bg, fg, hover = TEXT, "#ffffff", TEXT_2
        else:  # outline
            bg, fg, hover = SURFACE, TEXT, SURFACE_ALT

        font = (FONT_FAMILY, 10 if small else 11,
                "bold" if variant in ("primary", "dark") else "normal")
        pad_x = 12 if small else 18
        pad_y = 6 if small else 9

        super().__init__(master, text=text, bg=bg, fg=fg, font=font,
                         padx=pad_x, pady=pad_y, cursor="hand2",
                         bd=0, highlightthickness=(
                             1 if variant in ("outline", "ghost") else 0),
                         highlightbackground=BORDER, **kwargs)

        self._command = command
        self._bg = bg
        self._fg = fg
        self._hover = hover
        self._disabled = False

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _e=None):
        if not self._disabled:
            self.configure(bg=self._hover)

    def _on_leave(self, _e=None):
        if not self._disabled:
            self.configure(bg=self._bg)

    def _on_press(self, _e=None):
        if not self._disabled:
            self.configure(bg=self._bg)

    def _on_release(self, _e=None):
        if self._disabled:
            return
        self.configure(bg=self._hover)
        if self._command is not None:
            try:
                self._command()
            except Exception as exc:  # pragma: no cover - UI safety
                print(f"HoverButton command failed: {exc}")

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = bool(disabled)
        if disabled:
            self.configure(bg=SURFACE_ALT, fg=MUTED_2, cursor="arrow")
        else:
            self.configure(bg=self._bg, fg=self._fg, cursor="hand2")


def make_card(parent, padding: int = 16, alt: bool = False) -> Frame:
    """Modern bordered card. `.inner` is where content is placed."""
    bg = SURFACE_ALT if alt else SURFACE
    outer = Frame(parent, bg=BORDER, highlightthickness=0)
    inner = Frame(outer, bg=bg)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    inner.configure(padx=padding, pady=padding)
    outer.inner = inner  # type: ignore[attr-defined]
    return outer


def make_logo(parent, size: int = 42, bg: str = SIDEBAR_BG):
    """Show the MotionBloom logo image, with a drawn fallback."""
    if LOGO_ASSET.exists():
        try:
            image = Image.open(LOGO_ASSET).convert("RGBA")
            image.thumbnail((size, size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            logo_label = Label(parent, image=photo, bg=bg, width=size, height=size)
            logo_label.image = photo
            return logo_label
        except Exception:
            pass

    logo = Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)
    center = size / 2
    petal_r = size * 0.16
    orbit = size * 0.27
    logo.create_oval(2, 2, size - 2, size - 2, outline=GOLD, width=max(2, size // 18))
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        cx = center + orbit * np.cos(angle)
        cy = center + orbit * np.sin(angle)
        logo.create_oval(
            cx - petal_r,
            cy - petal_r,
            cx + petal_r,
            cy + petal_r,
            fill=RED,
            outline=RED_DEEP,
        )
    logo.create_oval(
        center - size * 0.12,
        center - size * 0.12,
        center + size * 0.12,
        center + size * 0.12,
        fill=GOLD,
        outline=GOLD,
    )
    return logo


def section_title(parent, text: str, bg: str = SURFACE) -> Label:
    return Label(parent, text=text.upper(),
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=MUTED, bg=bg, anchor="w")


def help_text(parent, text: str, bg: str = SURFACE, wraplength: int = 360) -> Label:
    return Label(parent, text=text, font=(FONT_FAMILY, 13), fg=MUTED,
                 bg=bg, justify="left", wraplength=wraplength)


# ---------- App -------------------------------------------------------------
class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(f"{APP_NAME} - Hand Movement Check")
        self.root.geometry("1360x880")
        self.root.configure(bg=BG)
        self.root.minsize(1180, 760)
        print("[STARTUP] MotionBloom customer UI active")
        print("[STARTUP] Tremor pipeline, palm-relative tracking, and diagnostics enabled")

        self.tracker = TremorTracker()
        self.baseline_rms: float | None = None
        self.baseline = AdaptiveBaseline()
        self.session_rows: list[dict] = []
        self.recording = False
        
        # Task mode for conditional metric computation
        self.current_task_mode = TaskMode.POSTURAL_GENERAL  # default
        self.task_mode_options = {
            "Regular check": TaskMode.POSTURAL_GENERAL,
            "Moving check": TaskMode.MOVEMENT_TREMOR,
            "Resting check": TaskMode.REST_TREMOR,
            "Reach check": TaskMode.REHAB_REACH,
        }
        self.task_mode_var = StringVar(value="Regular check")
        self.lm_var = StringVar(value="Index fingertip")
        self.record_var = BooleanVar(value=False)
        self.video_file_var = StringVar(value="No video selected")
        self.video_status_var = StringVar(value="Choose a downloaded video to play here.")
        self.youtube_query_var = StringVar(value="")
        self.youtube_link_var = StringVar(value="")
        self.youtube_status_var = StringVar(value="YouTube stays inside this page. No new windows will open.")
        self.service_login_status_var = StringVar(value="Connect with official account sign-in only. MotionBloom will never ask for your password.")
        self.video_player: LocalVideoPlayer | None = None
        self.video_loaded_path: str | None = None
        self.video_progress_after_id: str | None = None
        self.video_progress_var = DoubleVar(value=0.0)
        self.video_time_var = StringVar(value="0:00 / 0:00")
        self.video_motion_gate_enabled = BooleanVar(value=False)
        self.video_seeking = False
        self.local_video_cap = None
        self.local_video_after_id: str | None = None
        self.local_video_playing = False
        self.local_video_photo = None
        self.startup_video_player: LocalVideoPlayer | None = None
        self.startup_video_after_id: str | None = None
        self.startup_video_started_at: float | None = None
        self.startup_video_window = None
        self.startup_video_surface = None
        self.startup_video_label = None
        self.video_challenge_active = False
        self.video_next_challenge_at: float | None = None
        self.video_challenge_score_target: float | None = None
        self.video_challenge_action: str | None = None
        self.video_challenge_started_at: float | None = None
        self.video_challenge_prompt_var = StringVar(value="Every 3 seconds, the video may pause for a quick hand action.")
        self.video_challenge_status_var = StringVar(value="Play a video to begin.")

        self.hist_t: deque = deque(maxlen=HISTORY_MAX)
        self.hist_score: deque = deque(maxlen=HISTORY_MAX)
        self.hist_peak: deque = deque(maxlen=HISTORY_MAX)
        self.hist_amp: deque = deque(maxlen=HISTORY_MAX)

        self.last_metrics = None
        self.hand_motion_state = "NO_HAND"
        self.hand_motion_state_since = time.time()
        self.hand_steady_since: float | None = None
        self.hand_moving_since: float | None = None
        self.last_stable_score: int | None = None
        self.last_stable_status = "No clear shaking reading yet"
        self.last_stable_metrics = None

        # Reports / session tracking
        self.report_store = SessionReportStore()

        self.exercises = {e.key: ExerciseSession(e) for e in EXERCISES}
        self.active_exercise_key: str | None = None

        self._configure_ttk_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._t0 = time.time()

        self.root.withdraw()

        self.root.after(VIDEO_MS, self._video_loop)
        self.root.after(ANALYSIS_MS, self._analysis_loop)
        # Auto-start the camera shortly after the UI is visible so the
        # user doesn't have to click "Start camera" manually.
        self.root.after(400, self._auto_start_camera)
        self.root.after(150, self._show_startup_intro)

    def _auto_start_camera(self) -> None:
        if self.tracker.thread is None:
            self.start()

    def _show_startup_intro(self) -> None:
        if not STARTUP_VIDEO_ASSET.exists():
            return

        self._stop_startup_intro()

        win = Toplevel(self.root)
        win.title(APP_NAME)
        win.configure(bg="#000000")
        win.attributes("-fullscreen", True)
        win.protocol("WM_DELETE_WINDOW", self._stop_startup_intro)
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

        surface = Frame(win, bg="#000000")
        surface.pack(fill="both", expand=True)
        # Label for rendering OpenCV frames (HybridVideoPlayer)
        startup_label = Label(surface, bg="#000000")
        startup_label.pack(fill="both", expand=True)
        self.startup_video_label = startup_label
        win.update_idletasks()

        self.startup_video_window = win
        self.startup_video_surface = surface

        try:
            player = create_video_player(surface)
            player.load(str(STARTUP_VIDEO_ASSET))
            player.play()
            self.startup_video_player = player
            self.startup_video_started_at = time.time()
            self._startup_video_poll()
        except Exception as exc:
            print(f"Startup intro failed: {exc}", flush=True)
            self._stop_startup_intro()

    def _stop_startup_intro(self) -> None:
        if self.startup_video_after_id is not None:
            try:
                self.root.after_cancel(self.startup_video_after_id)
            except Exception:
                pass
            self.startup_video_after_id = None
        if self.startup_video_player is not None:
            try:
                self.startup_video_player.stop()
                self.startup_video_player.cleanup()
            except Exception:
                pass
            self.startup_video_player = None
        self.startup_video_started_at = None
        if self.startup_video_window is not None:
            try:
                self.startup_video_window.destroy()
            except Exception:
                pass
            self.startup_video_window = None
        self.startup_video_surface = None
        self.startup_video_label = None
        try:
            self.root.deiconify()
        except Exception:
            pass

    def _startup_video_poll(self) -> None:
        if self.startup_video_player is None:
            return

        started_at = self.startup_video_started_at or time.time()
        elapsed = time.time() - started_at

        if elapsed >= STARTUP_INTRO_SECONDS:
            self._stop_startup_intro()
            return

        # Render OpenCV frame to startup label (HybridVideoPlayer)
        try:
            frame = self.startup_video_player.render_frame()
            if frame is not None and getattr(self, "startup_video_label", None) is not None:
                self._render_video(self.startup_video_label, frame)
        except Exception as exc:
            print(f"[STARTUP] render error: {exc}", flush=True)

        if not self.startup_video_player.is_playing():
            self._stop_startup_intro()
            return

        # ~30 FPS for smooth playback
        self.startup_video_after_id = self.root.after(33, self._startup_video_poll)

    # --------------------------------------------------------------- styling
    
    def _load_mascot_asset(self, asset_name: str) -> None:
        """Load mascot image asset into mascot_frame.
        
        Falls back gracefully if asset missing; frame remains empty with bg color.
        """
        if not hasattr(self, "mascot_frame"):
            return
        
        try:
            photo = assets.load_asset_image(asset_name, size=(60, 60))
            if photo is None:
                # No asset file; leave frame empty but styled
                return
            
            # Clear frame and add image label
            for child in self.mascot_frame.winfo_children():
                child.destroy()
            
            img_label = Label(self.mascot_frame, image=photo, bg=SURFACE, borderwidth=0)
            img_label.pack(fill="both", expand=True)
            img_label.image = photo  # Keep reference
            self.mascot_photo = photo
        except Exception as e:
            print(f"[MASCOT] Failed to load {asset_name}: {e}", flush=True)
    def _configure_ttk_styles(self) -> None:
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass

        # Visible default notebook (not used in main layout, but referenced)
        s.configure("TNotebook", background=BG, borderwidth=0,
                    tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab",
                    background=BG, foreground=MUTED,
                padding=(22, 10), borderwidth=0,
                font=(FONT_FAMILY, 13))
        s.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", RED)],
              expand=[("selected", (0, 0, 0, 0))])

        # Tab-less notebook used by the sidebar-driven layout
        s.layout("Hidden.TNotebook.Tab", [])
        s.configure("Hidden.TNotebook", background=BG,
                    borderwidth=0, tabmargins=0)
        s.configure("Hidden.TNotebook.Tab", padding=0, borderwidth=0)

        s.configure("MB.Horizontal.TProgressbar",
                    background=RED, troughcolor=SURFACE_ALT,
                    bordercolor=BORDER, lightcolor=RED, darkcolor=RED_DEEP,
                    thickness=12)

        s.configure("TCombobox",
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, arrowcolor=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)

        s.configure("TCheckbutton", background=SURFACE, foreground=TEXT,
                    font=(FONT_FAMILY, 12))

    # ------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        if MOTIONBLOOM_UI_EXPERIMENTAL and HAS_CTK:
            self._build_layout_experimental()
        else:
            self._build_layout_legacy()

    def _build_layout_legacy(self) -> None:
        # Root split: sidebar | main area (topbar + notebook + footer)
        root_grid = Frame(self.root, bg=BG)
        root_grid.pack(fill="both", expand=True)
        root_grid.columnconfigure(1, weight=1)
        root_grid.rowconfigure(0, weight=1)

        self._build_sidebar(root_grid)

        main = Frame(root_grid, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self._main = main

        self._build_topbar(main)

        self.nb = ttk.Notebook(main, style="Hidden.TNotebook")
        self.nb.grid(row=1, column=0, sticky="nsew",
                     padx=24, pady=(0, 8))
        self.nb.bind("<<NotebookTabChanged>>",
                     lambda _e: self._on_tab_change())

        self._build_live_tab()
        self._build_video_tab()
        self._build_reports_tab()

        self._build_footer(main)
        # Select home
        self._select_nav(0)

    def _build_layout_experimental(self) -> None:
        """CTkFrame-based shell wrapper (experimental).
        
        Wraps sidebar + topbar in CTkFrames, but keeps tab bodies as plain Tk
        to preserve matplotlib, canvas, and video widgets without porting.
        """
        # Root split: sidebar | main area
        root_grid = ctk.CTkFrame(self.root, fg_color=theme.BG)
        root_grid.pack(fill="both", expand=True)
        root_grid.columnconfigure(1, weight=1)
        root_grid.rowconfigure(0, weight=1)

        self._build_sidebar_experimental(root_grid)

        main = ctk.CTkFrame(root_grid, fg_color=theme.BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self._main = main

        self._build_topbar_experimental(main)

        # Keep notebook as plain Tk (ttk.Notebook) for tab body compatibility
        self.nb = ttk.Notebook(main, style="Hidden.TNotebook")
        self.nb.grid(row=1, column=0, sticky="nsew",
                     padx=24, pady=(0, 8))
        self.nb.bind("<<NotebookTabChanged>>",
                     lambda _e: self._on_tab_change())

        self._build_live_tab()
        self._build_video_tab()
        self._build_reports_tab()

        self._build_footer_experimental(main)
        # Select home
        self._select_nav(0)

    def _build_sidebar_experimental(self, parent: ctk.CTkFrame) -> None:
        """CTk sidebar wrapper with rounded corners and dark theme."""
        rail = ctk.CTkFrame(parent, fg_color=theme.SIDEBAR_BG, width=260)
        rail.grid(row=0, column=0, sticky="ns")
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)

        # Brand
        brand = ctk.CTkFrame(rail, fg_color=theme.SIDEBAR_BG)
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(24, 22))
        logo = make_logo(brand, size=44, bg=theme.SIDEBAR_BG)
        logo.pack(side="left")
        wm = ctk.CTkFrame(brand, fg_color=theme.SIDEBAR_BG)
        wm.pack(side="left", padx=(12, 0))
        Label(wm, text="MotionBloom",
              font=(theme.FONT_FAMILY, 17, "bold"),
              fg=theme.SIDEBAR_TEXT, bg=theme.SIDEBAR_BG).pack(anchor="w")
        Label(wm, text="Simple hand check",
              font=(theme.FONT_FAMILY, 11),
              fg=theme.GOLD, bg=theme.SIDEBAR_BG).pack(anchor="w")

        # Nav section label
        Label(rail, text="MAIN",
              font=(theme.FONT_FAMILY, 9, "bold"),
              fg=theme.SIDEBAR_MUTED, bg=theme.SIDEBAR_BG,
              anchor="w").grid(row=1, column=0, sticky="ew",
                               padx=22, pady=(0, 8))

        items = [
            ("Check My Hand", "●"),
            ("Video", "▶"),
            ("Reports", "≡"),
        ]
        self._nav_buttons: list[Frame] = []
        nav = ctk.CTkFrame(rail, fg_color=theme.SIDEBAR_BG)
        nav.grid(row=2, column=0, sticky="ew", padx=12)
        for i, (label, glyph) in enumerate(items):
            self._nav_buttons.append(
                self._make_nav_item_experimental(nav, i, label, glyph))

        # Spacer + footer with status chip
        spacer = ctk.CTkFrame(rail, fg_color=theme.SIDEBAR_BG)
        spacer.grid(row=3, column=0, sticky="nsew")
        rail.rowconfigure(3, weight=1)

        foot = ctk.CTkFrame(rail, fg_color=theme.SIDEBAR_BG)
        foot.grid(row=4, column=0, sticky="ew", padx=18, pady=18)
        self.status_var = StringVar(value="Idle")
        self.status_chip = Label(
            foot, textvariable=self.status_var,
            bg=theme.SIDEBAR_ALT, fg=theme.SIDEBAR_TEXT,
            padx=14, pady=10,
            font=(theme.FONT_FAMILY, 12, "bold"),
            anchor="w")
        self.status_chip.pack(fill="x")
        self.fps_var = StringVar(value="- fps")
        Label(foot, textvariable=self.fps_var,
              fg=theme.SIDEBAR_MUTED, bg=theme.SIDEBAR_BG,
              font=(theme.FONT_FAMILY, 9)).pack(anchor="w", pady=(6, 0))
        self.tracking_source_var = StringVar(value="Source: -")
        Label(foot, textvariable=self.tracking_source_var,
              fg=theme.SIDEBAR_MUTED, bg=theme.SIDEBAR_BG,
              font=(theme.FONT_FAMILY, 9)).pack(anchor="w", pady=(4, 0))
        self.flow_status_var = StringVar(value="Flow: -")
        Label(foot, textvariable=self.flow_status_var,
              fg=theme.SIDEBAR_MUTED, bg=theme.SIDEBAR_BG,
              font=(theme.FONT_FAMILY, 9)).pack(anchor="w", pady=(2, 0))

    def _make_nav_item_experimental(self, parent: ctk.CTkFrame, index: int,
                       label: str, glyph: str) -> ctk.CTkFrame:
        """CTk nav item with rounded corners and hover effects."""
        row = ctk.CTkFrame(parent, fg_color=theme.SIDEBAR_BG, cursor="hand2")
        row.pack(fill="x", pady=4)
        inner = ctk.CTkFrame(row, fg_color=theme.SIDEBAR_BG)
        inner.pack(fill="x", padx=14, pady=12)

        g = Label(inner, text=glyph, fg=theme.SIDEBAR_MUTED, bg=theme.SIDEBAR_BG,
              font=(theme.FONT_FAMILY, 16, "bold"))
        g.pack(side="left")
        t = Label(inner, text=label, fg=theme.SIDEBAR_TEXT, bg=theme.SIDEBAR_BG,
              font=(theme.FONT_FAMILY, 13), anchor="w")
        t.pack(side="left", padx=(12, 0), fill="x", expand=True)

        # Store widgets and state for _select_nav() compatibility
        row._children_refs = (inner, g, t)  # type: ignore[attr-defined]
        row._index = index  # type: ignore[attr-defined]
        row._selected = False  # type: ignore[attr-defined]

        # Bind click to nav selection
        def click_nav(e, idx=index):
            self._select_nav(idx)

        for w in (row, inner, g, t):
            w.bind("<Button-1>", click_nav)
        return row

    def _build_topbar_experimental(self, parent: ctk.CTkFrame) -> None:
        """CTk topbar with rounded corners."""
        topbar = ctk.CTkFrame(parent, fg_color=theme.BG, height=80)
        topbar.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 8))
        topbar.columnconfigure(1, weight=1)

        title = ctk.CTkLabel(topbar, text="MotionBloom",
                            font=(theme.FONT_FAMILY, 24, "bold"),
                            text_color=theme.TEXT)
        title.pack(side="left")

        badge = ctk.CTkLabel(topbar, text="🔒 Private & local",
                            font=(theme.FONT_FAMILY, 11),
                            text_color=theme.MUTED)
        badge.pack(side="right")

    def _build_footer_experimental(self, parent: ctk.CTkFrame) -> None:
        """CTk footer with rounded corners."""
        footer = ctk.CTkFrame(parent, fg_color=theme.BG)
        footer.grid(row=2, column=0, sticky="ew", padx=24, pady=(8, 16))
        footer.columnconfigure(0, weight=1)

        label = ctk.CTkLabel(footer, text="© 2024 MotionBloom. MIT License.",
                            font=(theme.FONT_FAMILY, 9),
                            text_color=theme.MUTED)
        label.pack(side="left")

    # ----------------------------------------------------------- sidebar
    def _build_sidebar(self, parent: Frame) -> None:
        rail = Frame(parent, bg=SIDEBAR_BG, width=260)
        rail.grid(row=0, column=0, sticky="ns")
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)

        # Brand
        brand = Frame(rail, bg=SIDEBAR_BG)
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(24, 22))
        logo = make_logo(brand, size=44, bg=SIDEBAR_BG)
        logo.pack(side="left")
        wm = Frame(brand, bg=SIDEBAR_BG)
        wm.pack(side="left", padx=(12, 0))
        Label(wm, text="MotionBloom",
              font=(FONT_FAMILY, 17, "bold"),
              fg=SIDEBAR_TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        Label(wm, text="Simple hand check",
              font=(FONT_FAMILY, 11),
              fg=GOLD, bg=SIDEBAR_BG).pack(anchor="w")

        # Nav section label
        Label(rail, text="MAIN",
              font=(FONT_FAMILY, 9, "bold"),
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              anchor="w").grid(row=1, column=0, sticky="ew",
                               padx=22, pady=(0, 8))

        items = [
            ("Check My Hand", "●"),
            ("Video", "▶"),
            ("Reports", "≡"),
        ]
        self._nav_buttons: list[Frame] = []
        nav = Frame(rail, bg=SIDEBAR_BG)
        nav.grid(row=2, column=0, sticky="ew", padx=12)
        for i, (label, glyph) in enumerate(items):
            self._nav_buttons.append(
                self._make_nav_item(nav, i, label, glyph))

        # Spacer + footer with status chip
        spacer = Frame(rail, bg=SIDEBAR_BG)
        spacer.grid(row=3, column=0, sticky="nsew")
        rail.rowconfigure(3, weight=1)

        foot = Frame(rail, bg=SIDEBAR_BG)
        foot.grid(row=4, column=0, sticky="ew", padx=18, pady=18)
        self.status_var = StringVar(value="Idle")
        self.status_chip = Label(
            foot, textvariable=self.status_var,
            bg=SIDEBAR_ALT, fg=SIDEBAR_TEXT,
            padx=14, pady=10,
            font=(FONT_FAMILY, 12, "bold"),
            anchor="w")
        self.status_chip.pack(fill="x")
        self.fps_var = StringVar(value="- fps")
        Label(foot, textvariable=self.fps_var,
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(6, 0))
        # Tracking source diagnostics: optical_flow / mediapipe_landmark / fallback
        self.tracking_source_var = StringVar(value="Source: -")
        Label(foot, textvariable=self.tracking_source_var,
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(4, 0))
        self.flow_status_var = StringVar(value="Flow: -")
        Label(foot, textvariable=self.flow_status_var,
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(2, 0))

    def _make_nav_item(self, parent: Frame, index: int,
                       label: str, glyph: str) -> Frame:
        row = Frame(parent, bg=SIDEBAR_BG, cursor="hand2")
        row.pack(fill="x", pady=4)
        inner = Frame(row, bg=SIDEBAR_BG, padx=14, pady=12)
        inner.pack(fill="x")

        g = Label(inner, text=glyph, fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 16, "bold"))
        g.pack(side="left")
        t = Label(inner, text=label, fg=SIDEBAR_TEXT, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 13))
        t.pack(side="left", padx=(12, 0))

        # store widgets so we can restyle on select
        row._children_refs = (inner, g, t)  # type: ignore[attr-defined]
        row._index = index  # type: ignore[attr-defined]
        row._selected = False  # type: ignore[attr-defined]

        def click(_e=None, i=index):
            self._select_nav(i)

        for w in (row, inner, g, t):
            w.bind("<Button-1>", click)

        def on_enter(_e=None, r=row):
            if not r._selected:
                r._children_refs[0].configure(bg=SIDEBAR_ALT)
                for c in r._children_refs[1:]:
                    c.configure(bg=SIDEBAR_ALT)

        def on_leave(_e=None, r=row):
            if not r._selected:
                r._children_refs[0].configure(bg=SIDEBAR_BG)
                for c in r._children_refs[1:]:
                    c.configure(bg=SIDEBAR_BG)

        for w in (row, inner, g, t):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

        return row

    def _select_nav(self, index: int) -> None:
        for i, row in enumerate(self._nav_buttons):
            selected = i == index
            row._selected = selected  # type: ignore[attr-defined]
            inner, glyph, text = row._children_refs  # type: ignore[attr-defined]
            bg = SIDEBAR_ACTIVE if selected else SIDEBAR_BG
            
            # Handle both Tk Frame (legacy) and CTkFrame (experimental)
            # Try Tk Frame method first (bg), then CTkFrame method (fg_color)
            try:
                inner.configure(bg=bg)
            except (TypeError, ValueError):
                # Fall back to CTkFrame method (fg_color)
                inner.configure(fg_color=bg)
            
            glyph.configure(bg=bg,
                            fg=RED if selected else SIDEBAR_MUTED)
            text.configure(bg=bg,
                           fg=RED_DEEP if selected else SIDEBAR_TEXT,
                                                     font=(FONT_FAMILY, 13,
                                 "bold" if selected else "normal"))
        self.nb.select(index)

    def _build_topbar(self, parent: Frame) -> None:
        top = Frame(parent, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 8))
        top.columnconfigure(0, weight=1)

        left = Frame(top, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        self.page_title_var = StringVar(value="Check My Hand")
        Label(left, textvariable=self.page_title_var,
              font=(FONT_FAMILY, 28, "bold"),
              fg=TEXT, bg=BG).pack(anchor="w")
        Label(left, text=APP_TAGLINE,
              font=(FONT_FAMILY, 15), fg=MUTED, bg=BG).pack(anchor="w", pady=(4, 0))

        trust = Frame(top, bg=OK_TINT, highlightthickness=1,
                    highlightbackground="#bbf7d0")
        trust.grid(row=0, column=1, sticky="e")
        Label(trust, text="Private & local", font=(FONT_FAMILY, 13, "bold"),
              fg=OK_DEEP, bg=OK_TINT, padx=16, pady=8).pack(side="left")
        Label(trust, text="No video is uploaded", font=(FONT_FAMILY, 12),
              fg=OK_DEEP, bg=OK_TINT, padx=16, pady=8).pack(side="left")

    def _build_actionbar(self, parent: Frame) -> None:
          bar_outer = Frame(parent, bg=BG)
          bar_outer.grid(row=1, column=0, sticky="ew", padx=28, pady=(12, 12))
          bar = Frame(bar_outer, bg=SURFACE, highlightthickness=1,
                  highlightbackground=BORDER)
          bar.pack(fill="x")
          inner = Frame(bar, bg=SURFACE, padx=14, pady=10)
          inner.pack(fill="x")

          self.start_btn = HoverButton(inner, text="Start camera",
                             variant="primary", command=self.start)
          self.start_btn.pack(side="left")

          self.stop_btn = HoverButton(inner, text="Stop camera",
                            variant="outline", command=self.stop)
          self.stop_btn.pack(side="left", padx=(8, 16))
          self.stop_btn.set_disabled(True)

          Label(inner, text="Hand point", fg=MUTED, bg=SURFACE,
              font=(FONT_FAMILY, 12, "bold")).pack(side="left")
          lm = ttk.Combobox(inner, textvariable=self.lm_var,
                      values=list(LANDMARK_CHOICES.keys()),
                      state="readonly", width=18)
          lm.pack(side="left", padx=(8, 16))
          lm.bind("<<ComboboxSelected>>", self._on_landmark_change)

          Label(inner, text="Check", fg=MUTED, bg=SURFACE,
              font=(FONT_FAMILY, 12, "bold")).pack(side="left")
          mode = ttk.Combobox(inner, textvariable=self.task_mode_var,
                        values=list(self.task_mode_options.keys()),
                        state="readonly", width=18)
          mode.pack(side="left", padx=(8, 16))
          mode.bind("<<ComboboxSelected>>", self._on_task_mode_change)

          self.calib_btn = HoverButton(inner, text="Save Calm Hand",
                             variant="ghost", command=self.calibrate)
          self.calib_btn.pack(side="left")
          self.calib_btn.set_disabled(True)

          right = Frame(inner, bg=SURFACE)
          right.pack(side="right")

          ttk.Checkbutton(right, text="Record session",
                    variable=self.record_var,
                    command=self._toggle_record,
                    style="TCheckbutton").pack(side="left", padx=(0, 8))
          HoverButton(right, text="Export CSV", variant="neutral",
                  command=self.export_csv).pack(side="left")

    def _build_footer(self, parent: Frame) -> None:
        foot = Frame(parent, bg=BG)
        foot.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 14))
        Label(
            foot,
            text=(
                "MotionBloom is a wellness-oriented tool, not a medical device. "
                "If you are concerned about shaking or changes in movement, please consult a clinician."
            ),
            fg=MUTED_2,
            bg=BG,
            font=(FONT_FAMILY, 9),
            wraplength=1200,
            justify="left",
        ).pack(anchor="w")

    def make_ctk_card(self, parent, padding: int = 16, alt: bool = False):
        """
        Premium styled card widget with graceful CTkFrame fallback.
        Returns object with .inner attribute for content placement.
        Uses CTkFrame with shadow & rounded corners if experimental mode + CTk available, else Tk Frame.
        """
        # If not experimental or CTk not available, fall back to original make_card()
        if not MOTIONBLOOM_UI_EXPERIMENTAL or not HAS_CTK:
            return make_card(parent, padding=padding, alt=alt)
        
        # Experimental path: premium CTkFrame styling
        from motionbloom.ui import theme
        
        bg = SURFACE_ALT if alt else SURFACE
        
        # Outer frame: shadow/border effect with rounded corners
        outer = ctk.CTkFrame(
            parent,
            fg_color=theme.BORDER,  # Subtle red border as shadow
            corner_radius=theme.CTK_FRAME_CORNER_RADIUS,
            border_width=0
        )
        
        # Inner content frame: actual background with rounded corners
        inner = ctk.CTkFrame(
            outer,
            fg_color=bg,
            corner_radius=theme.CTK_FRAME_CORNER_RADIUS - 2,
            border_width=0
        )
        # Pack with 1px border spacing to show shadow effect
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        
        # Content wrapper: apply padding
        content = ctk.CTkFrame(inner, fg_color=bg, corner_radius=0, border_width=0)
        content.pack(fill="both", expand=True, padx=padding, pady=padding)
        
        # Attach content as .inner for compatibility
        outer.inner = content  # type: ignore[attr-defined]
        return outer

    # ---------------------------------------------------------- Live tab
    def _build_live_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Check My Hand  ")
        tab.columnconfigure(0, weight=7)
        tab.columnconfigure(1, weight=4)
        tab.rowconfigure(1, weight=1)

        guide = make_card(tab, padding=16, alt=True)
        guide.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        gi = guide.inner
        gi.columnconfigure(1, weight=1)
        make_logo(gi, size=54, bg=SURFACE_ALT).grid(
            row=0, column=0, rowspan=2, sticky="nw", padx=(0, 14)
        )
        Label(
            gi,
            text="Place your hand in the camera view and relax.",
            font=(FONT_FAMILY, 18, "bold"),
            fg=TEXT,
            bg=SURFACE_ALT,
        ).grid(row=0, column=1, sticky="w")
        Label(
            gi,
            text="The score updates live. The green dot is only a movement guide. It will not freeze the reading.",
            font=(FONT_FAMILY, 13),
            fg=MUTED,
            bg=SURFACE_ALT,
            wraplength=980,
            justify="left",
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        video_card = make_card(tab, padding=10)
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 14), pady=8)
        section_title(video_card.inner, "Live Camera View").pack(anchor="w", pady=(0, 8))
        self.video_label = Label(video_card.inner, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        right = Frame(tab, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        hero = self.make_ctk_card(right, padding=20)
        hero.grid(row=0, column=0, sticky="ew")
        inner = hero.inner
        inner.columnconfigure(0, weight=0)  # Mascot (fixed)
        inner.columnconfigure(1, weight=1)  # Content (flexible)

        Label(
            inner,
            text="Movement Score",
            fg=TEXT,
            bg=SURFACE,
            font=(FONT_FAMILY, 18, "bold"),
        ).grid(row=1, column=1, sticky="w", pady=(10, 0))

        self.score_var = StringVar(value="-")
        self.score_lbl = Label(
            inner,
            textvariable=self.score_var,
            font=(FONT_FAMILY, 72, "bold"),
            fg=TEXT,
            bg=SURFACE,
        )
        self.score_lbl.grid(row=2, column=1, sticky="w", pady=(0, 0))

        Label(
            inner,
            text="0 = calm movement   •   100 = strong movement",
            fg=MUTED,
            bg=SURFACE,
            font=(FONT_FAMILY, 13),
        ).grid(row=3, column=1, sticky="w")

        self.verdict_var = StringVar(value="Place your hand in view. The camera starts automatically.")
        self.verdict_lbl = Label(
            inner,
            textvariable=self.verdict_var,
            fg=TEXT,
            bg=RED_SOFT,
            font=(FONT_FAMILY, 15, "bold"),
            padx=16,
            pady=14,
            anchor="w",
            justify="left",
            wraplength=320,
        )
        self.verdict_lbl.grid(row=4, column=1, sticky="ew", pady=(16, 0))
        
        # Mascot frame (left column, anchored at top-left)
        self.mascot_frame = Frame(inner, bg=SURFACE, width=80, height=80)
        self.mascot_frame.grid(row=0, column=0, sticky="nw", rowspan=5, padx=(0, 16), pady=(0, 0))
        self.mascot_frame.grid_propagate(False)  # Keep fixed 80x80 size
        self.mascot_photo = None
        self._load_mascot_asset("bloom_idle")

        metrics = self.make_ctk_card(right, padding=14)
        metrics.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        mi = metrics.inner
        mi.columnconfigure(1, weight=1)
        mi.columnconfigure(3, weight=1)

        section_title(mi, "Simple Details").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )

        self.metric_vars = {k: StringVar(value="-") for k in [
            "peak", "amp", "amp_mm", "band", "snr", "reg",
            "sharp", "class", "fs", "samples",
            "live_score", "confidence", "validity", "tremor_candidate",
            "prominence", "fps_cv", "velocity", "path_ratio", "center_drift",
            "movement_score", "tremor_overlay", "dominant_tremor",
            "tremor_source", "tracking_quality", "movement_mode", "box_stability",
            "palm_motion", "relative_displacement", "relative_power", "raw_fingertip_power",
            "relative_peaks", "relative_agreement", "relative_veto",
            "palm_path_length_px", "screen_travel_ratio", "hand_relative_travel", "veto_reason",
        ]}
        rows = [
            ("Check", "movement_mode"),
            ("Hand stillness", "palm_motion"),
            ("Finger check", "relative_agreement"),
            ("Shake speed", "dominant_tremor"),
            ("Reading quality", "confidence"),
            ("Camera view", "tracking_quality"),
        ]
        for i, (lbl, key) in enumerate(rows):
            r, c = divmod(i, 2)
            Label(mi, text=lbl, fg=MUTED, bg=SURFACE, font=(FONT_FAMILY, 12)).grid(
                row=r + 1, column=c * 2, sticky="w", pady=3, padx=(0, 10)
            )
            Label(
                mi,
                textvariable=self.metric_vars[key],
                fg=TEXT,
                bg=SURFACE,
                font=(FONT_FAMILY, 13, "bold"),
            ).grid(row=r + 1, column=c * 2 + 1, sticky="e", pady=3)

        reassurance = self.make_ctk_card(right, padding=14, alt=True)
        reassurance.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        Label(
            reassurance.inner,
            text="What to do",
            fg=TEXT,
            bg=SURFACE_ALT,
            font=(FONT_FAMILY, 15, "bold"),
        ).pack(anchor="w")
        help_text(
            reassurance.inner,
            "Keep your hand in the camera and rest your arm. Small movement is okay. If you move too far, MotionBloom will ask you to hold still.",
            bg=SURFACE_ALT,
            wraplength=420,
        ).pack(anchor="w", pady=(6, 0))

        wave_card = make_card(right, padding=10)
        wave_card.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        section_title(wave_card.inner, "Hand Movement").pack(anchor="w", pady=(0, 6))

        self.wave_fig = Figure(figsize=(4.5, 2.0), dpi=100, facecolor=SURFACE)
        self.wave_ax = self.wave_fig.add_subplot(111)
        _style_axes(self.wave_ax, "Time", "Movement")
        self.wave_fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.22)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=wave_card.inner)
        self.wave_canvas.get_tk_widget().pack(fill="both", expand=True)
        (self._wave_lx,) = self.wave_ax.plot([], [], color=RED, lw=1.4)
        (self._wave_ly,) = self.wave_ax.plot([], [], color="#94a3b8", lw=1.2)
        self.wave_ax.set_xlim(0, WINDOW_SECONDS)
        self.wave_ax.set_ylim(-0.01, 0.01)

    # ---------------------------------------------------- Exercises tab
    def _build_exercise_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Exercises  ")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        # Hero
        hero = Frame(tab, bg=BG)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        Label(hero, text="Guided Movement Tests",
              font=(FONT_FAMILY, 24, "bold"),
              fg=TEXT, bg=BG).pack(anchor="w")
        Label(hero, text="Choose one simple activity. MotionBloom will tell you what to do next.",
              fg=MUTED, bg=BG, font=(FONT_FAMILY, 14)).pack(anchor="w", pady=(4, 0))

        # Left: exercise cards
        cards_col = Frame(tab, bg=BG)
        cards_col.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=8)
        cards_col.columnconfigure(0, weight=1)

        self.ex_card_widgets: dict[str, dict] = {}
        for i, ex in enumerate(EXERCISES):
            self._make_exercise_card(cards_col, ex, i)

        # Right: running session card + mirror video
        right = Frame(tab, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        sess_card = make_card(right, padding=18)
        sess_card.grid(row=0, column=0, sticky="ew")
        si = sess_card.inner
        si.columnconfigure(0, weight=1)

        self.ex_name_var = StringVar(value=EXERCISES[0].name)
        Label(si, textvariable=self.ex_name_var,
              font=(FONT_FAMILY, 20, "bold"),
              fg=TEXT, bg=SURFACE).grid(row=0, column=0, sticky="w")

        self.ex_desc_var = StringVar(value=EXERCISES[0].description)
        Label(si, textvariable=self.ex_desc_var,
              wraplength=440, fg=MUTED, bg=SURFACE, justify="left",
              font=(FONT_FAMILY, 13)).grid(row=1, column=0, sticky="w",
                                           pady=(6, 14))

        self.ex_stage_var = StringVar(value="Ready when you are.")
        self.ex_stage_lbl = Label(si, textvariable=self.ex_stage_var,
                                  bg=RED_SOFT, fg=RED, anchor="w",
                                  padx=12, pady=10,
                                  font=(FONT_FAMILY, 15, "bold"))
        self.ex_stage_lbl.grid(row=2, column=0, sticky="ew")

        # Pose-correctness chip
        self.ex_pose_var = StringVar(value="Pose: waiting")
        self.ex_pose_chip = Label(si, textvariable=self.ex_pose_var,
                                  bg=SURFACE_ALT, fg=MUTED, anchor="w",
                                  padx=10, pady=6,
                                  font=(FONT_FAMILY, 12, "bold"))
        self.ex_pose_chip.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.ex_feedback_var = StringVar(value="")
        Label(si, textvariable=self.ex_feedback_var,
              wraplength=440, fg=TEXT, bg=SURFACE, justify="left",
              font=(FONT_FAMILY, 14)).grid(row=4, column=0, sticky="w",
                                           pady=(10, 4))

        self.ex_progress = ttk.Progressbar(
            si, orient="horizontal", mode="determinate",
            style="MB.Horizontal.TProgressbar", maximum=100)
        self.ex_progress.grid(row=5, column=0, sticky="ew", pady=(10, 8))

        # Buttons
        btn_row = Frame(si, bg=SURFACE)
        btn_row.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        self.ex_start_btn = HoverButton(btn_row, text="Start guided test",
                                        variant="primary",
                                        command=self._exercise_start)
        self.ex_start_btn.pack(side="left")
        self.ex_start_btn.set_disabled(True)
        HoverButton(btn_row, text="Cancel", variant="outline",
                    command=self._exercise_cancel).pack(side="left",
                                                        padx=(8, 0))

        self.ex_result_var = StringVar(value="")
        Label(si, textvariable=self.ex_result_var,
              wraplength=440, fg=RED, bg=SURFACE,
              font=(FONT_FAMILY, 13, "bold"), justify="left").grid(
            row=7, column=0, sticky="w", pady=(14, 0))

        # Mirror video below session card
        mirror = make_card(right, padding=8)
        mirror.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.ex_video_label = Label(mirror.inner, bg="#000000")
        self.ex_video_label.pack(fill="both", expand=True)

        # Select first exercise now that all widgets exist
        self._select_exercise_key(EXERCISES[0].key)

    def _make_exercise_card(self, parent: Frame, ex: Exercise,
                            row: int) -> None:
        outer = Frame(parent, bg=BORDER)
        outer.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        outer.columnconfigure(0, weight=1)

        card = Frame(outer, bg=SURFACE, cursor="hand2")
        card.pack(fill="x", padx=1, pady=1)
        card.configure(padx=16, pady=14)

        left = Frame(card, bg=SURFACE)
        left.pack(side="left", fill="x", expand=True)

        name = Label(left, text=ex.name,
                     font=(FONT_FAMILY, 15, "bold"),
                     fg=TEXT, bg=SURFACE, anchor="w")
        name.pack(anchor="w")
        desc = Label(left, text=ex.description,
                     wraplength=380, justify="left",
                     fg=MUTED, bg=SURFACE,
                     font=(FONT_FAMILY, 12))
        desc.pack(anchor="w", pady=(4, 0))
        duration = Label(left,
                         text=f"≈ {int(ex.prepare_secs + ex.hold_secs)} s",
                         fg=MUTED_2, bg=SURFACE,
                         font=(FONT_FAMILY, 11))
        duration.pack(anchor="w", pady=(6, 0))

        indicator = Label(card, text="", bg=SURFACE, fg=RED,
                          font=("Helvetica", 16, "bold"),
                          padx=8)
        indicator.pack(side="right")

        widgets = {
            "outer": outer, "card": card, "name": name, "desc": desc,
            "duration": duration, "indicator": indicator,
        }
        self.ex_card_widgets[ex.key] = widgets

        def click(_e=None, k=ex.key):
            self._select_exercise_key(k)

        for w in (card, left, name, desc, duration, indicator):
            w.bind("<Button-1>", click)

    def _select_exercise_key(self, key: str) -> None:
        self.active_exercise_key_selected = key
        ex = next(e for e in EXERCISES if e.key == key)
        self.ex_name_var.set(ex.name)
        self.ex_desc_var.set(ex.description)
        for k, w in self.ex_card_widgets.items():
            selected = k == key
            w["outer"].configure(bg=RED if selected else BORDER)
            w["card"].configure(bg=RED_SOFT if selected else SURFACE)
            for child in ("name", "desc", "duration", "indicator"):
                w[child].configure(bg=RED_SOFT if selected else SURFACE)
            w["indicator"].configure(
                text="●" if selected else "", fg=RED)

    # --------------------------------------------------- Video tab
    def _build_video_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Video  ")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        main = make_card(tab, padding=24)
        main.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        section_title(main.inner, "Downloaded Video").pack(anchor="w", pady=(0, 8))
        help_text(
            main.inner,
            "Choose a video file from your computer. It will play here inside MotionBloom.",
            wraplength=1000,
        ).pack(anchor="w", pady=(0, 20))

        # Horizontal stage: video player on left, live camera preview on right
        stage = Frame(main.inner, bg=SURFACE)
        stage.pack(fill="both", expand=True, pady=(0, 16))

        # Live camera preview on the right (what the model sees while scoring)
        cam_col = Frame(stage, bg=SURFACE, width=340)
        cam_col.pack(side="right", fill="y", padx=(16, 0))
        cam_col.pack_propagate(False)
        Label(
            cam_col,
            text="Your Camera",
            fg=TEXT,
            bg=SURFACE,
            font=(FONT_FAMILY, 13, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))
        Label(
            cam_col,
            text="Live view the model is scoring",
            fg=MUTED,
            bg=SURFACE,
            font=(FONT_FAMILY, 11),
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))
        self.video_cam_preview = Label(cam_col, bg="#000000", width=320, height=240)
        self.video_cam_preview.pack(fill="both", expand=True)

        # Use Canvas for VLC video rendering (not Frame+Label)
        self.video_surface = Canvas(stage, bg="#000000", cursor="hand2", highlightthickness=0)
        self.video_surface.pack(side="left", fill="both", expand=True)
        self.video_surface.bind("<Button-1>", lambda _e: self._toggle_video_playback())

        # Overlay label shown when no video is loaded
        self.video_player_label = Label(
            self.video_surface,
            text="Choose a downloaded video",
            fg="#ffffff",
            bg="#000000",
            font=(FONT_FAMILY, 24, "bold"),
            justify="center",
            cursor="hand2",
        )
        self.video_player_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_player_label.bind("<Button-1>", lambda _e: self._toggle_video_playback())

        controls = Frame(main.inner, bg=SURFACE)
        controls.pack(fill="x", pady=(0, 10))
        HoverButton(controls, text="Choose Video", variant="primary", command=self._choose_video_file).pack(side="left", padx=(0, 8))
        self.video_play_pause_btn = HoverButton(controls, text="Pause", variant="neutral", command=self._toggle_video_playback)
        self.video_play_pause_btn.pack(side="left", padx=(0, 8))
        self.video_end_session_btn = HoverButton(controls, text="End Session", variant="outline", command=self._end_session_clicked)
        self.video_end_session_btn.pack(side="left", padx=(0, 8))
        self.video_gate_btn = HoverButton(controls, text="Enable Motion Gate", variant="neutral", command=self._toggle_motion_gate)
        self.video_gate_btn.pack(side="left", padx=(0, 8))
        Label(
            controls,
            textvariable=self.video_time_var,
            fg=TEXT,
            bg=SURFACE,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(side="right")

        self.video_progress = ttk.Scale(
            main.inner,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.video_progress_var,
            command=self._on_video_progress_drag,
        )
        self.video_progress.pack(fill="x", pady=(0, 14))
        self.video_progress.bind("<ButtonPress-1>", self._on_video_seek_start)
        self.video_progress.bind("<ButtonRelease-1>", self._on_video_seek_end)

        status_frame = Frame(main.inner, bg=SURFACE)
        status_frame.pack(fill="x")
        Label(
            status_frame,
            textvariable=self.video_file_var,
            fg=MUTED,
            bg=SURFACE,
            font=(FONT_FAMILY, 13),
            anchor="w",
            justify="left",
        ).pack(anchor="w", fill="x", pady=(0, 6))
        Label(
            status_frame,
            textvariable=self.video_status_var,
            fg=TEXT,
            bg=SURFACE,
            font=(FONT_FAMILY, 14, "bold"),
            anchor="w",
        ).pack(anchor="w", fill="x", pady=(0, 16))

        challenge_card = Frame(main.inner, bg=RED_SOFT, highlightthickness=1, highlightbackground=RED_TINT)
        challenge_card.pack(fill="x")
        Label(
            challenge_card,
            textvariable=self.video_challenge_prompt_var,
            fg=RED_DEEP,
            bg=RED_SOFT,
            font=(FONT_FAMILY, 15, "bold"),
            wraplength=1000,
            justify="left",
            padx=16,
        ).pack(anchor="w", fill="x", pady=(10, 5))
        Label(
            challenge_card,
            textvariable=self.video_challenge_status_var,
            fg=TEXT,
            bg=RED_SOFT,
            font=(FONT_FAMILY, 13),
            wraplength=1000,
            justify="left",
            padx=16,
        ).pack(anchor="w", fill="x", pady=(0, 10))

    # ---------------------------------------------------- Chart tabs
    def _build_spectrum_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Spectrum  ")
        card = make_card(tab, padding=14)
        card.pack(fill="both", expand=True, padx=0, pady=10)

        section_title(card.inner,
                  "Signal detail - rhythm strength by frequency").pack(
            anchor="w", pady=(0, 8))

        self.spec_fig = Figure(figsize=(8, 5), dpi=100, facecolor=SURFACE)
        self.spec_ax = self.spec_fig.add_subplot(111)
        _style_axes(self.spec_ax, "Frequency (Hz)", "Power (a.u.)")
        self.spec_fig.subplots_adjust(left=0.08, right=0.98,
                                      top=0.96, bottom=0.1)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=card.inner)
        self.spec_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_spectrogram_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Spectrogram  ")
        card = make_card(tab, padding=14)
        card.pack(fill="both", expand=True, padx=0, pady=10)

        section_title(card.inner,
                  "Deep view - rhythm over time").pack(
            anchor="w", pady=(0, 8))

        self.sg_fig = Figure(figsize=(8, 5), dpi=100, facecolor=SURFACE)
        self.sg_ax = self.sg_fig.add_subplot(111)
        _style_axes(self.sg_ax, "Time (s)", "Frequency (Hz)")
        self.sg_fig.subplots_adjust(left=0.08, right=0.98,
                                    top=0.96, bottom=0.1)
        self.sg_canvas = FigureCanvasTkAgg(self.sg_fig, master=card.inner)
        self.sg_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_history_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  History  ")
        card = make_card(tab, padding=14)
        card.pack(fill="both", expand=True, padx=0, pady=10)

        section_title(card.inner,
                  "History - score, rhythm, and movement size").pack(
            anchor="w", pady=(0, 8))

        self.hist_fig = Figure(figsize=(8, 5), dpi=100, facecolor=SURFACE)
        self.hist_ax1 = self.hist_fig.add_subplot(311)
        self.hist_ax2 = self.hist_fig.add_subplot(312, sharex=self.hist_ax1)
        self.hist_ax3 = self.hist_fig.add_subplot(313, sharex=self.hist_ax1)
        for ax, ylab in [(self.hist_ax1, "Score"),
                         (self.hist_ax2, "Peak Hz"),
                         (self.hist_ax3, "Amp (mm)")]:
            _style_axes(ax, "", ylab)
        self.hist_ax3.set_xlabel("Time (s)", color=MUTED, fontsize=9)
        self.hist_fig.subplots_adjust(left=0.1, right=0.98,
                                      top=0.97, bottom=0.08, hspace=0.35)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=card.inner)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------ actions
    def _enable_video_drop(self, widget: Frame) -> None:
        try:
            widget.tk.call("package", "require", "tkdnd")
            widget.tk.call("tkdnd::drop_target", "register", widget, "DND_Files")
            widget.bind("<<Drop>>", self._on_video_drop)
        except Exception:
            pass

    def _on_video_drop(self, event) -> None:
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [str(getattr(event, "data", ""))]
        if not paths:
            return
        self._set_video_file(paths[0])

    def _build_reports_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Reports  ")
        card = make_card(tab, padding=14)
        card.pack(fill="both", expand=True, padx=0, pady=10)

        section_title(card.inner, "Video session reports").pack(anchor="w", pady=(0, 8))
        Label(
            card.inner,
            text="Every video you play records an average tremor score. Lower is better.",
            fg=MUTED, bg=SURFACE, font=(FONT_FAMILY, 11), justify="left",
        ).pack(anchor="w", pady=(0, 10))

        # Summary row
        summary = Frame(card.inner, bg=SURFACE)
        summary.pack(fill="x", pady=(0, 12))
        self.report_summary_var = StringVar(value="No sessions yet")
        Label(
            summary, textvariable=self.report_summary_var,
            fg=TEXT, bg=SURFACE, font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w")

        # Session table
        cols = ("date", "video", "avg", "min", "max", "samples", "duration", "recording")
        headings = {
            "date": "Date",
            "video": "Video",
            "avg": "Avg Score",
            "min": "Min",
            "max": "Max",
            "samples": "Samples",
            "duration": "Duration (s)",
            "recording": "Recording",
        }
        widths = {"date": 140, "video": 200, "avg": 80, "min": 50, "max": 50,
                  "samples": 70, "duration": 90, "recording": 160}
        tree_frame = Frame(card.inner, bg=SURFACE)
        tree_frame.pack(fill="both", expand=True)
        self.report_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
        for c in cols:
            self.report_tree.heading(c, text=headings[c])
            self.report_tree.column(c, width=widths[c], anchor="w")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=vsb.set)
        self.report_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        # Double-click a row to open the recorded webcam clip
        self.report_tree.bind("<Double-1>", self._open_selected_recording)

        # Action row
        actions = Frame(card.inner, bg=SURFACE)
        actions.pack(fill="x", pady=(10, 0))
        HoverButton(actions, text="Refresh", variant="neutral",
                    command=self._refresh_reports).pack(side="left", padx=(0, 8))
        HoverButton(actions, text="Open Recording", variant="neutral",
                    command=self._open_selected_recording).pack(side="left", padx=(0, 8))
        HoverButton(actions, text="Open Folder", variant="neutral",
                    command=self._open_recordings_folder).pack(side="left", padx=(0, 8))
        HoverButton(actions, text="Clear All", variant="neutral",
                    command=self._clear_reports).pack(side="left")

        self._refresh_reports()

    def _refresh_reports(self) -> None:
        if not hasattr(self, "report_tree"):
            return
        # Clear existing rows
        for iid in self.report_tree.get_children():
            self.report_tree.delete(iid)
        records = self.report_store.all_records()
        self._report_row_to_record: dict[str, "SessionRecord"] = {}
        for rec in reversed(records):  # newest first
            rec_label = Path(rec.recording_path).name if rec.recording_path else "-"
            iid = self.report_tree.insert(
                "", "end",
                values=(
                    rec.started_at.replace("T", " "),
                    rec.video_name or "(unknown)",
                    f"{rec.avg_score:.1f}" if rec.avg_score is not None else "-",
                    rec.min_score if rec.min_score is not None else "-",
                    rec.max_score if rec.max_score is not None else "-",
                    rec.samples,
                    f"{rec.duration_sec:.1f}",
                    rec_label,
                ),
            )
            self._report_row_to_record[iid] = rec
        # Summary
        n = len(records)
        if n == 0:
            self.report_summary_var.set("No sessions recorded yet. Play a video on the Video tab to start a session.")
            return
        avg = self.report_store.overall_average()
        best = self.report_store.best_score()
        worst = self.report_store.worst_score()
        parts = [f"Sessions: {n}"]
        if avg is not None:
            parts.append(f"Lifetime average: {avg:.1f}")
        if best is not None:
            parts.append(f"Best (lowest): {best:.1f}")
        if worst is not None:
            parts.append(f"Worst (highest): {worst:.1f}")
        self.report_summary_var.set("   ".join(parts))

    def _clear_reports(self) -> None:
        if not messagebox.askyesno("Clear Reports",
                                   "Delete all recorded video sessions?\nThis cannot be undone."):
            return
        self.report_store.clear()
        self._refresh_reports()

    def _open_selected_recording(self, _event=None) -> None:
        """Open the webcam recording for the selected report row."""
        if not hasattr(self, "report_tree"):
            return
        sel = self.report_tree.selection()
        if not sel:
            messagebox.showinfo("Recording", "Select a session row first.")
            return
        rec = getattr(self, "_report_row_to_record", {}).get(sel[0])
        if rec is None or not rec.recording_path:
            messagebox.showinfo("Recording", "No webcam recording was saved for that session.")
            return
        p = Path(rec.recording_path)
        if not p.exists():
            messagebox.showwarning("Recording", f"File not found:\n{p}")
            return
        try:
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as exc:
            messagebox.showwarning("Recording", f"Could not open file:\n{exc}")

    def _open_recordings_folder(self) -> None:
        folder = Path.home() / ".motionbloom" / "recordings"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            messagebox.showwarning("Recordings", f"Could not open folder:\n{exc}")

    def _choose_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            print(f"[VIDEO] User selected file: {path}", flush=True)
            self._set_video_file(path)
        else:
            print(f"[VIDEO] User cancelled file selection", flush=True)

    def _set_video_file(self, path: str) -> None:
        try:
            video_path = Path(path).expanduser()
            print(f"[VIDEO] _set_video_file called with: {video_path}", flush=True)
            self._stop_local_video(clear_frame=False)
            self.video_file_var.set(str(video_path))
            self.video_status_var.set("Loading video inside MotionBloom.")
            self.video_challenge_prompt_var.set("Motion Gate is off")
            self.video_challenge_status_var.set("Video will play continuously unless Motion Gate is enabled.")
            self.status_var.set("Video selected")
            self._open_selected_video()
        except Exception as exc:
            print(f"[VIDEO] _set_video_file failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Video Error", f"Failed to load video.\n\n{type(exc).__name__}: {exc}")
            self.video_status_var.set(f"Failed to load video: {type(exc).__name__}")

    def _format_video_time(self, ms: int) -> str:
        seconds = max(0, int(ms / 1000))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _show_video_overlay(self, text: str) -> None:
        if hasattr(self, "video_player_label"):
            self.video_player_label.configure(image="", text=text, fg="#ffffff", bg="#000000")
            self.video_player_label.image = None
            self.video_player_label.place(relx=0.5, rely=0.5, anchor="center")
            self.video_player_label.lift()

    def _hide_video_overlay(self) -> None:
        if hasattr(self, "video_player_label"):
            self.video_player_label.place_forget()

    def _update_video_buttons(self) -> None:
        if not hasattr(self, "video_play_pause_btn"):
            return
        if self.video_player is None:
            self.video_play_pause_btn.configure(text="Play")
            return
        self.video_play_pause_btn.configure(text="Pause" if self.video_player.is_playing() else "Play")

    def _ensure_video_player(self) -> LocalVideoPlayer:
        if self.video_player is None:
            self.root.update_idletasks()
            try:
                self.video_player = create_video_player(self.video_surface)
                print(f"[VIDEO] Created player: {type(self.video_player).__name__}", flush=True)
            except Exception as exc:
                print(f"[VIDEO] Failed to create player: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Could not initialize video player: {exc}") from exc
        return self.video_player

    def _video_audio_status_text(self, player: LocalVideoPlayer, *, early: bool = False) -> str:
        if not player.audio_supported:
            return "OpenCV fallback: no audio support."
        state = player.debug_state()
        count = state.get("audio_track_count")
        has_audio = bool(state.get("has_audio_stream"))
        if has_audio or (isinstance(count, int) and count > 0):
            return "VLC audio track detected."
        if early:
            return "VLC loaded, no audio track detected yet."
        return "VLC loaded, no audio track found."

    def _log_video_audio_state(self, label: str = "app_delayed") -> None:
        player = self.video_player
        if player is None:
            return
        player.log_debug_state(label)
        self.video_status_var.set(self._video_audio_status_text(player))

    def _open_selected_video(self) -> None:
        path_text = self.video_file_var.get()
        if not path_text or path_text == "No video selected":
            messagebox.showinfo("Video", "Choose a video file first.")
            return
        path = Path(path_text).expanduser()
        if not path.exists():
            messagebox.showwarning("Video", "That video file could not be found.")
            return
        self._stop_local_video(clear_frame=False)
        player = self._ensure_video_player()
        try:
            player.load(str(path))
        except Exception as exc:
            print(f"[VIDEO] Load failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            messagebox.showwarning("Video", f"Could not load video file.\n\n{type(exc).__name__}: {exc}")
            self.video_status_var.set(f"Load failed: {type(exc).__name__}")
            return
        self.video_loaded_path = str(path)
        # Begin a new reports session for this video
        try:
            rec = self.report_store.start_session(str(path))
            # Start recording the webcam to a local file for this session
            try:
                rec_dir = Path.home() / ".motionbloom" / "recordings"
                rec_dir.mkdir(parents=True, exist_ok=True)
                rec_path = rec_dir / f"{rec.id}.mp4"
                fps = getattr(self.tracker, "fps_meas", 0.0) or 20.0
                self.tracker.start_recording(str(rec_path), fps=fps)
                rec.recording_path = str(rec_path)
                print(f"[REC] started -> {rec_path}", flush=True)
            except Exception as rec_exc:
                print(f"[REC] start_recording failed: {rec_exc}", flush=True)
        except Exception as exc:
            print(f"[REPORTS] start_session failed: {exc}", flush=True)
        self._clear_video_challenge()
        self.video_motion_gate_enabled.set(False)
        if hasattr(self, "video_gate_btn"):
            self.video_gate_btn.configure(text="Enable Motion Gate")
        if player.audio_supported:
            self._hide_video_overlay()
            self.video_status_var.set(self._video_audio_status_text(player, early=True))
        else:
            self._show_video_overlay("Preview mode\nAudio unavailable")
            self.video_status_var.set(player.warning or self._video_audio_status_text(player))
        self.video_challenge_prompt_var.set("Motion Gate is off")
        self.video_challenge_status_var.set("Enable Motion Gate only when you want hand-action pauses.")
        try:
            player.play()
        except Exception as exc:
            print(f"[VIDEO] Play failed: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            messagebox.showwarning("Video", f"Could not start playback.\n\n{type(exc).__name__}: {exc}")
            self.video_status_var.set(f"Play failed: {type(exc).__name__}")
            return
        self.root.after(750, lambda: self._log_video_audio_state("app_after_750ms"))
        self._start_video_progress_poll()
        self._update_video_buttons()
        self.status_var.set("Playing video")

    def _toggle_video_playback(self) -> None:
        if self.video_player is None:
            self._choose_video_file()
            return
        self._pause_local_video()

    def _pause_local_video(self) -> None:
        if self.video_player is None:
            self.video_status_var.set("Choose a downloaded video first.")
            return
        if self.video_challenge_active:
            self.video_status_var.set("Finish the hand action to continue.")
            return
        if self.video_player.is_playing():
            self.video_player.pause()
            self.video_status_var.set("Paused. Click the video or Play to continue.")
            self.status_var.set("Video paused")
        else:
            self.video_player.play()
            self.root.after(750, lambda: self._log_video_audio_state("app_resume_after_750ms"))
            self.video_status_var.set(self._video_audio_status_text(self.video_player, early=True))
            self.status_var.set("Playing video")
            if self.video_motion_gate_enabled.get() and not self.video_challenge_active:
                self.video_next_challenge_at = time.time() + 3.0
        self._start_video_progress_poll()
        self._update_video_buttons()

    def _end_session_clicked(self) -> None:
        """User-triggered end of the current video/scoring session."""
        active = getattr(self.report_store, "_active", None) is not None
        if not active:
            self.video_status_var.set("No active session to end.")
            self.status_var.set("No active session")
            return
        # Stop playback and finalize the report session
        self._stop_local_video(clear_frame=False)
        self.video_status_var.set("Session ended. See the Reports tab for your score.")
        self.status_var.set("Session ended")
        # Auto-switch to Reports tab so the user sees the result
        try:
            for i, tab_id in enumerate(self.nb.tabs()):
                if self.nb.tab(tab_id, "text").strip().lower().startswith("reports"):
                    self.nb.select(i)
                    break
        except Exception:
            pass

    def _stop_local_video(self, clear_frame: bool = True) -> None:
        # Stop webcam recording first so the file is finalized before
        # the reports session is closed.
        try:
            rec_path, n_frames = self.tracker.stop_recording()
            if rec_path:
                print(f"[REC] stopped: {rec_path} frames={n_frames}", flush=True)
                active = getattr(self.report_store, "_active", None)
                if active is not None:
                    active.recording_path = rec_path
        except Exception as exc:
            print(f"[REC] stop_recording failed: {exc}", flush=True)
        # End any active reports session
        try:
            if self.report_store.end_session() is not None and hasattr(self, "report_tree"):
                self._refresh_reports()
        except Exception as exc:
            print(f"[REPORTS] end_session failed: {exc}", flush=True)
        if self.local_video_after_id is not None:
            try:
                self.root.after_cancel(self.local_video_after_id)
            except Exception:
                pass
            self.local_video_after_id = None
        if self.video_progress_after_id is not None:
            try:
                self.root.after_cancel(self.video_progress_after_id)
            except Exception:
                pass
            self.video_progress_after_id = None
        self.local_video_playing = False
        self._clear_video_challenge()
        self.video_motion_gate_enabled.set(False)
        if hasattr(self, "video_gate_btn"):
            self.video_gate_btn.configure(text="Enable Motion Gate")
        if self.video_player is not None:
            self.video_player.cleanup()
            self.video_player = None
        self.video_loaded_path = None
        if self.local_video_cap is not None:
            self.local_video_cap.release()
            self.local_video_cap = None
        if clear_frame and hasattr(self, "video_player_label"):
            self._show_video_overlay("Choose a downloaded video")
            self.local_video_photo = None
            self.video_status_var.set("Stopped.")
            self.video_progress_var.set(0.0)
            self.video_time_var.set("0:00 / 0:00")
            self.video_challenge_prompt_var.set("Motion Gate is off")
            self.video_challenge_status_var.set("Choose a video to begin.")
        self._update_video_buttons()

    def _local_video_loop(self) -> None:
        self._poll_video_progress()

    def _start_video_progress_poll(self) -> None:
        if self.video_progress_after_id is None:
            self.video_progress_after_id = self.root.after(100, self._poll_video_progress)

    def _poll_video_progress(self) -> None:
        self.video_progress_after_id = None
        player = self.video_player
        if player is None:
            return
        try:
            current_ms = player.get_time_ms()
            duration_ms = player.get_duration_ms()
            if not self.video_seeking:
                self.video_progress_var.set(player.get_position_ratio())
            self.video_time_var.set(f"{self._format_video_time(current_ms)} / {self._format_video_time(duration_ms)}")
            frame = player.render_frame()
            if frame is not None and hasattr(self, "video_player_label"):
                # Show label on top of Canvas to display frame as image
                if not self.video_player_label.winfo_ismapped():
                    self.video_player_label.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
                self.video_player_label.configure(text="", bg="#000000")
                self._render_video(self.video_player_label, frame)
            self._update_video_buttons()
        except Exception as exc:
            print(f"[VIDEO] poll error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        # Schedule next frame at ~30 FPS for smooth playback
        self.video_progress_after_id = self.root.after(33, self._poll_video_progress)

    def _on_video_progress_drag(self, _value=None) -> None:
        if self.video_seeking:
            ratio = float(self.video_progress_var.get())
            duration_ms = self.video_player.get_duration_ms() if self.video_player is not None else 0
            self.video_time_var.set(f"{self._format_video_time(int(duration_ms * ratio))} / {self._format_video_time(duration_ms)}")

    def _on_video_seek_start(self, _event=None) -> None:
        self.video_seeking = True

    def _on_video_seek_end(self, _event=None) -> None:
        if self.video_player is not None:
            self.video_player.seek_ratio(float(self.video_progress_var.get()))
        self.video_seeking = False
        self._start_video_progress_poll()

    def _toggle_motion_gate(self) -> None:
        if self.video_player is None:
            self.video_status_var.set("Choose a downloaded video first.")
            return
        enabled = not self.video_motion_gate_enabled.get()
        self.video_motion_gate_enabled.set(enabled)
        self._clear_video_challenge()
        if enabled:
            if hasattr(self, "video_gate_btn"):
                self.video_gate_btn.configure(text="Disable Motion Gate")
            self.video_next_challenge_at = time.time() + 3.0
            self.video_challenge_prompt_var.set("Motion Gate is on")
            self.video_challenge_status_var.set("First hand action starts in about 3 seconds.")
            self.video_status_var.set("Playing with Motion Gate enabled.")
        else:
            if hasattr(self, "video_gate_btn"):
                self.video_gate_btn.configure(text="Enable Motion Gate")
            self.video_challenge_prompt_var.set("Motion Gate is off")
            self.video_challenge_status_var.set("Video will play continuously.")
            self.video_status_var.set("Playing with audio inside MotionBloom." if self.video_player.audio_supported else "Preview mode. Audio is unavailable.")

    def _clear_video_challenge(self) -> None:
        self.video_challenge_active = False
        self.video_challenge_action = None
        self.video_challenge_score_target = None
        self.video_challenge_started_at = None
        self.video_next_challenge_at = None

    def _current_video_score(self, metrics=None) -> float | None:
        source = metrics or self.last_metrics
        if source is None:
            return None
        value = getattr(source, "live_motion_score", None)
        if value is None:
            value = getattr(source, "final_tremor_score", None)
        if value is None:
            value = getattr(source, "score", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _start_video_challenge(self) -> None:
        if not self.video_motion_gate_enabled.get() or self.video_player is None:
            self.video_next_challenge_at = None
            return
        self.video_challenge_active = True
        if self.video_player.is_playing():
            self.video_player.pause()
        self.local_video_playing = False
        self.local_video_after_id = None
        self.video_challenge_started_at = time.time()
        self.video_challenge_score_target = self._current_video_score()
        self.video_challenge_action = random.choice(("nose", "hold_object", "squeeze_ball"))
        prompts = {
            "nose": "Touch your nose",
            "hold_object": "Hold object steadily",
            "squeeze_ball": "Squeeze rubber ball",
        }
        baseline = "previous score" if self.video_challenge_score_target is None else f"previous score {self.video_challenge_score_target:.0f}"
        self.video_challenge_prompt_var.set(prompts[self.video_challenge_action])
        self.video_challenge_status_var.set(f"Video paused. Do the action, then lower your hand score below the {baseline}.")
        self.video_status_var.set("Paused for hand action")
        self.status_var.set("Do the hand action")
        self._update_video_buttons()

    def _verify_video_challenge_action(self) -> tuple[bool, str]:
        pose = self.tracker.get_pose()
        tip = self.tracker.get_hand_tip_norm()
        grip = self.tracker.get_grip_strength()
        action = self.video_challenge_action
        if action == "nose":
            result = verify_touch_nose(pose, tip)
            return result.ok, result.message
        if action == "hold_object":
            result = verify_hold_object(pose, tip, grip=grip)
            return result.ok, result.message
        if action == "squeeze_ball":
            if tip is None:
                return False, "Show your hand to the camera"
            if grip >= 0.65:
                return True, "Good squeeze. Now relax until your score drops."
            return False, "Squeeze the ball harder"
        return False, "Waiting for action"

    def _update_video_challenge(self, metrics) -> None:
        if not self.video_motion_gate_enabled.get():
            return
        if not self.video_challenge_active:
            if (
                self.video_player is not None
                and self.video_player.is_playing()
                and self.video_next_challenge_at is not None
                and time.time() >= self.video_next_challenge_at
            ):
                self._start_video_challenge()
            return
        action_ok, action_message = self._verify_video_challenge_action()
        current_score = self._current_video_score(metrics)
        target_score = self.video_challenge_score_target
        if not action_ok:
            self.video_challenge_status_var.set(action_message)
            return
        if current_score is None:
            self.video_challenge_status_var.set("Action found. Waiting for a clear score.")
            return
        if target_score is None:
            target_score = current_score + 0.1
        if current_score < target_score:
            self.video_challenge_active = False
            self.video_challenge_action = None
            self.video_challenge_score_target = None
            self.video_challenge_started_at = None
            self.video_next_challenge_at = time.time() + 3.0
            self.video_challenge_prompt_var.set("Good job")
            if self.video_player is not None:
                self.video_player.play()
                self.video_challenge_status_var.set(f"Score improved to {current_score:.0f}. Video continuing.")
                self.video_status_var.set("Playing with Motion Gate enabled.")
                self._start_video_progress_poll()
                self._update_video_buttons()
            else:
                self.video_challenge_status_var.set(f"Score improved to {current_score:.0f}.")
                self.video_status_var.set("Motion Gate ready.")
            self.status_var.set("Playing video")
        else:
            self.video_challenge_status_var.set(
                f"{action_message} Current score {current_score:.0f}; get below {target_score:.0f} to continue."
            )

    def _search_youtube(self) -> None:
        query = self.youtube_query_var.get().strip()
        if not query:
            messagebox.showinfo("YouTube", "Type what you want to find first.")
            return
        self.youtube_status_var.set(f"Search saved here: {query}. YouTube playback needs an approved in-app player.")
        self.status_var.set("YouTube search saved")

    def _extract_youtube_id(self, text: str) -> str | None:
        value = text.strip()
        if not value:
            return None
        if not value.startswith(("http://", "https://")):
            value = f"https://{value}"
        parsed = urlparse(value)
        host = parsed.netloc.lower().replace("www.", "")
        if host == "youtu.be":
            candidate = parsed.path.strip("/").split("/")[0]
            return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
        if host.endswith("youtube.com"):
            if parsed.path == "/watch":
                candidate = parse_qs(parsed.query).get("v", [None])[0]
                return candidate if candidate and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
            match = re.search(r"/(embed|shorts)/([A-Za-z0-9_-]{11})", parsed.path)
            if match:
                return match.group(2)
        match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", value)
        return match.group(1) if match else None

    def _youtube_watch_url(self) -> str | None:
        text = self.youtube_link_var.get().strip()
        video_id = self._extract_youtube_id(text)
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        if text:
            return f"https://www.youtube.com/results?search_query={quote_plus(text)}"
        return None

    def _load_youtube_video(self) -> None:
        text = self.youtube_link_var.get().strip()
        if not text:
            messagebox.showinfo("YouTube", "Paste a YouTube link or type a video name.")
            return
        self._stop_local_video(clear_frame=False)
        video_id = self._extract_youtube_id(text)
        if video_id:
            self.youtube_status_var.set(f"YouTube video loaded: {video_id}")
            self.video_status_var.set("Press Open YouTube to play. Press Start Hand Actions for the exercise pauses.")
            self.video_player_label.configure(
                image="",
                text="▶\nYouTube video ready",
                fg="#ffffff",
                bg="#000000",
            )
            self.video_player_label.image = None
        else:
            self.youtube_status_var.set(f"YouTube search ready: {text}")
            self.video_status_var.set("Press Open YouTube to search and play the video.")
            self.video_player_label.configure(
                image="",
                text="⌕\nYouTube search ready",
                fg="#ffffff",
                bg="#000000",
            )
            self.video_player_label.image = None
        self.video_challenge_prompt_var.set("Start Hand Actions when the video is playing.")
        self.video_challenge_status_var.set("MotionBloom will pause you with quick hand actions.")
        self.status_var.set("YouTube ready")

    def _open_youtube_video(self) -> None:
        url = self._youtube_watch_url()
        if not url:
            messagebox.showinfo("YouTube", "Paste a YouTube link or type a video name first.")
            return
        self._load_youtube_video()
        webbrowser.open(url)
        self.video_status_var.set("Playing in YouTube. Keep MotionBloom open for hand actions.")
        self.status_var.set("YouTube opened")

    def _start_youtube_challenges(self) -> None:
        if not self.youtube_link_var.get().strip():
            messagebox.showinfo("YouTube", "Load a YouTube video first.")
            return
        self._clear_video_challenge()
        self.video_next_challenge_at = time.time() + 3.0
        self.video_challenge_prompt_var.set("Hand actions are on")
        self.video_challenge_status_var.set("First action starts in about 3 seconds.")
        self.video_status_var.set("Hand actions running with YouTube.")
        self.status_var.set("YouTube hand actions on")

    def _stop_youtube_video(self) -> None:
        self._stop_local_video(clear_frame=False)
        self._clear_video_challenge()
        if hasattr(self, "video_player_label"):
            self.video_player_label.configure(image="", text="▶\nPaste a YouTube link", fg="#ffffff", bg="#000000")
            self.video_player_label.image = None
        self.youtube_status_var.set("Paste a YouTube link or type what you want to watch.")
        self.video_status_var.set("Stopped.")
        self.video_challenge_prompt_var.set("Start Hand Actions when the video is playing.")
        self.video_challenge_status_var.set("Load a YouTube video to begin.")
        self.status_var.set("YouTube stopped")

    def _open_youtube_link(self) -> None:
        link = self.youtube_link_var.get().strip()
        if not link:
            messagebox.showinfo("YouTube", "Paste a YouTube link first.")
            return
        if not link.startswith(("http://", "https://")):
            link = f"https://{link}"
        if "youtube.com" not in link and "youtu.be" not in link:
            messagebox.showwarning("YouTube", "Please paste a YouTube link.")
            return
        self.youtube_status_var.set("YouTube link saved here. It will not open a new window.")
        self.status_var.set("YouTube link saved")

    def _show_streaming_status(self, service: str) -> None:
        self.youtube_status_var.set(f"{service} needs its official approved player for in-app playback. No new window opened.")
        self.status_var.set(f"{service} stays in app")

    def _connect_video_service(self, service: str) -> None:
        if service == "YouTube":
            self.service_login_status_var.set(
                "YouTube can use official Google sign-in after a Google client ID is added. No password is stored here."
            )
            self.youtube_status_var.set("YouTube account connection is waiting for official Google sign-in setup.")
        else:
            self.service_login_status_var.set(
                f"{service} does not offer public in-app playback login. MotionBloom needs an official partner player before it can connect."
            )
        self.status_var.set(f"{service} login checked")

    def _on_landmark_change(self, _evt=None) -> None:
        self.tracker.set_landmark(LANDMARK_CHOICES.get(self.lm_var.get(), 8))

    def _on_task_mode_change(self, _evt=None) -> None:
        self.current_task_mode = self.task_mode_options.get(
            self.task_mode_var.get(), TaskMode.POSTURAL_GENERAL
        )
        self.status_var.set(f"Check: {self.task_mode_var.get()}")

    def _on_tab_change(self) -> None:
        self._refresh_active_chart(force=True)
        titles = ["Check My Hand", "Video"]
        try:
            idx = self.nb.index("current")
            if 0 <= idx < len(titles):
                self.page_title_var.set(titles[idx])
        except Exception:
            pass

    def _toggle_record(self) -> None:
        self.recording = self.record_var.get()
        self.status_var.set("Recording" if self.recording else "Capturing…"
                            if self.tracker.thread else "Idle")

    def export_csv(self) -> None:
        if not self.session_rows:
            messagebox.showinfo("Export",
                                "No recorded data yet. Tick "
                                "\"Record session\" first and capture "
                                "for a bit.")
            return
        default = Path.home() / f"motionbloom_{int(time.time())}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=str(default.name),
            initialdir=str(default.parent),
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        keys = list(self.session_rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(self.session_rows)
        messagebox.showinfo("Export",
                            f"Saved {len(self.session_rows)} rows to\n{path}")

    def calibrate(self) -> None:
        snap = self.tracker.snapshot(3.0)
        if snap is None:
            messagebox.showwarning(
                "Save Calm Hand",
                "Please hold your hand steady for about 3 seconds, then try "
                "again.")
            return
        t, x, y, _ = snap
        fs_est = float(np.clip(len(t) / max(1e-3, t[-1] - t[0]), 15, 60))
        res = resample_uniform(t, x, y, fs_est)
        if res is None:
            return
        _, xu, yu = res
        xf = bandpass(highpass(xu, fs_est), fs_est)
        yf = bandpass(highpass(yu, fs_est), fs_est)
        rms = float(np.sqrt(np.mean(xf * xf + yf * yf)))
        self.baseline_rms = rms
        self.status_var.set("Calm hand saved")

    def _open_camera(self):
        """Try several camera indices and backends. Return an opened
        VideoCapture, or None if nothing worked."""
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for index in range(4):
            for backend in backends:
                try:
                    with CV_LOCK:
                        cap = cv2.VideoCapture(index, backend)
                except Exception:
                    continue
                if not cap or not cap.isOpened():
                    if cap is not None:
                        with CV_LOCK:
                            cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                ok = False
                for _ in range(30):
                    with CV_LOCK:
                        ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    return cap
                with CV_LOCK:
                    cap.release()
        return None

    def start(self) -> None:
        cap = self._open_camera()
        if cap is None:
            if sys.platform.startswith("win"):
                guidance = (
                    "We couldn't access your webcam.\n\n"
                    "On Windows, open Settings → Privacy & security → "
                    "Camera and make sure camera access is turned on for "
                    "this app (and for desktop apps). Also close any other "
                    "app that may be using the camera (Teams, Zoom, "
                    "Camera app), then try again."
                )
            elif sys.platform == "darwin":
                guidance = (
                    "We couldn't access your webcam.\n\n"
                    "On macOS, grant camera permission in System Settings "
                    "→ Privacy & Security → Camera for MotionBloom (or the "
                    "app running Python), then try again."
                )
            else:
                guidance = (
                    "We couldn't access your webcam. Please check that a "
                    "camera is connected, that no other app is using it, "
                    "and that this app has permission to use it, then try "
                    "again."
                )
            messagebox.showerror("Camera unavailable", guidance)
            return
        self.tracker.set_landmark(LANDMARK_CHOICES[self.lm_var.get()])
        self.tracker.start(cap)
        self._t0 = time.time()
        self.hist_t.clear(); self.hist_score.clear()
        self.hist_peak.clear(); self.hist_amp.clear()
        self._reset_hand_motion_state()
        if hasattr(self, "start_btn"):
            self.start_btn.set_disabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.set_disabled(False)
        if hasattr(self, "calib_btn"):
            self.calib_btn.set_disabled(False)
        if hasattr(self, "ex_start_btn"):
            self.ex_start_btn.set_disabled(False)
        self.status_var.set("Capturing…")

    def stop(self) -> None:
        self.tracker.stop()
        self._reset_hand_motion_state()
        if hasattr(self, "start_btn"):
            self.start_btn.set_disabled(False)
        if hasattr(self, "stop_btn"):
            self.stop_btn.set_disabled(True)
        if hasattr(self, "calib_btn"):
            self.calib_btn.set_disabled(True)
        if hasattr(self, "ex_start_btn"):
            self.ex_start_btn.set_disabled(True)
        self.status_var.set("Idle")
        self.video_label.configure(image="")
        if hasattr(self, "ex_video_label"):
            self.ex_video_label.configure(image="")
        for v in self.metric_vars.values():
            v.set("-")
        self.score_var.set("-")
        self.verdict_var.set("Place your hand in view. The camera will start automatically.")
        self.verdict_lbl.configure(bg=RED_SOFT, fg=RED)
        self.active_exercise_key = None
        if hasattr(self, "ex_stage_var"):
            self.ex_stage_var.set("Ready when you are.")
            self.ex_feedback_var.set("")
            self.ex_progress["value"] = 0

    def _reset_hand_motion_state(self) -> None:
        now = time.time()
        self.hand_motion_state = HAND_STATE_NO_HAND
        self.hand_motion_state_since = now
        self.hand_steady_since = None
        self.hand_moving_since = None
        self.last_stable_score = None
        self.last_stable_status = "No clear shaking reading yet"
        self.last_stable_metrics = None

    def _advance_hand_motion_state(self, palm_gate: dict | None, now: float) -> dict:
        state = classify_hand_motion_state(
            palm_gate,
            now,
            self.hand_motion_state,
            self.hand_motion_state_since,
            self.hand_steady_since,
            self.hand_moving_since,
        )
        self.hand_motion_state = state["state"]
        self.hand_motion_state_since = state["state_since"]
        self.hand_steady_since = state["steady_since"]
        self.hand_moving_since = state["moving_since"]
        return state

    def _format_palm_motion_label(self, state: str, palm_gate: dict | None) -> str:
        if palm_gate is None:
            return "No hand"
        if state == HAND_STATE_MOVING_WIDE:
            return "Too much movement"
        if state == HAND_STATE_MOVING:
            return "Moving"
        if state == HAND_STATE_DRIFTING:
            return "Moving slowly"
        if state == HAND_STATE_SETTLING:
            return "Almost still"
        if state == HAND_STATE_ACTIVE:
            return "Still enough"
        return "-"

    def _apply_paused_tremor_ui(self, state: str, message: str, palm_gate: dict | None = None) -> None:
        m = self.metric_vars
        m["palm_motion"].set(self._format_palm_motion_label(state, palm_gate))
        m["validity"].set("Getting ready")
        m["class"].set(state.replace("_", " ").title())
        m["confidence"].set("Building")
        m["movement_mode"].set(self.task_mode_var.get())
        m["movement_score"].set("-")
        m["tremor_overlay"].set("-")
        m["dominant_tremor"].set("-")
        m["tremor_source"].set("-")
        for key in (
            "peak", "amp", "amp_mm", "band", "snr", "reg", "sharp",
            "fs", "samples", "live_score", "tremor_candidate", "prominence",
            "fps_cv", "velocity", "path_ratio", "center_drift",
            "relative_displacement", "relative_power", "raw_fingertip_power",
            "relative_peaks", "relative_agreement", "relative_veto",
        ):
            m[key].set("-")
        if self.last_stable_score is None:
            self.score_var.set("-")
            self.verdict_var.set(f"Current status: {message}\n{self.last_stable_status}")
        else:
            self.score_var.set(f"{self.last_stable_score}")
            self.verdict_var.set(
                f"Last clear shaking score: {self.last_stable_score}%\n"
                f"Current status: {message}"
            )
        self.score_lbl.configure(fg=MUTED)
        self.verdict_lbl.configure(bg=WARN_TINT, fg=WARN_DEEP)
        self.status_var.set(message)

    def _record_last_stable_score(self, metrics) -> None:
        if metrics.final_tremor_score is None:
            return
        self.last_stable_score = int(metrics.final_tremor_score)
        self.last_stable_status = (
            f"Last clear shaking score: {self.last_stable_score}%"
        )
        self.last_stable_metrics = metrics
        # Feed the score into the active reports session (if any)
        try:
            self.report_store.add_score(self.last_stable_score)
        except Exception as exc:
            print(f"[REPORTS] add_score failed: {exc}", flush=True)

    # --------------------------------------------------- exercise controls
    def _exercise_start(self) -> None:
        key = getattr(self, "active_exercise_key_selected", None)
        if key is None:
            return
        sess = self.exercises[key]
        sess.start()
        self.active_exercise_key = key
        self.ex_result_var.set("")
        self.ex_stage_var.set("Get into position…")
        self.ex_stage_lbl.configure(bg=RED_SOFT, fg=RED)

    def _exercise_cancel(self) -> None:
        if self.active_exercise_key:
            self.exercises[self.active_exercise_key].cancel()
        self.active_exercise_key = None
        self.ex_stage_var.set("Cancelled.")
        self.ex_stage_lbl.configure(bg=SURFACE_ALT, fg=MUTED)
        self.ex_feedback_var.set("")
        self.ex_progress["value"] = 0

    # --------------------------------------------------------- loops
    def _video_loop(self) -> None:
        try:
            frame = self.tracker.get_frame()
            if frame is not None:
                try:
                    idx = self.nb.index("current") if self.nb.tabs() else 0
                except Exception:
                    idx = 0
                try:
                    if idx == 0:
                        self._render_video(self.video_label, frame)
                    elif idx == 1 and hasattr(self, "video_cam_preview"):
                        self._render_video(self.video_cam_preview, frame)
                    elif hasattr(self, "ex_video_label"):
                        self._render_video(self.ex_video_label, frame)
                except Exception:
                    pass
                if self.tracker.fps_meas > 0:
                    self.fps_var.set(f"{self.tracker.fps_meas:.0f} fps")
        finally:
            self.root.after(VIDEO_MS, self._video_loop)

    def _render_video(self, label: Label, frame: np.ndarray) -> None:
        lbl_w = max(label.winfo_width(), 320)
        lbl_h = max(label.winfo_height(), 240)
        h, w = frame.shape[:2]
        scale = min(lbl_w / w, lbl_h / h)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
        photo = ImageTk.PhotoImage(image=img)
        label.configure(image=photo)
        label.image = photo  # keep ref

    def _analysis_loop(self) -> None:
        try:
            metrics = self._refresh_analysis()
            self.last_metrics = metrics
            self._refresh_active_chart()

            if metrics and self.recording:
                row = {"t": round(time.time() - self._t0, 3)}
                d = asdict(metrics)
                d.pop("samples", None)
                row.update(d)
                self.session_rows.append(row)

            self._update_exercise(metrics)
            self._update_video_challenge(metrics)
        finally:
            self.root.after(ANALYSIS_MS, self._analysis_loop)

    def _refresh_analysis(self):
        local_x = local_y = global_x = global_y = None
        tracking_quality_value = None
        box_width = box_height = box_area = None
        palm_x = palm_y = palm_w = palm_h = hand_box_size = None
        relative_fingertip_signals = None
        raw_fingertip_power = 0.0
        raw_primary_x = raw_primary_y = None

        snap_box = self.tracker.snapshot_box_micro(WINDOW_SECONDS)
        snap_palm = self.tracker.snapshot_palm_center(WINDOW_SECONDS)
        snap_relative = self.tracker.snapshot_palm_relative(WINDOW_SECONDS)
        # --- ROI optical-flow channel (Lucas-Kanade hand interior) ---
        snap_flow = self.tracker.snapshot_flow(WINDOW_SECONDS)
        flow_status = self.tracker.get_flow_status()
        tracking_source, tracking_reason = select_tracking_source(
            snap_flow, flow_status, palm_relative_available=snap_relative is not None
        )
        # Publish to UI immediately so it always reflects current state
        try:
            self.tracking_source_var.set(f"Source: {tracking_source}")
            if flow_status is not None:
                bg = "yes" if flow_status.get("bg_subtracted") else "no"
                self.flow_status_var.set(
                    f"Flow: q={flow_status.get('flow_quality', 0.0):.2f}  "
                    f"pts={int(flow_status.get('valid_points', 0))}  "
                    f"surv={flow_status.get('survival_rate', 0.0):.2f}  bg={bg}"
                )
            else:
                self.flow_status_var.set("Flow: warming up")
        except Exception:
            pass

        palm_gate = None
        if snap_palm is not None:
            _palm_t, raw_palm_x, raw_palm_y, _palm_w, _palm_h, raw_hand_box_size, _palm_quality = snap_palm
            palm_gate = compute_palm_center_motion_gate(
                raw_palm_x,
                raw_palm_y,
                raw_hand_box_size,
                frame_width=CAM_WIDTH,
                frame_height=CAM_HEIGHT,
                hand_box_width=_palm_w,
                hand_box_height=_palm_h,
            )
        hand_state = self._advance_hand_motion_state(palm_gate, time.time())

        # Try multi-fingertip first (more robust)
        snap_multi = self.tracker.snapshot_multi_finger(WINDOW_SECONDS)
        use_palm_relative = snap_relative is not None
        use_box_micro = snap_box is not None and not use_palm_relative
        use_multi_finger = snap_multi is not None and not use_palm_relative and not use_box_micro
        
        # Check buffer status first
        movement_count = len(self.tracker.box_micro_samples)
        relative_count = len(self.tracker.palm_relative_samples)
        multi_count = len(self.tracker.multi_finger_samples)
        single_count = len(self.tracker.samples)
        
        if use_palm_relative:
            (
                t,
                ix_rel, iy_rel, mid_rel_x, mid_rel_y, ring_rel_x, ring_rel_y,
                thumb_rel_x, thumb_rel_y, pinky_rel_x, pinky_rel_y,
                index_mcp_rel_x, index_mcp_rel_y, middle_mcp_rel_x, middle_mcp_rel_y,
                rel_palm_x, rel_palm_y, rel_hand_size, quality,
            ) = snap_relative
            x = (ix_rel + mid_rel_x + ring_rel_x) / 3.0
            y = (iy_rel + mid_rel_y + ring_rel_y) / 3.0
            global_x = rel_palm_x
            global_y = rel_palm_y
            local_x = x
            local_y = y
            confidence = quality
            tracking_quality_value = float(np.median(quality)) if len(quality) else 1.0
            ref = rel_hand_size
            raw_index_x = ix_rel * rel_hand_size + rel_palm_x
            raw_index_y = iy_rel * rel_hand_size + rel_palm_y
            raw_middle_x = mid_rel_x * rel_hand_size + rel_palm_x
            raw_middle_y = mid_rel_y * rel_hand_size + rel_palm_y
            raw_ring_x = ring_rel_x * rel_hand_size + rel_palm_x
            raw_ring_y = ring_rel_y * rel_hand_size + rel_palm_y
            raw_primary_x = (raw_index_x + raw_middle_x + raw_ring_x) / 3.0
            raw_primary_y = (raw_index_y + raw_middle_y + raw_ring_y) / 3.0
            mode_str = "Palm-Relative"
        elif use_box_micro:
            (t, local_x, local_y, global_x, global_y,
             box_width, box_height, box_area, quality) = snap_box
            x = global_x
            y = global_y
            confidence = quality
            tracking_quality_value = float(np.median(quality)) if len(quality) else 1.0
            ref = box_width
            mode_str = "Box-Normalized"
        elif use_multi_finger:
            t, ix, iy, mx, my, rx, ry, ref, conf = snap_multi
            # Average the three fingertips for robust tremor signal
            x = (ix + mx + rx) / 3.0
            y = (iy + my + ry) / 3.0
            confidence = conf
            mode_str = "Multi-Finger"
        else:
            # Fallback to single landmark
            snap = self.tracker.snapshot(WINDOW_SECONDS)
            if snap is None:
                # Show buffering status
                required = max(16, int(WINDOW_SECONDS * 25))  # Minimum samples needed
                if relative_count > 0 or movement_count > 0 or multi_count > 0 or single_count > 0:
                    message = f"Hold your hand still. Getting ready ({max(relative_count, movement_count, multi_count, single_count)}/{required})."
                else:
                    message = "Hold your hand still. Getting ready."
                self._apply_paused_tremor_ui(HAND_STATE_STEADY, message, palm_gate)
                return None
            t, x, y, ref = snap
            confidence = None  # No confidence data for single landmark
            mode_str = "Single-Landmark"
        
        # If optical flow wins, replace the tremor x/y with body-frame
        # pseudo-position (cumulative sum of frame-to-frame body_dx/dy).
        # Position semantics keep the existing analyzer (highpass + Welch)
        # happy without per-axis amplitude recalibration. We keep the
        # palm-relative `local_*` / `global_*` and gating streams as-is
        # but interpolate them onto the flow timestamps so all downstream
        # resampling sees consistent array lengths.
        if tracking_source == "optical_flow" and snap_flow is not None:
            (ft, fdx, fdy, fpts, fsurv, fq, fbgx, fbgy) = snap_flow
            # Physiological gate: reject windows containing motion that
            # cannot be tremor (whole-arm waves, drops, big repositions).
            allow_flow, gate_reason, gross_fraction, fdx, fdy = gate_flow_for_microtremor(
                fdx, fdy,
            )
            try:
                self.flow_status_var.set(
                    self.flow_status_var.get()
                    + f"  gross={int(gross_fraction*100)}%"
                )
            except Exception:
                pass
            if not allow_flow:
                self._apply_paused_tremor_ui(
                    HAND_STATE_MOVING_WIDE,
                    gate_reason,
                    palm_gate,
                )
                return None
            t_old = np.asarray(t, dtype=np.float64)
            t = ft
            x = np.cumsum(fdx)
            y = np.cumsum(fdy)
            confidence = fq
            mode_str = "Optical-Flow"
            # Re-align palm-relative-derived streams onto the flow grid.
            if use_palm_relative and local_x is not None and local_y is not None:
                local_x = np.interp(t, t_old, np.asarray(local_x))
                local_y = np.interp(t, t_old, np.asarray(local_y))
            if use_palm_relative and global_x is not None and global_y is not None:
                global_x = np.interp(t, t_old, np.asarray(global_x))
                global_y = np.interp(t, t_old, np.asarray(global_y))
            # ref (hand reference scale) -- align too, so np.median works
            if isinstance(ref, np.ndarray) and ref.shape == t_old.shape:
                ref = np.interp(t, t_old, ref)
            # tracking quality reflects flow quality going forward
            tracking_quality_value = float(np.median(fq)) if len(fq) else tracking_quality_value
        n_samples = len(t)
        print(f"[analysis] mode={mode_str} n={n_samples} multi_buf={multi_count} single_buf={single_count}")
        
        # Assess trial quality
        from .signal import assess_trial_quality
        quality_result = assess_trial_quality(t, confidence)
        
        quality_status = quality_result.get("quality_status", "invalid")
        quality_score = quality_result.get("quality_score", 0.0)
        fps_cv = quality_result.get("fps_cv", 0.0)
        reasons = quality_result.get("reasons", [])
        
        print(f"[analysis] quality_status={quality_status} score={quality_score:.2f} fps_cv={fps_cv:.3f} reasons={reasons}")
        
        # Show quality warning suffix if low_quality or invalid
        if quality_status == "invalid":
            status_suffix = f" (⚠️ {reasons[0]})" if reasons else " (⚠️ invalid quality)"
        elif quality_status == "low_quality":
            status_suffix = f" (⚠️ {reasons[0]})" if reasons else " (⚠️ low quality)"
        else:
            status_suffix = ""
        
        dur = float(t[-1] - t[0])
        if dur <= 0.5:
            return None
        fs_est = float(np.clip(len(t) / dur, 15.0, 60.0))
        res = resample_uniform(t, x, y, fs_est)
        if res is None:
            # resample_uniform returns None if long gaps detected
            return None
        tu, xu, yu = res
        local_xu = local_yu = global_xu = global_yu = None
        box_width_u = box_height_u = box_area_u = None
        palm_xu = palm_yu = palm_w_u = palm_h_u = hand_box_size_u = None
        if use_palm_relative:
            local_res = resample_uniform(t, local_x, local_y, fs_est)
            global_res = resample_uniform(t, global_x, global_y, fs_est)
            if local_res is None or global_res is None:
                return None
            _, local_xu, local_yu = local_res
            _, global_xu, global_yu = global_res
            relative_fingertip_signals = {
                "index_tip": (np.interp(tu, t, ix_rel), np.interp(tu, t, iy_rel)),
                "middle_tip": (np.interp(tu, t, mid_rel_x), np.interp(tu, t, mid_rel_y)),
                "ring_tip": (np.interp(tu, t, ring_rel_x), np.interp(tu, t, ring_rel_y)),
            }
            if raw_primary_x is not None and raw_primary_y is not None:
                raw_xu = np.interp(tu, t, raw_primary_x)
                raw_yu = np.interp(tu, t, raw_primary_y)
                raw_features = movement_residual_features(raw_xu, raw_yu, fs_est)
                raw_fingertip_power = float(raw_features.get("tremor_power", 0.0))
        elif use_box_micro:
            local_res = resample_uniform(t, local_x, local_y, fs_est)
            global_res = resample_uniform(t, global_x, global_y, fs_est)
            if local_res is None or global_res is None:
                return None
            _, local_xu, local_yu = local_res
            _, global_xu, global_yu = global_res
            box_width_u = np.interp(tu, t, box_width)
            box_height_u = np.interp(tu, t, box_height)
            box_area_u = np.interp(tu, t, box_area)
        if snap_palm is not None:
            palm_t, palm_x, palm_y, palm_w, palm_h, hand_box_size, palm_quality = snap_palm
            palm_xu = np.interp(tu, palm_t, palm_x)
            palm_yu = np.interp(tu, palm_t, palm_y)
            hand_box_size_u = np.interp(tu, palm_t, hand_box_size)
            palm_w_u = np.interp(tu, palm_t, palm_w)
            palm_h_u = np.interp(tu, palm_t, palm_h)
            if tracking_quality_value is None and len(palm_quality):
                tracking_quality_value = float(np.median(palm_quality))
        hand_ref = float(np.median(ref))
        
        # Task-aware metric computation
        metrics = compute_metrics(
            xu, yu, fs_est,
            hand_ref_pixels=hand_ref,
            baseline_rms=self.baseline_rms,
            task_mode=self.current_task_mode,  # Pass task mode
            local_xu=local_xu,
            local_yu=local_yu,
            global_xu=global_xu,
            global_yu=global_yu,
            tracking_quality=tracking_quality_value,
            box_width=box_width_u,
            box_height=box_height_u,
            box_area=box_area_u,
            palm_center_x=palm_xu,
            palm_center_y=palm_yu,
            hand_box_size=hand_box_size_u,
            frame_width=CAM_WIDTH,
            frame_height=CAM_HEIGHT,
            hand_box_width=palm_w_u,
            hand_box_height=palm_h_u,
            relative_fingertip_signals=relative_fingertip_signals,
            raw_fingertip_tremor_power=raw_fingertip_power,
        )
        if metrics is None:
            print("[analysis] compute_metrics returned None")
            return None
        
        # Override quality layer from assess_trial_quality
        metrics.quality_status = quality_status
        metrics.quality_reason = ", ".join(reasons) if reasons else ""
        metrics.fps_cv = fps_cv
        
        # Compute final research_valid: both quality AND motion must pass
        motion_valid = (metrics.motion_classification == "valid_tremor")
        quality_valid = (quality_status == "valid")
        metrics.research_valid = motion_valid and quality_valid
        
        # Gate final_tremor_score based on research_valid
        if not metrics.research_valid:
            metrics.final_tremor_score = None
            # Update reason for display
            if quality_status == "invalid":
                metrics.reason = reasons[0] if reasons else "Invalid quality"
            elif quality_status == "low_quality":
                metrics.reason = reasons[0] if reasons else "Low quality"
            else:
                metrics.reason = metrics.motion_reason
        
        motion_class = metrics.motion_classification
        
        print(f"[analysis] quality={quality_status} motion={motion_class} research_valid={metrics.research_valid}")
        print(f"[analysis] f={metrics.peak_hz:.2f}Hz v95={metrics.velocity_p95:.3f} ratio={metrics.path_ratio:.1f} palm_ratio={metrics.palm_gross_motion_ratio:.3f}")
        print(f"[analysis] live={metrics.live_motion_score:.1f} candidate={metrics.tremor_candidate_score:.1f} final={metrics.final_tremor_score} confidence={metrics.confidence_level}")
        
        # ALWAYS show live motion score (never use dashes)
        self.score_var.set(f"{int(metrics.live_motion_score)}")

        # Feed the live motion score into the active video reports session
        # (if any) so every play populates Avg/Min/Max/Samples on the
        # Reports tab, even when research-grade validity is not reached.
        try:
            if getattr(self.report_store, "_active", None) is not None:
                self.report_store.add_score(int(metrics.live_motion_score))
        except Exception as exc:
            print(f"[REPORTS] live add_score failed: {exc}", flush=True)

        # Update three-score model metrics (ALWAYS visible)
        m = self.metric_vars
        m["live_score"].set(f"{metrics.live_motion_score:.0f}")
        quality_words = {"high": "Strong", "medium": "Okay", "low": "Still checking"}
        m["confidence"].set(quality_words.get(metrics.confidence_level, "Still checking"))
        m["tremor_candidate"].set(f"{metrics.tremor_candidate_score:.0f}")
        m["prominence"].set(f"{metrics.peak_prominence:.2f}×" if metrics.peak_prominence > 0 else "-")
        m["fps_cv"].set(f"{fps_cv:.3f}")
        m["velocity"].set(f"{metrics.velocity_p95:.3f}")
        m["path_ratio"].set(f"{metrics.path_ratio:.1f}")
        m["center_drift"].set(f"{metrics.center_drift:.4f}")
        m["movement_mode"].set("Moving check" if metrics.movement_mode_active else self.task_mode_var.get())
        m["movement_score"].set(f"{metrics.movement_score:.0f}")
        m["relative_displacement"].set(f"{metrics.palm_relative_displacement:.4f}" if metrics.palm_relative_displacement > 0 else "-")
        m["relative_power"].set(f"{metrics.palm_relative_tremor_power:.2e}" if metrics.palm_relative_tremor_power > 0 else "-")
        m["raw_fingertip_power"].set(f"{metrics.raw_fingertip_tremor_power:.2e}" if metrics.raw_fingertip_tremor_power > 0 else "-")
        if any(v > 0 for v in (metrics.palm_relative_index_peak_hz, metrics.palm_relative_middle_peak_hz, metrics.palm_relative_ring_peak_hz)):
            m["relative_peaks"].set(
                f"{metrics.palm_relative_index_peak_hz:.1f}/"
                f"{metrics.palm_relative_middle_peak_hz:.1f}/"
                f"{metrics.palm_relative_ring_peak_hz:.1f} Hz"
            )
        else:
            m["relative_peaks"].set("-")
        if metrics.palm_relative_agreement_count >= 2:
            m["relative_agreement"].set("Good")
        elif metrics.palm_relative_agreement_count == 1:
            m["relative_agreement"].set("Checking")
        else:
            m["relative_agreement"].set("-")
        m["relative_veto"].set(metrics.palm_relative_veto_reason or "-")
        m["palm_path_length_px"].set(f"{metrics.palm_path_length_px:.0f} px" if metrics.palm_path_length_px > 0 else "-")
        m["screen_travel_ratio"].set(f"{metrics.screen_travel_ratio:.3f}" if metrics.screen_travel_ratio > 0 else "-")
        m["hand_relative_travel"].set(f"{metrics.hand_relative_travel:.2f}" if metrics.hand_relative_travel > 0 else "-")
        m["veto_reason"].set(metrics.physical_veto_reason or "Clear")
        m["tremor_overlay"].set(f"{metrics.tremor_overlay_score:.0f}" if metrics.movement_mode_active else "-")
        if metrics.dominant_tremor_frequency_hz <= 0:
            m["dominant_tremor"].set("-")
        elif metrics.dominant_tremor_frequency_hz < 4:
            m["dominant_tremor"].set("Slow")
        elif metrics.dominant_tremor_frequency_hz <= 7:
            m["dominant_tremor"].set("Medium")
        else:
            m["dominant_tremor"].set("Fast")
        m["tremor_source"].set(metrics.tremor_source if metrics.movement_mode_active else "-")
        m["tracking_quality"].set(f"{metrics.tracking_quality * 100:.0f}%")
        if metrics.box_stability_status == "stable":
            m["box_stability"].set("Good")
        elif metrics.box_stability_status == "box_size_warning":
            m["box_stability"].set("Changing")
        else:
            m["box_stability"].set("Needs steady view")
        if metrics.palm_motion_state == "wide_range_hand_movement":
            m["palm_motion"].set("Too much movement")
        elif metrics.palm_motion_state == "motion_too_high_for_tremor":
            m["palm_motion"].set("Moving")
        elif metrics.palm_motion_state == "steady":
            m["palm_motion"].set("Still enough")
        else:
            m["palm_motion"].set("-")
        
        # Determine validity status and reason
        if quality_status == "invalid":
            validity_text = "Camera issue"
            reason_text = reasons[0] if reasons else "Unknown"
            status_msg = "Camera needs a clearer view"
        elif quality_status == "low_quality":
            validity_text = "Needs clearer view"
            reason_text = reasons[0] if reasons else "Unknown"
            status_msg = "Keep your hand in view"
        elif not motion_valid:
            validity_text = "Keep still"
            if motion_class == "wide_range_hand_movement":
                reason_text = "Too much hand movement"
                status_msg = "Hold your hand still in one place"
            elif motion_class == "motion_too_high_for_tremor":
                reason_text = "Hand is moving too much"
                status_msg = "Hold your hand still"
            elif motion_class == "gross_translation":
                reason_text = "Hand moving around"
                status_msg = "Hold hand still in one place"
            elif motion_class == "gross_motion":
                reason_text = "Large movements detected"
                status_msg = "Try moving more gently"
            elif motion_class == "uncertain":
                reason_text = "Reading is not clear yet"
                status_msg = "Keep your hand still a little longer"
            elif motion_class == "tracking_unstable":
                reason_text = "Camera view is not steady"
                status_msg = "Keep hand in view"
            elif motion_class == "low_frequency":
                reason_text = "Movement is too slow"
                status_msg = "Movement too slow"
            elif motion_class == "high_frequency_noise":
                reason_text = "Movement is too fast"
                status_msg = "Movement too fast"
            else:
                reason_text = metrics.motion_reason
                status_msg = reason_text
        else:
            # Research-valid tremor
            validity_text = "Shaking found"
            reason_text = "Steady shaking found"
            if metrics.movement_mode_active:
                status_msg = "✓ Shaking found while moving"
            else:
                status_msg = "✓ Shaking found"
        
        m["validity"].set(validity_text)
        self.status_var.set(status_msg)
        if metrics.research_valid and metrics.final_tremor_score is not None:
            self.verdict_var.set(
                f"Current shaking score: {int(metrics.final_tremor_score)}%\n"
                f"{status_msg}"
            )
            self.score_lbl.configure(fg=OK_GREEN)
            self.verdict_lbl.configure(bg=OK_TINT, fg=OK_DEEP)
        else:
            self.verdict_var.set(f"Current status: {status_msg}")
            self.score_lbl.configure(fg=TEXT)
            self.verdict_lbl.configure(bg=WARN_TINT, fg=WARN_DEEP)
        
        # Show spectral features
        m["peak"].set(f"{metrics.peak_hz:.2f} Hz")
        m["band"].set(f"{metrics.band_ratio * 100:.0f}%")
        
        # Only show full amplitude/quality metrics for valid tremor
        if metrics.research_valid:
            m["amp"].set(f"{metrics.rms_amp:.5f}")
            m["amp_mm"].set(f"{metrics.rms_amp_mm:.1f} mm")
            m["snr"].set(f"{metrics.snr_db:.1f} dB")
            m["reg"].set(f"{metrics.regularity:.2f}")
            m["sharp"].set(f"{metrics.peak_sharpness:.1f}×")
            m["class"].set(metrics.class_label)
            m["fs"].set(f"{metrics.fs:.0f} Hz")
            m["samples"].set(str(metrics.samples))
            
            # Update status with mode and band
            self.status_var.set(status_msg)
            self._record_last_stable_score(metrics)
            
            return metrics
        else:
            # Non-valid: show limited info
            m["amp"].set("-")
            m["amp_mm"].set("-")
            m["snr"].set("-")
            m["reg"].set("-")
            m["sharp"].set("-")
            m["class"].set(motion_class.replace('_', ' ').title())
            m["fs"].set(f"{metrics.fs:.0f} Hz")
            m["samples"].set(str(metrics.samples))
            
            return metrics

        # Update verdict based on live score (not gated)
        live_score = int(metrics.live_motion_score)
        if live_score < 15:
            txt, color, fg, sbg = "Very little movement", OK_GREEN, "#0f5132", "#ecfdf3"
        elif live_score < 40:
            txt, color, fg, sbg = (f"Light movement detected ({metrics.peak_hz:.1f} cycles/sec)",
                                   WARN, "#854d0e", "#fef9c3")
        elif live_score < 70:
            if metrics.research_valid:
                txt, color, fg, sbg = ("Shaking found",
                                       "#ea580c", "#9a3412", "#ffedd5")
            else:
                txt, color, fg, sbg = (f"Moderate movement ({metrics.peak_hz:.1f} cycles/sec)",
                                       WARN, "#854d0e", "#fef9c3")
        else:
            if metrics.research_valid:
                txt, color, fg, sbg = ("Strong shaking found",
                                       BAD, "#7f1d1d", RED_TINT)
            else:
                txt, color, fg, sbg = (f"Strong movement ({metrics.peak_hz:.1f} cycles/sec)",
                                       "#ea580c", "#9a3412", "#ffedd5")
        
        self.verdict_var.set(txt)
        self.score_lbl.configure(fg=color)
        self.verdict_lbl.configure(bg=sbg, fg=fg)

        self.hist_t.append(time.time() - self._t0)
        self.hist_score.append(live_score)
        self.hist_peak.append(metrics.peak_hz)
        self.hist_amp.append(metrics.rms_amp_mm)

        # Feed adaptive baseline so the video-gate threshold personalises.
        self.baseline.update(metrics.rms_amp, metrics.band_ratio,
                             metrics.score)
        # If the user never explicitly calibrated, use the learned floor
        # once we have enough quiet samples.
        if (self.baseline_rms is None and self.baseline.rms is not None
                and self.baseline.samples > 30):
            self.baseline_rms = self.baseline.rms
        return metrics

    def _refresh_active_chart(self, force: bool = False) -> None:
        if not self.nb.tabs():
            return
        idx = self.nb.index("current")
        if idx == 0:
            self._plot_waveform()
        elif idx == 3:
            self._plot_spectrum()
        elif idx == 4:
            self._plot_spectrogram()
        elif idx == 5:
            self._plot_history()

    # ---------------------------------- plots
    def _plot_waveform(self) -> None:
        snap = self.tracker.snapshot(WINDOW_SECONDS)
        if snap is None:
            self._wave_lx.set_data([], [])
            self._wave_ly.set_data([], [])
            self.wave_canvas.draw_idle()
            return
        t, x, y, _ = snap
        fs = float(np.clip(len(t) / max(1e-3, t[-1] - t[0]), 15, 60))
        res = resample_uniform(t, x, y, fs)
        if res is None:
            return
        tu, xu, yu = res
        xf = bandpass(highpass(xu, fs), fs)
        yf = bandpass(highpass(yu, fs), fs)
        tt = tu - tu[0]
        self._wave_lx.set_data(tt, xf)
        self._wave_ly.set_data(tt, yf)
        self.wave_ax.set_xlim(0, max(WINDOW_SECONDS, float(tt[-1])))
        m = float(max(np.max(np.abs(xf)), np.max(np.abs(yf)), 1e-4))
        self.wave_ax.set_ylim(-m * 1.2, m * 1.2)
        self.wave_canvas.draw_idle()

    def _plot_spectrum(self) -> None:
        ax = self.spec_ax
        ax.clear()
        _style_axes(ax, "Frequency (Hz)", "Power (a.u.)")
        ax.axvspan(TREMOR_BAND[0], TREMOR_BAND[1],
                   color=RED, alpha=0.05, label="Tremor band")
        m = self.last_metrics
        snap = self.tracker.snapshot(WINDOW_SECONDS)
        if m is not None and snap is not None:
            t, x, y, _ = snap
            fs = m.fs
            res = resample_uniform(t, x, y, fs)
            if res is not None:
                _, xu, yu = res
                xf = bandpass(highpass(xu, fs), fs)
                yf = bandpass(highpass(yu, fs), fs)
                nperseg = int(min(xf.size, max(64, fs * 2)))
                fxx, pxx = welch(xf, fs=fs, nperseg=nperseg)
                _, pyy = welch(yf, fs=fs, nperseg=nperseg)
                psd = pxx + pyy
                if psd.max() > 0:
                    psd_n = psd / psd.max()
                    ax.fill_between(fxx, 0, psd_n, color=RED, alpha=0.25)
                    ax.plot(fxx, psd_n, color=RED, linewidth=1.6)
                    ax.axvline(m.peak_hz, color=RED_DEEP, linewidth=1.5,
                               linestyle="--",
                               label=f"Peak {m.peak_hz:.2f} Hz")
                ax.set_xlim(0, 15); ax.set_ylim(0, 1.05)
                leg = ax.legend(loc="upper right", fontsize=9,
                                facecolor=SURFACE, edgecolor=BORDER,
                                labelcolor=TEXT)
                leg.get_frame().set_linewidth(0.8)
        self.spec_canvas.draw_idle()

    def _plot_spectrogram(self) -> None:
        ax = self.sg_ax
        ax.clear()
        _style_axes(ax, "Time (s)", "Frequency (Hz)")
        snap = self.tracker.snapshot(15.0)
        if snap is None:
            self.sg_canvas.draw_idle()
            return
        t, x, y, _ = snap
        fs = float(np.clip(len(t) / max(1e-3, t[-1] - t[0]), 15, 60))
        res = resample_uniform(t, x, y, fs)
        if res is None:
            return
        _, xu, yu = res
        mag = np.hypot(bandpass(highpass(xu, fs), fs),
                       bandpass(highpass(yu, fs), fs))
        sg = rolling_spectrogram(mag, fs)
        if sg is None:
            return
        f, tt, Sxx = sg
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        vmax = float(np.max(Sxx_db))
        vmin = vmax - 35
        # Red-and-white colormap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "mb_reds", ["#ffffff", "#ffe3e3", "#ffa8a8", "#ff6b6b",
                        "#e53935", "#7f1d1d"])
        ax.pcolormesh(tt, f, Sxx_db, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="auto")
        ax.axhspan(TREMOR_BAND[0], TREMOR_BAND[1], color=RED, alpha=0.05)
        ax.set_ylim(0, 15)
        self.sg_canvas.draw_idle()

    def _plot_history(self) -> None:
        if not self.hist_t:
            return
        t = list(self.hist_t)
        for ax, data, color, ylab, ylim in [
            (self.hist_ax1, list(self.hist_score), RED, "Score", (0, 100)),
            (self.hist_ax2, list(self.hist_peak), "#94a3b8", "Peak Hz", (0, 15)),
            (self.hist_ax3, list(self.hist_amp), RED_DEEP, "Amp (mm)", None),
        ]:
            ax.clear()
            _style_axes(ax, "", ylab)
            ax.plot(t, data, color=color, linewidth=1.6)
            ax.fill_between(t, 0, data, color=color, alpha=0.12)
            if ylim:
                ax.set_ylim(*ylim)
        self.hist_ax3.set_xlabel("Time (s)", color=MUTED, fontsize=9)
        self.hist_canvas.draw_idle()

    # ------------------------------------------- exercise state update
    def _update_exercise(self, metrics) -> None:
        key = self.active_exercise_key
        if key is None:
            return
        sess = self.exercises[key]
        pose = self.tracker.get_pose()
        tip = self.tracker.get_hand_tip_norm()

        score = metrics.score if metrics else None
        peak = metrics.peak_hz if metrics else None
        amp = metrics.rms_amp_mm if metrics else None
        grip = self.tracker.get_grip_strength()

        v = sess.update(pose, tip, score, peak, amp, grip=grip)

        # Pose-correctness chip - green when user is actually in the pose
        if v.ok:
            self.ex_pose_var.set("Pose: correct ✓")
            self.ex_pose_chip.configure(bg="#ecfdf3", fg="#0f5132")
        else:
            self.ex_pose_var.set("Pose: not yet")
            self.ex_pose_chip.configure(bg=RED_TINT, fg=RED_DEEP)

        if sess.stage == Stage.PREPARE:
            if sess.prepare_ready_since is None:
                prog = 0
                self.ex_stage_var.set("Get into position…")
                self.ex_stage_lbl.configure(bg=RED_SOFT, fg=RED)
            else:
                frac = min(1.0,
                           (time.time() - sess.prepare_ready_since)
                           / max(0.1, sess.exercise.prepare_secs))
                prog = int(frac * 100)
                self.ex_stage_var.set(f"Holding position… {prog}%")
                self.ex_stage_lbl.configure(bg=RED_SOFT, fg=RED)
            self.ex_feedback_var.set(v.message)
            self.ex_progress["value"] = prog
        elif sess.stage == Stage.HOLD:
            frac = min(1.0, sess.elapsed() / sess.exercise.hold_secs)
            prog = int(frac * 100)
            remaining = max(0, int(sess.exercise.hold_secs - sess.elapsed()))
            self.ex_stage_var.set(f"Measuring… {remaining}s left")
            self.ex_stage_lbl.configure(bg=RED_TINT, fg=RED_DEEP)
            msg = v.message if v.ok else f"Hold the pose. {v.message.lower()}"
            self.ex_feedback_var.set(msg)
            self.ex_progress["value"] = prog
        elif sess.stage == Stage.DONE:
            self.ex_stage_var.set("Done. Results ready")
            self.ex_stage_lbl.configure(bg="#ecfdf3", fg="#0f5132")
            self.ex_feedback_var.set("")
            self.ex_result_var.set(sess.result_summary)
            self.ex_progress["value"] = 100
            self.active_exercise_key = None

    # -------------------------------------------------------- shutdown
    def _on_close(self) -> None:
        try:
            self._stop_startup_intro()
            self._stop_local_video(clear_frame=False)
            self.tracker.stop()
        finally:
            self.root.destroy()


def main() -> None:
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
