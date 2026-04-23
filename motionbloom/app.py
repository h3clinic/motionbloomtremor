"""MotionBloom Lite — consumer-grade tremor-sensing desktop app.

Red & white theme, card-based layout, conversational copy.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from tkinter import (
    Tk, Canvas, Frame, Label, Button, ttk,
    StringVar, BooleanVar, filedialog, messagebox,
)

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from scipy.signal import welch

from .exercises import EXERCISES, Exercise, ExerciseSession, Stage
from .signal import (
    TREMOR_BAND,
    AdaptiveBaseline,
    bandpass,
    compute_metrics,
    highpass,
    resample_uniform,
    rolling_spectrogram,
)
from .tracker import CAM_HEIGHT, CAM_WIDTH, LANDMARK_CHOICES, TremorTracker
from .video_gate import VideoGate

# ---------- Brand palette (modern light theme + red accent) ---------------
BG = "#f6f7f9"               # app background (cool off-white)
CANVAS = "#ffffff"            # main content panel
SURFACE = "#ffffff"           # cards
SURFACE_ALT = "#f3f4f6"       # subtle alt surface / chips
SURFACE_ELEV = "#ffffff"
BORDER = "#e5e7eb"
BORDER_STRONG = "#d1d5db"
TEXT = "#0f172a"              # near-black
TEXT_2 = "#1f2937"
MUTED = "#64748b"
MUTED_2 = "#94a3b8"

# Sidebar (dark)
SIDEBAR_BG = "#0f172a"
SIDEBAR_ALT = "#111b30"
SIDEBAR_TEXT = "#e2e8f0"
SIDEBAR_MUTED = "#94a3b8"
SIDEBAR_ACTIVE = "#1e293b"

RED = "#ef4444"               # primary accent
RED_DEEP = "#dc2626"
RED_SOFT = "#fef2f2"
RED_TINT = "#fee2e2"
PLOT_BG = "#ffffff"
PLOT_GRID = "#eef0f4"

OK_GREEN = "#16a34a"
OK_TINT = "#ecfdf5"
OK_DEEP = "#065f46"
WARN = "#f59e0b"
WARN_TINT = "#fffbeb"
WARN_DEEP = "#92400e"
BAD = "#dc2626"

FONT_FAMILY = "Helvetica"

WINDOW_SECONDS = 4.0
VIDEO_MS = 33
ANALYSIS_MS = 200
HISTORY_MAX = 600

APP_NAME = "MotionBloom Lite"
APP_TAGLINE = "Gentle tremor sensing, right from your camera."


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


def section_title(parent, text: str, bg: str = SURFACE) -> Label:
    return Label(parent, text=text.upper(),
                 font=(FONT_FAMILY, 9, "bold"),
                 fg=MUTED, bg=bg, anchor="w")


# ---------- App -------------------------------------------------------------
class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1320x860")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 720)

        self.tracker = TremorTracker()
        self.baseline_rms: float | None = None
        self.baseline = AdaptiveBaseline()
        self.session_rows: list[dict] = []
        self.recording = False

        self.hist_t: deque = deque(maxlen=HISTORY_MAX)
        self.hist_score: deque = deque(maxlen=HISTORY_MAX)
        self.hist_peak: deque = deque(maxlen=HISTORY_MAX)
        self.hist_amp: deque = deque(maxlen=HISTORY_MAX)

        self.last_metrics = None

        self.exercises = {e.key: ExerciseSession(e) for e in EXERCISES}
        self.active_exercise_key: str | None = None

        self._configure_ttk_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._t0 = time.time()

        self.root.after(VIDEO_MS, self._video_loop)
        self.root.after(ANALYSIS_MS, self._analysis_loop)
        # Auto-start the camera shortly after the UI is visible so the
        # user doesn't have to click "Start camera" manually.
        self.root.after(400, self._auto_start_camera)

    def _auto_start_camera(self) -> None:
        if self.tracker.thread is None:
            self.start()

    # --------------------------------------------------------------- styling
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
                    padding=(18, 8), borderwidth=0,
                    font=(FONT_FAMILY, 11))
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
                    thickness=8)

        s.configure("TCombobox",
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=TEXT, arrowcolor=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)

        s.configure("TCheckbutton", background=SURFACE, foreground=TEXT,
                    font=(FONT_FAMILY, 10))

    # ------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        # Root split: sidebar | main area (topbar + notebook + footer)
        root_grid = Frame(self.root, bg=BG)
        root_grid.pack(fill="both", expand=True)
        root_grid.columnconfigure(1, weight=1)
        root_grid.rowconfigure(0, weight=1)

        self._build_sidebar(root_grid)

        main = Frame(root_grid, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        self._main = main

        self._build_topbar(main)
        self._build_actionbar(main)

        self.nb = ttk.Notebook(main, style="Hidden.TNotebook")
        self.nb.grid(row=2, column=0, sticky="nsew",
                     padx=24, pady=(0, 8))
        self.nb.bind("<<NotebookTabChanged>>",
                     lambda _e: self._on_tab_change())

        self._build_live_tab()
        self._build_exercise_tab()
        self._build_video_tab()
        self._build_spectrum_tab()
        self._build_spectrogram_tab()
        self._build_history_tab()

        self._build_footer(main)
        # Select home
        self._select_nav(0)

    # ----------------------------------------------------------- sidebar
    def _build_sidebar(self, parent: Frame) -> None:
        rail = Frame(parent, bg=SIDEBAR_BG, width=232)
        rail.grid(row=0, column=0, sticky="ns")
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)

        # Brand
        brand = Frame(rail, bg=SIDEBAR_BG)
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(22, 18))
        logo = Canvas(brand, width=30, height=30, bg=SIDEBAR_BG,
                      highlightthickness=0)
        logo.pack(side="left")
        logo.create_oval(2, 2, 28, 28, outline=RED, width=2)
        for cx, cy in [(15, 7), (23, 15), (15, 23), (7, 15)]:
            logo.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                             fill=RED, outline=RED)
        logo.create_oval(12, 12, 18, 18, fill=SIDEBAR_BG,
                         outline=RED, width=1)
        wm = Frame(brand, bg=SIDEBAR_BG)
        wm.pack(side="left", padx=(10, 0))
        Label(wm, text="MotionBloom",
              font=(FONT_FAMILY, 14, "bold"),
              fg=SIDEBAR_TEXT, bg=SIDEBAR_BG).pack(anchor="w")
        Label(wm, text="Lite",
              font=(FONT_FAMILY, 10),
              fg=RED, bg=SIDEBAR_BG).pack(anchor="w")

        # Nav section label
        Label(rail, text="WORKSPACE",
              font=(FONT_FAMILY, 8, "bold"),
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              anchor="w").grid(row=1, column=0, sticky="ew",
                               padx=22, pady=(0, 8))

        items = [
            ("Home", "◉"),
            ("Exercises", "◎"),
            ("Focus Video", "▶"),
            ("Spectrum", "∿"),
            ("Spectrogram", "▦"),
            ("History", "☷"),
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
            padx=12, pady=7,
            font=(FONT_FAMILY, 10, "bold"),
            anchor="w")
        self.status_chip.pack(fill="x")
        self.fps_var = StringVar(value="— fps")
        Label(foot, textvariable=self.fps_var,
              fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
              font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(6, 0))

    def _make_nav_item(self, parent: Frame, index: int,
                       label: str, glyph: str) -> Frame:
        row = Frame(parent, bg=SIDEBAR_BG, cursor="hand2")
        row.pack(fill="x", pady=2)
        inner = Frame(row, bg=SIDEBAR_BG, padx=12, pady=9)
        inner.pack(fill="x")

        g = Label(inner, text=glyph, fg=SIDEBAR_MUTED, bg=SIDEBAR_BG,
                  font=(FONT_FAMILY, 13))
        g.pack(side="left")
        t = Label(inner, text=label, fg=SIDEBAR_TEXT, bg=SIDEBAR_BG,
                  font=(FONT_FAMILY, 11))
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
            inner.configure(bg=bg)
            glyph.configure(bg=bg,
                            fg=RED if selected else SIDEBAR_MUTED)
            text.configure(bg=bg,
                           fg="#ffffff" if selected else SIDEBAR_TEXT,
                           font=(FONT_FAMILY, 11,
                                 "bold" if selected else "normal"))
        self.nb.select(index)

    def _build_topbar(self, parent: Frame) -> None:
        top = Frame(parent, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 4))
        top.columnconfigure(0, weight=1)

        left = Frame(top, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        self.page_title_var = StringVar(value="Home")
        Label(left, textvariable=self.page_title_var,
              font=(FONT_FAMILY, 22, "bold"),
              fg=TEXT, bg=BG).pack(anchor="w")
        Label(left, text=APP_TAGLINE,
              font=(FONT_FAMILY, 11), fg=MUTED, bg=BG).pack(anchor="w")

    def _build_actionbar(self, parent: Frame) -> None:
        bar_outer = Frame(parent, bg=BG)
        bar_outer.grid(row=1, column=0, sticky="ew", padx=24,
                       pady=(14, 10))
        bar = Frame(bar_outer, bg=SURFACE, highlightthickness=1,
                    highlightbackground=BORDER)
        bar.pack(fill="x")
        inner = Frame(bar, bg=SURFACE, padx=14, pady=10)
        inner.pack(fill="x")

        self.start_btn = HoverButton(inner, text="Start camera",
                                     variant="primary", command=self.start)
        self.start_btn.pack(side="left")

        self.stop_btn = HoverButton(inner, text="Stop",
                                    variant="outline", command=self.stop)
        self.stop_btn.pack(side="left", padx=(8, 16))
        self.stop_btn.set_disabled(True)

        Label(inner, text="Tracked point", fg=MUTED, bg=SURFACE,
              font=(FONT_FAMILY, 10)).pack(side="left")
        self.lm_var = StringVar(value="Index fingertip")
        lm = ttk.Combobox(inner, textvariable=self.lm_var,
                          values=list(LANDMARK_CHOICES.keys()),
                          state="readonly", width=18)
        lm.pack(side="left", padx=(8, 16))
        lm.bind("<<ComboboxSelected>>", self._on_landmark_change)

        self.calib_btn = HoverButton(inner, text="Calibrate (3 s still)",
                                     variant="ghost",
                                     command=self.calibrate)
        self.calib_btn.pack(side="left")
        self.calib_btn.set_disabled(True)

        right = Frame(inner, bg=SURFACE)
        right.pack(side="right")

        self.record_var = BooleanVar(value=False)
        ttk.Checkbutton(right, text="Record session",
                        variable=self.record_var,
                        command=self._toggle_record,
                        style="TCheckbutton").pack(side="left", padx=(0, 8))
        HoverButton(right, text="Export CSV", variant="neutral",
                    command=self.export_csv).pack(side="left")

    def _build_footer(self, parent: Frame) -> None:
        foot = Frame(parent, bg=BG)
        foot.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 12))
        Label(foot,
              text="MotionBloom Lite is a wellness-oriented demo and not "
                   "a medical device. If you are concerned about tremor, "
                   "please consult a clinician.",
              fg=MUTED_2, bg=BG, font=(FONT_FAMILY, 9),
              wraplength=1200, justify="left").pack(anchor="w")

    # ---------------------------------------------------------- Live tab
    def _build_live_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Home  ")
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)

        # Video card
        video_card = make_card(tab, padding=8)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=8)
        self.video_label = Label(video_card.inner, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

        # Right column
        right = Frame(tab, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # Score hero
        hero = make_card(right, padding=20)
        hero.grid(row=0, column=0, sticky="ew")
        inner = hero.inner
        inner.columnconfigure(0, weight=1)

        section_title(inner, "Tremor score").grid(row=0, column=0, sticky="w")

        self.score_var = StringVar(value="—")
        self.score_lbl = Label(inner, textvariable=self.score_var,
                               font=("Helvetica", 48, "bold"),
                               fg=TEXT, bg=SURFACE)
        self.score_lbl.grid(row=1, column=0, sticky="w", pady=(4, 0))

        Label(inner, text="out of 100", fg=MUTED, bg=SURFACE,
              font=("Helvetica", 10)).grid(row=2, column=0, sticky="w")

        self.verdict_var = StringVar(value="Press Start to begin.")
        self.verdict_lbl = Label(inner, textvariable=self.verdict_var,
                                 fg=TEXT, bg=RED_SOFT,
                                 font=("Helvetica", 12, "bold"),
                                 padx=14, pady=10, anchor="w")
        self.verdict_lbl.grid(row=3, column=0, sticky="ew", pady=(14, 0))

        # Metrics grid
        metrics = make_card(right, padding=14)
        metrics.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        mi = metrics.inner
        mi.columnconfigure(1, weight=1)
        mi.columnconfigure(3, weight=1)

        section_title(mi, "Live metrics").grid(row=0, column=0,
                                               columnspan=4, sticky="w",
                                               pady=(0, 8))

        self.metric_vars = {k: StringVar(value="—") for k in [
            "peak", "amp", "amp_mm", "band", "snr", "reg",
            "sharp", "class", "fs", "samples",
        ]}
        rows = [
            ("Dominant freq", "peak"),
            ("Amplitude (RMS)", "amp"),
            ("Amplitude (est)", "amp_mm"),
            ("Band power", "band"),
            ("Signal/noise", "snr"),
            ("Regularity", "reg"),
            ("Peak sharpness", "sharp"),
            ("Pattern", "class"),
            ("Sample rate", "fs"),
            ("Samples", "samples"),
        ]
        for i, (lbl, key) in enumerate(rows):
            r, c = divmod(i, 2)
            Label(mi, text=lbl, fg=MUTED, bg=SURFACE,
                  font=("Helvetica", 10)).grid(
                row=r + 1, column=c * 2, sticky="w", pady=3, padx=(0, 10))
            Label(mi, textvariable=self.metric_vars[key],
                  fg=TEXT, bg=SURFACE,
                  font=("Helvetica", 12, "bold")).grid(
                row=r + 1, column=c * 2 + 1, sticky="e", pady=3)

        # Waveform
        wave_card = make_card(right, padding=10)
        wave_card.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        section_title(wave_card.inner, "Filtered motion (3–12 Hz)").pack(
            anchor="w", pady=(0, 6))

        self.wave_fig = Figure(figsize=(4.5, 2.0), dpi=100, facecolor=SURFACE)
        self.wave_ax = self.wave_fig.add_subplot(111)
        _style_axes(self.wave_ax, "Time (s)", "Displacement")
        self.wave_fig.subplots_adjust(left=0.12, right=0.98,
                                      top=0.96, bottom=0.22)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig,
                                             master=wave_card.inner)
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
        Label(hero, text="Guided assessments",
              font=("Helvetica", 18, "bold"),
              fg=TEXT, bg=BG).pack(anchor="w")
        Label(hero, text="Pick an exercise — we'll walk you through it and "
                         "measure your tremor while you hold the pose.",
              fg=MUTED, bg=BG, font=("Helvetica", 11)).pack(anchor="w")

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
              font=("Helvetica", 17, "bold"),
              fg=TEXT, bg=SURFACE).grid(row=0, column=0, sticky="w")

        self.ex_desc_var = StringVar(value=EXERCISES[0].description)
        Label(si, textvariable=self.ex_desc_var,
              wraplength=440, fg=MUTED, bg=SURFACE, justify="left",
              font=("Helvetica", 11)).grid(row=1, column=0, sticky="w",
                                           pady=(6, 14))

        self.ex_stage_var = StringVar(value="Ready when you are.")
        self.ex_stage_lbl = Label(si, textvariable=self.ex_stage_var,
                                  bg=RED_SOFT, fg=RED, anchor="w",
                                  padx=12, pady=10,
                                  font=("Helvetica", 12, "bold"))
        self.ex_stage_lbl.grid(row=2, column=0, sticky="ew")

        # Pose-correctness chip
        self.ex_pose_var = StringVar(value="Pose: waiting")
        self.ex_pose_chip = Label(si, textvariable=self.ex_pose_var,
                                  bg=SURFACE_ALT, fg=MUTED, anchor="w",
                                  padx=10, pady=6,
                                  font=("Helvetica", 10, "bold"))
        self.ex_pose_chip.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.ex_feedback_var = StringVar(value="")
        Label(si, textvariable=self.ex_feedback_var,
              wraplength=440, fg=TEXT, bg=SURFACE, justify="left",
              font=("Helvetica", 12)).grid(row=4, column=0, sticky="w",
                                           pady=(10, 4))

        self.ex_progress = ttk.Progressbar(
            si, orient="horizontal", mode="determinate",
            style="MB.Horizontal.TProgressbar", maximum=100)
        self.ex_progress.grid(row=5, column=0, sticky="ew", pady=(10, 8))

        # Buttons
        btn_row = Frame(si, bg=SURFACE)
        btn_row.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        self.ex_start_btn = HoverButton(btn_row, text="Begin exercise",
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
              font=("Helvetica", 11, "bold"), justify="left").grid(
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
                     font=("Helvetica", 13, "bold"),
                     fg=TEXT, bg=SURFACE, anchor="w")
        name.pack(anchor="w")
        desc = Label(left, text=ex.description,
                     wraplength=380, justify="left",
                     fg=MUTED, bg=SURFACE,
                     font=("Helvetica", 10))
        desc.pack(anchor="w", pady=(4, 0))
        duration = Label(left,
                         text=f"≈ {int(ex.prepare_secs + ex.hold_secs)} s",
                         fg=MUTED_2, bg=SURFACE,
                         font=("Helvetica", 9))
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

    # --------------------------------------------------- Video gate tab
    def _build_video_tab(self) -> None:
        self.video_gate = VideoGate(self.nb, self,
                                    make_card=make_card,
                                    HoverButton=HoverButton,
                                    section_title=section_title)
        self.nb.add(self.video_gate, text="  Focus Video  ")

    # ---------------------------------------------------- Chart tabs
    def _build_spectrum_tab(self) -> None:
        tab = Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  Spectrum  ")
        card = make_card(tab, padding=14)
        card.pack(fill="both", expand=True, padx=0, pady=10)

        section_title(card.inner,
                      "Motion spectrum — power vs frequency").pack(
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
                      "Spectrogram — rolling 15-second window").pack(
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
                      "Session history — score, frequency, amplitude").pack(
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
    def _on_landmark_change(self, _evt=None) -> None:
        self.tracker.set_landmark(LANDMARK_CHOICES.get(self.lm_var.get(), 8))

    def _on_tab_change(self) -> None:
        self._refresh_active_chart(force=True)
        titles = ["Home", "Exercises", "Focus Video",
                  "Spectrum", "Spectrogram", "History"]
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
                "Calibrate",
                "Please hold your hand steady for about 3 seconds, then try "
                "calibrating again.")
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
        self.status_var.set("Calibrated")

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
                    cap = cv2.VideoCapture(index, backend)
                except Exception:
                    continue
                if not cap or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                ok = False
                for _ in range(30):
                    ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    return cap
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
        self.start_btn.set_disabled(True)
        self.stop_btn.set_disabled(False)
        self.calib_btn.set_disabled(False)
        self.ex_start_btn.set_disabled(False)
        self.status_var.set("Capturing…")

    def stop(self) -> None:
        self.tracker.stop()
        self.start_btn.set_disabled(False)
        self.stop_btn.set_disabled(True)
        self.calib_btn.set_disabled(True)
        self.ex_start_btn.set_disabled(True)
        self.status_var.set("Idle")
        self.video_label.configure(image="")
        self.ex_video_label.configure(image="")
        for v in self.metric_vars.values():
            v.set("—")
        self.score_var.set("—")
        self.verdict_var.set("Press Start to begin.")
        self.verdict_lbl.configure(bg=RED_SOFT, fg=RED)
        self.active_exercise_key = None
        self.ex_stage_var.set("Ready when you are.")
        self.ex_feedback_var.set("")
        self.ex_progress["value"] = 0

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
                    elif idx == 1:
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
        finally:
            self.root.after(ANALYSIS_MS, self._analysis_loop)

    def _refresh_analysis(self):
        snap = self.tracker.snapshot(WINDOW_SECONDS)
        if snap is None:
            return None
        t, x, y, ref = snap
        dur = float(t[-1] - t[0])
        if dur <= 0.5:
            return None
        fs_est = float(np.clip(len(t) / dur, 15.0, 60.0))
        res = resample_uniform(t, x, y, fs_est)
        if res is None:
            return None
        _, xu, yu = res
        hand_ref = float(np.median(ref))
        metrics = compute_metrics(
            xu, yu, fs_est,
            hand_ref_pixels=hand_ref,
            baseline_rms=self.baseline_rms,
        )
        if metrics is None:
            return None

        m = self.metric_vars
        m["peak"].set(f"{metrics.peak_hz:.2f} Hz")
        m["amp"].set(f"{metrics.rms_amp:.5f}")
        m["amp_mm"].set(f"{metrics.rms_amp_mm:.1f} mm")
        m["band"].set(f"{metrics.band_ratio * 100:.0f}%")
        m["snr"].set(f"{metrics.snr_db:.1f} dB")
        m["reg"].set(f"{metrics.regularity:.2f}")
        m["sharp"].set(f"{metrics.peak_sharpness:.1f}×")
        m["class"].set(metrics.class_label)
        m["fs"].set(f"{metrics.fs:.0f} Hz")
        m["samples"].set(str(metrics.samples))

        self.score_var.set(str(metrics.score))
        if metrics.score < 15:
            txt, color, fg, sbg = "All quiet — no notable tremor", OK_GREEN, "#0f5132", "#ecfdf3"
        elif metrics.score < 40:
            txt, color, fg, sbg = (f"Mild oscillation at {metrics.peak_hz:.1f} Hz",
                                   WARN, "#854d0e", "#fef9c3")
        elif metrics.score < 70:
            txt, color, fg, sbg = (f"Moderate tremor at {metrics.peak_hz:.1f} Hz "
                                   f"({metrics.class_label})",
                                   "#ea580c", "#9a3412", "#ffedd5")
        else:
            txt, color, fg, sbg = (f"Strong tremor at {metrics.peak_hz:.1f} Hz "
                                   f"({metrics.class_label})",
                                   BAD, "#7f1d1d", RED_TINT)
        self.verdict_var.set(txt)
        self.score_lbl.configure(fg=color)
        self.verdict_lbl.configure(bg=sbg, fg=fg)

        self.hist_t.append(time.time() - self._t0)
        self.hist_score.append(metrics.score)
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

        # Pose-correctness chip — green when user is actually in the pose
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
            msg = v.message if v.ok else f"Hold the pose — {v.message.lower()}"
            self.ex_feedback_var.set(msg)
            self.ex_progress["value"] = prog
        elif sess.stage == Stage.DONE:
            self.ex_stage_var.set("Done — results ready")
            self.ex_stage_lbl.configure(bg="#ecfdf3", fg="#0f5132")
            self.ex_feedback_var.set("")
            self.ex_result_var.set(sess.result_summary)
            self.ex_progress["value"] = 100
            self.active_exercise_key = None

    # -------------------------------------------------------- shutdown
    def _on_close(self) -> None:
        try:
            self.tracker.stop()
        finally:
            self.root.destroy()


def main() -> None:
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
