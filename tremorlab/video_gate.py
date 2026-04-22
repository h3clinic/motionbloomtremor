"""Video gate — plays a video and pauses for tremor-check challenges.

The gate runs on the same tracker/exercise machinery as the Exercises tab.
Every `interval` seconds of playback, the video pauses and prompts the user
to perform a randomly picked exercise. The pass threshold is adaptive: it
starts from `AdaptiveBaseline.personal_pass_threshold` and relaxes after
repeated failures so users with naturally higher tremor can still progress.
"""

from __future__ import annotations

import random
import time
from tkinter import (
    BooleanVar, DoubleVar, Frame, Label, Scale, StringVar,
    filedialog, ttk,
)

import numpy as np
from PIL import Image, ImageTk

try:
    from ffpyplayer.player import MediaPlayer
    try:
        from ffpyplayer.tools import set_loglevel as _set_ll
    except ImportError:  # older naming
        from ffpyplayer.tools import set_log_level as _set_ll  # type: ignore
    _set_ll("error")
    HAS_FFPY = True
except Exception:  # pragma: no cover - optional dependency
    MediaPlayer = None  # type: ignore
    HAS_FFPY = False

from .exercises import EXERCISES, ExerciseSession, Stage


# Palette values re-declared here to keep this module self-contained if
# someone wants to split it out later; kept in sync with app.py.
BG = "#ffffff"
SURFACE = "#ffffff"
SURFACE_ALT = "#fafafa"
BORDER = "#ececec"
TEXT = "#1a1a1a"
MUTED = "#6b7280"
MUTED_2 = "#9ca3af"
RED = "#e53935"
RED_DEEP = "#c62828"
RED_SOFT = "#fff5f5"
RED_TINT = "#ffe3e3"
OK_GREEN = "#16a34a"


class VideoGate(Frame):
    """Frame hosting the video + challenge panel.

    The host App must pass:
      - `app`: reference to the main App (for `tracker`, `baseline` and
        `last_metrics`).
      - `make_card`, `HoverButton`, `section_title`: UI factories.
    """

    PLAY_MS = 33                  # fallback tick if no frame pacing info
    CHECK_MS = 200                # challenge refresh cadence
    DEFAULT_INTERVAL = 20.0       # seconds of playback between gates
    MIN_INTERVAL = 10.0
    MAX_INTERVAL = 60.0

    def __init__(self, master, app, make_card, HoverButton,
                 section_title) -> None:
        super().__init__(master, bg=BG)
        self.app = app
        self._make_card = make_card
        self._Btn = HoverButton
        self._section = section_title

        self.player: "MediaPlayer | None" = None
        self.video_path: str | None = None
        self.playing = False
        self.play_since: float | None = None
        self.playback_credit = 0.0    # seconds of play accumulated toward gate
        self.interval = self.DEFAULT_INTERVAL

        self.session: ExerciseSession | None = None
        self.gate_active = False
        self.fail_streak = 0
        self.pass_streak = 0
        self.personal_threshold: float = 50.0
        self.last_threshold_shown: float = 50.0

        self._last_display_size: tuple[int, int] = (0, 0)
        self._last_photo: ImageTk.PhotoImage | None = None
        self._last_frame_wall = 0.0
        self._fps_avg = 0.0
        self._duration = 0.0
        self._volume = 1.0
        self._muted = False
        self._seeking = False

        self._build()
        self.after(self.PLAY_MS, self._video_loop)
        self.after(self.CHECK_MS, self._gate_loop)

    # -------------------------------------------------------------- UI
    def _build(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        # Hero row
        hero = Frame(self, bg=BG)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        Label(hero, text="Focus mode",
              font=("Helvetica", 18, "bold"),
              fg=TEXT, bg=BG).pack(anchor="w")
        Label(hero, text="Watch a video — we'll pause every so often for a "
                         "quick tremor check. Pass the check to keep "
                         "watching. The difficulty adapts to you.",
              fg=MUTED, bg=BG, font=("Helvetica", 11),
              wraplength=1100, justify="left").pack(anchor="w")

        # Video card
        video_card = self._make_card(self, padding=8)
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12), pady=8)
        video_card.inner.configure(bg="#000000")
        self.video_label = Label(video_card.inner, bg="#000000",
                                 text="Load a video to begin",
                                 fg="#cccccc",
                                 font=("Helvetica", 12))
        self.video_label.pack(fill="both", expand=True)

        # Controls below video — seek bar + player controls + gate setting
        ctrl_wrap = Frame(self, bg=BG)
        ctrl_wrap.grid(row=2, column=0, sticky="ew", padx=(0, 12),
                       pady=(4, 10))
        ctrl_wrap.columnconfigure(0, weight=1)

        # Seek + time row
        seek_row = Frame(ctrl_wrap, bg=BG)
        seek_row.grid(row=0, column=0, sticky="ew")
        seek_row.columnconfigure(1, weight=1)
        self.time_cur_var = StringVar(value="0:00")
        Label(seek_row, textvariable=self.time_cur_var,
              fg=MUTED, bg=BG,
              font=("Helvetica", 10)).grid(row=0, column=0, padx=(0, 8))
        self.seek_var = DoubleVar(value=0.0)
        self.seek = Scale(seek_row, from_=0.0, to=100.0,
                          orient="horizontal", showvalue=0,
                          variable=self.seek_var,
                          bg=BG, troughcolor=SURFACE_ALT,
                          highlightthickness=0, bd=0,
                          activebackground=RED_DEEP,
                          sliderrelief="flat", length=600)
        self.seek.grid(row=0, column=1, sticky="ew")
        self.seek.bind("<ButtonPress-1>", lambda _e: self._seek_start())
        self.seek.bind("<ButtonRelease-1>", lambda _e: self._seek_end())
        self.time_dur_var = StringVar(value="0:00")
        Label(seek_row, textvariable=self.time_dur_var,
              fg=MUTED, bg=BG,
              font=("Helvetica", 10)).grid(row=0, column=2, padx=(8, 0))

        # Buttons + gate row
        btn_row = Frame(ctrl_wrap, bg=BG)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.load_btn = self._Btn(btn_row, text="Load video",
                                  variant="neutral",
                                  command=self._load_video)
        self.load_btn.pack(side="left")

        self.play_btn = self._Btn(btn_row, text="▶  Play",
                                  variant="primary",
                                  command=self._toggle_play)
        self.play_btn.pack(side="left", padx=(8, 8))
        self.play_btn.set_disabled(True)

        self.back_btn = self._Btn(btn_row, text="◀ 10s",
                                  variant="ghost", small=True,
                                  command=lambda: self._skip(-10))
        self.back_btn.pack(side="left")
        self.fwd_btn = self._Btn(btn_row, text="10s ▶",
                                 variant="ghost", small=True,
                                 command=lambda: self._skip(+10))
        self.fwd_btn.pack(side="left", padx=(4, 12))

        self.mute_btn = self._Btn(btn_row, text="🔊",
                                  variant="ghost", small=True,
                                  command=self._toggle_mute)
        self.mute_btn.pack(side="left")

        self.vol_var = DoubleVar(value=100.0)
        self.vol = Scale(btn_row, from_=0.0, to=100.0,
                         orient="horizontal", showvalue=0,
                         variable=self.vol_var,
                         bg=BG, troughcolor=SURFACE_ALT,
                         highlightthickness=0, bd=0,
                         activebackground=RED_DEEP,
                         sliderrelief="flat", length=100,
                         command=self._on_volume)
        self.vol.pack(side="left", padx=(6, 16))

        Label(btn_row, text="Gate every",
              fg=MUTED, bg=BG,
              font=("Helvetica", 10)).pack(side="left")
        self.interval_var = StringVar(value="20 s")
        cb = ttk.Combobox(btn_row, textvariable=self.interval_var,
                          values=["10 s", "15 s", "20 s", "30 s",
                                  "45 s", "60 s"],
                          state="readonly", width=8)
        cb.pack(side="left", padx=(6, 0))
        cb.bind("<<ComboboxSelected>>", self._on_interval_change)

        self.file_var = StringVar(value="No video loaded")
        Label(ctrl_wrap, textvariable=self.file_var, fg=MUTED_2, bg=BG,
              font=("Helvetica", 10)).grid(row=2, column=0,
                                           sticky="w", pady=(6, 0))

        # Right panel: gate status
        side = Frame(self, bg=BG)
        side.grid(row=1, column=1, rowspan=2, sticky="nsew", pady=8)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(4, weight=1)

        # Camera mirror — shows the user what the tracker sees
        cam_card = self._make_card(side, padding=8)
        cam_card.grid(row=0, column=0, sticky="ew")
        cam_card.inner.configure(bg="#000000")
        self.cam_label = Label(cam_card.inner, bg="#000000",
                               text="Start camera to see yourself",
                               fg="#cccccc",
                               font=("Helvetica", 10),
                               height=9)
        self.cam_label.pack(fill="both", expand=True)
        self._cam_photo: ImageTk.PhotoImage | None = None
        self.after(66, self._cam_loop)

        # Progress toward next gate
        status_card = self._make_card(side, padding=16)
        status_card.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        si = status_card.inner
        si.columnconfigure(0, weight=1)
        self._section(si, "Next gate in").grid(row=0, column=0, sticky="w")
        self.next_gate_var = StringVar(value="— s")
        Label(si, textvariable=self.next_gate_var,
              font=("Helvetica", 26, "bold"),
              fg=TEXT, bg=SURFACE).grid(row=1, column=0, sticky="w",
                                        pady=(4, 6))
        self.gate_progress = ttk.Progressbar(
            si, orient="horizontal", mode="determinate",
            style="MB.Horizontal.TProgressbar", maximum=100)
        self.gate_progress.grid(row=2, column=0, sticky="ew")

        # Threshold / adaptation panel
        thr_card = self._make_card(side, padding=16)
        thr_card.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ti = thr_card.inner
        ti.columnconfigure(0, weight=1)
        self._section(ti, "Adaptive threshold").grid(row=0, column=0,
                                                     sticky="w")
        self.thr_var = StringVar(value="— /100")
        Label(ti, textvariable=self.thr_var,
              font=("Helvetica", 22, "bold"),
              fg=RED, bg=SURFACE).grid(row=1, column=0, sticky="w",
                                       pady=(2, 4))
        self.thr_hint_var = StringVar(
            value="Will personalise as we learn your baseline.")
        Label(ti, textvariable=self.thr_hint_var, wraplength=300,
              justify="left", fg=MUTED, bg=SURFACE,
              font=("Helvetica", 10)).grid(row=2, column=0, sticky="w")

        # Challenge card
        chal = self._make_card(side, padding=18)
        chal.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ci = chal.inner
        ci.columnconfigure(0, weight=1)
        self._section(ci, "Challenge").grid(row=0, column=0, sticky="w")

        self.chal_name_var = StringVar(value="—")
        Label(ci, textvariable=self.chal_name_var,
              font=("Helvetica", 15, "bold"),
              fg=TEXT, bg=SURFACE).grid(row=1, column=0, sticky="w",
                                        pady=(4, 4))

        self.chal_desc_var = StringVar(
            value="When the video pauses, follow the exercise prompt. "
                  "Stay under the adaptive threshold to pass.")
        Label(ci, textvariable=self.chal_desc_var,
              wraplength=320, justify="left",
              fg=MUTED, bg=SURFACE, font=("Helvetica", 10)).grid(
            row=2, column=0, sticky="w")

        self.chal_stage_var = StringVar(value="Idle")
        self.chal_stage_lbl = Label(ci, textvariable=self.chal_stage_var,
                                    bg=SURFACE_ALT, fg=MUTED,
                                    padx=12, pady=10,
                                    font=("Helvetica", 11, "bold"))
        self.chal_stage_lbl.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        # Pose-correctness chip — turns green when user is in the required pose
        self.pose_chip_var = StringVar(value="Pose: waiting")
        self.pose_chip = Label(ci, textvariable=self.pose_chip_var,
                               bg=SURFACE_ALT, fg=MUTED,
                               padx=10, pady=6,
                               font=("Helvetica", 10, "bold"))
        self.pose_chip.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        self.chal_feedback_var = StringVar(value="")
        Label(ci, textvariable=self.chal_feedback_var,
              wraplength=320, justify="left",
              fg=TEXT, bg=SURFACE, font=("Helvetica", 10)).grid(
            row=5, column=0, sticky="w", pady=(6, 0))

        self.chal_progress = ttk.Progressbar(
            ci, orient="horizontal", mode="determinate",
            style="MB.Horizontal.TProgressbar", maximum=100)
        self.chal_progress.grid(row=6, column=0, sticky="ew", pady=(10, 8))

        self.chal_result_var = StringVar(value="")
        self.chal_result_lbl = Label(ci, textvariable=self.chal_result_var,
                                     wraplength=320, justify="left",
                                     fg=TEXT, bg=SURFACE,
                                     font=("Helvetica", 11, "bold"))
        self.chal_result_lbl.grid(row=7, column=0, sticky="w")

        btn_row = Frame(ci, bg=SURFACE)
        btn_row.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        self.skip_btn = self._Btn(btn_row, text="Retry",
                                  variant="ghost",
                                  command=self._retry_challenge)
        self.skip_btn.pack(side="left")
        self.skip_btn.set_disabled(True)

        self.force_btn = self._Btn(btn_row, text="Trigger gate now",
                                   variant="neutral", small=True,
                                   command=self._force_gate)
        self.force_btn.pack(side="right")

    # ---------------------------------------------------- video handling
    def _load_video(self) -> None:
        if not HAS_FFPY:
            self.file_var.set("ffpyplayer not installed")
            return
        path = filedialog.askopenfilename(
            title="Pick a video",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        if self.player is not None:
            try:
                self.player.close_player()
            except Exception:
                pass
            self.player = None

        # Auto-start playback so the user sees the video immediately.
        ff_opts = {"paused": False, "sync": "audio", "out_fmt": "rgb24"}
        try:
            self.player = MediaPlayer(path, ff_opts=ff_opts)
        except Exception as e:
            self.file_var.set(f"Could not open: {e}")
            return
        self.video_path = path
        self.playing = True
        self.playback_credit = 0.0
        self.play_since = time.time()
        self._last_frame_wall = 0.0
        self._duration = 0.0
        self.play_btn.set_disabled(False)
        self.play_btn.configure(text="❚❚  Pause")
        self.file_var.set(path.rsplit("/", 1)[-1])
        try:
            self.player.set_volume(self._volume)
            self.player.set_mute(self._muted)
        except Exception:
            pass

    def _toggle_play(self) -> None:
        if self.player is None:
            return
        if self.gate_active:
            return
        self.playing = not self.playing
        try:
            self.player.set_pause(not self.playing)
        except Exception:
            pass
        self.play_btn.configure(text="❚❚  Pause" if self.playing
                                else "▶  Play")
        self.play_since = time.time() if self.playing else None
        if self.playing:
            self._last_frame_wall = 0.0

    def _skip(self, seconds: float) -> None:
        if self.player is None:
            return
        try:
            self.player.seek(float(seconds), relative=True,
                             accurate=False)
        except Exception:
            pass
        self._last_frame_wall = 0.0

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        if self.player is not None:
            try:
                self.player.set_mute(self._muted)
            except Exception:
                pass
        self.mute_btn.configure(text="🔇" if self._muted else "🔊")

    def _on_volume(self, _val=None) -> None:
        self._volume = max(0.0, min(1.0, self.vol_var.get() / 100.0))
        if self.player is not None:
            try:
                self.player.set_volume(self._volume)
            except Exception:
                pass

    def _seek_start(self) -> None:
        self._seeking = True

    def _seek_end(self) -> None:
        self._seeking = False
        if self.player is None or self._duration <= 0:
            return
        target = max(0.0, min(self._duration,
                              self._duration * self.seek_var.get() / 100.0))
        try:
            self.player.seek(target, relative=False, accurate=False)
        except Exception:
            pass
        self._last_frame_wall = 0.0

    def _on_interval_change(self, _e=None) -> None:
        raw = self.interval_var.get().split()[0]
        try:
            val = float(raw)
        except ValueError:
            return
        self.interval = max(self.MIN_INTERVAL,
                            min(self.MAX_INTERVAL, val))

    def _force_gate(self) -> None:
        if self.player is None or self.gate_active:
            return
        self.playback_credit = self.interval
        self._trigger_gate()

    # ----------------------------------------------------- main loops
    def _video_loop(self) -> None:
        next_delay = self.PLAY_MS
        try:
            if (self.player is not None and self.playing
                    and not self.gate_active):
                frame, val = self.player.get_frame()
                if val == "eof":
                    # Loop
                    try:
                        self.player.seek(0, relative=False)
                    except Exception:
                        pass
                    next_delay = 5
                elif val == "paused":
                    next_delay = 50
                elif frame is not None:
                    img, _ = frame
                    self._render_ff_frame(img)
                    self._update_time_ui()
                    now = time.time()
                    if self._last_frame_wall > 0:
                        dt = now - self._last_frame_wall
                        if 0 < dt < 0.5:
                            self.playback_credit += dt
                            inst_fps = 1.0 / max(dt, 1e-3)
                            self._fps_avg = (0.85 * self._fps_avg
                                             + 0.15 * inst_fps)
                    self._last_frame_wall = now
                    self._update_gate_progress()
                    if self.playback_credit >= self.interval:
                        self._trigger_gate()
                    # Use ffpyplayer's pacing hint (seconds until next frame)
                    if isinstance(val, (int, float)) and val > 0:
                        next_delay = int(max(1, min(100, val * 1000)))
                    else:
                        next_delay = 5
                else:
                    # No frame ready yet — poll again soon but not in a
                    # tight loop.
                    next_delay = 10
        finally:
            self.after(next_delay, self._video_loop)

    def _update_time_ui(self) -> None:
        if self.player is None:
            return
        try:
            pts = float(self.player.get_pts() or 0.0)
        except Exception:
            pts = 0.0
        if self._duration <= 0.0:
            try:
                dur = float(self.player.get_metadata().get("duration") or 0.0)
                if dur and dur > 0:
                    self._duration = dur
                    self.time_dur_var.set(self._fmt_time(dur))
                    self.seek.configure(to=100.0)
            except Exception:
                pass
        self.time_cur_var.set(self._fmt_time(pts))
        if not self._seeking and self._duration > 0:
            pct = 100.0 * pts / self._duration
            self.seek_var.set(max(0.0, min(100.0, pct)))

    @staticmethod
    def _fmt_time(secs: float) -> str:
        secs = max(0, int(secs))
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _cam_loop(self) -> None:
        try:
            frame = self.app.tracker.get_frame()
            if frame is not None:
                lbl_w = max(self.cam_label.winfo_width(), 240)
                lbl_h = max(self.cam_label.winfo_height(), 160)
                h, w = frame.shape[:2]
                scale = min(lbl_w / w, lbl_h / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                pil = Image.fromarray(frame)
                if (new_w, new_h) != (w, h):
                    pil = pil.resize((new_w, new_h), Image.BILINEAR)
                photo = ImageTk.PhotoImage(image=pil)
                self.cam_label.configure(image=photo, text="")
                self.cam_label.image = photo
                self._cam_photo = photo
            else:
                if self._cam_photo is not None:
                    self.cam_label.configure(image="",
                                             text="Camera stopped")
                    self.cam_label.image = None
                    self._cam_photo = None
        finally:
            self.after(66, self._cam_loop)

    def _render_ff_frame(self, img) -> None:
        """Convert an ffpyplayer Image to a Tk PhotoImage and show it."""
        lbl_w = max(self.video_label.winfo_width(), 320)
        lbl_h = max(self.video_label.winfo_height(), 240)
        w, h = img.get_size()
        scale = min(lbl_w / w, lbl_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # ffpyplayer gives us a planar image; convert via bytearray.
        data_bytes = bytes(img.to_bytearray()[0])
        pil = Image.frombytes("RGB", (w, h), data_bytes)
        if (new_w, new_h) != (w, h):
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
        photo = ImageTk.PhotoImage(image=pil)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # keep ref
        self._last_photo = photo

    def _gate_loop(self) -> None:
        try:
            # Threshold = your running average score + small margin,
            # so you pass by being roughly as steady as usual.
            thr = self.app.baseline.personal_avg_threshold(
                margin=8.0, default=50.0)
            # apply failure-streak relaxation so users don't get stuck
            thr += self.fail_streak * 6.0
            thr -= self.pass_streak * 2.0
            thr = float(max(25.0, min(90.0, thr)))
            self.personal_threshold = thr
            if abs(thr - self.last_threshold_shown) > 0.5:
                self.last_threshold_shown = thr
                self.thr_var.set(f"{thr:.0f} /100")
                avg = self.app.baseline.score_avg
                if avg is None or self.app.baseline.samples < 20:
                    hint = "Learning your average tremor score…"
                else:
                    hint = (f"Avg {avg:.0f} + margin. Stay at or below "
                            "this to pass the gate.")
                if self.fail_streak >= 1:
                    hint = ("Relaxed after a miss — keep going, this "
                            "check will be easier.")
                self.thr_hint_var.set(hint)

            if self.gate_active and self.session is not None:
                self._advance_challenge()
        finally:
            self.after(self.CHECK_MS, self._gate_loop)

    def _update_gate_progress(self) -> None:
        remaining = max(0.0, self.interval - self.playback_credit)
        self.next_gate_var.set(f"{remaining:0.0f} s")
        pct = 100.0 * self.playback_credit / max(0.1, self.interval)
        self.gate_progress["value"] = min(100.0, pct)

    # ----------------------------------------------- challenge control
    def _trigger_gate(self) -> None:
        # Always pause the video when the gate fires, regardless of whether
        # the camera is running. If it's not, prompt the user to start it.
        self.playing = False
        if self.player is not None:
            try:
                self.player.set_pause(True)
            except Exception:
                pass
        self.play_btn.configure(text="▶  Play")
        self.gate_active = True
        self.chal_result_var.set("")
        ex = random.choice(EXERCISES)
        self.session = ExerciseSession(ex)
        self.session.start()
        self.chal_name_var.set(ex.name)
        self.chal_desc_var.set(ex.description)
        if self.app.tracker.thread is None:
            self.chal_stage_var.set("Start the camera to begin")
            self.chal_stage_lbl.configure(bg=RED_TINT, fg=RED_DEEP)
            self.chal_feedback_var.set(
                "Video paused. Press 'Start camera' on the top bar so we "
                "can verify your pose, then hold the requested exercise.")
        else:
            self.chal_stage_var.set("Get into position…")
            self.chal_stage_lbl.configure(bg=RED_SOFT, fg=RED)
            self.chal_feedback_var.set("")
        self.chal_progress["value"] = 0
        self.skip_btn.set_disabled(False)

    def _advance_challenge(self) -> None:
        assert self.session is not None
        sess = self.session
        pose = self.app.tracker.get_pose()
        tip = self.app.tracker.get_hand_tip_norm()
        m = self.app.last_metrics
        score = m.score if m else None
        peak = m.peak_hz if m else None
        amp = m.rms_amp_mm if m else None
        grip = self.app.tracker.get_grip_strength()
        v = sess.update(pose, tip, score, peak, amp, grip=grip)

        # Live pose-correctness chip — must be green to pass
        if v.ok:
            self.pose_chip_var.set("Pose: correct ✓")
            self.pose_chip.configure(bg="#ecfdf3", fg="#0f5132")
        else:
            self.pose_chip_var.set("Pose: not yet")
            self.pose_chip.configure(bg=RED_TINT, fg=RED_DEEP)

        if sess.stage == Stage.PREPARE:
            if sess.prepare_ready_since is None:
                self.chal_stage_var.set("Get into position…")
                self.chal_progress["value"] = 0
            else:
                frac = min(1.0,
                           (time.time() - sess.prepare_ready_since)
                           / max(0.1, sess.exercise.prepare_secs))
                self.chal_stage_var.set(
                    f"Holding position… {int(frac * 100)}%")
                self.chal_progress["value"] = frac * 100
            self.chal_stage_lbl.configure(bg=RED_SOFT, fg=RED)
            self.chal_feedback_var.set(v.message)
        elif sess.stage == Stage.HOLD:
            frac = min(1.0, sess.elapsed() / sess.exercise.hold_secs)
            remaining = max(0,
                            int(sess.exercise.hold_secs - sess.elapsed()))
            self.chal_stage_var.set(f"Measuring… {remaining}s left")
            self.chal_stage_lbl.configure(bg=RED_TINT, fg=RED_DEEP)
            self.chal_feedback_var.set(
                v.message if v.ok else
                f"Stay in position — {v.message.lower()}")
            self.chal_progress["value"] = frac * 100
        elif sess.stage == Stage.DONE:
            self._finish_challenge()

    def _finish_challenge(self) -> None:
        assert self.session is not None
        sess = self.session
        samples = sess.hold_samples
        mean_s = float(np.mean(samples)) if samples else 100.0
        passed = bool(samples) and mean_s <= self.personal_threshold
        self.chal_progress["value"] = 100

        if passed:
            self.pass_streak += 1
            self.fail_streak = max(0, self.fail_streak - 1)
            self.chal_stage_var.set("Passed — enjoy the video")
            self.chal_stage_lbl.configure(bg="#ecfdf3", fg="#0f5132")
            self.chal_result_var.set(
                f"Mean tremor score {mean_s:.0f}/100 — under your "
                f"threshold of {self.personal_threshold:.0f}. Resuming "
                "playback.")
            self.chal_result_lbl.configure(fg=OK_GREEN)
            self.gate_active = False
            self.session = None
            self.playback_credit = 0.0
            self.skip_btn.set_disabled(True)
            # resume
            self.playing = True
            if self.player is not None:
                try:
                    self.player.set_pause(False)
                except Exception:
                    pass
            self._last_frame_wall = 0.0
            self.play_btn.configure(text="❚❚  Pause")
        else:
            self.fail_streak += 1
            self.pass_streak = 0
            self.chal_stage_var.set("Not yet — let's try again")
            self.chal_stage_lbl.configure(bg=RED_TINT, fg=RED_DEEP)
            reason = ("pose wasn't held long enough" if not samples
                      else f"mean score {mean_s:.0f} exceeded your "
                           f"threshold of {self.personal_threshold:.0f}")
            self.chal_result_var.set(
                f"{reason}. Threshold will relax a bit on the next try.")
            self.chal_result_lbl.configure(fg=RED_DEEP)
            # leave gate active, enable retry button
            self.skip_btn.set_disabled(False)

    def _retry_challenge(self) -> None:
        if not self.gate_active:
            # allow user to manually fire a check after a miss
            self._force_gate()
            return
        ex = (self.session.exercise if self.session is not None
              else random.choice(EXERCISES))
        self.session = ExerciseSession(ex)
        self.session.start()
        self.chal_stage_var.set("Get into position…")
        self.chal_stage_lbl.configure(bg=RED_SOFT, fg=RED)
        self.chal_feedback_var.set("")
        self.chal_result_var.set("")
        self.chal_progress["value"] = 0

    def shutdown(self) -> None:
        try:
            if self.player is not None:
                self.player.close_player()
        except Exception:
            pass
