"""Real-Time Tremor Detector — desktop app.

Tracks a hand landmark via webcam (MediaPipe Hands), estimates dominant
oscillation frequency (1–15 Hz) using a Goertzel scan on a rolling 3-second
window, and shows live metrics plus a motion spectrum in a Tkinter window.

Disclaimer: technical demo, not a medical device.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Thread
from tkinter import Tk, ttk, StringVar, messagebox

import cv2
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

# ---------- Config ----------
WINDOW_SECONDS = 3.0
CAM_WIDTH = 640
CAM_HEIGHT = 480
LANDMARK_CHOICES = {
    "Index fingertip": 8,
    "Middle fingertip": 12,
    "Thumb tip": 4,
    "Wrist": 0,
}
FREQ_MIN, FREQ_MAX, FREQ_STEP = 1.0, 15.0, 0.25
TREMOR_BAND = (3.0, 12.0)


# ---------- Signal processing ----------
def detrend(arr: np.ndarray) -> np.ndarray:
    n = arr.size
    if n < 2:
        return arr.copy()
    x = np.arange(n, dtype=np.float64)
    m, b = np.polyfit(x, arr, 1)
    return arr - (m * x + b)


def hann_window(arr: np.ndarray) -> np.ndarray:
    return arr * np.hanning(arr.size)


def goertzel_power(arr: np.ndarray, fs: float, target_hz: float) -> float:
    n = arr.size
    if n == 0 or target_hz <= 0:
        return 0.0
    k = target_hz * n / fs
    w = 2.0 * np.pi * k / n
    cw, sw = np.cos(w), np.sin(w)
    coeff = 2.0 * cw
    s1 = 0.0
    s2 = 0.0
    for v in arr:
        s0 = v + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * cw
    imag = s2 * sw
    return (real * real + imag * imag) / (n * n)


def resample_uniform(samples: list[tuple[float, float, float]], fs: float):
    if len(samples) < 2:
        return None
    t = np.array([s[0] for s in samples], dtype=np.float64)
    x = np.array([s[1] for s in samples], dtype=np.float64)
    y = np.array([s[2] for s in samples], dtype=np.float64)
    dur = t[-1] - t[0]
    if dur <= 0:
        return None
    n = int(dur * fs)
    if n < 16:
        return None
    tu = t[0] + np.arange(n) / fs
    xu = np.interp(tu, t, x)
    yu = np.interp(tu, t, y)
    return xu, yu, fs, dur


@dataclass
class Analysis:
    fs: float
    dominant_hz: float
    amplitude: float
    band_ratio: float
    score: int
    freqs: np.ndarray
    power: np.ndarray


def analyse(samples: deque) -> Analysis | None:
    if len(samples) < 16:
        return None
    now = samples[-1][0]
    window = [s for s in samples if s[0] >= now - WINDOW_SECONDS]
    if len(window) < 16:
        return None
    dur = window[-1][0] - window[0][0]
    if dur <= 0:
        return None
    fs_est = len(window) / dur
    fs = float(np.clip(fs_est, 15.0, 60.0))
    res = resample_uniform(window, fs)
    if res is None:
        return None
    xu, yu, fs, _ = res
    dx = hann_window(detrend(xu))
    dy = hann_window(detrend(yu))

    freqs = np.arange(FREQ_MIN, FREQ_MAX + 1e-9, FREQ_STEP)
    power = np.array(
        [goertzel_power(dx, fs, f) + goertzel_power(dy, fs, f) for f in freqs]
    )

    # Dominant frequency limited to 2..14 Hz to avoid edge artifacts
    mask = (freqs >= 2.0) & (freqs <= 14.0)
    if np.any(mask) and power[mask].max() > 0:
        dom_idx = np.argmax(np.where(mask, power, -1))
        dom_hz = float(freqs[dom_idx])
    else:
        dom_hz = 0.0

    rms = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    total = float(power.sum())
    band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
    band_power = float(power[band_mask].sum())
    band_ratio = band_power / total if total > 0 else 0.0

    amp_norm = min(1.0, rms / 0.02)
    score = int(round(100 * min(1.0, amp_norm * (0.4 + 0.6 * band_ratio))))

    return Analysis(fs, dom_hz, rms, band_ratio, score, freqs, power)


# ---------- Capture / tracking thread ----------
class TremorTracker:
    def __init__(self) -> None:
        self.samples: deque = deque(maxlen=1024)
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.hand_present = False
        self.landmark_idx = 8
        self.fps_meas = 0.0
        self._cap = None

    def set_landmark(self, idx: int) -> None:
        self.landmark_idx = idx
        self.samples.clear()

    def start(self, cap: "cv2.VideoCapture") -> None:
        if self.thread and self.thread.is_alive():
            return
        self._cap = cap
        self.stop_event.clear()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.samples.clear()
        self.hand_present = False
        self.latest_frame = None

    def _run(self) -> None:
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        cap = self._cap
        if cap is None or not cap.isOpened():
            self.latest_frame = None
            return

        last_t = time.time()
        ema_fps = 0.0
        with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                    self.fps_meas = ema_fps

                h, w = frame.shape[:2]
                if results.multi_hand_landmarks:
                    self.hand_present = True
                    lm = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    p = lm.landmark[self.landmark_idx]
                    self.samples.append((now, float(p.x), float(p.y)))
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 8,
                               (229, 37, 133), -1)
                else:
                    self.hand_present = False

                self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        self._cap = None


# ---------- GUI ----------
class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Real-Time Tremor Detector")
        self.root.geometry("1100x640")
        self.tracker = TremorTracker()

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Video panel
        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.pack(fill="both", expand=True)

        # Right panel
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        # Status
        self.status_var = StringVar(value="Idle")
        status_box = ttk.LabelFrame(right, text="Status", padding=8)
        status_box.grid(row=0, column=0, sticky="ew")
        self.status_lbl = ttk.Label(
            status_box, textvariable=self.status_var,
            font=("Helvetica", 14, "bold"), anchor="center"
        )
        self.status_lbl.pack(fill="x")

        # Metrics
        metrics = ttk.LabelFrame(right, text="Metrics", padding=8)
        metrics.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        metrics.columnconfigure(1, weight=1)

        self.freq_var = StringVar(value="— Hz")
        self.amp_var = StringVar(value="—")
        self.score_var = StringVar(value="—")
        self.fs_var = StringVar(value="— fps")
        self.hand_var = StringVar(value="No")

        rows = [
            ("Dominant frequency", self.freq_var),
            ("Amplitude (norm.)", self.amp_var),
            ("Tremor score (0–100)", self.score_var),
            ("Capture rate", self.fs_var),
            ("Hand detected", self.hand_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(metrics, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(metrics, textvariable=var,
                      font=("Helvetica", 12, "bold")).grid(
                row=i, column=1, sticky="e", pady=2
            )

        # Controls
        controls = ttk.LabelFrame(right, text="Controls", padding=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(controls, text="Start camera", command=self.start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(controls, text="Tracked point:").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.lm_var = StringVar(value="Index fingertip")
        lm_box = ttk.Combobox(controls, textvariable=self.lm_var,
                              values=list(LANDMARK_CHOICES.keys()),
                              state="readonly")
        lm_box.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        lm_box.bind("<<ComboboxSelected>>", self._on_landmark_change)

        # Spectrum plot
        spec = ttk.LabelFrame(right, text="Motion spectrum (1–15 Hz)", padding=4)
        spec.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        right.rowconfigure(3, weight=1)

        self.fig = Figure(figsize=(4, 2.2), dpi=100, facecolor="#1b1f24")
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=spec)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Hint
        hint = (
            "Hold hand ~40–60 cm from camera with good lighting.\n"
            "Tremor band: 3–12 Hz. Essential/Parkinsonian: ~4–7 Hz.\n"
            "Disclaimer: demo only, not a medical device."
        )
        ttk.Label(right, text=hint, foreground="#6a737d",
                  wraplength=380, justify="left").grid(
            row=4, column=0, sticky="ew", pady=(8, 0)
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_loop()

    def _style_axes(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#0b0e13")
        self.ax.set_xlim(FREQ_MIN, FREQ_MAX)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Frequency (Hz)", color="#8b949e", fontsize=8)
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        for s in self.ax.spines.values():
            s.set_color("#30363d")
        self.ax.axvspan(TREMOR_BAND[0], TREMOR_BAND[1],
                        color="#4cc9f0", alpha=0.08)

    def _on_landmark_change(self, _evt=None) -> None:
        name = self.lm_var.get()
        self.tracker.set_landmark(LANDMARK_CHOICES.get(name, 8))

    def start(self) -> None:
        # Open the camera on the main thread so macOS can display the
        # authorization prompt via AVFoundation.
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        # Give macOS a moment to display the permission dialog, then probe.
        ok = False
        for _ in range(30):  # up to ~3 s
            if cap.isOpened():
                ok, _frame = cap.read()
                if ok:
                    break
            time.sleep(0.1)
        if not ok:
            cap.release()
            messagebox.showerror(
                "Camera unavailable",
                "Could not access the webcam. On macOS, grant camera "
                "permission in System Settings → Privacy & Security → Camera "
                "for Terminal (or your Python launcher), then try again.",
            )
            return
        self.tracker.set_landmark(LANDMARK_CHOICES[self.lm_var.get()])
        self.tracker.start(cap)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Collecting samples…")

    def stop(self) -> None:
        self.tracker.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopped")
        self.freq_var.set("— Hz")
        self.amp_var.set("—")
        self.score_var.set("—")
        self.fs_var.set("— fps")
        self.hand_var.set("No")
        self._style_axes()
        self.canvas.draw_idle()
        self.video_label.configure(image="")

    def _update_loop(self) -> None:
        # Video frame
        frame = self.tracker.latest_frame
        if frame is not None:
            lbl_w = max(self.video_label.winfo_width(), 320)
            lbl_h = max(self.video_label.winfo_height(), 240)
            h, w = frame.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # keep ref

        # Metrics
        self.hand_var.set("Yes" if self.tracker.hand_present else "No")
        if self.tracker.fps_meas > 0:
            self.fs_var.set(f"{self.tracker.fps_meas:.1f} fps")

        result = analyse(self.tracker.samples)
        self._update_status(result)
        self._update_spectrum(result)

        self.root.after(100, self._update_loop)

    def _update_status(self, r: Analysis | None) -> None:
        if not self.tracker.hand_present:
            if self.tracker.thread and self.tracker.thread.is_alive():
                self.status_var.set("Show your hand to the camera")
            return
        if r is None:
            self.status_var.set("Collecting samples…")
            return
        self.freq_var.set(f"{r.dominant_hz:.2f} Hz")
        self.amp_var.set(f"{r.amplitude:.4f}")
        self.score_var.set(str(r.score))
        if r.score < 20:
            self.status_var.set("No significant tremor detected")
        elif r.score < 50:
            self.status_var.set(f"Mild oscillation @ {r.dominant_hz:.1f} Hz")
        else:
            self.status_var.set(f"Tremor-like motion @ {r.dominant_hz:.1f} Hz")

    def _update_spectrum(self, r: Analysis | None) -> None:
        self._style_axes()
        if r is not None and r.power.max() > 0:
            norm = r.power / r.power.max()
            self.ax.bar(r.freqs, norm, width=FREQ_STEP * 0.9,
                        color="#4cc9f0", edgecolor="none")
            if r.dominant_hz > 0:
                self.ax.axvline(r.dominant_hz, color="#f72585", linewidth=1.5)
        self.fig.tight_layout()
        self.canvas.draw_idle()

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
