"""
hand_landscape/dots.py

Dense residual-dot amplitude field for tremor classification.

The tracked hand region is sampled with ~100 stable feature dots. Each dot is an
independent *micro-oscillator sensor*: after macro hand motion is removed
(residual = point_delta − macro_state_delta), every dot owns a short time-history
of its residual motion. Tremor is classified from the RHYTHMIC STRUCTURE of that
residual amplitude field — never from raw movement or point-latch behaviour.

Per-dot lifecycle (this module owns the classification; the bridge owns the
optical-flow point lifecycle / refill):
  - a dot must survive a WARMUP period before it may vote
  - each dot tracks dot_age, relatch_count and a residual-magnitude history
  - discontinuous single-frame jumps increment relatch_count
  - a dot whose relatch rate is high is rejected as RELATCHING

A dot supports tremor only if ALL hold:
  1. it is warmed-up (enough age + history) and high optical-flow confidence
  2. residual oscillation amplitude (RMS of 3–12 Hz bandpassed residual) is high
  3. its dominant frequency sits in the tremor band (3–12 Hz)
  4. that frequency persists across analysis windows
  5. it is not relatching / spike-dominated (tracking noise)

Global tremor is a ROBUST vote over VALID dots:
  - median / trimmed-mean (top-k) of valid dot amplitudes — never a flat average
  - a minimum valid_dot_count (coverage) is required
  - several valid dots must share a similar peak frequency (coherence)
so one bad dot can never explode the score and a single spike is rejected.

States per dot
--------------
    INVALID            – warming up / too short / low quality, cannot vote
    STABLE_NO_TREMOR   – valid but low tremor-band amplitude
    TREMOR_CANDIDATE   – rhythmic 3–12 Hz oscillation in this window
    TREMOR_CONFIRMED   – candidate whose frequency persisted across windows
    TRACKING_NOISE     – high amplitude but spiky / broadband (not rhythmic)
    RELATCHING         – dominated by single-frame jumps (slip / re-latch)
"""

from __future__ import annotations

import math
import os
from collections import deque, Counter

import numpy as np

from .tremor import _butter_bandpass, _spike_energy_fraction, TREMOR_LO_HZ, TREMOR_HI_HZ

try:
    from scipy.signal import welch  # type: ignore
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ─── debug mode ───────────────────────────────────────────────────────────────
# Separates *evidence* from *trust*. When on (MB_TREMOR_DEBUG=1) the early-stage
# voting thresholds are RELAXED and the score is no longer hard-zeroed on a
# non-VALID tracking state (the raw evidence is revealed and merely flagged
# UNTRUSTED). Default OFF so the library / test-suite keep the strict production
# thresholds and all false-positive guarantees. The Electron bridge enables it.
DEBUG_MODE = os.environ.get("MB_TREMOR_DEBUG", "0").lower() in ("1", "true", "yes", "on")


# ─── per-dot states ───────────────────────────────────────────────────────────
INVALID = "INVALID"
STABLE_NO_TREMOR = "STABLE_NO_TREMOR"
TREMOR_CANDIDATE = "TREMOR_CANDIDATE"
TREMOR_CONFIRMED = "TREMOR_CONFIRMED"
TRACKING_NOISE = "TRACKING_NOISE"
RELATCHING = "RELATCHING"

_TREMOR_POSITIVE = (TREMOR_CANDIDATE, TREMOR_CONFIRMED)
_VOTE_EXCLUDED = (INVALID, RELATCHING, TRACKING_NOISE)


# ─── thresholds ───────────────────────────────────────────────────────────────
HISTORY                = 180     # rolling residual samples kept per dot (~6 s)
WARMUP_FRAMES          = 18      # a dot must live this many frames before voting
MIN_DOT_SECONDS        = 1.5     # minimum residual duration before a dot can vote
MIN_DOT_SAMPLES        = 16      # minimum residual samples for spectral analysis
QUALITY_MIN            = 0.30    # median per-dot tracking quality floor
AMP_MIN_PX             = 0.30    # RMS bandpassed amplitude floor for tremor
# ── debug-relaxed thresholds (strict production value kept in the else) ───────
AMP_SCALE_PX           = (0.35 if DEBUG_MODE else 1.10)   # amplitude that saturates the normalised score
BPR_MIN                = (0.25 if DEBUG_MODE else 0.45)   # tremor-band / total power floor (RAW residual)
PROMINENCE_MIN         = (1.8  if DEBUG_MODE else 3.0)    # peak / mean band power for a sharp rhythm
FLATNESS_MAX           = (0.75 if DEBUG_MODE else 0.55)   # spectral flatness above this ⇒ broadband noise
SPIKE_MAX              = 0.50    # spike-energy fraction above this → noise/relatch
RELATCH_SPIKE          = 0.70    # spike fraction this high → RELATCHING
PEAK_CV_MAX            = 0.25    # peak-frequency CV ceiling for persistence
COHERENCE_HZ           = 1.0     # dots within ±this of median freq are coherent
COHERENT_TARGET        = 3       # this many coherent dots = full coherence
DOT_TIMEOUT_S          = 0.5     # drop a dot unseen for longer than this
PEAK_HISTORY           = 10      # windows of peak-frequency memory per dot

# Dense-field robustness (debug-relaxed; strict production values in the else)
MIN_VALID_DOTS         = (10 if DEBUG_MODE else 30)      # valid-dot coverage required for full confidence
COVERAGE_DENOM         = (12 if DEBUG_MODE else MIN_VALID_DOTS)  # coverage_factor denominator
TOPK_FRACTION          = 0.10    # fraction of valid dots used for top-k amplitude
TOPK_MIN               = 3       # never trust fewer than this many dots for top-k
# With a ~500-dot field, cap how many (highest-amplitude) dots run the EXPENSIVE
# spectral pass per window so analysis stays real-time. The vote is unaffected:
# coverage/coherence already saturate well below this, and the strongest dots —
# the tremor-relevant ones — are always the ones analysed.
MAX_SPECTRAL_DOTS      = 160

# Per-dot discontinuity / relatch detection
JUMP_K                 = 4.0     # residual-magnitude jump this × median ⇒ a jump
JUMP_FLOOR_PX          = 1.5     # ignore jumps smaller than this (sensor noise)
RELATCH_RATE_MAX       = 0.15    # jump fraction over a dot's life ⇒ RELATCHING


def _spectral_features(sig: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    """Return (peak_hz, band_power_ratio, prominence, flatness) of a RAW residual.

    Computed on the un-filtered (mean-removed) residual so that band_power_ratio
    and flatness actually discriminate a concentrated tremor tone from broadband
    tracking noise:

      band_power_ratio – tremor-band power / total power (tonal ⇒ high)
      prominence       – peak-bin power / mean band power (sharp peak ⇒ high)
      flatness         – geometric/arithmetic mean of PSD (white noise ⇒ ~1,
                         pure tone ⇒ ~0). The single strongest noise rejector.
    """
    if len(sig) < MIN_DOT_SAMPLES:
        return 0.0, 0.0, 1.0, 1.0
    if _SCIPY:
        try:
            # Short segments ⇒ several Welch averages ⇒ noise peaks flatten out.
            nperseg = min(len(sig), max(32, int(fs * 2)))
            freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.0, 0.0, 1.0, 1.0
    else:
        psd = np.abs(np.fft.rfft(sig)) ** 2
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)

    # Spectral flatness over everything above DC (drift excluded).
    above_dc = freqs > 0.5
    pdc = psd[above_dc]
    if pdc.size and float(np.mean(pdc)) > 1e-12:
        gmean = float(np.exp(np.mean(np.log(pdc + 1e-12))))
        flatness = float(np.clip(gmean / (float(np.mean(pdc)) + 1e-12), 0.0, 1.0))
    else:
        flatness = 1.0

    band = (freqs >= TREMOR_LO_HZ) & (freqs <= TREMOR_HI_HZ)
    if not band.any():
        return 0.0, 0.0, 1.0, flatness
    band_psd = psd[band]
    band_power = float(np.sum(band_psd))
    total_power = float(np.sum(psd[above_dc])) + 1e-12
    ratio = min(1.0, band_power / total_power)
    peak_idx = int(np.argmax(band_psd))
    peak_hz = float(freqs[band][peak_idx])
    mean_band = float(np.mean(band_psd)) + 1e-12
    prominence = float(band_psd[peak_idx]) / mean_band
    return peak_hz, ratio, prominence, flatness


class MicroOscillator:
    """One persistent tracked dot, classified from its residual rhythm."""

    __slots__ = (
        "id", "region", "x", "y", "age", "last_seen", "relatch_count", "last_jump",
        "_rx", "_ry", "_rmag", "_ts", "_q", "_peak_hist",
        "amplitude", "peak_hz", "band_power_ratio", "prominence", "flatness",
        "freq_confidence", "score", "state", "invalid_reason", "_raw_amp",
    )

    def __init__(self, dot_id: int, region: str = "hand") -> None:
        self.id = dot_id
        self.region = region
        self.x = 0.0
        self.y = 0.0
        self.age = 0                 # dot_age in frames (warmup uses this)
        self.last_seen = 0.0
        self.relatch_count = 0       # # of discontinuous residual jumps seen
        self.last_jump = False       # did the most recent sample jump?
        self._rx: deque = deque(maxlen=HISTORY)
        self._ry: deque = deque(maxlen=HISTORY)
        self._rmag: deque = deque(maxlen=HISTORY)
        self._ts: deque = deque(maxlen=HISTORY)
        self._q: deque = deque(maxlen=HISTORY)
        self._peak_hist: deque = deque(maxlen=PEAK_HISTORY)
        self.amplitude = 0.0
        self.peak_hz = 0.0
        self.band_power_ratio = 0.0
        self.prominence = 1.0
        self.flatness = 1.0
        self.freq_confidence = 0.0
        self.score = 0.0
        self.state = INVALID
        self.invalid_reason = "too_young"
        self._raw_amp = 0.0

    # ------------------------------------------------------------------ #
    @property
    def dot_age(self) -> int:
        return self.age

    @property
    def relatch_rate(self) -> float:
        """Fraction of this dot's life that was a discontinuous jump."""
        if self.age <= 0:
            return 0.0
        return float(self.relatch_count) / float(self.age)

    # ------------------------------------------------------------------ #
    def update(self, rx: float, ry: float, quality: float, ts: float,
               x: float, y: float, region: str) -> None:
        rx = float(rx)
        ry = float(ry)
        mag = math.hypot(rx, ry)

        # Discontinuous-jump detection: a residual magnitude far above this
        # dot's recent typical residual is a slip / re-latch, NOT tremor.
        self.last_jump = False
        if len(self._rmag) >= 6:
            recent = np.asarray(self._rmag, dtype=np.float64)[-12:]
            med = float(np.median(recent))
            if mag > JUMP_FLOOR_PX and mag > JUMP_K * (med + 1e-6):
                self.relatch_count += 1
                self.last_jump = True

        self._rx.append(rx)
        self._ry.append(ry)
        self._rmag.append(mag)
        self._q.append(float(quality))
        self._ts.append(float(ts))
        self.x = float(x)
        self.y = float(y)
        self.region = region
        self.age += 1
        self.last_seen = float(ts)

    # ------------------------------------------------------------------ #
    def analyze(self, fs: float) -> None:
        """Classify this dot from its residual rhythm. Sets state + score.

        Two-stage so a large field stays real-time: the cheap validity + raw
        amplitude gates run first, and the expensive spectral pass only if the
        dot actually moves enough to possibly be tremor.
        """
        if self._cheap_gate(fs):
            self._spectral_analyze(fs)

    def _cheap_gate(self, fs: float) -> bool:
        """Cheap validity gates + raw-amplitude pre-gate (no filtering / PSD).

        Returns True iff this dot needs the expensive spectral pass; otherwise a
        terminal state + score is set here and the spectral pass is skipped.
        """
        self.invalid_reason = ""
        self._raw_amp = 0.0
        # ── warmup gate: a freshly spawned dot may not vote yet ───────────────
        if self.age < WARMUP_FRAMES:
            self.state = INVALID
            self.score = 0.0
            self.invalid_reason = "too_young"
            return False
        n = len(self._rx)
        if n < MIN_DOT_SAMPLES:
            self.state = INVALID
            self.score = 0.0
            self.invalid_reason = "too_few_samples"
            return False
        span = self._ts[-1] - self._ts[0]
        median_q = float(np.median(self._q)) if self._q else 0.0
        if span < MIN_DOT_SECONDS:
            self.state = INVALID
            self.score = 0.0
            self.invalid_reason = "too_short_span"
            return False
        if median_q < QUALITY_MIN:
            self.state = INVALID
            self.score = 0.0
            self.invalid_reason = "low_quality"
            return False
        # ── relatch gate: a dot that keeps slipping is a tracking artefact ────
        if self.relatch_rate > RELATCH_RATE_MAX:
            self.state = RELATCHING
            self.score = 0.0
            self.invalid_reason = "relatching"
            return False

        rx = np.asarray(self._rx, dtype=np.float64)
        ry = np.asarray(self._ry, dtype=np.float64)
        rx -= np.mean(rx)
        ry -= np.mean(ry)

        # Cheap pre-gate (critical with a ~500-dot field): band-pass filtering
        # can only REMOVE spectral energy, so the filtered amplitude can never
        # exceed the RAW residual amplitude. If even the raw amplitude is below
        # the tremor floor the dot cannot be tremor — nor a spike, which would
        # inflate the raw std well past the floor — so skip the costly two
        # Butterworth filters + PSD for the (usually large) majority of still
        # dots. This is behaviour-preserving: such dots reach STABLE_NO_TREMOR
        # on the existing post-filter path too.
        raw_amp = float(math.hypot(float(np.std(rx)), float(np.std(ry))))
        self._raw_amp = raw_amp     # used by the field to rank under the cap
        if raw_amp < AMP_MIN_PX:
            self.amplitude = raw_amp
            self.peak_hz = 0.0
            self.band_power_ratio = 0.0
            self.prominence = 1.0
            self.flatness = 1.0
            self.freq_confidence = 0.0
            self.state = STABLE_NO_TREMOR
            self.score = 0.0
            self.invalid_reason = "low_amplitude"
            return False
        return True

    def mark_spectral_capped(self) -> None:
        """Overflow dot when more than MAX_SPECTRAL_DOTS move at once: it cleared
        the raw-amplitude gate but is not spectrally classified this window.
        Treated as non-tremor (it can never inflate the vote)."""
        self.amplitude = self._raw_amp
        self.peak_hz = 0.0
        self.band_power_ratio = 0.0
        self.prominence = 1.0
        self.flatness = 1.0
        self.freq_confidence = 0.0
        self.state = STABLE_NO_TREMOR
        self.score = 0.0
        self.invalid_reason = "spectral_capped"

    def _spectral_analyze(self, fs: float) -> None:
        """Expensive pass: band-pass amplitude + PSD features + classification.
        Assumes _cheap_gate(fs) returned True for this window."""
        rx = np.asarray(self._rx, dtype=np.float64)
        ry = np.asarray(self._ry, dtype=np.float64)
        rx -= np.mean(rx)
        ry -= np.mean(ry)
        median_q = float(np.median(self._q)) if self._q else 0.0

        bx = _butter_bandpass(rx, fs)
        by = _butter_bandpass(ry, fs)
        std_x, std_y = float(np.std(bx)), float(np.std(by))
        # Amplitude = RMS of the 3–12 Hz bandpassed residual (per spec).
        self.amplitude = float(math.hypot(std_x, std_y))

        spike = max(_spike_energy_fraction(bx), _spike_energy_fraction(by))

        # Fast path for the dense field: a dot whose tremor-band amplitude is
        # negligible and which is not spiking cannot be tremor — skip the costly
        # per-dot PSD. With ~100 mostly-still dots this avoids ~100 Welch
        # transforms per analysis window.
        if self.amplitude < AMP_MIN_PX and spike < SPIKE_MAX:
            self.peak_hz = 0.0
            self.band_power_ratio = 0.0
            self.prominence = 1.0
            self.flatness = 1.0
            self.freq_confidence = 0.0
            self.state = STABLE_NO_TREMOR
            self.score = 0.0
            self.invalid_reason = "low_amplitude"
            return

        # Spectral features come from the RAW (unfiltered) residual so that
        # band-power ratio and flatness genuinely separate a tremor tone from
        # broadband tracking noise. Use the axis with the stronger oscillation.
        raw_axis = rx if std_x >= std_y else ry
        peak_hz, bpr, prom, flatness = _spectral_features(raw_axis, fs)
        self.peak_hz = peak_hz
        self.band_power_ratio = bpr
        self.prominence = prom
        self.flatness = flatness

        in_band = (TREMOR_LO_HZ <= peak_hz <= TREMOR_HI_HZ)
        if in_band:
            self._peak_hist.append(peak_hz)
        self.freq_confidence = self._frequency_confidence(prom)

        rhythmic = (prom >= PROMINENCE_MIN and flatness <= FLATNESS_MAX and bpr >= BPR_MIN)

        # ── classification ────────────────────────────────────────────────────
        if spike >= RELATCH_SPIKE:
            self.state = RELATCHING
            self.invalid_reason = "relatching"
        elif self.amplitude >= AMP_MIN_PX and (spike >= SPIKE_MAX or not rhythmic):
            # Moving a lot, but flat / spiky spectrum → tracking noise, not tremor.
            self.state = TRACKING_NOISE
            self.invalid_reason = "noisy_spectrum"
        elif self.amplitude < AMP_MIN_PX or not in_band:
            self.state = STABLE_NO_TREMOR
            self.invalid_reason = "low_amplitude" if self.amplitude < AMP_MIN_PX else "out_of_band"
        elif len(self._peak_hist) >= 3 and self.freq_confidence >= 0.6:
            self.state = TREMOR_CONFIRMED
            self.invalid_reason = ""
        else:
            self.state = TREMOR_CANDIDATE
            self.invalid_reason = ""

        # ── per-dot tremor score (only tremor-positive dots score) ────────────
        if self.state in _TREMOR_POSITIVE:
            amp_norm = math.tanh(self.amplitude / AMP_SCALE_PX)
            tonality = float(np.clip(1.0 - flatness / FLATNESS_MAX, 0.0, 1.0))
            self.score = float(np.clip(
                amp_norm * np.clip(bpr, 0.0, 1.0) * tonality
                * (0.5 + 0.5 * self.freq_confidence) * median_q,
                0.0, 1.0,
            ))
        else:
            self.score = 0.0

    def _frequency_confidence(self, prominence: float) -> float:
        """Blend peak sharpness with cross-window peak-frequency stability."""
        prom_conf = float(np.clip((prominence - 1.0) / (PROMINENCE_MIN + 1.0), 0.0, 1.0))
        peaks = np.asarray(self._peak_hist, dtype=np.float64)
        if len(peaks) >= 3:
            mean = float(np.mean(peaks))
            if mean < 1e-6:
                return 0.0
            cv = float(np.std(peaks)) / mean
            cv_conf = float(np.clip(1.0 - cv / PEAK_CV_MAX, 0.0, 1.0))
            return float(np.clip(0.5 * prom_conf + 0.5 * cv_conf, 0.0, 1.0))
        # Not enough persistence yet — cap confidence so CONFIRMED needs history.
        return float(np.clip(0.5 * prom_conf, 0.0, 0.5))

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "region": self.region,
            "amplitude": round(self.amplitude, 3),
            "peak_hz": round(self.peak_hz, 2),
            "band_power_ratio": round(self.band_power_ratio, 3),
            "dot_age": int(self.age),
            "relatch_count": int(self.relatch_count),
            "state": self.state,
            "invalid_reason": self.invalid_reason,
            "score": round(self.score, 3),
        }


class DotFieldAnalyzer:
    """Manages all micro-oscillators and produces the global tremor vote."""

    def __init__(self) -> None:
        self._dots: dict[int, MicroOscillator] = {}

    # ------------------------------------------------------------------ #
    def update(self, ids, xs, ys, residuals_xy, qualities, timestamp, regions=None) -> None:
        """Route this frame's per-point residuals to their persistent dots.

        ids:           (N,) int array of stable dot identities.
        xs, ys:        (N,) current pixel positions.
        residuals_xy:  (N, 2) per-point residual (point_delta − macro_delta).
        qualities:     (N,) per-point tracking quality in [0, 1].
        regions:       optional (N,) region label per point.
        """
        if ids is None or residuals_xy is None or len(ids) == 0:
            return
        residuals_xy = np.asarray(residuals_xy, dtype=np.float64)
        for k in range(len(ids)):
            dot_id = int(ids[k])
            region = regions[k] if regions is not None else "hand"
            dot = self._dots.get(dot_id)
            if dot is None:
                dot = MicroOscillator(dot_id, region)
                self._dots[dot_id] = dot
            dot.update(
                residuals_xy[k, 0], residuals_xy[k, 1],
                float(qualities[k]) if qualities is not None else 1.0,
                timestamp, float(xs[k]), float(ys[k]), region,
            )

    def prune(self, now: float) -> None:
        """Drop dots not seen recently (they latched off / left the hand)."""
        stale = [d_id for d_id, d in self._dots.items() if now - d.last_seen > DOT_TIMEOUT_S]
        for d_id in stale:
            del self._dots[d_id]

    def state_of(self, dot_id: int) -> str:
        d = self._dots.get(int(dot_id))
        return d.state if d is not None else INVALID

    def amplitude_of(self, dot_id: int) -> float:
        """Last computed filtered residual amplitude for a dot (for the UI)."""
        d = self._dots.get(int(dot_id))
        return d.amplitude if d is not None else 0.0

    # ------------------------------------------------------------------ #
    def analyze(self, fs: float, tracking_quality: float = 1.0,
                min_valid_dots: int = MIN_VALID_DOTS) -> dict:
        """Classify every dot and produce the robust dense-field tremor vote."""
        # Two-pass for real-time behaviour with a large field: cheap gates on
        # every dot, then the expensive spectral pass on at most MAX_SPECTRAL_DOTS
        # of the highest raw-amplitude movers (the tremor-relevant ones).
        movers = [d for d in self._dots.values() if d._cheap_gate(fs)]
        if len(movers) > MAX_SPECTRAL_DOTS:
            movers.sort(key=lambda d: d._raw_amp, reverse=True)
            for d in movers[:MAX_SPECTRAL_DOTS]:
                d._spectral_analyze(fs)
            for d in movers[MAX_SPECTRAL_DOTS:]:
                d.mark_spectral_capped()
        else:
            for d in movers:
                d._spectral_analyze(fs)

        dots = list(self._dots.values())
        positive = [d for d in dots if d.state in _TREMOR_POSITIVE]
        valid = [d for d in dots if d.state not in (INVALID,)]
        warming = [d for d in dots if d.state == INVALID]
        confirmed = [d for d in dots if d.state == TREMOR_CONFIRMED]
        noise = [d for d in dots if d.state in (TRACKING_NOISE, RELATCHING)]
        relatching = [d for d in dots if d.state == RELATCHING]
        # Finer partitions + invalid-reason tally for the debug decomposition.
        stable = [d for d in dots if d.state == STABLE_NO_TREMOR]
        candidate_dots = [d for d in dots if d.state == TREMOR_CANDIDATE]
        tracking_noise_only = [d for d in dots if d.state == TRACKING_NOISE]
        invalid_reason_counts = dict(Counter(
            d.invalid_reason for d in dots if d.invalid_reason))

        # ── dense-field amplitude statistics (over VALID dots) ────────────────
        # median_dot_amplitude  : robust field baseline (low unless many shake)
        # topk_dot_amplitude    : robust localized tremor amplitude — the mean of
        #                         the strongest few valid dots, so a fingertip
        #                         tremor is captured without one dot dominating.
        valid_amps = np.array([d.amplitude for d in valid], dtype=np.float64)
        if valid_amps.size:
            median_dot_amplitude = float(np.median(valid_amps))
            k = max(TOPK_MIN, math.ceil(TOPK_FRACTION * valid_amps.size))
            k = min(k, valid_amps.size)
            topk_dot_amplitude = float(np.mean(np.sort(valid_amps)[::-1][:k]))
        else:
            median_dot_amplitude = 0.0
            topk_dot_amplitude = 0.0

        # Amplitude distribution — reveals whether the amplitude *scale* is the
        # thing crushing the score (e.g. residuals all < 0.2 px).
        if valid_amps.size:
            p75_dot_amplitude = float(np.percentile(valid_amps, 75))
            p90_dot_amplitude = float(np.percentile(valid_amps, 90))
            max_dot_amplitude = float(np.max(valid_amps))
        else:
            p75_dot_amplitude = p90_dot_amplitude = max_dot_amplitude = 0.0

        # Spectral medians over MOVING dots (those that ran the PSD path) so we
        # can see whether the rhythm thresholds (prominence / flatness / bpr)
        # are what is rejecting a real tremor.
        spectral_dots = positive + tracking_noise_only
        if spectral_dots:
            median_band_power_ratio = float(np.median([d.band_power_ratio for d in spectral_dots]))
            median_prominence = float(np.median([d.prominence for d in spectral_dots]))
            median_flatness = float(np.median([d.flatness for d in spectral_dots]))
        else:
            median_band_power_ratio = 0.0
            median_prominence = 0.0
            median_flatness = 1.0
        coherent: list[MicroOscillator] = []
        global_freq = 0.0
        if positive:
            freqs = np.array([d.peak_hz for d in positive], dtype=np.float64)
            median_freq = float(np.median(freqs))
            coherent = [d for d in positive if abs(d.peak_hz - median_freq) <= COHERENCE_HZ]
            global_freq = float(np.median([d.peak_hz for d in coherent])) if coherent else median_freq
        coherent_count = len(coherent)
        # Coherence factor: a single isolated positive dot scores 0 here, two
        # dots 0.5, three or more 1.0 — so one dot can never dominate the vote.
        coherence_factor = float(np.clip(
            (coherent_count - 1) / max(1, (COHERENT_TARGET - 1)), 0.0, 1.0))

        # ── coverage gate: require a minimum number of valid sensing dots ─────
        # A sparse field (few warmed-up dots) cannot be trusted; confidence
        # ramps from 0 to full as valid coverage reaches the coverage denom.
        coverage_factor = float(np.clip(len(valid) / float(max(1, COVERAGE_DENOM)), 0.0, 1.0))

        # ── robust aggregate: trimmed top-30% of positive dot scores ──────────
        # Trimming to the top fraction (≥2 dots once enough are positive) means
        # neither a single hot dot nor a long tail of weak dots sets the score.
        if positive:
            scores = sorted((d.score for d in positive), reverse=True)
            k = max(1, math.ceil(0.30 * len(scores)))
            if len(scores) >= 2:
                k = max(2, k)
            global_dot_score = float(np.mean(scores[:k]))
        else:
            global_dot_score = 0.0

        # Confidence is GATED by coherence, coverage and tracking quality — all
        # multiplicative, so any one being poor collapses the score.
        tremor_confidence = float(np.clip(
            global_dot_score
            * (0.25 + 0.75 * coherence_factor)
            * coverage_factor
            * float(np.clip(tracking_quality, 0.0, 1.0)),
            0.0, 1.0,
        ))
        global_tremor_score = int(round(100.0 * tremor_confidence))

        # Decomposition for the debug UI: EVIDENCE (raw dot oscillation) kept
        # separate from TRUST (coherence × coverage × tracking) so a low-trust
        # frame still SHOWS its evidence instead of collapsing to zero.
        tq_clip = float(np.clip(tracking_quality, 0.0, 1.0))
        trust_factor = float(np.clip(coherence_factor * coverage_factor * tq_clip, 0.0, 1.0))
        raw_evidence_score = int(round(100.0 * global_dot_score))
        ref = coherent or positive
        global_amplitude = float(np.median([d.amplitude for d in ref])) if ref else 0.0
        global_bpr = float(np.median([d.band_power_ratio for d in ref])) if ref else 0.0
        peak_stability = float(np.clip(
            0.5 * coherence_factor
            + 0.5 * (float(np.median([d.freq_confidence for d in positive])) if positive else 0.0),
            0.0, 1.0,
        ))

        # ── field-level validity note ─────────────────────────────────────────
        if len(valid) < min_valid_dots:
            field_validity = "SPARSE_FIELD"
        elif positive and coherent_count < 2:
            field_validity = "INCOHERENT"
        else:
            field_validity = "OK"

        # ── per-region aggregation ────────────────────────────────────────────
        region_tremor = self._aggregate_regions(positive)

        # ── compact per-dot list for the UI (top by score, then noise sample) ─
        top = sorted(dots, key=lambda d: -d.score)[:60]
        per_dot = [d.to_dict() for d in top]

        return {
            "global_tremor_score": global_tremor_score,        # 0..100 (pre-suppression)
            "global_dot_score": round(global_dot_score, 4),     # raw evidence in [0,1]
            "raw_evidence_score": raw_evidence_score,           # 100×global_dot_score
            "coherence_factor": round(coherence_factor, 3),
            "trust_factor": round(trust_factor, 4),             # coh×cov×tq in [0,1]
            "p75_dot_amplitude": round(p75_dot_amplitude, 4),
            "p90_dot_amplitude": round(p90_dot_amplitude, 4),
            "max_dot_amplitude": round(max_dot_amplitude, 4),
            "median_band_power_ratio": round(median_band_power_ratio, 4),
            "median_prominence": round(median_prominence, 4),
            "median_flatness": round(median_flatness, 4),
            "stable_dot_count": len(stable),
            "candidate_dot_count": len(candidate_dots),
            "tracking_noise_dot_count": len(tracking_noise_only),
            "invalid_reason_counts": invalid_reason_counts,
            "tremor_confidence": round(tremor_confidence, 4),
            "global_tremor_amplitude": round(global_amplitude, 4),
            "global_tremor_frequency_hz": round(global_freq, 2),
            "dominant_tremor_frequency_hz": round(global_freq, 2),
            "global_band_power_ratio": round(global_bpr, 4),
            "median_dot_amplitude": round(median_dot_amplitude, 4),
            "topk_dot_amplitude": round(topk_dot_amplitude, 4),
            "peak_stability": round(peak_stability, 4),
            "coverage_factor": round(coverage_factor, 3),
            "field_validity": field_validity,
            "tracking_quality": round(float(np.clip(tracking_quality, 0.0, 1.0)), 3),
            "valid_dot_count": len(valid),
            "warming_dot_count": len(warming),
            "positive_dot_count": len(positive),
            "coherent_dot_count": coherent_count,
            "tremor_confirmed_count": len(confirmed),
            "noise_dot_count": len(noise),
            "relatching_dot_count": len(relatching),
            "total_dot_count": len(dots),
            "min_valid_dots": int(min_valid_dots),
            "region_tremor": region_tremor,
            "per_dot": per_dot,
        }

    def _aggregate_regions(self, positive: list[MicroOscillator]) -> dict:
        regions: dict[str, list[MicroOscillator]] = {}
        for d in positive:
            regions.setdefault(d.region, []).append(d)
        out = {}
        for name, members in regions.items():
            amps = [m.amplitude for m in members]
            freqs = [m.peak_hz for m in members]
            scores = [m.score for m in members]
            out[name] = {
                "count": len(members),
                "amplitude": round(float(np.median(amps)), 3),
                "frequency_hz": round(float(np.median(freqs)), 2),
                "score": int(round(100.0 * float(np.median(scores)) * float(np.clip(len(members) / 2.0, 0.0, 1.0)))),
            }
        return out

    def debug(self) -> dict:
        return {"active_dots": len(self._dots)}
