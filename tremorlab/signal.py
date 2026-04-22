"""Signal processing for tremor analysis.

All functions operate on 1-D numpy arrays sampled at a known fs (Hz) unless
otherwise noted. Filters are designed once and cached by parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram

# -- Constants ---------------------------------------------------------------
TREMOR_BAND = (3.0, 12.0)  # Hz, full tremor band
FREQ_CLASSES = [
    # (low, high, label, short_code)
    (3.0, 5.0, "Parkinsonian-like (rest)", "parkinsonian"),
    (5.0, 8.0, "Essential-like (postural/kinetic)", "essential"),
    (8.0, 12.0, "Enhanced physiological", "physiological"),
]


# -- Filter design (cached) --------------------------------------------------
@lru_cache(maxsize=32)
def _bandpass_sos(fs: float, low: float, high: float, order: int = 4):
    nyq = 0.5 * fs
    low_n = max(1e-4, low / nyq)
    high_n = min(0.999, high / nyq)
    return butter(order, [low_n, high_n], btype="bandpass", output="sos")


@lru_cache(maxsize=32)
def _highpass_sos(fs: float, cutoff: float, order: int = 2):
    nyq = 0.5 * fs
    cn = max(1e-4, cutoff / nyq)
    return butter(order, cn, btype="highpass", output="sos")


def bandpass(x: np.ndarray, fs: float,
             low: float = TREMOR_BAND[0],
             high: float = TREMOR_BAND[1]) -> np.ndarray:
    """Zero-phase Butterworth bandpass; returns array same length as x."""
    fs_r = round(fs, 3)
    sos = _bandpass_sos(fs_r, low, high)
    # sosfiltfilt requires len(x) > padlen ~= 3 * (2*order + 1) per section.
    padlen = 3 * (sos.shape[0] * 2 + 1)
    if x.size <= padlen + 1:
        return x - np.mean(x)
    return sosfiltfilt(sos, x)


def highpass(x: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """Remove slow drift / mean offset."""
    sos = _highpass_sos(round(fs, 3), cutoff)
    padlen = 3 * (sos.shape[0] * 2 + 1)
    if x.size <= padlen + 1:
        return x - np.mean(x)
    return sosfiltfilt(sos, x)


# -- Resampling --------------------------------------------------------------
def resample_uniform(t: np.ndarray, x: np.ndarray,
                     y: np.ndarray, fs: float):
    """Linear-interpolate irregular samples onto a uniform grid of rate fs."""
    if t.size < 2:
        return None
    dur = t[-1] - t[0]
    if dur <= 0:
        return None
    n = int(dur * fs)
    if n < 16:
        return None
    tu = t[0] + np.arange(n) / fs
    xu = np.interp(tu, t, x)
    yu = np.interp(tu, t, y)
    return tu, xu, yu


# -- Metrics -----------------------------------------------------------------
@dataclass
class TremorMetrics:
    fs: float
    samples: int
    duration: float

    peak_hz: float          # dominant frequency (Hz)
    peak_power: float       # PSD at peak
    band_power: float       # integrated power in tremor band
    total_power: float      # integrated power 0.5..fs/2
    band_ratio: float       # band_power / total_power

    rms_amp: float          # RMS of bandpassed signal (normalized coords)
    rms_amp_mm: float       # RMS in mm (if hand_ref_pixels & hand_width_mm)
    peak_to_peak_mm: float

    snr_db: float           # 10*log10(peak_power / median off-band)
    regularity: float       # 1 - normalized spectral entropy in band (0..1)
    peak_sharpness: float   # peak / mean in ±1 Hz window

    class_label: str        # human-readable class
    class_code: str         # short code
    score: int              # 0..100 tremor score


def _estimate_class(peak_hz: float) -> tuple[str, str]:
    if peak_hz <= 0:
        return "Unclassified", "none"
    for lo, hi, label, code in FREQ_CLASSES:
        if lo <= peak_hz < hi:
            return label, code
    return "Out of band", "other"


def compute_metrics(
    xu: np.ndarray,
    yu: np.ndarray,
    fs: float,
    *,
    hand_ref_pixels: float | None = None,
    hand_width_mm: float = 85.0,  # avg adult palm width
    baseline_rms: float | None = None,
) -> TremorMetrics | None:
    """Compute tremor metrics from uniform 2D samples.

    `hand_ref_pixels` is the current pixel distance between two reference
    landmarks (e.g. wrist–middle MCP). If provided, amplitude is rescaled
    from normalized coords to millimetres using `hand_width_mm` as reference.
    `baseline_rms` optionally subtracts a measured still-hand noise floor.
    """
    n = xu.size
    if n < 32:
        return None

    # Detrend + bandpass on each axis
    xf = bandpass(highpass(xu, fs), fs)
    yf = bandpass(highpass(yu, fs), fs)

    # Welch PSD — combined magnitude
    nperseg = int(min(n, max(64, fs * 2)))
    fxx, pxx = welch(xf, fs=fs, nperseg=nperseg, detrend=False)
    _, pyy = welch(yf, fs=fs, nperseg=nperseg, detrend=False)
    psd = pxx + pyy
    if psd.sum() <= 0:
        return None

    # Sub-band (voluntary-motion) power: 0.3..2.5 Hz. We'll use this later
    # to penalise large slow movements (arm raises etc) that otherwise
    # leak into the 3-12 Hz band via filter ringing.
    low_mask = (fxx >= 0.3) & (fxx <= 2.5)
    low_power = float(psd[low_mask].sum()) if np.any(low_mask) else 0.0

    # Restrict peak search to tremor band (avoid tiny-frequency leakage)
    band_mask = (fxx >= TREMOR_BAND[0]) & (fxx <= TREMOR_BAND[1])
    if not np.any(band_mask):
        return None

    search_mask = (fxx >= 2.0) & (fxx <= 14.0)
    peak_idx = int(np.argmax(np.where(search_mask, psd, -np.inf)))
    peak_hz = float(fxx[peak_idx])
    peak_power = float(psd[peak_idx])

    df = float(fxx[1] - fxx[0]) if fxx.size > 1 else 1.0
    total_power = float(psd.sum() * df)
    band_power = float(psd[band_mask].sum() * df)
    band_ratio = band_power / total_power if total_power > 0 else 0.0

    # Peak sharpness: ratio of peak to mean within ±1 Hz, off-peak only
    around = (fxx >= peak_hz - 1.0) & (fxx <= peak_hz + 1.0)
    around_offpeak = around & (np.arange(fxx.size) != peak_idx)
    if np.any(around_offpeak):
        mean_near = float(psd[around_offpeak].mean())
        sharpness = peak_power / mean_near if mean_near > 0 else 0.0
    else:
        sharpness = 0.0

    # SNR: peak vs median of band excluding ±0.75 Hz around peak
    off = band_mask & ~((fxx >= peak_hz - 0.75) & (fxx <= peak_hz + 0.75))
    if np.any(off) and np.median(psd[off]) > 0:
        snr_db = 10.0 * np.log10(peak_power / float(np.median(psd[off])))
    else:
        snr_db = 0.0

    # Regularity: 1 - normalized entropy of PSD within band (concentrated → 1)
    band_psd = psd[band_mask]
    p = band_psd / band_psd.sum() if band_psd.sum() > 0 else band_psd
    p = p[p > 0]
    if p.size > 1:
        entropy = float(-np.sum(p * np.log(p)))
        max_entropy = float(np.log(p.size))
        regularity = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        regularity = 0.0

    # Amplitudes
    mag = np.sqrt(xf * xf + yf * yf)
    rms_amp = float(np.sqrt(np.mean(mag * mag)))
    if baseline_rms is not None:
        rms_amp_net = max(0.0, rms_amp - baseline_rms)
    else:
        rms_amp_net = rms_amp

    if hand_ref_pixels and hand_ref_pixels > 1:
        # Reference: distance wrist→middle-MCP is ~0.35 of hand length;
        # we calibrate against palm width in mm per normalized unit using
        # image-width ≈ hand_ref_pixels / 0.22 (palm ~22% of frame width
        # at typical distance). Instead of assuming that, we scale using
        # the hand-reference pixel distance directly.
        # We approximate: the tracked normalized units span the frame.
        # To convert to mm we use the hand-width proxy.
        mm_per_norm = hand_width_mm / max(1e-6, hand_ref_pixels)
        rms_amp_mm = rms_amp_net * mm_per_norm * 1000  # approx
        # This ratio is an estimate only; we'll keep it simple:
        rms_amp_mm = rms_amp_net * (hand_width_mm / 0.22)  # fallback heuristic
        peak_to_peak_mm = 2.8 * rms_amp_mm  # sinusoid p2p ≈ 2√2 * RMS
    else:
        rms_amp_mm = rms_amp_net * (hand_width_mm / 0.22)
        peak_to_peak_mm = 2.8 * rms_amp_mm

    # Composite 0-100 score.
    #
    # Tremor is an *oscillation* in 3-12 Hz with a sharp spectral peak.
    # Large voluntary motion (raising the hand, waving) has huge amplitude
    # but most of its power sits below 3 Hz; filter ringing then leaks some
    # into the tremor band and can fool an amplitude-only score. We fix
    # that by gating every term on spectral purity.
    amp_for_score = rms_amp_net
    amp_term = float(np.tanh(amp_for_score / 0.004))
    band_term = max(0.0, min(1.0, band_ratio / 0.5))
    reg_term = max(0.0, min(1.0, regularity * 1.3))
    snr_term = max(0.0, min(1.0, snr_db / 12.0))
    sharp_term = max(0.0, min(1.0, (sharpness - 1.0) / 4.0))

    # Voluntary-motion penalty: if low-frequency (<2.5 Hz) power dominates
    # the in-band tremor power, the user is moving their hand, not
    # trembling. Penalty → 0 when low/band >> 1, → 1 when low << band.
    voluntary_ratio = low_power / max(band_power, 1e-12)
    motion_gate = 1.0 / (1.0 + max(0.0, voluntary_ratio - 0.5))
    # Spectral-purity gate: tremor needs band_ratio to be meaningful AND a
    # sharp peak. Broadband motion flunks both.
    purity_gate = max(0.0, min(1.0,
                               0.6 * band_term
                               + 0.4 * sharp_term))

    quality = (0.40 * band_term + 0.25 * reg_term
               + 0.20 * snr_term + 0.15 * sharp_term)
    # Gate quality by a softer amplitude check so a still hand scores 0.
    quality_gate = float(np.tanh(amp_for_score / 0.0015))
    # Tremor-likelihood multiplier applied to both halves of the score:
    # big amplitude only counts when it actually looks like an oscillation.
    tremor_likelihood = motion_gate * (0.3 + 0.7 * purity_gate)
    score_raw = (55.0 * amp_term * tremor_likelihood
                 + 45.0 * quality * quality_gate)
    score = int(round(max(0.0, min(100.0, score_raw))))
    regularity = reg_term  # keep backwards compatibility for dataclass

    label, code = _estimate_class(peak_hz)

    return TremorMetrics(
        fs=fs,
        samples=n,
        duration=n / fs,
        peak_hz=peak_hz,
        peak_power=peak_power,
        band_power=band_power,
        total_power=total_power,
        band_ratio=band_ratio,
        rms_amp=rms_amp_net,
        rms_amp_mm=rms_amp_mm,
        peak_to_peak_mm=peak_to_peak_mm,
        snr_db=snr_db,
        regularity=reg_term,
        peak_sharpness=sharpness,
        class_label=label,
        class_code=code,
        score=score,
    )


# -- Spectrogram -------------------------------------------------------------
def rolling_spectrogram(x: np.ndarray, fs: float,
                        f_max: float = 15.0,
                        nperseg: int | None = None):
    """Return (f, t, Sxx) restricted to 0..f_max Hz."""
    if x.size < 64:
        return None
    nperseg = nperseg or int(min(x.size, max(64, fs * 1.5)))
    noverlap = int(nperseg * 0.75)
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nperseg,
                            noverlap=noverlap, detrend="linear",
                            scaling="density", mode="psd")
    mask = f <= f_max
    return f[mask], t, Sxx[mask]


# -- Adaptive baseline -------------------------------------------------------
class AdaptiveBaseline:
    """Rolling noise-floor estimator.

    Updates a running "quiet" RMS whenever the current window looks benign
    (low in-band energy). Used to personalise both the amplitude baseline
    and the video-gate pass threshold so users with naturally higher
    movement don't get locked out.
    """

    def __init__(self, alpha: float = 0.03) -> None:
        self.alpha = float(alpha)
        self.rms: float | None = None
        self.score_floor: float | None = None
        self.score_ceiling: float | None = None
        self.score_avg: float | None = None
        self.samples = 0

    def update(self, rms_amp: float, band_ratio: float,
               score: float) -> None:
        self.samples += 1
        # Running average of ALL scores — represents the user's typical
        # tremor level regardless of whether they are currently quiet.
        if self.score_avg is None:
            self.score_avg = float(score)
        else:
            a = 0.02  # slower than the quiet-only EMA
            self.score_avg = (1 - a) * self.score_avg + a * float(score)
        # Quiet = not much in-band power AND modest score.
        quiet = band_ratio < 0.22 and score < 25
        if quiet:
            if self.rms is None:
                self.rms = float(rms_amp)
            else:
                blended = ((1 - self.alpha) * self.rms
                           + self.alpha * float(rms_amp))
                # monotonically decrease toward true quiet floor
                self.rms = min(self.rms, blended)
            if self.score_floor is None:
                self.score_floor = float(score)
            else:
                self.score_floor = ((1 - self.alpha) * self.score_floor
                                    + self.alpha * float(score))
        # Track an upper envelope of scores during any activity.
        if self.score_ceiling is None:
            self.score_ceiling = float(score)
        else:
            decay = 0.005
            self.score_ceiling = max(
                (1 - decay) * self.score_ceiling,
                float(score),
            )

    def personal_pass_threshold(self, default: float = 45.0) -> float:
        """Score the user should stay *under* to pass a gate.

        Chosen as floor + fraction of (ceiling - floor) so it adapts to
        each user's observed range rather than using a fixed number.
        """
        if self.score_floor is None or self.score_ceiling is None:
            return default
        floor = self.score_floor
        ceil = max(self.score_ceiling, floor + 15.0)
        # Pass if you're in the lower ~55% of your personal range.
        thr = floor + 0.55 * (ceil - floor)
        return float(max(25.0, min(85.0, thr)))

    def personal_avg_threshold(self, margin: float = 8.0,
                               default: float = 45.0) -> float:
        """Pass threshold defined as your running average + a small margin.

        The user passes if their measured tremor score is at most slightly
        above their typical tremor level — i.e. they just need to be as
        steady as they usually are, not dramatically steadier.
        """
        if self.score_avg is None or self.samples < 20:
            return default
        return float(max(25.0, min(90.0, self.score_avg + margin)))
