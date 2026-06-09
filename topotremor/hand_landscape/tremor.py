"""
hand_landscape/tremor.py

Rhythmic-residual tremor analysis — the ONLY legitimate tremor source.

Hard rule
---------
Tremor is NOT a single-frame error. Tremor is *time-coherent rhythmic motion*
in the 3–12 Hz band that survives macro-motion removal. A tracking glitch, a
one-frame point jump, or a non-periodic burst must NEVER score as tremor.

A serious tremor score therefore requires ALL of:

  1. Sustained band power           – energy concentrated in 3–12 Hz, not broadband.
  2. Stable peak frequency          – the dominant frequency holds across windows.
  3. Sufficient valid duration      – at least ~2 s of trustworthy residual.
  4. Low spike energy               – not dominated by isolated outliers.
  5. Good tracking quality          – gated by the TrackingQualityMonitor.

`run_tremor_analysis` is a stateless single-window estimate (used by tests).
`TremorAnalyzer` adds cross-window peak-frequency stability and is what the
live bridge uses.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, welch  # type: ignore
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from ripser import ripser as _ripser  # type: ignore
    _RIPSER = True
except ImportError:
    _RIPSER = False


# ─── constants ────────────────────────────────────────────────────────────────
TREMOR_LO_HZ = 3.0
TREMOR_HI_HZ = 12.0
WELCH_NPERSEG_FACTOR = 4

MIN_TREMOR_SECONDS  = 2.0    # below this no tremor score is trusted
FULL_TREMOR_SECONDS = 4.0    # duration at which the duration factor saturates
POWER_RATIO_FLOOR   = 0.20   # band/total power must exceed this to count
POWER_RATIO_FULL    = 0.55   # ratio at which the rhythm factor saturates
PEAK_CV_MAX         = 0.25   # peak-frequency coefficient-of-variation ceiling
SPIKE_MAD_K         = 4.0    # |x-med| > K*MAD is treated as a spike


# ─── bandpass ─────────────────────────────────────────────────────────────────

def _butter_bandpass(signal: np.ndarray, fs: float,
                     lo: float = TREMOR_LO_HZ, hi: float = TREMOR_HI_HZ) -> np.ndarray:
    if len(signal) < 8:
        return signal.copy()
    if _SCIPY:
        nyq = fs / 2.0
        low = max(0.01, lo / nyq)
        high = min(0.99, hi / nyq)
        if low >= high:
            return signal.copy()
        sos = butter(4, [low, high], btype="bandpass", output="sos")
        try:
            return sosfiltfilt(sos, signal)
        except Exception:
            return signal.copy()
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    return np.fft.irfft(fft * mask, n=len(signal))


# ─── spike rejection ──────────────────────────────────────────────────────────

def _spike_energy_fraction(signal: np.ndarray) -> float:
    """Fraction of total signal energy contributed by isolated outliers.

    A single-frame jump dumps a lot of energy into one sample. We measure how
    much of the total energy comes from samples that are MAD-outliers; a high
    fraction means the "signal" is really spikes, not rhythm.
    """
    if len(signal) < 4:
        return 0.0
    med = float(np.median(signal))
    dev = np.abs(signal - med)
    mad = float(np.median(dev)) * 1.4826
    if mad < 1e-9:
        return 0.0
    threshold = SPIKE_MAD_K * mad
    spike_mask = dev > threshold
    total_energy = float(np.sum((signal - med) ** 2)) + 1e-12
    spike_energy = float(np.sum((signal[spike_mask] - med) ** 2))
    return min(1.0, spike_energy / total_energy)


# ─── spectral core ────────────────────────────────────────────────────────────

def _spectral_analysis(sig: np.ndarray, fs: float) -> dict:
    """Single-window spectral features. No gating, no confidence."""
    sig = sig - np.mean(sig)
    filtered = _butter_bandpass(sig, fs)
    band_rms = float(np.std(filtered))

    peak_hz = 0.0
    power_ratio = 0.0
    band_power = 0.0
    if _SCIPY and len(sig) >= 16:
        try:
            nperseg = min(len(sig), max(16, int(fs * WELCH_NPERSEG_FACTOR)))
            freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
            band_mask = (freqs >= TREMOR_LO_HZ) & (freqs <= TREMOR_HI_HZ)
            band_power = float(np.sum(psd[band_mask]))
            total_power = float(np.sum(psd)) + 1e-12
            power_ratio = min(1.0, band_power / total_power)
            if band_mask.any():
                peak_idx = int(np.argmax(psd[band_mask]))
                peak_hz = float(freqs[band_mask][peak_idx])
        except Exception:
            pass

    h1_lifetime = 0.0
    if _RIPSER and band_rms > 0.1 and len(filtered) >= 20:
        try:
            delay = max(1, int(round(fs / (2.0 * max(peak_hz, 5.0)))))
            delay = min(delay, len(filtered) // 4)
            if delay > 0:
                embedded = np.stack([filtered[:-delay], filtered[delay:]], axis=1)
            else:
                embedded = filtered.reshape(-1, 1)
            diagrams = _ripser(embedded, maxdim=1)["dgms"]
            if len(diagrams) > 1 and len(diagrams[1]) > 0:
                lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                if len(lifetimes):
                    h1_lifetime = float(np.max(lifetimes))
        except Exception:
            h1_lifetime = 0.0

    return {
        "filtered": filtered,
        "band_rms": band_rms,
        "peak_hz": peak_hz,
        "power_ratio": power_ratio,
        "band_power": band_power,
        "h1_lifetime": h1_lifetime,
        "embedding_delay": max(1, int(round(fs / (2.0 * max(peak_hz, 5.0))))),
    }


# ─── rhythm / confidence ──────────────────────────────────────────────────────

def _rhythm_factors(spec: dict, signal: np.ndarray, span_seconds: float,
                    peak_stability: float) -> dict:
    """Combine spectral features into a rhythm confidence in [0, 1]."""
    power_ratio = spec["power_ratio"]
    rhythm_power = np.clip(
        (power_ratio - POWER_RATIO_FLOOR) / (POWER_RATIO_FULL - POWER_RATIO_FLOOR),
        0.0, 1.0,
    )
    spike_fraction = _spike_energy_fraction(signal)
    duration_factor = float(np.clip(
        (span_seconds - MIN_TREMOR_SECONDS) / (FULL_TREMOR_SECONDS - MIN_TREMOR_SECONDS),
        0.0, 1.0,
    ))
    # Topological cyclicity nudges confidence up when a clean loop exists.
    h1 = spec["h1_lifetime"]
    topo_factor = float(np.clip(0.6 + h1, 0.6, 1.0)) if h1 > 0 else 0.6

    rhythm = float(
        rhythm_power * peak_stability * (1.0 - spike_fraction) * duration_factor * topo_factor
    )
    return {
        "rhythm": float(np.clip(rhythm, 0.0, 1.0)),
        "rhythm_power": float(rhythm_power),
        "spike_fraction": float(spike_fraction),
        "duration_factor": duration_factor,
        "peak_stability": float(peak_stability),
    }


def _assemble_result(spec: dict, factors: dict, signal: np.ndarray,
                     tracking_quality: float, suppress: bool) -> dict:
    band_rms = spec["band_rms"]
    amplitude_score = float(np.tanh(band_rms / 1.5)) * 100.0

    tremor_confidence = float(np.clip(factors["rhythm"] * tracking_quality, 0.0, 1.0))
    if suppress:
        tremor_confidence *= 0.0  # FLOW_LAG / RELATCHING → untrusted, zero it out

    tremor_score = int(round(np.clip(amplitude_score * tremor_confidence, 0.0, 100.0)))
    tremor_energy = (amplitude_score / 100.0) * tremor_confidence

    filtered = spec["filtered"]
    return {
        "filtered": [round(float(v), 4) for v in filtered[-200:]],
        "band_rms": round(band_rms, 4),
        "residual_rms": round(band_rms, 4),
        "peak_hz": round(spec["peak_hz"], 2),
        "residual_peak_frequency_hz": round(spec["peak_hz"], 2),
        "tremor_power_ratio": round(spec["power_ratio"], 4),
        "residual_band_power": round(spec["band_power"], 6),
        "h1_lifetime": round(spec["h1_lifetime"], 4),
        "peak_stability": round(factors["peak_stability"], 4),
        "spike_fraction": round(factors["spike_fraction"], 4),
        "duration_factor": round(factors["duration_factor"], 4),
        "tremor_confidence": round(tremor_confidence, 4),
        "tremor_score": tremor_score,
        "tremor_energy": round(float(tremor_energy), 4),
        "embedding_delay": spec["embedding_delay"],
    }


# ─── stateless single-window API (used by tests) ──────────────────────────────

def run_tremor_analysis(
    residual_x_buf: list[float] | np.ndarray,
    fs: float,
    landscape_confidence: float = 1.0,
    span_seconds: float | None = None,
    tracking_quality: float = 1.0,
    suppress: bool = False,
) -> dict:
    """Stateless tremor analysis of one residual window.

    Peak-frequency stability cannot be measured from a single window, so a
    neutral value (0.6) is used here. The live bridge uses TremorAnalyzer for
    true cross-window stability.
    """
    sig = np.asarray(residual_x_buf, dtype=np.float64)
    if len(sig) < 8 or fs < 8.0:
        return _empty_result()

    if span_seconds is None:
        span_seconds = len(sig) / max(fs, 1e-6)

    spec = _spectral_analysis(sig, fs)
    centered = sig - np.mean(sig)
    factors = _rhythm_factors(spec, centered, span_seconds, peak_stability=0.6)
    gate = min(max(tracking_quality, 0.0), 1.0) * min(max(landscape_confidence, 0.0), 1.0)
    return _assemble_result(spec, factors, centered, gate, suppress)


def _empty_result() -> dict:
    return {
        "filtered": [],
        "band_rms": 0.0,
        "residual_rms": 0.0,
        "peak_hz": 0.0,
        "residual_peak_frequency_hz": 0.0,
        "tremor_power_ratio": 0.0,
        "residual_band_power": 0.0,
        "h1_lifetime": 0.0,
        "peak_stability": 0.0,
        "spike_fraction": 0.0,
        "duration_factor": 0.0,
        "tremor_confidence": 0.0,
        "tremor_score": 0,
        "tremor_energy": 0.0,
        "embedding_delay": 1,
    }


# ─── stateful cross-window analyzer (used by the live bridge) ─────────────────

class TremorAnalyzer:
    """Adds cross-window peak-frequency stability to the stateless core."""

    def __init__(self, peak_history: int = 12) -> None:
        self._peak_hist: deque = deque(maxlen=peak_history)

    def _peak_stability(self, latest_peak: float) -> float:
        """Coefficient-of-variation-based stability of recent peak frequencies."""
        if latest_peak > 0:
            self._peak_hist.append(latest_peak)
        peaks = np.array([p for p in self._peak_hist if p > 0], dtype=np.float64)
        if len(peaks) < 3:
            return 0.3  # not enough history yet → low-ish, cannot trust rhythm
        mean = float(np.mean(peaks))
        if mean < 1e-6:
            return 0.0
        cv = float(np.std(peaks)) / mean
        return float(np.clip(1.0 - cv / PEAK_CV_MAX, 0.0, 1.0))

    def analyze(
        self,
        residual_x_buf: list[float] | np.ndarray,
        fs: float,
        span_seconds: float,
        tracking_quality: float,
        suppress: bool,
    ) -> dict:
        sig = np.asarray(residual_x_buf, dtype=np.float64)
        if len(sig) < 8 or fs < 8.0:
            return _empty_result()
        sig = sig - np.mean(sig)
        spec = _spectral_analysis(sig, fs)
        stability = self._peak_stability(spec["peak_hz"])
        factors = _rhythm_factors(spec, sig, span_seconds, peak_stability=stability)
        gate = float(np.clip(tracking_quality, 0.0, 1.0))
        return _assemble_result(spec, factors, sig, gate, suppress)
