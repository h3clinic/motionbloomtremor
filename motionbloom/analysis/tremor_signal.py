"""Explainable tremor-band analysis for ROI-local motion signals.

This module intentionally accepts generic x/y motion time series. It does not
know about MediaPipe, landmarks, UI state, or camera capture. MediaPipe may
locate a hand ROI elsewhere, but the tremor measurement here is signal
processing over ROI-local optical-flow motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt, welch


class TremorBand(StrEnum):
    PARKINSONIAN = "parkinsonian"
    BROAD = "broad"


@dataclass(frozen=True)
class TremorAnalysisConfig:
    primary_band: tuple[float, float] = (3.0, 7.0)
    broad_band: tuple[float, float] = (4.0, 12.0)
    gross_band: tuple[float, float] = (0.2, 2.0)
    default_window_sec: float = 4.0
    stable_analysis_window_sec: float = 6.0
    min_duration_sec: float = 3.0
    min_fps: float = 24.0
    min_cycles_required: float = 12.0
    likely_snr: float = 3.0
    possible_snr: float = 1.8
    max_likely_peak_width_hz: float = 1.25
    max_possible_peak_width_hz: float = 2.0
    xy_frequency_tolerance_hz: float = 0.75
    min_band_power: float = 1e-7
    max_gross_to_tremor_ratio: float = 6.0
    nperseg_max: int = 256


@dataclass
class TremorAnalysisResult:
    label: str
    peak_frequency_hz: float
    snr: float
    peak_width_hz: float
    band_power: float
    gross_power: float
    cycles: float
    xy_agreement: bool
    valid_points: int
    track_survival_rate: float
    fps: float
    window_sec: float
    reasons: list[str] = field(default_factory=list)
    x_peak_frequency_hz: float = 0.0
    y_peak_frequency_hz: float = 0.0
    x_snr: float = 0.0
    y_snr: float = 0.0
    total_power: float = 0.0
    spectral_entropy: float = 1.0
    feature_group_consistency: bool = True

    def as_diagnostic_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "peak_frequency_hz": round(self.peak_frequency_hz, 3),
            "snr": round(self.snr, 3),
            "peak_width_hz": round(self.peak_width_hz, 3),
            "band_power": float(self.band_power),
            "gross_power": float(self.gross_power),
            "cycles": round(self.cycles, 3),
            "xy_agreement": self.xy_agreement,
            "valid_points": self.valid_points,
            "track_survival_rate": round(self.track_survival_rate, 3),
            "fps": round(self.fps, 3),
            "window_sec": round(self.window_sec, 3),
            "reasons": list(self.reasons),
            "x_peak_frequency_hz": round(self.x_peak_frequency_hz, 3),
            "y_peak_frequency_hz": round(self.y_peak_frequency_hz, 3),
            "x_snr": round(self.x_snr, 3),
            "y_snr": round(self.y_snr, 3),
            "total_power": float(self.total_power),
            "spectral_entropy": round(self.spectral_entropy, 3),
            "feature_group_consistency": self.feature_group_consistency,
        }


def estimate_fps_and_duration(timestamps: np.ndarray) -> tuple[float, float]:
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size < 2:
        return 0.0, 0.0
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return 0.0, 0.0
    return float((t.size - 1) / duration), duration


def _uniform_resample(timestamps: np.ndarray, values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.size < 2 or fs <= 0:
        return t, v
    duration = float(t[-1] - t[0])
    n = max(2, int(round(duration * fs)) + 1)
    tu = t[0] + np.arange(n, dtype=np.float64) / fs
    tu = tu[tu <= t[-1] + 1e-9]
    vu = np.interp(tu, t, v)
    return tu, vu


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    return float(np.sum(psd[mask]) * df)


def _highpass(values: np.ndarray, fs: float, cutoff: float = 0.2) -> np.ndarray:
    if values.size < 12 or cutoff <= 0 or fs <= cutoff * 2:
        return detrend(values, type="linear")
    sos = butter(2, cutoff / (0.5 * fs), btype="highpass", output="sos")
    return sosfiltfilt(sos, detrend(values, type="linear"))


def _lowpass(values: np.ndarray, fs: float, cutoff: float = 2.0) -> np.ndarray:
    if values.size < 12 or cutoff <= 0 or fs <= cutoff * 2:
        return values - detrend(values, type="linear")
    sos = butter(2, min(0.99, cutoff / (0.5 * fs)), btype="lowpass", output="sos")
    return sosfiltfilt(sos, values)


def _axis_features(
    values: np.ndarray,
    fs: float,
    band: tuple[float, float],
    config: TremorAnalysisConfig,
) -> dict[str, float | np.ndarray]:
    conditioned = _highpass(values, fs, cutoff=0.2)
    nperseg = int(min(config.nperseg_max, max(32, conditioned.size)))
    freqs, psd = welch(conditioned, fs=fs, nperseg=nperseg, detrend=False)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if psd.size == 0 or not np.any(band_mask) or float(np.sum(psd)) <= 0:
        return {
            "peak_hz": 0.0,
            "peak_power": 0.0,
            "snr": 0.0,
            "peak_width_hz": 0.0,
            "band_power": 0.0,
            "total_power": 0.0,
            "gross_power": 0.0,
            "entropy": 1.0,
            "freqs": freqs,
            "psd": psd,
        }

    peak_idx = int(np.argmax(np.where(band_mask, psd, -np.inf)))
    peak_hz = float(freqs[peak_idx])
    peak_power = float(psd[peak_idx])
    off_peak = band_mask & ~((freqs >= peak_hz - 0.75) & (freqs <= peak_hz + 0.75))
    noise_floor = float(np.median(psd[off_peak])) if np.any(off_peak) else float(np.median(psd[band_mask]))
    snr = peak_power / max(noise_floor, 1e-12)

    half_power = peak_power * 0.5
    above_half = band_mask & (psd >= half_power)
    if np.any(above_half):
        peak_width_hz = float(freqs[above_half][-1] - freqs[above_half][0])
    else:
        peak_width_hz = 0.0

    band_power = _band_power(freqs, psd, band)
    total_power = float(np.sum(psd) * (freqs[1] - freqs[0])) if freqs.size > 1 else float(np.sum(psd))
    gross_power = _band_power(freqs, psd, config.gross_band)

    band_psd = psd[band_mask]
    probs = band_psd / max(float(np.sum(band_psd)), 1e-12)
    probs = probs[probs > 0]
    if probs.size > 1:
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(probs.size))
    else:
        entropy = 1.0

    return {
        "peak_hz": peak_hz,
        "peak_power": peak_power,
        "snr": float(snr),
        "peak_width_hz": peak_width_hz,
        "band_power": band_power,
        "total_power": total_power,
        "gross_power": gross_power,
        "entropy": entropy,
        "freqs": freqs,
        "psd": psd,
    }


def analyze_tremor_motion(
    timestamps: np.ndarray,
    motion_x: np.ndarray,
    motion_y: np.ndarray,
    *,
    config: TremorAnalysisConfig | None = None,
    band: TremorBand | str = TremorBand.PARKINSONIAN,
    valid_points: int = 0,
    track_survival_rate: float = 1.0,
    roi_visibility: float = 1.0,
    feature_group_consistency: bool = True,
    background_motion_x: np.ndarray | None = None,
    background_motion_y: np.ndarray | None = None,
) -> TremorAnalysisResult:
    """Analyze ROI-local optical-flow motion and return explainable diagnostics."""
    config = config or TremorAnalysisConfig()
    label_unusable = "Unusable recording"
    label_unreliable = "No reliable tremor signal"
    reasons: list[str] = []

    timestamps = np.asarray(timestamps, dtype=np.float64)
    x = np.asarray(motion_x, dtype=np.float64)
    y = np.asarray(motion_y, dtype=np.float64)
    if background_motion_x is not None:
        x = x - np.asarray(background_motion_x, dtype=np.float64)
        reasons.append("background/global motion removed from x signal")
    if background_motion_y is not None:
        y = y - np.asarray(background_motion_y, dtype=np.float64)
        reasons.append("background/global motion removed from y signal")

    fps, duration = estimate_fps_and_duration(timestamps)
    if timestamps.size < 4 or x.size != timestamps.size or y.size != timestamps.size:
        return TremorAnalysisResult(label_unusable, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, valid_points, track_survival_rate, fps, duration, ["insufficient or mismatched samples"])
    if fps < config.min_fps:
        return TremorAnalysisResult(label_unusable, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, valid_points, track_survival_rate, fps, duration, [f"fps {fps:.1f} below {config.min_fps:.1f}"])
    if duration < config.min_duration_sec:
        return TremorAnalysisResult(label_unusable, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, valid_points, track_survival_rate, fps, duration, [f"window {duration:.1f}s shorter than {config.min_duration_sec:.1f}s"])
    if roi_visibility < 0.4:
        return TremorAnalysisResult(label_unusable, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, valid_points, track_survival_rate, fps, duration, ["hand ROI visibility too low"])

    band_tuple = config.primary_band if str(band) == str(TremorBand.PARKINSONIAN) else config.broad_band
    tu, xu = _uniform_resample(timestamps, x, fps)
    _, yu = _uniform_resample(timestamps, y, fps)

    x_features = _axis_features(xu, fps, band_tuple, config)
    y_features = _axis_features(yu, fps, band_tuple, config)
    x_score = float(x_features["band_power"]) * max(float(x_features["snr"]), 0.0)
    y_score = float(y_features["band_power"]) * max(float(y_features["snr"]), 0.0)
    selected = x_features if x_score >= y_score else y_features

    x_peak = float(x_features["peak_hz"])
    y_peak = float(y_features["peak_hz"])
    peak_frequency = float(selected["peak_hz"])
    snr = float(selected["snr"])
    peak_width = float(selected["peak_width_hz"])
    band_power = max(float(x_features["band_power"]), float(y_features["band_power"]))
    total_power = float(x_features["total_power"]) + float(y_features["total_power"])

    gross_x = _lowpass(xu, fps, cutoff=config.gross_band[1])
    gross_y = _lowpass(yu, fps, cutoff=config.gross_band[1])
    gross_power = float(np.var(gross_x) + np.var(gross_y))
    cycles = peak_frequency * duration
    xy_agreement = (
        x_peak > 0
        and y_peak > 0
        and abs(x_peak - y_peak) <= config.xy_frequency_tolerance_hz
    )

    if cycles >= config.min_cycles_required:
        reasons.append("enough tremor-band cycles in window")
    else:
        reasons.append(f"only {cycles:.1f} cycles; confidence downgraded")
    if snr >= config.likely_snr:
        reasons.append("tremor-band peak passed SNR threshold")
    elif snr >= config.possible_snr:
        reasons.append("tremor-band peak has borderline SNR")
    else:
        reasons.append("tremor-band SNR too low")
    if peak_width <= config.max_likely_peak_width_hz:
        reasons.append("spectral peak is narrow")
    elif peak_width <= config.max_possible_peak_width_hz:
        reasons.append("spectral peak is moderately broad")
    else:
        reasons.append("spectral peak too broad")
    if xy_agreement:
        reasons.append("x/y peak frequencies agree")
    else:
        reasons.append("x/y peak frequencies do not agree")
    if gross_power <= config.max_gross_to_tremor_ratio * max(band_power, 1e-12):
        reasons.append("gross motion below rejection threshold")
    else:
        reasons.append("gross motion dominates tremor-band power")
    if feature_group_consistency:
        reasons.append("feature groups are consistent")
    else:
        reasons.append("feature groups are inconsistent")

    if track_survival_rate < 0.35 or valid_points < 6:
        label = label_unreliable
        reasons.append("too few stable optical-flow tracks")
    elif band_power < config.min_band_power:
        label = "No tremor detected"
        reasons.append("tremor-band power near noise floor")
    elif gross_power > config.max_gross_to_tremor_ratio * max(band_power, 1e-12):
        label = "No reliable tremor signal"
    elif snr >= config.likely_snr and peak_width <= config.max_likely_peak_width_hz and cycles >= config.min_cycles_required and xy_agreement and feature_group_consistency:
        label = "Likely rhythmic tremor"
    elif snr >= config.possible_snr and peak_width <= config.max_possible_peak_width_hz and cycles >= config.min_cycles_required:
        label = "Possible rhythmic tremor"
    elif snr < config.possible_snr and band_power < config.min_band_power * 10:
        label = "No tremor detected"
    else:
        label = "No reliable tremor signal"

    return TremorAnalysisResult(
        label=label,
        peak_frequency_hz=peak_frequency,
        snr=snr,
        peak_width_hz=peak_width,
        band_power=band_power,
        gross_power=gross_power,
        cycles=cycles,
        xy_agreement=xy_agreement,
        valid_points=valid_points,
        track_survival_rate=track_survival_rate,
        fps=fps,
        window_sec=duration,
        reasons=reasons,
        x_peak_frequency_hz=x_peak,
        y_peak_frequency_hz=y_peak,
        x_snr=float(x_features["snr"]),
        y_snr=float(y_features["snr"]),
        total_power=total_power,
        spectral_entropy=min(float(x_features["entropy"]), float(y_features["entropy"])),
        feature_group_consistency=feature_group_consistency,
    )
