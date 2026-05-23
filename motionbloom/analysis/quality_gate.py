"""Quality gates for explainable ROI tremor confidence labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TremorConfidenceLabel(StrEnum):
    UNUSABLE = "Unusable recording"
    NO_RELIABLE_SIGNAL = "No reliable tremor signal"
    NO_TREMOR = "No tremor detected"
    POSSIBLE = "Possible rhythmic tremor"
    LIKELY = "Likely rhythmic tremor"


@dataclass(frozen=True)
class TremorQualityConfig:
    min_fps: float = 24.0
    min_window_sec: float = 3.0
    min_cycles: float = 12.0
    min_valid_points: int = 6
    min_track_survival_rate: float = 0.35
    min_roi_visibility: float = 0.4
    possible_snr: float = 1.8
    likely_snr: float = 3.0
    max_possible_peak_width_hz: float = 2.0
    max_likely_peak_width_hz: float = 1.25
    max_gross_to_tremor_ratio: float = 6.0
    min_band_power: float = 1e-7


@dataclass
class TremorQualityDecision:
    label: TremorConfidenceLabel
    reasons: list[str] = field(default_factory=list)


def classify_tremor_quality(
    *,
    fps: float,
    window_sec: float,
    cycles: float,
    valid_points: int,
    track_survival_rate: float,
    roi_visibility: float,
    snr: float,
    peak_width_hz: float,
    band_power: float,
    gross_power: float,
    xy_agreement: bool,
    feature_group_consistency: bool = True,
    config: TremorQualityConfig | None = None,
) -> TremorQualityDecision:
    """Return a conservative confidence label and plain-language reasons."""
    cfg = config or TremorQualityConfig()
    reasons: list[str] = []

    if fps < cfg.min_fps:
        return TremorQualityDecision(
            TremorConfidenceLabel.UNUSABLE,
            [f"fps {fps:.1f} below {cfg.min_fps:.1f}"],
        )
    if window_sec < cfg.min_window_sec:
        return TremorQualityDecision(
            TremorConfidenceLabel.UNUSABLE,
            [f"window {window_sec:.1f}s shorter than {cfg.min_window_sec:.1f}s"],
        )
    if roi_visibility < cfg.min_roi_visibility:
        return TremorQualityDecision(
            TremorConfidenceLabel.UNUSABLE,
            ["hand ROI visibility too low"],
        )

    if valid_points < cfg.min_valid_points:
        reasons.append("too few valid optical-flow points")
    else:
        reasons.append("valid optical-flow point count passed")

    if track_survival_rate < cfg.min_track_survival_rate:
        reasons.append("track survival rate too low")
    else:
        reasons.append("track survival rate passed")

    if cycles < cfg.min_cycles:
        reasons.append(f"only {cycles:.1f} tremor-band cycles")
    else:
        reasons.append("enough tremor-band cycles in window")

    if snr >= cfg.likely_snr:
        reasons.append("tremor-band peak passed SNR threshold")
    elif snr >= cfg.possible_snr:
        reasons.append("tremor-band peak has borderline SNR")
    else:
        reasons.append("tremor-band SNR too low")

    if peak_width_hz <= cfg.max_likely_peak_width_hz:
        reasons.append("spectral peak is narrow")
    elif peak_width_hz <= cfg.max_possible_peak_width_hz:
        reasons.append("spectral peak is moderately broad")
    else:
        reasons.append("spectral peak too broad")

    if xy_agreement:
        reasons.append("x/y peak frequencies agree")
    else:
        reasons.append("x/y peak frequencies do not agree")

    if feature_group_consistency:
        reasons.append("feature groups are consistent")
    else:
        reasons.append("feature groups are inconsistent")

    gross_ratio = gross_power / max(band_power, 1e-12)
    if gross_ratio <= cfg.max_gross_to_tremor_ratio:
        reasons.append("gross motion below rejection threshold")
    else:
        reasons.append("gross motion dominates tremor-band power")

    if valid_points < cfg.min_valid_points or track_survival_rate < cfg.min_track_survival_rate:
        return TremorQualityDecision(TremorConfidenceLabel.NO_RELIABLE_SIGNAL, reasons)
    if band_power < cfg.min_band_power:
        return TremorQualityDecision(TremorConfidenceLabel.NO_TREMOR, reasons + ["tremor-band power near noise floor"])
    if gross_ratio > cfg.max_gross_to_tremor_ratio:
        return TremorQualityDecision(TremorConfidenceLabel.NO_RELIABLE_SIGNAL, reasons)
    if snr >= cfg.likely_snr and peak_width_hz <= cfg.max_likely_peak_width_hz and cycles >= cfg.min_cycles and xy_agreement and feature_group_consistency:
        return TremorQualityDecision(TremorConfidenceLabel.LIKELY, reasons)
    if snr >= cfg.possible_snr and peak_width_hz <= cfg.max_possible_peak_width_hz and cycles >= cfg.min_cycles:
        return TremorQualityDecision(TremorConfidenceLabel.POSSIBLE, reasons)
    if snr < cfg.possible_snr and band_power < cfg.min_band_power * 10:
        return TremorQualityDecision(TremorConfidenceLabel.NO_TREMOR, reasons)
    return TremorQualityDecision(TremorConfidenceLabel.NO_RELIABLE_SIGNAL, reasons)
