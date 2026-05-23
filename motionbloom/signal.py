"""Signal processing for tremor analysis.

All functions operate on 1-D numpy arrays sampled at a known fs (Hz) unless
otherwise noted. Filters are designed once and cached by parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, detrend

# -- Constants ---------------------------------------------------------------
# SOTA Update (di Biase 2025): Primary pathological tremor band 3-7 Hz
PRIMARY_TREMOR_BAND = (3.0, 7.0)   # Hz, primary pathological tremor (SOTA)
SECONDARY_BAND = (7.0, 12.0)        # Hz, enhanced physiological tremor
TREMOR_BAND = (3.0, 12.0)           # Hz, full tremor band (legacy compatibility)

# Tremor band presets (VisionMD uses 2-10 Hz for general postural tremor)
TREMOR_BANDS = {
    "parkinsonian": (3.0, 7.0),      # Parkinsonian rest/postural tremor
    "general": (2.0, 10.0),          # General tremor screening (VisionMD)
    "essential": (4.0, 12.0),        # Essential tremor / broader detection
    "legacy": (3.0, 12.0),           # Original MotionBloom range
}

# Updated classification (4 classes for better specificity)
FREQ_CLASSES = [
    # (low, high, label, short_code)
    (3.0, 5.0, "Parkinsonian-like (rest)", "parkinsonian"),
    (5.0, 7.0, "Essential-like (postural/kinetic)", "essential"),
    (7.0, 10.0, "Enhanced physiological (mild)", "physiological_mild"),
    (10.0, 12.0, "Enhanced physiological (high)", "physiological_high"),
]

# Quality thresholds for trial validation (loosened for live debugging)
MIN_FPS = 20.0                      # Minimum acceptable frame rate (was 25)
MIN_CONFIDENCE = 0.4                # Minimum MediaPipe landmark confidence (was 0.5)
MAX_MISSING_FRAMES_PCT = 25.0      # Max percentage of missing/low-confidence frames (was 15)
MAX_SHORT_GAP_SEC = 0.3             # Interpolate gaps <= 0.3s (was 0.2s)
MAX_LONG_GAP_SEC = 0.8              # Reject trials with gaps > 0.8s (was 0.5s)

# FPS quality gates (distinguish mild vs severe instability)
FPS_CV_WARN_THRESHOLD = 0.25    # mild instability -> low_quality
FPS_CV_INVALID_THRESHOLD = 0.45  # severe instability -> invalid

# Gross motion detection thresholds with periodicity requirements (STRICT for research)
HIGH_VELOCITY_P95_THRESHOLD = 0.30  # 95th percentile velocity for gross motion (norm/sec)
HIGH_PATH_RATIO_THRESHOLD = 8.0     # path_length / displacement ratio (chaotic motion)
HIGH_JUMP_RATE_THRESHOLD = 0.40     # Fraction of frames with large jumps
JUMP_DELTA_THRESHOLD = 0.04         # Frame-to-frame delta considered a "jump"

# Center stability threshold for translation detection
CENTER_DRIFT_THRESHOLD = 2.0        # center_drift / motion_span ratio (translation if > 2.0)

# Periodicity requirements for valid tremor (STRICT for research validity)
MIN_PEAK_PROMINENCE = 3.0           # Peak power / median PSD ratio (narrow peak required)
MIN_BAND_POWER_RATIO = 0.25         # Tremor band power / total power (band concentration)
MIN_TREMOR_FREQ = 2.0               # Hz, minimum for valid tremor (below = gross motion)
MAX_TREMOR_FREQ = 12.0              # Hz, maximum for valid tremor (above = noise)

# Live score calibration (saturating formula scales - ONLY tune these for UX)
LIVE_AMP_SCALE = 0.015              # Amplitude saturation scale
LIVE_VELOCITY_SCALE = 0.35          # Velocity saturation scale
LIVE_POWER_SCALE = 0.002            # Power saturation scale

MOVEMENT_TREMOR_HIGHPASS_HZ = 2.5   # Remove slow intentional movement
MOVEMENT_TREMOR_POWER_SCALE = 0.001 # Residual PSD score calibration
MOVEMENT_TREMOR_RMS_SCALE = 0.010   # Residual amplitude score calibration

BOX_AREA_CV_WARN = 0.20
BOX_AREA_CV_INVALID = 0.35
BOX_SIZE_CV_WARN = 0.20
BOX_SIZE_CV_INVALID = 0.35
BOX_JUMP_RATE_INVALID = 0.40

PALM_STEADY_GROSS_RATIO = 0.15
PALM_UNCERTAIN_GROSS_RATIO = 0.35
PALM_STEADY_FRAME_MOTION = 0.02
PALM_MODERATE_FRAME_MOTION = 0.06
PALM_DRIFT_RATIO_THRESHOLD = 0.12
PALM_SCREEN_TRAVEL_RATIO_MAX = 0.08
PALM_SCREEN_NET_RATIO_MAX = 0.05
PALM_HAND_RELATIVE_TRAVEL_MAX = 0.50
PALM_HAND_RELATIVE_NET_MAX = 0.35
TREMOR_AMP_RATIO_MAX = 0.12
TREMOR_AMP_RATIO_MIN = 0.002
VERY_STRONG_TREMOR_SNR_DB = 12.0

PALM_RELATIVE_PRIMARY_FINGERS = ("index_tip", "middle_tip", "ring_tip")
PALM_RELATIVE_MIN_SNR_DB = 6.0
PALM_RELATIVE_MAX_PEAK_WIDTH_HZ = 2.0
PALM_RELATIVE_MIN_CYCLES = 4.0
PALM_RELATIVE_AGREEMENT_HZ = 1.0
PALM_RELATIVE_MIN_BAND_POWER = 1e-7


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


# -- Smoothness (Jerk) Computation -------------------------------------------
def compute_jerk(x: np.ndarray, y: np.ndarray, fs: float) -> float:
    """Compute RMS jerk (3rd derivative) for smoothness analysis.
    
    Based on Rupprechter et al. (2021): Lower jerk indicates smoother,
    more controlled movement. High jerk indicates erratic, tremor-like motion.
    
    Args:
        x, y: Position arrays (normalized coordinates)
        fs: Sampling frequency (Hz)
        
    Returns:
        RMS jerk value (lower = smoother movement)
    """
    if len(x) < 4 or len(y) < 4:
        return 0.0
    
    dt = 1.0 / fs
    
    # 1st derivative: Velocity (dx/dt, dy/dt)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    
    # 2nd derivative: Acceleration (dv/dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    
    # 3rd derivative: Jerk (da/dt)
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    
    # Jerk magnitude (combined axes)
    jerk_mag = np.sqrt(jx**2 + jy**2)
    
    # RMS jerk (root mean square for overall smoothness)
    rms_jerk = float(np.sqrt(np.mean(jerk_mag**2)))
    
    return rms_jerk


def compute_gross_motion_features(xu: np.ndarray, yu: np.ndarray, fs: float) -> dict:
    """Compute features to detect gross motion vs tremor.
    
    Returns dict with:
    - velocity_p95: 95th percentile velocity (normalized coords/sec)
    - path_length: Total path traveled
    - displacement: Net displacement (start to end)
    - path_ratio: path_length / displacement (high = chaotic)
    - jump_rate: Fraction of frames with large sudden jumps
    - mean_jump: Average frame-to-frame delta
    - center_drift: Net center displacement / motion span (translation if large)
    - motion_span: 95th percentile distance from center
    """
    n = xu.size
    if n < 3:
        return {"velocity_p95": 0.0, "path_length": 0.0, "displacement": 0.0,
                "path_ratio": 1.0, "jump_rate": 0.0, "mean_jump": 0.0,
                "center_drift": 0.0, "motion_span": 0.0}
    
    dt = 1.0 / fs
    
    # Frame-to-frame deltas
    dx = np.diff(xu)
    dy = np.diff(yu)
    delta = np.hypot(dx, dy)
    
    # Velocity
    velocity = delta / dt
    velocity_p95 = float(np.percentile(velocity, 95))
    
    # Path metrics
    path_length = float(np.sum(delta))
    displacement = float(np.hypot(xu[-1] - xu[0], yu[-1] - yu[0]))
    path_ratio = path_length / max(displacement, 1e-6)
    
    # Jump detection
    jump_mask = delta > JUMP_DELTA_THRESHOLD
    jump_rate = float(np.mean(jump_mask))
    mean_jump = float(np.mean(delta))
    
    # Center stability (translation detection)
    center_x = float(np.mean(xu))
    center_y = float(np.mean(yu))
    distances = np.hypot(xu - center_x, yu - center_y)
    motion_span = float(np.percentile(distances, 95))
    
    # Net center drift (start/end midpoint vs overall center)
    start_end_center_x = (xu[0] + xu[-1]) / 2.0
    start_end_center_y = (yu[0] + yu[-1]) / 2.0
    center_drift_abs = float(np.hypot(
        start_end_center_x - center_x,
        start_end_center_y - center_y
    ))
    center_drift = center_drift_abs / max(motion_span, 1e-6)
    
    return {
        "velocity_p95": velocity_p95,
        "path_length": path_length,
        "displacement": displacement,
        "path_ratio": path_ratio,
        "jump_rate": jump_rate,
        "mean_jump": mean_jump,
        "center_drift": center_drift,
        "motion_span": motion_span,
    }


def classify_motion_type(
    peak_hz: float,
    gross_motion_features: dict,
    band_power_ratio: float,
    peak_prominence: float,
) -> tuple[str, str]:
    """Classify motion as tremor vs gross motion vs noise.
    
    Tremor requires:
    - Frequency in tremor band (2-12 Hz)
    - Strong periodicity (high peak prominence)
    - Energy concentrated in tremor band
    - Stable center (low drift)
    - Moderate velocity (not aggressive waving)
    
    Returns:
        (classification, reason)
    """
    velocity_p95 = gross_motion_features["velocity_p95"]
    path_ratio = gross_motion_features["path_ratio"]
    jump_rate = gross_motion_features["jump_rate"]
    center_drift = gross_motion_features["center_drift"]
    displacement = gross_motion_features["displacement"]
    path_length = gross_motion_features["path_length"]
    
    # Check for tracking instability (large jumps, chaotic path)
    if jump_rate > HIGH_JUMP_RATE_THRESHOLD:
        return (MotionClassification.TRACKING_UNSTABLE,
                f"Tracking unstable: {jump_rate:.1%} frames with jumps")
    
    if path_ratio > HIGH_PATH_RATIO_THRESHOLD and peak_prominence < MIN_PEAK_PROMINENCE:
        return (MotionClassification.TRACKING_UNSTABLE,
                f"Chaotic path: ratio {path_ratio:.1f}")
    
    # Check for gross translation (moving hand across frame)
    # Large displacement with low center stability = translation not tremor
    if displacement > 0.35 * path_length and peak_hz < MIN_TREMOR_FREQ:
        return (MotionClassification.GROSS_TRANSLATION,
                f"Hand translation detected: {peak_hz:.1f} Hz")
    
    if center_drift > CENTER_DRIFT_THRESHOLD:
        return (MotionClassification.GROSS_TRANSLATION,
                f"Center drifted: ratio {center_drift:.1f}")
    
    # Check for gross voluntary motion (aggressive waving, fast movement)
    if velocity_p95 > HIGH_VELOCITY_P95_THRESHOLD and peak_prominence < MIN_PEAK_PROMINENCE:
        return (MotionClassification.GROSS_MOTION,
                f"Gross motion: velocity {velocity_p95:.2f}, weak peak")
    
    # Check frequency range
    if peak_hz < MIN_TREMOR_FREQ:
        return (MotionClassification.LOW_FREQUENCY,
                f"Frequency too low: {peak_hz:.1f} Hz")
    
    if peak_hz > MAX_TREMOR_FREQ:
        return (MotionClassification.HIGH_FREQUENCY_NOISE,
                f"Frequency too high: {peak_hz:.1f} Hz")
    
    # Check periodicity (tremor should have narrow peak)
    if peak_prominence < MIN_PEAK_PROMINENCE:
        return (MotionClassification.UNCERTAIN,
                f"Weak periodicity: prominence {peak_prominence:.1f}")
    
    # Check band power concentration
    if band_power_ratio < MIN_BAND_POWER_RATIO:
        return (MotionClassification.UNCERTAIN,
                f"Weak tremor band: {band_power_ratio:.2f}")
    
    # All gates passed: valid tremor
    return (MotionClassification.VALID_TREMOR,
            f"Tremor detected: {peak_hz:.1f} Hz")


def compute_speed(xu: np.ndarray, yu: np.ndarray, fs: float) -> float:
    """Compute RMS speed (velocity magnitude) for movement analysis.
    
    Args:
        xu, yu: Position arrays (normalized coordinates)
        fs: Sampling frequency (Hz)
        
    Returns:
        RMS speed value
    """
    if len(xu) < 2 or len(yu) < 2:
        return 0.0
    
    dt = 1.0 / fs
    
    # Velocity components
    vx = np.gradient(xu, dt)
    vy = np.gradient(yu, dt)
    
    # Speed magnitude
    speed = np.sqrt(vx**2 + vy**2)
    
    # RMS speed
    rms_speed = float(np.sqrt(np.mean(speed**2)))
    
    return rms_speed


def compute_sparc(x: np.ndarray, y: np.ndarray, fs: float) -> float:
    """Compute SPARC (Spectral Arc Length) smoothness metric.
    
    SPARC is a validated smoothness metric for upper-limb reaching movements
    after stroke (Balasubramanian et al. 2021, J NeuroEng Rehab).
    
    More negative = less smooth (more submovements/corrections)
    Closer to 0 = smoother movement
    
    Args:
        x, y: Position arrays (normalized coordinates)
        fs: Sampling frequency (Hz)
        
    Returns:
        SPARC value (typically -5 to -1.5 for reaching movements)
    """
    if len(x) < 8 or len(y) < 8:
        return 0.0
    
    # Compute velocity magnitude
    dt = 1.0 / fs
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed = np.sqrt(vx**2 + vy**2)
    
    # Compute FFT of speed profile
    n = len(speed)
    fft_speed = np.fft.rfft(speed)
    freqs = np.fft.rfftfreq(n, d=dt)
    
    # Magnitude spectrum (normalized)
    magnitude = np.abs(fft_speed)
    magnitude = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
    
    # Compute arc length in frequency domain
    # Arc length = integral of sqrt(1 + (dM/df)^2)
    dmag = np.diff(magnitude)
    dfreq = np.diff(freqs)
    
    # Avoid division by zero
    dfreq = np.where(dfreq == 0, 1e-10, dfreq)
    
    # Arc length segments
    arc_segments = np.sqrt(1 + (dmag / dfreq)**2)
    arc_length = np.sum(arc_segments * dfreq)
    
    # SPARC = -arc_length (negative for convention)
    sparc = -arc_length
    
    return float(sparc)


def compute_log_dimensionless_jerk(x: np.ndarray, y: np.ndarray, fs: float, 
                                    movement_duration: float | None = None) -> float:
    """Compute Log Dimensionless Jerk (LDLJ) smoothness metric.
    
    Normalized jerk metric that accounts for movement duration and amplitude.
    Lower values indicate smoother movement.
    
    Args:
        x, y: Position arrays (normalized coordinates)
        fs: Sampling frequency (Hz)
        movement_duration: Duration in seconds (auto-computed if None)
        
    Returns:
        LDLJ value (typically -4 to 0 for reaching movements)
    """
    if len(x) < 4 or len(y) < 4:
        return 0.0
    
    dt = 1.0 / fs
    duration = movement_duration if movement_duration else len(x) * dt
    
    # Compute jerk magnitude
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    jx = np.gradient(ax, dt)
    jy = np.gradient(ay, dt)
    jerk_mag = np.sqrt(jx**2 + jy**2)
    
    # Compute path length (movement amplitude)
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    if path_length < 1e-6:
        return 0.0
    
    # Dimensionless jerk = (duration^5 / path_length^2) * integral(jerk^2)
    jerk_squared_integral = np.trapz(jerk_mag**2, dx=dt)
    dimensionless_jerk = (duration**5 / path_length**2) * jerk_squared_integral
    
    # Log transform
    ldlj = float(np.log(dimensionless_jerk)) if dimensionless_jerk > 0 else 0.0
    
    return ldlj


# -- Trial Quality Assessment ------------------------------------------------
def detect_gaps(t: np.ndarray, max_gap: float = MAX_LONG_GAP_SEC) -> dict:
    """Detect temporal gaps in timestamp array.
    
    Returns dict with:
        - has_long_gap: bool, whether any gap exceeds max_gap
        - max_gap_sec: float, largest gap found
        - gap_count: int, number of gaps > expected frame time
    """
    if t.size < 2:
        return {"has_long_gap": False, "max_gap_sec": 0.0, "gap_count": 0}
    
    dt = np.diff(t)
    expected_dt = np.median(dt)
    
    # Gaps are frames that took >2x the median frame time
    gaps = dt[dt > 2.0 * expected_dt]
    
    max_gap_sec = float(np.max(dt)) if len(dt) > 0 else 0.0
    has_long_gap = max_gap_sec > max_gap
    
    return {
        "has_long_gap": has_long_gap,
        "max_gap_sec": max_gap_sec,
        "gap_count": int(len(gaps)),
    }


def assess_trial_quality(t: np.ndarray, confidence: np.ndarray | None = None) -> dict:
    """Assess quality of a video trial for valid analysis.
    
    Returns dict with:
        - quality_status: "valid" | "low_quality" | "invalid"
        - quality_score: float 0-1
        - fps_mean: float
        - fps_cv: float (coefficient of variation)
        - missing_pct: float (if confidence provided)
        - has_long_gap: bool
        - reasons: list of str (issues found)
    """
    reasons = []
    
    # Check FPS stability
    if t.size < 2:
        return {"quality_status": "invalid", "quality_score": 0.0, "fps_cv": 0.0, "reasons": ["insufficient samples"]}
    
    dt = np.diff(t)
    if len(dt) == 0 or np.any(dt <= 0):
        return {"quality_status": "invalid", "quality_score": 0.0, "fps_cv": 0.0, "reasons": ["invalid timestamps"]}
    
    fps_inst = 1.0 / dt
    fps_mean = float(np.mean(fps_inst))
    fps_std = float(np.std(fps_inst))
    fps_cv = fps_std / fps_mean if fps_mean > 0 else 0.0
    fps_stable = fps_cv < 0.20  # CV < 20%
    
    if fps_mean < MIN_FPS:
        reasons.append(f"low FPS ({fps_mean:.1f} < {MIN_FPS})")
    
    # Determine quality status based on FPS CV
    if fps_cv >= FPS_CV_INVALID_THRESHOLD:
        quality_status = "invalid"
        reasons.append(f"severely unstable FPS (cv={fps_cv:.2f})")
    elif fps_cv >= FPS_CV_WARN_THRESHOLD:
        quality_status = "low_quality"
        reasons.append(f"unstable FPS (cv={fps_cv:.2f})")
    else:
        quality_status = "valid"
    
    # Check for long gaps
    gap_info = detect_gaps(t)
    if gap_info["has_long_gap"]:
        reasons.append(f"long gap detected ({gap_info['max_gap_sec']:.2f}s > {MAX_LONG_GAP_SEC}s)")
        quality_status = "invalid"
    
    # Check landmark confidence if provided
    missing_pct = 0.0
    if confidence is not None:
        low_conf = np.sum(confidence < MIN_CONFIDENCE)
        missing_pct = 100.0 * low_conf / len(confidence)
        if missing_pct > MAX_MISSING_FRAMES_PCT:
            reasons.append(f"too many low-confidence frames ({missing_pct:.1f}%)")
            if quality_status == "valid":
                quality_status = "low_quality"
    
    # Compute quality score
    fps_score = min(1.0, fps_mean / 60.0)  # Normalize to 60 fps target
    stability_score = 1.0 if fps_cv < FPS_CV_WARN_THRESHOLD else (0.5 if fps_cv < FPS_CV_INVALID_THRESHOLD else 0.0)
    confidence_score = 1.0 - (missing_pct / 100.0) if confidence is not None else 1.0
    gap_score = 0.0 if gap_info["has_long_gap"] else 1.0
    
    quality_score = (fps_score + stability_score + confidence_score + gap_score) / 4.0
    
    return {
        "quality_status": quality_status,
        "quality_score": float(quality_score),
        "fps_mean": fps_mean,
        "fps_cv": float(fps_cv),
        "fps_stable": fps_stable,
        "missing_pct": missing_pct,
        "has_long_gap": gap_info["has_long_gap"],
        "max_gap_sec": gap_info["max_gap_sec"],
        "reasons": reasons,
    }


# -- Resampling --------------------------------------------------------------
def resample_uniform(t: np.ndarray, x: np.ndarray,
                     y: np.ndarray, fs: float,
                     max_gap: float = MAX_SHORT_GAP_SEC):
    """Linear-interpolate irregular samples onto a uniform grid.
    
    Handles dropped frames by interpolating short gaps (<= max_gap).
    Returns None if long gaps exist (data is unreliable).
    
    Args:
        t: Timestamps (seconds)
        x, y: Position arrays
        fs: Target sampling frequency (Hz)
        max_gap: Maximum gap to interpolate (seconds)
        
    Returns:
        (tu, xu, yu) or None if invalid
    """
    if t.size < 2:
        return None
    
    # Check for long gaps
    gap_info = detect_gaps(t, max_gap=MAX_LONG_GAP_SEC)
    if gap_info["has_long_gap"]:
        return None  # Reject trials with long gaps
    
    dur = t[-1] - t[0]
    if dur <= 0:
        return None
    n = int(dur * fs)
    if n < 16:
        return None
    
    # Create uniform time grid
    tu = t[0] + np.arange(n) / fs
    
    # Interpolate positions
    # This handles short gaps automatically via linear interpolation
    xu = np.interp(tu, t, x)
    yu = np.interp(tu, t, y)
    
    return tu, xu, yu


# -- Task Modes --------------------------------------------------------------
class TaskMode:
    """Task mode definitions for metric selection and band configuration."""
    REST_TREMOR = "rest_tremor"                    # 3-7 Hz, Parkinsonian rest tremor
    POSTURAL_GENERAL = "postural_tremor_general"  # 2-10 Hz, general postural tremor
    ESSENTIAL_SCREEN = "essential_tremor_screen"  # 4-12 Hz, essential tremor
    MOVEMENT_TREMOR = "movement_tremor"            # Tremor residual during intentional movement
    REHAB_REACH = "rehab_reach"                    # Movement quality only
    FINGER_TAPPING = "finger_tapping"              # Repetition metrics


class MotionClassification:
    """Motion type classification to distinguish tremor from gross motion."""
    VALID_TREMOR = "valid_tremor"              # Clean rhythmic tremor in frequency band
    GROSS_MOTION = "gross_motion"              # Large voluntary movement, not tremor
    GROSS_TRANSLATION = "gross_translation"    # Hand translation/repositioning
    WIDE_RANGE_HAND_MOVEMENT = "wide_range_hand_movement"
    MOTION_TOO_HIGH_FOR_TREMOR = "motion_too_high_for_tremor"
    TRACKING_UNSTABLE = "tracking_unstable"    # High landmark jitter/jumps
    LOW_FREQUENCY = "low_frequency_motion"     # Motion < 2.5 Hz (voluntary)
    HIGH_FREQUENCY_NOISE = "high_frequency_noise"  # > 12 Hz, likely tracking artifacts
    INSUFFICIENT_DATA = "insufficient_data"    # Too few samples
    UNCERTAIN = "uncertain"                    # Cannot classify confidently


TASK_BAND_MAP = {
    TaskMode.REST_TREMOR: (3.0, 7.0),
    TaskMode.POSTURAL_GENERAL: (2.0, 10.0),
    TaskMode.ESSENTIAL_SCREEN: (4.0, 12.0),
    TaskMode.MOVEMENT_TREMOR: TREMOR_BAND,
    TaskMode.REHAB_REACH: None,      # No tremor analysis
    TaskMode.FINGER_TAPPING: None,   # No tremor analysis
}


# -- Metrics -----------------------------------------------------------------
@dataclass
class MovementQualityMetrics:
    """Rehabilitation movement quality metrics (separate from tremor)."""
    fs: float
    samples: int
    duration: float
    
    # Movement characteristics
    range_of_motion: float = 0.0      # Total path length
    path_efficiency: float = 0.0      # Straight-line distance / path length
    peak_velocity: float = 0.0        # Maximum velocity reached
    mean_velocity: float = 0.0        # Average velocity
    
    # Smoothness metrics (validated for stroke rehab)
    sparc: float = 0.0                # SPARC (more negative = less smooth)
    ldlj: float = 0.0                 # Log dimensionless jerk
    rms_jerk: float = 0.0             # Raw RMS jerk (use with caution)
    
    # Task completion
    repetition_count: int = 0         # Number of completed reps
    
    # Quality assessment
    trial_quality: float = 1.0        # 0-1 measurement confidence
    
    # Status
    status: str = "valid"             # valid, invalid, incomplete
    status_reason: str = ""           # Reason if not valid


@dataclass
class TremorMetrics:
    """Tremor-specific metrics with 2-layer validation (quality + motion)."""
    fs: float
    samples: int
    duration: float

    # === QUALITY LAYER (FPS stability, confidence, gaps) ===
    quality_status: str = "invalid"          # valid | low_quality | invalid
    quality_reason: str = ""
    fps_cv: float = 0.0

    # === MOTION LAYER (tremor vs gross motion vs noise) ===
    motion_classification: str = "unknown"   # valid_tremor | gross_translation | gross_motion | uncertain | tracking_unstable | low_frequency | high_frequency_noise
    motion_reason: str = ""

    # === THREE-SCORE MODEL (live feedback + research-valid certification) ===
    live_motion_score: float = 0.0           # ALWAYS shown - general motion/tremor-like activity (0-100)
    tremor_candidate_score: float = 0.0      # What tremor score would be if all gates pass (0-100)
    final_tremor_score: int | None = None    # ONLY when research_valid=True (0-100)
    movement_score: float = 0.0              # Voluntary movement context score (0-100)
    tremor_overlay_score: float = 0.0        # Residual tremor during movement (0-100)
    local_tremor_score: float = 0.0          # Finger-relative residual score (0-100)
    global_residual_tremor_score: float = 0.0 # Whole-hand residual score (0-100)
    dominant_tremor_frequency_hz: float = 0.0 # Dominant residual tremor frequency
    tremor_overlay_ratio: float = 0.0        # Tremor residual / movement+tremor power
    movement_mode_active: bool = False       # True for TaskMode.MOVEMENT_TREMOR
    tremor_source: str = "-"                 # local | global | mixed | none
    tracking_quality: float = 1.0            # 0-1 tracking confidence proxy
    box_stability_status: str = "stable"     # stable | unstable_box_size | unstable_box_motion
    box_area_cv: float = 0.0                  # box area coefficient of variation
    box_width_cv: float = 0.0                 # box width coefficient of variation
    box_height_cv: float = 0.0                # box height coefficient of variation
    box_jump_rate: float = 0.0                # frame-to-frame macro box jump rate
    palm_gross_motion_ratio: float = 0.0      # max palm-center displacement / median hand size
    palm_max_displacement: float = 0.0        # max palm-center displacement over analysis window
    palm_median_box_size: float = 0.0         # median hand box size used for normalization
    palm_velocity_p95: float = 0.0            # per-frame palm velocity normalized by hand size
    palm_vertical_drift: float = 0.0           # window end-start palm y movement
    palm_horizontal_drift: float = 0.0         # window end-start palm x movement
    palm_drift_ratio: float = 0.0              # net palm drift / median hand size
    palm_motion_state: str = "unknown"        # steady | moderate_movement | wide_range_hand_movement
    tremor_analysis_paused: bool = False      # True when palm macro-motion veto is active
    palm_path_length_px: float = 0.0           # palm-center path length over window in pixels
    palm_net_displacement_px: float = 0.0      # palm-center start/end displacement in pixels
    screen_travel_ratio: float = 0.0           # palm path length / frame diagonal
    net_screen_displacement_ratio: float = 0.0 # palm net displacement / frame diagonal
    hand_relative_travel: float = 0.0          # palm path length / median hand-box diagonal
    hand_relative_net: float = 0.0             # palm net displacement / median hand-box diagonal
    physical_veto_reason: str = ""             # hard physics gate explanation
    tremor_amp_ratio: float = 0.0              # tremor p2p displacement / median hand-box diagonal
    raw_fingertip_tremor_power: float = 0.0    # tremor-band power from raw screen-space fingertip motion
    palm_relative_tremor_power: float = 0.0    # tremor-band power from palm-relative primary fingertips
    palm_relative_displacement: float = 0.0    # RMS displacement across primary palm-relative fingertips
    palm_relative_index_peak_hz: float = 0.0
    palm_relative_middle_peak_hz: float = 0.0
    palm_relative_ring_peak_hz: float = 0.0
    palm_relative_agreement_count: int = 0
    palm_relative_veto_reason: str = ""
    
    research_valid: bool = False             # Both quality AND motion layers pass
    confidence_level: str = "low"            # low | medium | high

    # === REASON (shown when not research-valid) ===
    reason: str = ""                         # User-friendly explanation

    # Legacy compatibility (do NOT use for UI truth)
    score: int = 0

    # === SPECTRAL FEATURES ===
    peak_hz: float = 0.0          # dominant frequency (Hz)
    peak_power: float = 0.0       # PSD at peak
    band_power: float = 0.0       # integrated power in tremor band
    total_power: float = 0.0      # integrated power 0.5..fs/2
    band_ratio: float = 0.0       # band_power / total_power
    peak_prominence: float = 0.0  # peak power / median PSD (for periodicity check)

    # === AMPLITUDE ===
    rms_amp: float = 0.0          # RMS of bandpassed signal (normalized coords)
    rms_amp_mm: float = 0.0       # RMS in mm (ONLY if calibrated, else 0.0)
    peak_to_peak_mm: float = 0.0  # Peak-to-peak amplitude (ONLY if calibrated)

    # === QUALITY METRICS ===
    snr_db: float = 0.0           # 10*log10(peak_power / median off-band)
    regularity: float = 0.0       # 1 - normalized spectral entropy in band (0..1)
    peak_sharpness: float = 0.0   # peak / mean in ±1 Hz window

    # === CLASSIFICATION ===
    class_label: str = "-"        # human-readable class
    class_code: str = "-"         # short code

    # === GROSS MOTION FEATURES ===
    velocity_p95: float = 0.0     # 95th percentile velocity
    path_ratio: float = 1.0       # Path length / displacement
    jump_rate: float = 0.0        # Frame-to-frame jump rate
    center_drift: float = 0.0     # Center drift / motion span

    # === LEGACY KINEMATIC METRICS ===
    rms_jerk: float = 0.0   # RMS jerk (use with caution - noisy)
    speed_rms: float = 0.0  # RMS velocity
    sparc: float = 0.0      # SPARC (use MovementQualityMetrics instead)
    ldlj: float = 0.0       # LDLJ (use MovementQualityMetrics instead)
    
    # === LEGACY STATUS FIELDS (keep for compatibility) ===
    trial_quality: float = 1.0  # 0-1 measurement confidence
    status: str = "invalid"       # valid, invalid, low_quality
    status_reason: str = ""     # Reason if not valid


def compute_raw_motion_score(rms_amp: float, velocity_p95: float, band_power: float) -> float:
    """Compute live motion intensity score (0-100) using saturating exponential formula.
    
    This is motion intensity feedback for UX, NOT tremor certification.
    Uses exponential saturation to allow full 0-100 range without artificial caps.
    Weighted toward visible features (amp, velocity) over spectral analysis (power).
    
    Args:
        rms_amp: RMS amplitude of motion (normalized coords)
        velocity_p95: 95th percentile velocity
        band_power: Power in tremor band
        
    Returns:
        Live motion intensity score 0-100 (always shown to user)
    """
    def saturate(value: float, scale: float) -> float:
        """Exponential saturation: 100 * (1 - exp(-x/scale))"""
        value = max(float(value), 0.0)
        return 100.0 * (1.0 - np.exp(-value / max(scale, 1e-6)))
    
    amp_component = saturate(rms_amp, LIVE_AMP_SCALE)
    vel_component = saturate(velocity_p95, LIVE_VELOCITY_SCALE)
    power_component = saturate(band_power, LIVE_POWER_SCALE)
    
    # Weight toward visible features (45-45-10) not equal weighting
    return float(np.clip(0.45 * amp_component + 0.45 * vel_component + 0.10 * power_component, 0, 100))


def compute_palm_center_motion_gate(
    palm_center_x: np.ndarray | None,
    palm_center_y: np.ndarray | None,
    hand_box_size: np.ndarray | None,
    *,
    frame_width: float = 640.0,
    frame_height: float = 480.0,
    hand_box_width: np.ndarray | None = None,
    hand_box_height: np.ndarray | None = None,
) -> dict:
    """Compute palm-center macro-motion gate before tremor classification.

    The palm center is a stable whole-hand anchor. If it travels too far
    relative to the visible hand size, the window is intentional/wide-range
    hand movement and tremor scoring should be paused before PSD analysis.
    """
    empty_physics = {
        "palm_path_length_px": 0.0,
        "palm_net_displacement_px": 0.0,
        "screen_travel_ratio": 0.0,
        "net_screen_displacement_ratio": 0.0,
        "median_hand_box_diagonal_px": 0.0,
        "hand_relative_travel": 0.0,
        "hand_relative_net": 0.0,
        "physical_veto_reason": "",
        "physical_veto": False,
    }
    if palm_center_x is None or palm_center_y is None or hand_box_size is None:
        return {
            "state": "unknown",
            "gross_motion_ratio": 0.0,
            "max_displacement": 0.0,
            "median_box_size": 0.0,
            "per_frame_velocity_p95": 0.0,
            "vertical_drift": 0.0,
            "horizontal_drift": 0.0,
            "drift_ratio": 0.0,
            "allow_tremor": True,
            "pause_tremor": False,
            "reason": "Palm-center gate unavailable",
            **empty_physics,
        }

    x = np.asarray(palm_center_x, dtype=np.float64)
    y = np.asarray(palm_center_y, dtype=np.float64)
    size = np.asarray(hand_box_size, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(size) & (size > 1e-6)
    if np.count_nonzero(valid) < 3:
        return {
            "state": "unknown",
            "gross_motion_ratio": 0.0,
            "max_displacement": 0.0,
            "median_box_size": 0.0,
            "per_frame_velocity_p95": 0.0,
            "vertical_drift": 0.0,
            "horizontal_drift": 0.0,
            "drift_ratio": 0.0,
            "allow_tremor": True,
            "pause_tremor": False,
            "reason": "Insufficient palm-center samples",
            **empty_physics,
        }

    x = x[valid]
    y = y[valid]
    size = size[valid]
    if hand_box_width is not None and hand_box_height is not None:
        width = np.asarray(hand_box_width, dtype=np.float64)[valid]
        height = np.asarray(hand_box_height, dtype=np.float64)[valid]
    else:
        width = height = None
    median_box_size = float(np.median(size))
    max_displacement = float(np.hypot(np.max(x) - np.min(x), np.max(y) - np.min(y)))
    gross_motion_ratio = max_displacement / max(median_box_size, 1e-6)

    frame_diagonal = float(np.hypot(frame_width, frame_height))
    x_px = x * float(frame_width)
    y_px = y * float(frame_height)
    frame_delta_px = np.hypot(np.diff(x_px), np.diff(y_px))
    palm_path_length_px = float(np.sum(frame_delta_px)) if frame_delta_px.size else 0.0
    palm_net_displacement_px = float(np.hypot(x_px[-1] - x_px[0], y_px[-1] - y_px[0]))
    screen_travel_ratio = palm_path_length_px / max(frame_diagonal, 1e-6)
    net_screen_displacement_ratio = palm_net_displacement_px / max(frame_diagonal, 1e-6)
    if width is not None and height is not None and width.size == x.size:
        hand_box_diagonal_px = np.hypot(width * float(frame_width), height * float(frame_height))
        hand_box_diagonal_px = hand_box_diagonal_px[np.isfinite(hand_box_diagonal_px) & (hand_box_diagonal_px > 1e-3)]
        median_hand_box_diagonal_px = float(np.median(hand_box_diagonal_px)) if hand_box_diagonal_px.size else median_box_size * frame_diagonal
    else:
        median_hand_box_diagonal_px = median_box_size * frame_diagonal
    hand_relative_travel = palm_path_length_px / max(median_hand_box_diagonal_px, 1e-6)
    hand_relative_net = palm_net_displacement_px / max(median_hand_box_diagonal_px, 1e-6)

    frame_delta = np.hypot(np.diff(x), np.diff(y))
    per_frame_velocity_p95 = float(np.percentile(frame_delta / max(median_box_size, 1e-6), 95)) if frame_delta.size else 0.0
    horizontal_drift = float(x[-1] - x[0])
    vertical_drift = float(y[-1] - y[0])
    drift_ratio = float(np.hypot(horizontal_drift, vertical_drift) / max(median_box_size, 1e-6))

    physical_veto_reason = ""
    if screen_travel_ratio > PALM_SCREEN_TRAVEL_RATIO_MAX:
        physical_veto_reason = f"Screen path too large ({screen_travel_ratio:.3f})"
    elif net_screen_displacement_ratio > PALM_SCREEN_NET_RATIO_MAX:
        physical_veto_reason = f"Screen displacement too large ({net_screen_displacement_ratio:.3f})"
    elif hand_relative_travel > PALM_HAND_RELATIVE_TRAVEL_MAX:
        physical_veto_reason = f"Palm path exceeds hand-size gate ({hand_relative_travel:.2f})"
    elif hand_relative_net > PALM_HAND_RELATIVE_NET_MAX:
        physical_veto_reason = f"Palm displacement exceeds hand-size gate ({hand_relative_net:.2f})"

    if physical_veto_reason or gross_motion_ratio > PALM_UNCERTAIN_GROSS_RATIO:
        state = MotionClassification.WIDE_RANGE_HAND_MOVEMENT
        allow_tremor = False
        pause_tremor = True
        reason = "Too much hand movement. Hold still."
    elif gross_motion_ratio >= PALM_STEADY_GROSS_RATIO:
        state = MotionClassification.MOTION_TOO_HIGH_FOR_TREMOR
        allow_tremor = False
        pause_tremor = False
        reason = "Movement too high for tremor scoring"
    else:
        state = "steady"
        allow_tremor = True
        pause_tremor = False
        reason = "Palm center steady enough for tremor analysis"

    return {
        "state": state,
        "gross_motion_ratio": float(gross_motion_ratio),
        "max_displacement": max_displacement,
        "median_box_size": median_box_size,
        "per_frame_velocity_p95": per_frame_velocity_p95,
        "vertical_drift": vertical_drift,
        "horizontal_drift": horizontal_drift,
        "drift_ratio": drift_ratio,
        "allow_tremor": allow_tremor,
        "pause_tremor": pause_tremor,
        "reason": reason,
        "palm_path_length_px": palm_path_length_px,
        "palm_net_displacement_px": palm_net_displacement_px,
        "screen_travel_ratio": float(screen_travel_ratio),
        "net_screen_displacement_ratio": float(net_screen_displacement_ratio),
        "median_hand_box_diagonal_px": float(median_hand_box_diagonal_px),
        "hand_relative_travel": float(hand_relative_travel),
        "hand_relative_net": float(hand_relative_net),
        "physical_veto_reason": physical_veto_reason,
        "physical_veto": bool(physical_veto_reason),
    }


def _saturating_score(value: float, scale: float) -> float:
    value = max(float(value), 0.0)
    return 100.0 * (1.0 - np.exp(-value / max(scale, 1e-9)))


def project_dominant_axis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project a 2D trajectory onto its dominant motion axis using SVD."""
    xy = np.column_stack((np.asarray(x, dtype=np.float64),
                          np.asarray(y, dtype=np.float64)))
    if xy.shape[0] < 2:
        return xy[:, 0] if xy.size else np.array([], dtype=np.float64)
    xy_centered = xy - xy.mean(axis=0, keepdims=True)
    if not np.any(np.isfinite(xy_centered)):
        return np.zeros(xy.shape[0], dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(xy_centered, full_matrices=False)
        return xy_centered @ vh[0]
    except np.linalg.LinAlgError:
        return xy_centered[:, 0]


def power_between(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    return float(psd[mask].sum() * df)


def movement_residual_features(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    lowcut: float = MOVEMENT_TREMOR_HIGHPASS_HZ,
    tremor_band: tuple[float, float] = TREMOR_BAND,
) -> dict:
    """Extract high-frequency tremor residual features from a 2D stream."""
    axis_signal = project_dominant_axis(x, y)
    if axis_signal.size < 16:
        return {
            "score": 0.0, "peak_hz": 0.0, "tremor_power": 0.0,
            "total_power": 0.0, "rms": 0.0, "prominence": 0.0,
            "ratio": 0.0, "residual": axis_signal,
        }

    raw_signal = detrend(axis_signal, type="linear")
    residual = highpass(raw_signal, fs, cutoff=lowcut)
    nperseg = int(min(128, residual.size))
    freqs, psd = welch(residual, fs=fs, nperseg=nperseg, detrend=False)
    if psd.sum() <= 0:
        return {
            "score": 0.0, "peak_hz": 0.0, "tremor_power": 0.0,
            "total_power": 0.0, "rms": 0.0, "prominence": 0.0,
            "ratio": 0.0, "residual": residual,
        }

    tremor_mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])
    tremor_power = power_between(freqs, psd, tremor_band[0], tremor_band[1])
    total_power = float(psd.sum() * (freqs[1] - freqs[0])) if freqs.size > 1 else float(psd.sum())
    peak_hz = 0.0
    peak_power = 0.0
    prominence = 0.0
    if np.any(tremor_mask):
        peak_idx = int(np.argmax(np.where(tremor_mask, psd, -np.inf)))
        peak_hz = float(freqs[peak_idx])
        peak_power = float(psd[peak_idx])
        median_psd = float(np.median(psd[tremor_mask]))
        prominence = peak_power / max(median_psd, 1e-12)
    rms = float(np.sqrt(np.mean(residual * residual)))
    amp_score = _saturating_score(rms, MOVEMENT_TREMOR_RMS_SCALE)
    power_score = _saturating_score(tremor_power, MOVEMENT_TREMOR_POWER_SCALE)
    score = float(np.clip(0.55 * amp_score + 0.45 * power_score, 0, 100))
    ratio = tremor_power / max(total_power, 1e-12)
    return {
        "score": score,
        "peak_hz": peak_hz,
        "tremor_power": tremor_power,
        "total_power": total_power,
        "rms": rms,
        "prominence": prominence,
        "ratio": ratio,
        "residual": residual,
    }


def box_stability_features(
    box_width: np.ndarray | None,
    box_height: np.ndarray | None,
    box_area: np.ndarray | None,
    box_center_x: np.ndarray | None,
    box_center_y: np.ndarray | None,
) -> dict:
    """Measure whether the instantaneous hand box is stable enough to trust."""
    def cv(values: np.ndarray | None) -> float:
        if values is None or len(values) < 2:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        return float(np.std(arr) / max(abs(mean), 1e-9))

    area_cv = cv(box_area)
    width_cv = cv(box_width)
    height_cv = cv(box_height)
    jump_rate = 0.0
    if box_center_x is not None and box_center_y is not None and len(box_center_x) > 2:
        dx = np.diff(np.asarray(box_center_x, dtype=np.float64))
        dy = np.diff(np.asarray(box_center_y, dtype=np.float64))
        jumps = np.sqrt(dx * dx + dy * dy)
        jump_rate = float(np.mean(jumps > JUMP_DELTA_THRESHOLD))

    if area_cv > BOX_AREA_CV_INVALID or width_cv > BOX_SIZE_CV_INVALID or height_cv > BOX_SIZE_CV_INVALID:
        status = "unstable_box_size"
    elif jump_rate > BOX_JUMP_RATE_INVALID:
        status = "unstable_box_motion"
    elif area_cv > BOX_AREA_CV_WARN or width_cv > BOX_SIZE_CV_WARN or height_cv > BOX_SIZE_CV_WARN:
        status = "box_size_warning"
    else:
        status = "stable"

    return {
        "status": status,
        "area_cv": area_cv,
        "width_cv": width_cv,
        "height_cv": height_cv,
        "jump_rate": jump_rate,
    }


def compute_box_normalized_tremor_metrics(
    micro_x: np.ndarray,
    micro_y: np.ndarray,
    fs: float,
    tremor_band: tuple[float, float] = TREMOR_BAND,
) -> dict:
    """Score tremor from fingertip motion inside the instantaneous hand box."""
    return movement_residual_features(
        micro_x,
        micro_y,
        fs,
        lowcut=MOVEMENT_TREMOR_HIGHPASS_HZ,
        tremor_band=tremor_band,
    )


def analyze_palm_relative_fingertips(
    fingertip_signals: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    fs: float,
    *,
    tremor_band: tuple[float, float] = PRIMARY_TREMOR_BAND,
) -> dict:
    """Analyze palm-relative index/middle/ring fingertips separately.

    A likely tremor requires at least 2 of 3 primary fingertips to pass
    peak/SNR/cycle checks and agree on peak frequency within ~1 Hz.
    """
    empty = {
        "classification": "unavailable",
        "agreement_count": 0,
        "veto_reason": "Palm-relative fingertip signals unavailable",
        "relative_power": 0.0,
        "relative_displacement": 0.0,
        "median_peak_hz": 0.0,
        "features": {},
    }
    if not fingertip_signals:
        return empty

    features: dict[str, dict] = {}
    passed_names: list[str] = []
    peak_values: list[float] = []
    powers: list[float] = []
    displacements: list[float] = []

    for name in PALM_RELATIVE_PRIMARY_FINGERS:
        pair = fingertip_signals.get(name)
        if pair is None:
            continue
        x = np.asarray(pair[0], dtype=np.float64)
        y = np.asarray(pair[1], dtype=np.float64)
        if x.size != y.size or x.size < 16:
            continue

        axis_signal = project_dominant_axis(x, y)
        residual = highpass(detrend(axis_signal, type="linear"), fs, cutoff=2.0)
        nperseg = int(min(residual.size, max(32, fs * 1.5)))
        freqs, psd = welch(residual, fs=fs, nperseg=nperseg, detrend=False)
        band_mask = (freqs >= tremor_band[0]) & (freqs <= tremor_band[1])
        if psd.size == 0 or not np.any(band_mask) or float(np.sum(psd)) <= 0:
            feature = {
                "peak_hz": 0.0,
                "snr_db": 0.0,
                "peak_width_hz": 0.0,
                "band_power": 0.0,
                "cycles": 0.0,
                "passed": False,
            }
        else:
            peak_idx = int(np.argmax(np.where(band_mask, psd, -np.inf)))
            peak_hz = float(freqs[peak_idx])
            peak_power = float(psd[peak_idx])
            off_peak = band_mask & ~((freqs >= peak_hz - 0.75) & (freqs <= peak_hz + 0.75))
            noise = float(np.median(psd[off_peak])) if np.any(off_peak) else float(np.median(psd[band_mask]))
            snr_db = 10.0 * np.log10(peak_power / max(noise, 1e-12))
            half_power = peak_power * 0.5
            above_half = band_mask & (psd >= half_power)
            peak_width_hz = float(freqs[above_half][-1] - freqs[above_half][0]) if np.any(above_half) else 0.0
            band_power = power_between(freqs, psd, tremor_band[0], tremor_band[1])
            duration = x.size / max(fs, 1e-6)
            cycles = peak_hz * duration
            passed = (
                tremor_band[0] <= peak_hz <= tremor_band[1]
                and snr_db >= PALM_RELATIVE_MIN_SNR_DB
                and peak_width_hz <= PALM_RELATIVE_MAX_PEAK_WIDTH_HZ
                and cycles >= PALM_RELATIVE_MIN_CYCLES
                and band_power >= PALM_RELATIVE_MIN_BAND_POWER
            )
            feature = {
                "peak_hz": peak_hz,
                "snr_db": float(snr_db),
                "peak_width_hz": peak_width_hz,
                "band_power": band_power,
                "cycles": cycles,
                "passed": passed,
            }
            powers.append(band_power)
            if passed:
                passed_names.append(name)
                peak_values.append(peak_hz)

        displacements.append(float(np.sqrt(np.mean((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))))
        features[name] = feature

    agreement_count = 0
    median_peak = float(np.median(peak_values)) if peak_values else 0.0
    if len(peak_values) >= 2:
        agreement_count = int(np.count_nonzero(np.abs(np.asarray(peak_values) - median_peak) <= PALM_RELATIVE_AGREEMENT_HZ))

    if agreement_count >= 2:
        classification = "likely"
        veto_reason = "Palm-relative primary fingertips agree"
    elif len(passed_names) >= 1:
        classification = "possible"
        veto_reason = "Only one primary fingertip passed palm-relative tremor checks"
    else:
        classification = "none"
        veto_reason = "Palm-relative primary fingertips did not show consistent tremor"

    return {
        "classification": classification,
        "agreement_count": agreement_count,
        "veto_reason": veto_reason,
        "relative_power": float(np.median(powers)) if powers else 0.0,
        "relative_displacement": float(np.median(displacements)) if displacements else 0.0,
        "median_peak_hz": median_peak,
        "features": features,
    }


def _estimate_class(peak_hz: float) -> tuple[str, str]:
    if peak_hz <= 0:
        return "Unclassified", "none"
    for lo, hi, label, code in FREQ_CLASSES:
        if lo <= peak_hz < hi:
            return label, code
    return "Out of band", "other"


def compute_movement_quality(
    xu: np.ndarray,
    yu: np.ndarray,
    fs: float,
    trial_quality: float = 1.0,
) -> MovementQualityMetrics:
    """Compute rehabilitation movement quality metrics (no tremor analysis).
    
    For reach-to-target, finger-tapping, and other rehab tasks.
    Focuses on smoothness, ROM, path efficiency, not tremor frequency.
    """
    n = xu.size
    duration = n / fs
    
    # Compute velocity
    dt = 1.0 / fs
    vx = np.gradient(xu, dt)
    vy = np.gradient(yu, dt)
    speed = np.sqrt(vx**2 + vy**2)
    
    peak_velocity = float(np.max(speed))
    mean_velocity = float(np.mean(speed))
    
    # Path length (ROM proxy)
    path_segments = np.sqrt(np.diff(xu)**2 + np.diff(yu)**2)
    range_of_motion = float(np.sum(path_segments))
    
    # Path efficiency (straight-line / actual path)
    straight_dist = np.sqrt((xu[-1] - xu[0])**2 + (yu[-1] - yu[0])**2)
    path_efficiency = float(straight_dist / range_of_motion) if range_of_motion > 1e-6 else 0.0
    
    # Smoothness metrics
    sparc = compute_sparc(xu, yu, fs)
    ldlj = compute_log_dimensionless_jerk(xu, yu, fs, movement_duration=duration)
    rms_jerk = compute_jerk(xu, yu, fs)
    
    # Determine status
    status = "valid" if trial_quality >= 0.7 else "low_quality"
    status_reason = "" if status == "valid" else f"trial_quality={trial_quality:.2f} < 0.7"
    
    return MovementQualityMetrics(
        fs=fs,
        samples=n,
        duration=duration,
        range_of_motion=range_of_motion,
        path_efficiency=path_efficiency,
        peak_velocity=peak_velocity,
        mean_velocity=mean_velocity,
        sparc=sparc,
        ldlj=ldlj,
        rms_jerk=rms_jerk,
        repetition_count=0,  # TODO: Add repetition detection
        trial_quality=trial_quality,
        status=status,
        status_reason=status_reason,
    )


def compute_metrics(
    xu: np.ndarray,
    yu: np.ndarray,
    fs: float,
    *,
    hand_ref_pixels: float | None = None,
    hand_width_mm: float = 85.0,  # avg adult palm width
    baseline_rms: float | None = None,
    task_mode: str | None = None,  # Task mode for frequency band selection
    local_xu: np.ndarray | None = None,
    local_yu: np.ndarray | None = None,
    global_xu: np.ndarray | None = None,
    global_yu: np.ndarray | None = None,
    tracking_quality: float | None = None,
    box_width: np.ndarray | None = None,
    box_height: np.ndarray | None = None,
    box_area: np.ndarray | None = None,
    palm_center_x: np.ndarray | None = None,
    palm_center_y: np.ndarray | None = None,
    hand_box_size: np.ndarray | None = None,
    frame_width: float = 640.0,
    frame_height: float = 480.0,
    hand_box_width: np.ndarray | None = None,
    hand_box_height: np.ndarray | None = None,
    relative_fingertip_signals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    raw_fingertip_tremor_power: float = 0.0,
) -> TremorMetrics | None:
    """Compute tremor metrics from uniform 2D samples.

    `hand_ref_pixels` is the current pixel distance between two reference
    landmarks (e.g. wrist–middle MCP). If provided, amplitude is rescaled
    from normalized coords to millimetres using `hand_width_mm` as reference.
    `baseline_rms` optionally subtracts a measured still-hand noise floor.
    `task_mode` selects appropriate frequency band (defaults to PRIMARY_TREMOR_BAND).
    If `local_*` and `global_*` are provided, tremor is measured from the
    local/micro stream while gross movement is measured from the global/macro
    stream. In box-normalized mode, local is fingertip-in-box motion and global
    is hand-box center motion.
    """
    n = xu.size
    if n < 32:
        return None

    tremor_x = local_xu if local_xu is not None and local_xu.size == n else xu
    tremor_y = local_yu if local_yu is not None and local_yu.size == n else yu
    macro_x = global_xu if global_xu is not None and global_xu.size == n else xu
    macro_y = global_yu if global_yu is not None and global_yu.size == n else yu

    # Compute gross motion features FIRST (before any filtering)
    gross_motion_features = compute_gross_motion_features(macro_x, macro_y, fs)
    palm_gate = compute_palm_center_motion_gate(
        palm_center_x,
        palm_center_y,
        hand_box_size,
        frame_width=frame_width,
        frame_height=frame_height,
        hand_box_width=hand_box_width,
        hand_box_height=hand_box_height,
    )

    if palm_gate["pause_tremor"]:
        center_x = float(np.mean(macro_x))
        center_y = float(np.mean(macro_y))
        live_motion_score = compute_raw_motion_score(
            gross_motion_features["motion_span"],
            gross_motion_features["velocity_p95"],
            0.0,
        )
        return TremorMetrics(
            fs=fs,
            samples=n,
            duration=n / fs,
            quality_status="valid",
            motion_classification=MotionClassification.WIDE_RANGE_HAND_MOVEMENT,
            motion_reason=palm_gate["reason"],
            live_motion_score=live_motion_score,
            tremor_candidate_score=0.0,
            final_tremor_score=None,
            movement_score=live_motion_score,
            palm_gross_motion_ratio=palm_gate["gross_motion_ratio"],
            palm_max_displacement=palm_gate["max_displacement"],
            palm_median_box_size=palm_gate["median_box_size"],
            palm_velocity_p95=palm_gate["per_frame_velocity_p95"],
            palm_vertical_drift=palm_gate["vertical_drift"],
            palm_horizontal_drift=palm_gate["horizontal_drift"],
            palm_drift_ratio=palm_gate["drift_ratio"],
            palm_motion_state=str(palm_gate["state"]),
            tremor_analysis_paused=True,
            palm_path_length_px=palm_gate["palm_path_length_px"],
            palm_net_displacement_px=palm_gate["palm_net_displacement_px"],
            screen_travel_ratio=palm_gate["screen_travel_ratio"],
            net_screen_displacement_ratio=palm_gate["net_screen_displacement_ratio"],
            hand_relative_travel=palm_gate["hand_relative_travel"],
            hand_relative_net=palm_gate["hand_relative_net"],
            physical_veto_reason=palm_gate["physical_veto_reason"] or palm_gate["reason"],
            raw_fingertip_tremor_power=raw_fingertip_tremor_power,
            research_valid=False,
            confidence_level="low",
            reason=palm_gate["reason"],
            velocity_p95=gross_motion_features["velocity_p95"],
            path_ratio=gross_motion_features["path_ratio"],
            jump_rate=gross_motion_features["jump_rate"],
            center_drift=float(np.sqrt(center_x**2 + center_y**2)),
            status=MotionClassification.WIDE_RANGE_HAND_MOVEMENT,
            status_reason=palm_gate["reason"],
        )

    # Compute clinical kinematic metrics (before filtering)
    rms_jerk = compute_jerk(macro_x, macro_y, fs)
    speed_rms = compute_speed(macro_x, macro_y, fs)
    sparc = compute_sparc(macro_x, macro_y, fs)
    ldlj = compute_log_dimensionless_jerk(macro_x, macro_y, fs)

    # Preprocess for tremor analysis: remove linear drift + high-pass at 2 Hz
    # This eliminates slow hand translation from contaminating tremor band
    xu_detrend = detrend(tremor_x, type="linear")
    yu_detrend = detrend(tremor_y, type="linear")
    xu_hp = highpass(xu_detrend, fs, cutoff=2.0)
    yu_hp = highpass(yu_detrend, fs, cutoff=2.0)

    # Bandpass for tremor analysis (primary 3-7 Hz band for pathological tremor)
    xf = bandpass(xu_hp, fs, low=PRIMARY_TREMOR_BAND[0], high=PRIMARY_TREMOR_BAND[1])
    yf = bandpass(yu_hp, fs, low=PRIMARY_TREMOR_BAND[0], high=PRIMARY_TREMOR_BAND[1])

    # Welch PSD - combined magnitude
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

    # Restrict peak search to primary tremor band (3-7 Hz pathological)
    primary_mask = (fxx >= PRIMARY_TREMOR_BAND[0]) & (fxx <= PRIMARY_TREMOR_BAND[1])
    # Also track secondary band for comparison
    secondary_mask = (fxx >= SECONDARY_BAND[0]) & (fxx <= SECONDARY_BAND[1])
    band_mask = (fxx >= TREMOR_BAND[0]) & (fxx <= TREMOR_BAND[1])  # Full band for legacy
    
    if not np.any(primary_mask):
        return None

    # Search for peak in primary band (3-7 Hz) for pathological tremor
    search_mask = (fxx >= 2.5) & (fxx <= 7.5)  # Slightly wider for edge cases
    peak_idx = int(np.argmax(np.where(primary_mask, psd, -np.inf)))
    peak_hz = float(fxx[peak_idx])
    peak_power = float(psd[peak_idx])

    df = float(fxx[1] - fxx[0]) if fxx.size > 1 else 1.0
    total_power = float(psd.sum() * df)
    # Use primary band for power calculation (3-7 Hz pathological focus)
    band_power = float(psd[primary_mask].sum() * df)
    secondary_power = float(psd[secondary_mask].sum() * df) if np.any(secondary_mask) else 0.0
    band_ratio = band_power / total_power if total_power > 0 else 0.0

    # Peak sharpness: ratio of peak to mean within ±1 Hz, off-peak only
    around = (fxx >= peak_hz - 1.0) & (fxx <= peak_hz + 1.0)
    around_offpeak = around & (np.arange(fxx.size) != peak_idx)
    if np.any(around_offpeak):
        mean_near = float(psd[around_offpeak].mean())
        sharpness = peak_power / mean_near if mean_near > 0 else 0.0
    else:
        sharpness = 0.0

    # SNR: peak vs median of primary band excluding ±0.75 Hz around peak
    off = primary_mask & ~((fxx >= peak_hz - 0.75) & (fxx <= peak_hz + 0.75))
    if np.any(off) and np.median(psd[off]) > 0:
        snr_db = 10.0 * np.log10(peak_power / float(np.median(psd[off])))
    else:
        snr_db = 0.0
    
    # Peak prominence: ratio of peak to median PSD (for periodicity check)
    median_psd = float(np.median(psd[primary_mask]))
    peak_prominence = peak_power / max(median_psd, 1e-12)

    # Regularity: 1 - normalized entropy of PSD within primary band (concentrated → 1)
    band_psd = psd[primary_mask]
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
    tremor_peak_to_peak_displacement = float(np.hypot(np.max(xf) - np.min(xf), np.max(yf) - np.min(yf)))
    tremor_amp_ratio = tremor_peak_to_peak_displacement / max(float(palm_gate["median_box_size"]), 1e-6)
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
        rms_amp_mm = 0.0  # Not calibrated - do not report mm
        peak_to_peak_mm = 0.0  # Not calibrated

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
    
    # Classify motion type BEFORE scoring (tremor vs gross motion vs noise)
    motion_class, motion_reason = classify_motion_type(
        peak_hz, gross_motion_features, band_ratio, peak_prominence
    )
    movement_mode_active = task_mode == TaskMode.MOVEMENT_TREMOR
    relative_features = analyze_palm_relative_fingertips(relative_fingertip_signals, fs)
    relative_index_peak = float(relative_features.get("features", {}).get("index_tip", {}).get("peak_hz", 0.0))
    relative_middle_peak = float(relative_features.get("features", {}).get("middle_tip", {}).get("peak_hz", 0.0))
    relative_ring_peak = float(relative_features.get("features", {}).get("ring_tip", {}).get("peak_hz", 0.0))
    relative_agreement_count = int(relative_features.get("agreement_count", 0))
    relative_veto_reason = str(relative_features.get("veto_reason", ""))

    if relative_fingertip_signals is not None and not movement_mode_active:
        relative_classification = str(relative_features.get("classification", "unavailable"))
        if relative_classification == "likely":
            motion_class = MotionClassification.VALID_TREMOR
            motion_reason = f"Palm-relative fingertip tremor agreement ({relative_agreement_count}/3)"
            if relative_features.get("median_peak_hz", 0.0):
                peak_hz = float(relative_features["median_peak_hz"])
                label, code = _estimate_class(peak_hz)
        elif relative_classification == "possible":
            motion_class = MotionClassification.UNCERTAIN
            motion_reason = relative_veto_reason
        else:
            motion_class = MotionClassification.UNCERTAIN
            motion_reason = relative_veto_reason

    box_features = box_stability_features(
        box_width,
        box_height,
        box_area,
        macro_x,
        macro_y,
    )

    movement_score = compute_raw_motion_score(
        gross_motion_features["motion_span"],
        gross_motion_features["velocity_p95"],
        low_power,
    )
    tremor_overlay_score = 0.0
    local_tremor_score = 0.0
    global_residual_tremor_score = 0.0
    dominant_tremor_frequency_hz = peak_hz
    tremor_overlay_ratio = 0.0
    tremor_source = "-"
    tracking_quality_value = 1.0 if tracking_quality is None else float(np.clip(tracking_quality, 0.0, 1.0))

    if movement_mode_active:
        micro_features = compute_box_normalized_tremor_metrics(tremor_x, tremor_y, fs)
        local_tremor_score = micro_features["score"]
        global_residual_tremor_score = 0.0
        tremor_overlay_score = local_tremor_score
        tremor_overlay_ratio = micro_features["ratio"]
        selected = micro_features
        tremor_source = "inside hand box" if tremor_overlay_score >= 10 else "none"

        peak_hz = selected["peak_hz"]
        peak_power = selected["tremor_power"]
        band_power = selected["tremor_power"]
        total_power = max(selected["total_power"], 1e-12)
        band_ratio = selected["ratio"]
        peak_prominence = selected["prominence"]
        dominant_tremor_frequency_hz = selected["peak_hz"]
        rms_amp_net = selected["rms"]
        tremor_candidate_score = tremor_overlay_score
        score = int(round(tremor_overlay_score))
        confidence_level = "high" if tremor_overlay_score >= 60 and peak_prominence >= 3.0 else "medium" if tremor_overlay_score >= 30 else "low"

        tracking_unstable = (
            gross_motion_features["jump_rate"] > HIGH_JUMP_RATE_THRESHOLD
            or tracking_quality_value < 0.4
            or box_features["status"] in {"unstable_box_size", "unstable_box_motion"}
        )
        if tracking_unstable:
            motion_class = MotionClassification.TRACKING_UNSTABLE
            if box_features["status"] == "unstable_box_size":
                motion_reason = "Camera view is changing too much"
            elif box_features["status"] == "unstable_box_motion":
                motion_reason = "Camera view is jumping"
            else:
                motion_reason = "Keep your hand visible"
        elif tremor_overlay_score >= 15 and peak_prominence >= 2.0 and TREMOR_BAND[0] <= peak_hz <= TREMOR_BAND[1]:
            motion_class = MotionClassification.VALID_TREMOR
            motion_reason = f"Tremor during movement: {peak_hz:.1f} Hz"
        else:
            motion_class = MotionClassification.UNCERTAIN
            motion_reason = "Movement detected; tremor overlay is low"
    else:
        tremor_candidate_score = float(score)  # What tremor score would be if all gates pass

    if not movement_mode_active and box_features["status"] in {"unstable_box_size", "unstable_box_motion"}:
        motion_class = MotionClassification.TRACKING_UNSTABLE
        if box_features["status"] == "unstable_box_size":
            motion_reason = "Camera view is changing too much"
        else:
            motion_reason = "Camera view is jumping"

    if not movement_mode_active:
        if tremor_amp_ratio > TREMOR_AMP_RATIO_MAX:
            motion_class = MotionClassification.GROSS_MOTION
            motion_reason = f"Tremor amplitude too large for microtremor ({tremor_amp_ratio:.2f} hand diagonals)"
        elif tremor_amp_ratio < TREMOR_AMP_RATIO_MIN and snr_db < VERY_STRONG_TREMOR_SNR_DB:
            motion_class = MotionClassification.UNCERTAIN
            motion_reason = "No reliable tremor signal"
    
    # Three-score model: always compute live feedback
    live_motion_score = compute_raw_motion_score(rms_amp_net, gross_motion_features["velocity_p95"], band_power)
    if movement_mode_active:
        live_motion_score = movement_score
        label, code = _estimate_class(peak_hz)
    
    # Determine confidence level based on spectral quality
    if not movement_mode_active:
        if band_ratio >= 0.35 and peak_prominence >= 4.0 and regularity >= 0.6:
            confidence_level = "high"
        elif band_ratio >= 0.20 and peak_prominence >= 2.5 and regularity >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
    
    # 2-Layer validation gate:
    # Layer 1: Quality (FPS, gaps, confidence) - will be overridden by caller with assess_trial_quality
    # Layer 2: Motion classification (tremor vs gross motion vs noise)
    motion_valid = (motion_class == MotionClassification.VALID_TREMOR)
    
    # Determine final output based on motion layer (quality layer handled by caller)
    if motion_valid:
        # Valid tremor motion - keep tremor score and classification
        # (Quality layer in caller may still suppress if quality_status != "valid")
        final_tremor_score_value = score
        final_label = label
        final_code = code
        final_amp_mm = rms_amp_mm
        final_p2p_mm = peak_to_peak_mm
        legacy_score = score
        research_valid_default = True  # Caller will override based on quality_status
        reason_text = ""  # No issue
    else:
        # Non-tremor motion - suppress final tremor score but keep live feedback
        final_tremor_score_value = None
        final_label = "-"
        final_code = "-"
        final_amp_mm = 0.0
        final_p2p_mm = 0.0
        legacy_score = 0
        research_valid_default = False
        reason_text = motion_reason  # Show why it's not tremor-valid
    
    # Compute center drift for debug visibility (from original xu, yu before detrending)
    center_x = float(np.mean(macro_x))
    center_y = float(np.mean(macro_y))
    center_drift = float(np.sqrt(center_x**2 + center_y**2))
    
    return TremorMetrics(
        # === QUALITY LAYER (caller will override from assess_trial_quality) ===
        quality_status="valid",  # Placeholder - caller MUST override
        quality_reason="",
        fps_cv=0.0,
        
        # === MOTION LAYER ===
        motion_classification=motion_class,
        motion_reason=motion_reason,
        
        # === THREE-SCORE MODEL ===
        live_motion_score=live_motion_score,
        tremor_candidate_score=tremor_candidate_score,
        final_tremor_score=final_tremor_score_value,
        movement_score=movement_score,
        tremor_overlay_score=tremor_overlay_score,
        local_tremor_score=local_tremor_score,
        global_residual_tremor_score=global_residual_tremor_score,
        dominant_tremor_frequency_hz=dominant_tremor_frequency_hz,
        tremor_overlay_ratio=tremor_overlay_ratio,
        movement_mode_active=movement_mode_active,
        tremor_source=tremor_source,
        tracking_quality=tracking_quality_value,
        box_stability_status=box_features["status"],
        box_area_cv=box_features["area_cv"],
        box_width_cv=box_features["width_cv"],
        box_height_cv=box_features["height_cv"],
        box_jump_rate=box_features["jump_rate"],
        palm_gross_motion_ratio=palm_gate["gross_motion_ratio"],
        palm_max_displacement=palm_gate["max_displacement"],
        palm_median_box_size=palm_gate["median_box_size"],
        palm_velocity_p95=palm_gate["per_frame_velocity_p95"],
        palm_vertical_drift=palm_gate["vertical_drift"],
        palm_horizontal_drift=palm_gate["horizontal_drift"],
        palm_drift_ratio=palm_gate["drift_ratio"],
        palm_motion_state=str(palm_gate["state"]),
        tremor_analysis_paused=False,
        palm_path_length_px=palm_gate["palm_path_length_px"],
        palm_net_displacement_px=palm_gate["palm_net_displacement_px"],
        screen_travel_ratio=palm_gate["screen_travel_ratio"],
        net_screen_displacement_ratio=palm_gate["net_screen_displacement_ratio"],
        hand_relative_travel=palm_gate["hand_relative_travel"],
        hand_relative_net=palm_gate["hand_relative_net"],
        physical_veto_reason=palm_gate["physical_veto_reason"],
        tremor_amp_ratio=tremor_amp_ratio,
        raw_fingertip_tremor_power=raw_fingertip_tremor_power,
        palm_relative_tremor_power=float(relative_features.get("relative_power", 0.0)),
        palm_relative_displacement=float(relative_features.get("relative_displacement", 0.0)),
        palm_relative_index_peak_hz=relative_index_peak,
        palm_relative_middle_peak_hz=relative_middle_peak,
        palm_relative_ring_peak_hz=relative_ring_peak,
        palm_relative_agreement_count=relative_agreement_count,
        palm_relative_veto_reason=relative_veto_reason,
        research_valid=research_valid_default,  # Caller will override
        confidence_level=confidence_level,
        reason=reason_text,
        
        # === SPECTRAL FEATURES ===
        fs=fs,
        samples=n,
        duration=n / fs,
        peak_hz=peak_hz,
        peak_power=peak_power,
        band_power=band_power,
        total_power=total_power,
        band_ratio=band_ratio,
        snr_db=snr_db,
        regularity=reg_term,
        peak_sharpness=sharpness,
        
        # === AMPLITUDE FEATURES ===
        rms_amp=rms_amp_net,
        rms_amp_mm=final_amp_mm,
        peak_to_peak_mm=final_p2p_mm,
        
        # === CLASSIFICATION (gated by motion layer) ===
        class_label=final_label,
        class_code=final_code,
        peak_prominence=peak_prominence,
        
        # === GROSS MOTION FEATURES ===
        velocity_p95=gross_motion_features["velocity_p95"],
        path_ratio=gross_motion_features["path_ratio"],
        jump_rate=gross_motion_features["jump_rate"],
        center_drift=center_drift,
        
        # === SMOOTHNESS METRICS ===
        rms_jerk=rms_jerk,
        speed_rms=speed_rms,
        sparc=sparc,
        ldlj=ldlj,
        
        # === LEGACY STATUS FIELDS ===
        trial_quality=1.0,
        status="valid" if motion_valid else motion_class,
        status_reason="" if motion_valid else motion_reason,
        score=legacy_score,  # Legacy only - do NOT use for UI truth
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
        # Running average of ALL scores - represents the user's typical
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
        above their typical tremor level - i.e. they just need to be as
        steady as they usually are, not dramatically steadier.
        """
        if self.score_avg is None or self.samples < 20:
            return default
        return float(max(25.0, min(90.0, self.score_avg + margin)))
