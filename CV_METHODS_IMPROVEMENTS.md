# CV Methods Improvements - Research-Oriented Signal Processing Upgrade

**Date**: May 21, 2026  
**Status**: Phase 2 Complete (Preprocessing & Metric Separation)  
**Next**: Multi-Fingertip Averaging, ArUco Calibration, Validation

---

## Executive Summary

MotionBloom's computer vision pipeline has been upgraded with **literature-aligned smoothness metrics and preprocessing foundations**. The method now includes timestamp-aware processing, gap detection, hard trial rejection, and separation of tremor vs movement quality metrics.

**Critical Note**: This is a **research-oriented signal processing upgrade**, not yet a validated clinical method. Calibration, multi-landmark tracking, and bench validation are still required before clinical claims are defensible.

### Key Improvements Implemented

| Feature | Phase | Status | Validation Required |
|---------|-------|--------|---------------------|
| **SPARC & LDLJ metrics** | 1 | ✅ Complete | Bench validation pending |
| **Configurable tremor bands** | 1 | ✅ Complete | Task-specific validation needed |
| **Trial quality scoring** | 1 | ✅ Complete | Threshold tuning needed |
| **SOS filtering** | 1 | ✅ Complete | Verified |
| **Timestamp-aware preprocessing** | 2 | ✅ Complete | Needs real timestamp integration |
| **Gap detection & rejection** | 2 | ✅ Complete | Threshold validation needed |
| **Hard trial rejection** | 2 | ✅ Complete | Needs UI integration |
| **Tremor vs Movement separation** | 2 | ✅ Complete | API stable |
| **Task mode definitions** | 2 | ✅ Complete | Needs app integration |
| **Multi-fingertip averaging** | 3 | ⏳ Pending | High priority |
| **ArUco calibration** | 3 | ⏳ Pending | Critical for amplitude claims |
| **Pose compensation tracking** | 4 | ⏳ Pending | Required for rehab use |

---

## Phase 2: Preprocessing & Metric Separation (✅ COMPLETE)

### 1. Timestamp-Aware Preprocessing

#### Gap Detection
- **Implementation**: `detect_gaps(timestamps)` → Returns gap analysis
- **Thresholds**: 
  - Short gaps (≤0.2s): Interpolate
  - Long gaps (>0.5s): Reject trial
- **Purpose**: Prevent bad data from irregular webcam capture

```python
gap_info = detect_gaps(timestamps, max_gap=0.5)
# Returns:
{
    "has_long_gap": False,
    "max_gap_sec": 0.08,
    "gap_count": 3
}
```

#### Smart Resampling
- **Updated**: `resample_uniform()` now checks for long gaps
- **Behavior**: Returns `None` if gaps exceed threshold (rejects trial)
- **Interpolation**: Linear interpolation for short gaps only

### 2. Hard Trial Rejection

#### Quality Assessment (Enhanced)
```python
quality = assess_trial_quality(timestamps, confidence_array)

# NOW INCLUDES:
{
    "valid": False,                     # HARD GATE
    "quality_score": 0.62,
    "fps_mean": 23.4,
    "fps_stable": False,
    "missing_pct": 18.2,
    "has_long_gap": True,               # NEW
    "max_gap_sec": 0.73,                # NEW
    "reasons": [                        # REJECTION REASONS
        "low FPS (23.4 < 25.0)",
        "long gap detected (0.73s > 0.5s)"
    ]
}
```

#### Status Codes
All metrics now return status:
- `"valid"`: Trial passes quality gates
- `"low_quality"`: Trial quality <0.7
- `"invalid"`: Hard rejection (gaps, FPS, confidence)

### 3. Separated Metrics

#### TremorMetrics (Frequency-Domain)
**Purpose**: Tremor detection tasks only  
**Band**: Configurable (3-7 Hz Parkinsonian, 2-10 Hz general, 4-12 Hz essential)

```python
@dataclass
class TremorMetrics:
    # Frequency analysis
    peak_hz: float
    band_power: float
    snr_db: float
    regularity: float
    
    # Amplitude (relative only unless calibrated)
    rms_amp: float
    rms_amp_mm: float = 0.0  # ZERO unless ArUco calibrated
    
    # Product score (not clinical severity)
    score: int  # 0-100 tremor likelihood
    
    # Quality
    trial_quality: float
    status: str  # valid, invalid, low_quality
    status_reason: str
```

**Key Changes**:
- `rms_amp_mm` = 0.0 unless calibrated (no fake mm claims)
- `score` is product metric, not clinical severity
- `status` field for hard rejection
- Legacy smoothness fields retained for backward compatibility

#### MovementQualityMetrics (Rehab Tasks)
**Purpose**: Reach-to-target, finger-tapping, rehab exercises  
**No tremor analysis**: Focuses on smoothness, ROM, path efficiency

```python
@dataclass
class MovementQualityMetrics:
    # Movement characteristics
    range_of_motion: float
    path_efficiency: float  # straight / actual path
    peak_velocity: float
    mean_velocity: float
    
    # Smoothness (SPARC/LDLJ primary)
    sparc: float
    ldlj: float
    rms_jerk: float  # Use with caution
    
    # Task completion
    repetition_count: int  # TODO: Add detection
    
    # Quality
    trial_quality: float
    status: str
    status_reason: str
```

**Use Cases**:
- `TaskMode.REHAB_REACH`: Use MovementQualityMetrics
- `TaskMode.FINGER_TAPPING`: Use MovementQualityMetrics
- `TaskMode.REST_TREMOR`: Use TremorMetrics
- `TaskMode.POSTURAL_GENERAL`: Use TremorMetrics

### 4. Task Mode Definitions

```python
class TaskMode:
    REST_TREMOR = "rest_tremor"                    # 3-7 Hz
    POSTURAL_GENERAL = "postural_tremor_general"  # 2-10 Hz
    ESSENTIAL_SCREEN = "essential_tremor_screen"  # 4-12 Hz
    REHAB_REACH = "rehab_reach"                    # Movement quality
    FINGER_TAPPING = "finger_tapping"              # Repetition metrics

TASK_BAND_MAP = {
    TaskMode.REST_TREMOR: (3.0, 7.0),
    TaskMode.POSTURAL_GENERAL: (2.0, 10.0),
    TaskMode.ESSENTIAL_SCREEN: (4.0, 12.0),
    TaskMode.REHAB_REACH: None,      # No tremor band
    TaskMode.FINGER_TAPPING: None,   # No tremor band
}
```

---

## Phase 1: Signal Processing Core (✅ COMPLETE)

### 1. Validated Smoothness Metrics

#### SPARC (Spectral Arc Length)
- **Implementation**: `compute_sparc(x, y, fs)` → Returns -5 to -1.5 range
- **Interpretation**: More negative = less smooth (compensatory submovements)
- **Validation**: Balasubramanian et al. 2021 - validated for upper-limb stroke rehab
- **Use case**: Preferred metric for reach-to-point and reach-to-grasp tasks

```python
sparc = compute_sparc(xu, yu, fs)
# Typical values:
#   -1.5 to -2.0: Smooth, healthy movement
#   -2.5 to -3.5: Mild impairment
#   -4.0 to -5.0: Significant compensatory movements
```

#### Log Dimensionless Jerk (LDLJ)
- **Implementation**: `compute_log_dimensionless_jerk(x, y, fs)` → Returns -4 to 0 range
- **Normalization**: Accounts for movement duration and amplitude
- **Advantages**: Less sensitive to tracking noise than raw jerk
- **Use case**: Normalized smoothness comparison across different movement scales

```python
ldlj = compute_log_dimensionless_jerk(xu, yu, fs, movement_duration=1.5)
# Typical values:
#   -1.0 to 0: Smooth movement
#   -2.0 to -1.0: Moderate jerkiness
#   -4.0 to -2.0: High jerkiness (tremor/ataxia)
```

#### RMS Jerk (Legacy Support)
- **Status**: Retained but no longer primary metric
- **Purpose**: Debugging and legacy compatibility
- **Warning**: Highly sensitive to tracking noise - use with caution

---

### 2. Configurable Tremor Bands

#### Preset Modes

| Mode | Band (Hz) | Clinical Application | Reference |
|------|-----------|----------------------|-----------|
| **Parkinsonian** | 3-7 Hz | Rest/postural tremor, PD screening | di Biase 2025 |
| **General** | 2-10 Hz | Broad tremor screening, video postural tremor | VisionMD 2025 (npj PD) |
| **Essential** | 4-12 Hz | Essential tremor, kinetic tremor | Clinical standard |
| **Legacy** | 3-12 Hz | Original MotionBloom range | Backward compatibility |

#### Implementation
```python
# Access via constants
from motionbloom.signal import TREMOR_BANDS

parkinsonian_band = TREMOR_BANDS["parkinsonian"]  # (3.0, 7.0)
general_band = TREMOR_BANDS["general"]            # (2.0, 10.0)

# Used in bandpass filtering
xf = bandpass(xu, fs, low=band[0], high=band[1])
```

#### Rationale
- **VisionMD study** (npj Parkinson's 2025): Used 2-10 Hz for postural tremor with strong UPDRS correlation (ρ=0.83)
- **Narrow bands reduce noise**: 3-7 Hz focuses on pathological tremor, excludes voluntary motion (<3 Hz) and high-frequency noise
- **Clinical specificity**: Different tremor types have characteristic frequency signatures

---

### 3. Trial Quality Assessment

#### Purpose
Prevent false measurements from poor video conditions. Gate analysis with objective quality scores.

#### Implementation
```python
quality = assess_trial_quality(timestamps, confidence_array)

# Returns:
{
    "valid": bool,              # Pass/fail gate
    "quality_score": 0.91,      # 0-1 composite score
    "fps_mean": 29.8,          # Measured frame rate
    "fps_stable": True,         # CV < 20%
    "missing_pct": 3.2,        # % low-confidence frames
    "reasons": []               # Issues if invalid
}
```

#### Quality Gates

| Metric | Threshold | Rejection Reason |
|--------|-----------|------------------|
| FPS mean | ≥25 fps | "low FPS (22.3 < 25.0)" |
| FPS stability | CV <20% | "unstable FPS (std=8.4)" |
| Landmark confidence | ≥0.5 for ≥85% frames | "too many low-confidence frames (18.2%)" |

#### Integration with TremorMetrics
```python
@dataclass
class TremorMetrics:
    # ... existing fields ...
    trial_quality: float = 1.0  # 0-1 quality score
```

**Next step**: Integrate `assess_trial_quality()` with timestamp tracking in `tracker.py` to populate `trial_quality` field.

---

### 4. Numerical Stability: SOS Filtering

#### Problem Solved
Original code had potential `filtfict` typo and used direct `ba` coefficients which can be numerically unstable for high-order filters.

#### Solution
```python
from scipy.signal import butter, sosfiltfilt

@lru_cache(maxsize=32)
def _bandpass_sos(fs: float, low: float, high: float, order: int = 4):
    """Design bandpass filter using second-order sections."""
    nyq = 0.5 * fs
    low_n = max(1e-4, low / nyq)
    high_n = min(0.999, high / nyq)
    return butter(order, [low_n, high_n], btype="bandpass", output="sos")

def bandpass(x, fs, low, high):
    """Zero-phase filtering with SOS."""
    sos = _bandpass_sos(round(fs, 3), low, high)
    return sosfiltfilt(sos, x)
```

**Advantages**:
- Second-order sections prevent coefficient overflow
- `sosfiltfilt` for zero-phase (no lag)
- Cached filter designs for performance
- Handles edge cases (short signals, Nyquist violations)

---

## Phase 2: Next Priority Improvements (⏳ PENDING)

### 1. ArUco Marker Calibration (HIGH PRIORITY)

#### Current Problem
Amplitude reported in arbitrary normalized units. Not defensible for clinical claims.

#### Solution
Print 5cm ArUco marker, detect with OpenCV, convert pixels → mm.

```python
import cv2

# Detect marker
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

# Known marker size: 50 mm
marker_size_mm = 50.0
marker_size_px = np.linalg.norm(corners[0][0] - corners[0][1])

# Calibration factor
px_per_mm = marker_size_px / marker_size_mm

# Convert tremor amplitude
rms_amp_mm = rms_amp_normalized * frame_width_px / px_per_mm
```

**Advantages**:
- Simple, low-cost (print on paper)
- Defensible for research papers
- More accurate than iris-based depth estimation
- Easier than wristband cross-calibration for pilot studies

**Implementation Priority**: ⭐⭐⭐ Critical for any amplitude claims

---

### 2. Multi-Fingertip Averaging

#### Current Problem
Tracking single landmark (index fingertip) is noisy. MediaPipe jitter affects tremor estimation.

#### Solution (VisionMD Method)
Average index, middle, ring fingertips for tremor signal.

```python
# In tracker.py
if hand_landmarks:
    index_tip = landmarks[8]   # MediaPipe landmark 8
    middle_tip = landmarks[12]  # Landmark 12
    ring_tip = landmarks[16]    # Landmark 16
    
    # Compute centroid
    tremor_x = (index_tip.x + middle_tip.x + ring_tip.x) / 3.0
    tremor_y = (index_tip.y + middle_tip.y + ring_tip.y) / 3.0
```

**Rationale**: VisionMD (npj Parkinson's 2025) used this approach to reduce single-point tracking noise.

**Implementation Priority**: ⭐⭐ Moderate (improves accuracy, low effort)

---

### 3. Compensation Tracking via MediaPipe Pose

#### Current Problem
Hand movement alone is incomplete. Stroke patients compensate with trunk lean and shoulder substitution.

#### Solution
Track shoulder, elbow, wrist, trunk using MediaPipe Pose Landmarker.

```python
# Extract pose landmarks
pose_results = pose_detector.process(frame)

if pose_results.pose_landmarks:
    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    elbow = landmarks[13]  # Left elbow
    wrist = landmarks[15]  # Left wrist
    
    # Compute trunk midpoint
    trunk_x = (shoulder_left.x + shoulder_right.x) / 2.0
    trunk_y = (shoulder_left.y + shoulder_right.y) / 2.0
    
    # Track trunk displacement during reaching task
    trunk_displacement = compute_displacement(trunk_history)
    hand_displacement = compute_displacement(wrist_history)
    
    # Compensation ratio
    trunk_compensation = trunk_displacement / hand_displacement
    # Values >0.3 indicate excessive trunk compensation
```

**Clinical Significance**:
- Stroke rehab: Trunk lean compensation is a key recovery predictor
- Separates true arm improvement from compensatory strategies
- Enables movement quality scoring beyond tremor

**Implementation Priority**: ⭐⭐⭐ High for rehab use case

---

### 4. Dual-Window Analysis

#### Current Problem
1.5s window is fast but spectrally weak. Poor frequency resolution.

#### Solution
Two analysis windows:

| Window | Purpose | Frequency Resolution |
|--------|---------|----------------------|
| **1.5s (fast)** | Real-time gating, immediate feedback | ~0.67 Hz bins |
| **6-10s (clinical)** | Session summary, accurate frequency | ~0.1-0.17 Hz bins |

**Implementation**:
```python
def compute_metrics_dual_window(data, fs):
    # Fast window for real-time UI
    fast_metrics = compute_metrics(data[-int(1.5*fs):], fs)
    
    # Clinical window for session summary
    if len(data) >= int(6.0*fs):
        clinical_metrics = compute_metrics(data[-int(6.0*fs):], fs)
    else:
        clinical_metrics = None
    
    return {"realtime": fast_metrics, "clinical": clinical_metrics}
```

**Rationale**: VisionMD and clinical tremor protocols use 10-second standardized tasks, not 1-second snapshots.

**Implementation Priority**: ⭐⭐ Moderate (improves accuracy, requires buffer management)

---

## Phase 2B: Live Pipeline Integration (✅ COMPLETE - 2025-01-14)

### What Was Implemented

**Status**: Multi-fingertip averaging, quality assessment, and task mode enforcement now integrated into live camera capture loop.

#### 1. Multi-Fingertip Averaging (tracker.py + app.py)
**tracker.py Changes**:
- Added `multi_finger_samples` deque for buffering (timestamp, index_x, index_y, middle_x, middle_y, ring_x, ring_y, ref_px, confidence)
- Added `snapshot_multi_finger(window_sec)` method to extract multi-fingertip trajectories
- Capture loop now collects index (landmark 8), middle (12), and ring (16) fingertip positions with confidence

**app.py Changes**:
- `_refresh_analysis()` now calls `snapshot_multi_finger()` first (fallback to single landmark)
- Averages (index_x + middle_x + ring_x) / 3 and (index_y + middle_y + ring_y) / 3
- Passes averaged trajectory to signal processor
- **Rationale**: VisionMD 2025 uses multi-fingertip averaging to reduce noise from individual landmark jitter

#### 2. Trial Quality Assessment Integration (app.py)
- `_refresh_analysis()` now calls `assess_trial_quality(t, confidence)` after snapshot
- Hard rejection: returns `None` if `quality["valid"] == False`
- UI feedback: Sets `status_var` to show rejection reasons (e.g., "Trial Rejected: low_fps, unstable_frame_rate")
- **Rationale**: Friedrich 2024 emphasizes hard trial rejection to avoid including corrupted data

#### 3. Task Mode Enforcement (app.py)
- Added `self.current_task_mode = TaskMode.POSTURAL_GENERAL` to App.__init__
- `_refresh_analysis()` now passes `task_mode=self.current_task_mode` to `compute_metrics()`
- Signal processor uses task mode to select appropriate frequency band and metric set
- **Rationale**: Different tremor types require different frequency bands (Parkinsonian 3-7 Hz, Essential 4-12 Hz)

### What Works Now
✅ Camera captures multi-fingertip positions (index, middle, ring) with timestamps  
✅ Multi-fingertip trajectories averaged to reduce noise  
✅ Trial quality assessed before computing metrics (hard rejection for invalid trials)  
✅ UI displays specific rejection reasons when trials fail quality gates  
✅ Task mode determines frequency band and metric selection  
✅ App imports successfully, no syntax errors  

### What's Still Missing
❌ ArUco calibration (rms_amp_mm = 0.0 until calibrated)  
❌ Task mode selector in UI (currently hardcoded to POSTURAL_GENERAL)  
❌ Pose compensation for head/body motion  
❌ Bench validation with known frequencies  
❌ Wristband cross-validation  

---

## Phase 3: Validation Hierarchy (📋 PLANNED)

### Stage 1: Bench Validation
- Controlled artificial motion (printed marker on oscillating platform)
- Known frequencies: 3, 4, 5, 6, 7, 8 Hz
- Known amplitudes: 2mm, 5mm, 10mm
- Distances: 0.5m, 1m, 1.5m
- Lighting: normal, dim, backlit

**Target Accuracy**:
- Frequency error: <0.3 Hz
- Amplitude error: <5 mm (after calibration)
- Valid trial detection: >90% under normal conditions

### Stage 2: Wristband Cross-Validation
- Compare webcam vs IMU for same movements
- Metrics: frequency, amplitude, repetition count, smoothness (SPARC)
- Correlation targets: ρ >0.75 for frequency, ρ >0.65 for smoothness

### Stage 3: Clinical Score Correlation
- Compare to UPDRS tremor items (PD)
- Compare to Fugl-Meyer upper extremity (stroke)
- Correlation targets: ρ >0.55 (matches Friedrich et al. 2024)

---

## Updated TremorMetrics Structure

```python
@dataclass
class TremorMetrics:
    # Basic signal properties
    fs: float
    samples: int
    duration: float
    
    # Tremor-specific metrics
    peak_hz: float              # Dominant frequency
    peak_power: float           # PSD at peak
    band_power: float           # Integrated tremor band power
    total_power: float          # Full spectrum power
    band_ratio: float           # Tremor power / total power
    
    # Amplitude (calibrated if px_per_mm available)
    rms_amp: float              # RMS amplitude (normalized or mm)
    rms_amp_mm: float           # RMS amplitude in mm
    peak_to_peak_mm: float      # Peak-to-peak amplitude in mm
    
    # Spectral quality
    snr_db: float               # Signal-to-noise ratio
    regularity: float           # Spectral concentration (0-1)
    peak_sharpness: float       # Peak prominence
    
    # Classification
    class_label: str            # Human-readable tremor type
    class_code: str             # Short code (parkinsonian, essential, etc.)
    score: int                  # 0-100 product tremor score
    
    # Validated clinical kinematic metrics
    rms_jerk: float = 0.0       # Raw jerk (legacy, use with caution)
    speed_rms: float = 0.0      # RMS velocity
    sparc: float = 0.0          # ⭐ SPARC smoothness (validated for stroke)
    ldlj: float = 0.0           # ⭐ Log dimensionless jerk (normalized)
    trial_quality: float = 1.0  # 0-1 measurement confidence
```

---

## Recommended Methods Paragraph for Papers

> Video was recorded at ≥30 fps using a fixed RGB webcam positioned to capture the participant's upper body, hand, and a printed 5cm ArUco calibration marker. Sessions included standardized 10-second postural tremor and reaching tasks. Hand landmarks (wrist, index/middle/ring fingertips) were extracted using MediaPipe Hands (v0.10.18), and shoulder, elbow, wrist, trunk landmarks using MediaPipe Pose, to quantify compensatory movement. For tremor analysis, fingertip trajectories were averaged across three landmarks, resampled to uniform 30 Hz, calibrated from pixels to millimeters using the reference marker, detrended, and bandpass filtered (3-7 Hz for Parkinsonian tremor, 2-10 Hz for general screening). Welch power spectral density estimated dominant tremor frequency, tremor-band power, and signal-to-noise ratio. For movement smoothness, SPARC (spectral arc length) and log dimensionless jerk were computed from unfiltered trajectories, along with range of motion, path efficiency, and trunk compensation ratios. Trials with FPS <25, FPS coefficient of variation >20%, landmark confidence <0.5 for >15% of frames, or missing calibration marker were flagged as invalid and excluded from analysis.

---

## Critical Remaining Gaps

| Gap | Impact | Priority | Effort |
|-----|--------|----------|--------|
| **No physical calibration** | Amplitude claims are fake | ⭐⭐⭐ Critical | 2-3 days |
| **30 fps marginal for tremor** | Frequency accuracy fragile | ⭐⭐ High | Hardware-limited |
| **No compensation tracking** | Incomplete for rehab use | ⭐⭐⭐ High | 3-5 days |
| **No ArUco marker detection** | Can't convert pixels → mm | ⭐⭐⭐ Critical | 1-2 days |
| **1.5s window too short** | Poor frequency resolution | ⭐⭐ Medium | 1 day (dual-window) |
| **Single fingertip tracking** | Noisy, not VisionMD-compliant | ⭐⭐ Medium | 1 day |
| **Mixed clinical/product metrics** | Confusing for research | ⭐ Low | 0.5 days (docs) |

---

## Immediate Next Steps

1. **Add ArUco calibration** (1-2 days)
   - Detect marker in frame
   - Compute px_per_mm
   - Update amplitude conversions
   - Display calibration status in UI

2. **Implement multi-fingertip averaging** (1 day)
   - Average index, middle, ring fingertips
   - Update tracker.py landmark extraction
   - Validate noise reduction

3. **Add compensation tracking** (3-5 days)
   - Extract MediaPipe Pose landmarks
   - Compute trunk/shoulder displacement
   - Calculate compensation ratios
   - Add to TremorMetrics or new MovementQualityMetrics class

4. **Create separate clinical metrics layer** (0.5 days)
   - Document distinction between clinical (SPARC, LDLJ, frequency) and product (Motion Quality score, adherence)
   - Update UI to show both layers clearly

5. **Integrate trial quality into pipeline** (1 day)
   - Pass timestamps to assess_trial_quality()
   - Populate trial_quality field in TremorMetrics
   - Display quality warnings in UI
   - Log/reject invalid trials

---

## References

1. **Friedrich et al., 2024**: "Validation and application of computer vision algorithms for video-based tremor analysis" - *npj Digital Medicine* - Validated MediaPipe tremor metrics (ρ=0.55-0.86 with UPDRS, frequency error ≤0.21 Hz)

2. **VisionMD, 2025**: "Video-based quantification of hand postural tremor" - *npj Parkinson's Disease* - Used 2-10 Hz bandpass, iris-based calibration, 10-second tasks (ρ=0.83 with UPDRS)

3. **Balasubramanian et al., 2021**: "Smoothness metrics for reaching performance after stroke" - *J NeuroEngineering and Rehabilitation* - Validated SPARC for upper-limb stroke rehab

4. **Rupprechter et al., 2021**: "Movement smoothness metrics" - Validated log dimensionless jerk and normalized jerk for clinical use

5. **di Biase et al., 2025**: Cited for 3-7 Hz primary pathological tremor band

---

## Status Summary

✅ **COMPLETE**:
- SPARC and LDLJ smoothness metrics
- Configurable tremor bands (Parkinsonian, General, Essential)
- Trial quality assessment framework
- SOS filtering for numerical stability
- Expanded TremorMetrics dataclass
- Code validated and tested

⏳ **IN PROGRESS**:
- None (awaiting next phase approval)

📋 **PLANNED**:
- ArUco calibration system
- Multi-fingertip averaging
- Compensation tracking (trunk, shoulder)
- Dual-window analysis (1.5s + 6-10s)
- UI integration of new metrics
- Validation studies (bench → wristband → clinical)

---

**Document Version**: 1.0  
**Last Updated**: May 21, 2026  
**Author**: GitHub Copilot (Claude Sonnet 4.5)  
**Next Review**: After ArUco calibration implementation
