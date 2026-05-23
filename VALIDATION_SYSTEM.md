# 2-Layer Validation System for Tremor Scoring

## Overview

The MotionBloom app now implements a **2-layer validation system** that separates **quality assessment** from **motion classification**. This provides transparent debug feedback while maintaining strict research-valid tremor scoring.

## Design Principle

> **"Show more diagnostics, not more fake scores"**

Instead of loosening research thresholds to show more scores, we:
1. **Keep strict research thresholds unchanged** (MIN_PEAK_PROMINENCE=3.0, MIN_BAND_POWER_RATIO=0.25)
2. **Always compute motion energy and candidate scores** for debug visibility
3. **Gate final tremor score** behind both quality AND motion validation
4. **Display transparent reasons** when scores are suppressed

## Architecture

### Layer 1: Quality Assessment
Evaluates recording quality (FPS stability, gaps, confidence):

**Status:** `valid | low_quality | invalid`

**Thresholds:**
- `FPS_CV_WARN_THRESHOLD = 0.25` → low_quality (mild FPS instability)
- `FPS_CV_INVALID_THRESHOLD = 0.45` → invalid (severe FPS instability)
- `MAX_LONG_GAP_SEC = 0.8` → invalid (long tracking gaps)
- `MAX_MISSING_FRAMES_PCT = 20` → low_quality (too many low-confidence frames)

**Implementation:** `assess_trial_quality()` in signal.py returns:
- `quality_status`: "valid" | "low_quality" | "invalid"
- `fps_cv`: Coefficient of variation (fps_std / fps_mean)
- `reasons`: List of quality issues

### Layer 2: Motion Classification
Classifies motion type (tremor vs gross motion vs noise):

**Status:** `valid_tremor | gross_translation | gross_motion | uncertain | tracking_unstable | low_frequency | high_frequency_noise`

**Thresholds (STRICT - unchanged):**
- `MIN_PEAK_PROMINENCE = 3.0` → peak must be 3× median PSD
- `MIN_BAND_POWER_RATIO = 0.25` → 25% of power in tremor band
- `HIGH_VELOCITY_P95_THRESHOLD = 0.30` → gross motion detection
- `CENTER_DRIFT_THRESHOLD = 2.0` → translation detection

**Implementation:** `classify_motion_type()` in signal.py checks:
- Peak prominence (sharp spectral peak required)
- Band power ratio (25% in tremor band required)
- Velocity (p95 < 0.30 required)
- Center drift (< 2.0 normalized units required)

### Final Gate: research_valid

```python
research_valid = (quality_status == "valid") AND (motion_classification == "valid_tremor")
```

**Only when both layers pass:**
- `tremor_score` is set to computed score (int 0-100)
- Full tremor metrics displayed (amplitude_mm, peak_to_peak_mm, classification)

**When either layer fails:**
- `tremor_score = None` (suppressed)
- Score display shows "—"
- But debug metrics still visible:
  - `raw_motion_score` (0-100 motion energy)
  - `debug_tremor_candidate_score` (what score would be if gates open)
  - Classification features (prominence, band_ratio, velocity, path_ratio, center_drift, fps_cv)

## TremorMetrics Dataclass Structure

```python
@dataclass
class TremorMetrics:
    # === QUALITY LAYER ===
    quality_status: str        # "valid" | "low_quality" | "invalid"
    quality_reason: str        # Human-readable quality issues
    fps_cv: float             # Coefficient of variation
    
    # === MOTION LAYER ===
    motion_classification: str # "valid_tremor" | "gross_translation" | ...
    motion_reason: str        # Human-readable motion rejection reason
    
    # === FINAL OUTPUT (gated by both layers) ===
    tremor_score: Optional[int]  # None unless research_valid=True
    research_valid: bool         # True only when both layers pass
    
    # === DEBUG SCORES (always visible) ===
    raw_motion_score: float           # Motion energy 0-100
    debug_tremor_candidate_score: float  # What score would be if gates open
    
    # === CLASSIFICATION FEATURES ===
    peak_prominence: float      # Peak PSD / median PSD
    center_drift: float        # Translation magnitude
    velocity_p95: float        # 95th percentile velocity
    path_ratio: float         # Straight-line efficiency
    
    # === SPECTRAL FEATURES ===
    peak_hz: float
    band_power: float
    band_ratio: float
    peak_sharpness: float
    # ... (other spectral metrics)
    
    # === AMPLITUDE FEATURES ===
    rms_amp: float
    rms_amp_mm: float         # Only non-zero when research_valid
    peak_to_peak_mm: float    # Only non-zero when research_valid
    
    # === LEGACY FIELDS (backwards compatibility) ===
    score: int = 0            # DO NOT use for UI truth!
```

## UI Display Logic

### Main Score Display
```
IF research_valid AND tremor_score is not None:
    Display: "42 / 100"
ELSE:
    Display: "—"
```

### Tremor Candidate Display (Debug Section)
```
IF research_valid:
    Display: "42 / 100"
ELSE:
    Display: "61 / 100 (not research-valid)"
```

**Key insight:** Gross translation can have **high candidate score** (e.g., 61/100) and still be correctly rejected. This is transparent and defensible.

### Status Message Priority
1. **Invalid quality** → "Invalid quality: severely unstable FPS (cv=0.50)"
2. **Low quality** → "Low quality: unstable FPS (cv=0.32)" (show candidate, warn user)
3. **Motion rejection** → "Hold hand still — center drifted (gross translation)"
4. **Valid tremor** → "Multi-Fingertip | Resting | 3-12Hz"

### Debug Metrics (Always Visible)
- **Motion energy:** Raw motion 0-100 (amplitude + velocity + power)
- **Tremor candidate:** What tremor score would be (with "not research-valid" label if suppressed)
- **Peak prominence:** Ratio of peak PSD to median PSD (need ≥3.0 for tremor)
- **FPS CV:** Coefficient of variation (need <0.25 for valid quality)
- **Velocity (p95):** 95th percentile velocity (need <0.30 for tremor)
- **Path ratio:** Straight-line efficiency (need <3.0 for tremor)
- **Center drift:** Translation magnitude (need <2.0 for tremor)

## Example Scenarios

### Scenario 1: Valid Tremor (5 Hz clean)
```
quality_status = "valid"
motion_classification = "valid_tremor"
research_valid = True
tremor_score = 42

Display:
  Score: "42 / 100"
  Candidate: "42 / 100"
  Status: "Multi-Fingertip | Resting | 3-12Hz"
  Motion energy: 38
  Prominence: 4.2×
  FPS CV: 0.18
```

### Scenario 2: Gross Translation (moving hand)
```
quality_status = "valid"
motion_classification = "gross_translation"
research_valid = False
tremor_score = None
debug_tremor_candidate_score = 61
center_drift = 3.5

Display:
  Score: "—"
  Candidate: "61 / 100 (not research-valid)"
  Status: "Hold hand still — center drifted (gross translation)"
  Motion energy: 58
  Prominence: 2.1×
  Center drift: 3.5000
```

**Key insight:** High motion energy (58) and high candidate score (61) don't mean tremor! The **center drift (3.5 > 2.0)** correctly rejects this as gross translation. Users see WHY the score is suppressed.

### Scenario 3: Weak Periodicity (uncertain)
```
quality_status = "valid"
motion_classification = "uncertain"
research_valid = False
tremor_score = None
debug_tremor_candidate_score = 18
peak_prominence = 1.8

Display:
  Score: "—"
  Candidate: "18 / 100 (not research-valid)"
  Status: "Uncertain: Weak periodicity (prominence 1.8 < 3.0)"
  Motion energy: 22
  Prominence: 1.8×
  Band ratio: 18%
```

**Key insight:** Low prominence (1.8 < 3.0) means no sharp spectral peak. This is correctly classified as "uncertain" rather than tremor.

### Scenario 4: Low Quality FPS (cv=0.32)
```
quality_status = "low_quality"
motion_classification = "valid_tremor"
research_valid = False
tremor_score = None
debug_tremor_candidate_score = 45
fps_cv = 0.32

Display:
  Score: "—"
  Candidate: "45 / 100 (not research-valid)"
  Status: "Low quality: unstable FPS (cv=0.32)"
  Motion energy: 41
  Prominence: 3.8×
  FPS CV: 0.320
```

**Key insight:** Motion is valid tremor (prominence 3.8, good band ratio), but **FPS instability (cv=0.32 > 0.25)** makes recording low quality. Quality layer rejection is separate from motion layer.

### Scenario 5: Invalid FPS (cv=0.50)
```
quality_status = "invalid"
motion_classification = "valid_tremor"
research_valid = False
tremor_score = None
fps_cv = 0.50

Display:
  Score: "—"
  Candidate: "—" (not computed for invalid)
  Status: "Invalid quality: severely unstable FPS (cv=0.50)"
  Motion energy: 35
  FPS CV: 0.500
```

## Code Flow

### signal.py
1. `compute_metrics()`:
   - Computes all spectral/amplitude features
   - Calls `classify_motion_type()` → motion_classification
   - Calls `compute_raw_motion_score()` → raw_motion_score (ALWAYS)
   - Stores pre-gated score → debug_tremor_candidate_score
   - Gates tremor_score based on motion_classification:
     - If valid_tremor: tremor_score = score (int)
     - Else: tremor_score = None
   - Returns TremorMetrics with placeholder quality_status="valid"

2. `assess_trial_quality()`:
   - Calculates fps_cv = fps_std / fps_mean
   - Determines quality_status based on thresholds:
     - fps_cv ≥ 0.45 → "invalid"
     - fps_cv ≥ 0.25 → "low_quality"
     - else → "valid"
   - Returns dict with quality_status, fps_cv, reasons

### app.py
1. `_refresh_analysis()`:
   - Calls `assess_trial_quality()` → gets quality_status, fps_cv
   - Calls `compute_metrics()` → gets metrics with placeholder quality
   - **Overrides** metrics.quality_status, metrics.fps_cv with real values
   - Computes `research_valid = (quality_status == "valid" AND motion_classification == "valid_tremor")`
   - Gates metrics.tremor_score:
     - If not research_valid: tremor_score = None
   - Updates UI based on priority:
     1. Invalid quality → show reason, suppress score
     2. Low quality → show candidate with warning
     3. Motion rejection → show reason, suppress score
     4. Valid tremor → show full metrics

2. Display logic:
   - **Score:** tremor_score if research_valid else "—"
   - **Candidate:** tremor_score if research_valid else "candidate (not research-valid)"
   - **Debug metrics:** ALWAYS visible (motion_energy, prominence, fps_cv, velocity, etc.)

## Testing Checklist

- [x] Valid tremor (5 Hz clean) → shows final tremor_score
- [ ] Gross translation (moving hand) → shows "—" for score, high candidate_score, "Hold hand still — center drifted"
- [ ] Weak periodicity → shows "—" for score, low candidate_score, "Uncertain: Weak periodicity (prominence 1.8)"
- [ ] Low quality FPS (cv=0.32) → shows "—" for score, candidate visible, "Low quality: unstable FPS"
- [ ] Invalid FPS (cv=0.50) → shows "—" for score, "Invalid: severely unstable FPS"
- [ ] Verify MIN_PEAK_PROMINENCE=3.0 unchanged
- [ ] Verify MIN_BAND_POWER_RATIO=0.25 unchanged
- [ ] Verify debug metrics always visible (motion_energy, prominence, fps_cv, velocity, path_ratio, center_drift)

## Philosophy

### Original Problem
"Number keeps on disappearing and there is high delay"

### Bad Solution (Rejected)
Loosen thresholds globally:
- MIN_PEAK_PROMINENCE: 3.0 → 2.0
- MIN_BAND_POWER_RATIO: 0.25 → 0.15

**User rejection:** "That's a demo hack, not a research fix. It brings back the original problem: hand translation scoring as tremor."

### Correct Solution (Implemented)
Add transparent debug layer while keeping strict research gates:
- **Keep strict thresholds** (MIN_PEAK_PROMINENCE=3.0, MIN_BAND_POWER_RATIO=0.25)
- **Always compute debug scores** (raw_motion_score, debug_tremor_candidate_score)
- **Show classification features** (prominence, band_ratio, velocity, path_ratio, center_drift, fps_cv)
- **Gate final tremor_score** behind both quality AND motion validation
- **Display "Tremor Candidate: X / 100 (not research-valid)"** for rejected motion

### Key Principles
1. **"More transparency, not looser truth"**
2. **"Final tremor score should be hard to earn. Raw motion feedback can be easy to show."**
3. **Gross translation with high candidate score (61/100) is correctly rejected** → transparent and defensible
4. **Quality layer (FPS/gaps) separate from motion layer (tremor vs gross motion)** → cleaner architecture
5. **Optional[int] for tremor_score** → None is clearer than score=0 default

## Files Modified

### signal.py
- **Lines 43-47:** Added FPS_CV_WARN_THRESHOLD=0.25, FPS_CV_INVALID_THRESHOLD=0.45
- **Lines 49-62:** Marked all research thresholds as "(STRICT for research validity)"
- **Lines 437-501:** Updated assess_trial_quality() to return quality_status and fps_cv
- **Lines 576-654:** Restructured TremorMetrics dataclass with 2-layer validation fields
- **Lines 676-695:** Added compute_raw_motion_score() helper function
- **Lines 950-1020:** Updated compute_metrics() with 2-layer gating logic

### app.py
- **Lines 568-595:** Added debug metric variables (motion_energy, tremor_candidate, prominence, fps_cv, velocity, path_ratio, center_drift)
- **Lines 596-621:** Added debug section separator and display grid
- **Lines 1096-1120:** Updated quality assessment to use quality_status from assess_trial_quality
- **Lines 1154-1280:** Updated _refresh_analysis() display logic:
  - Override metrics.quality_status, quality_reason, fps_cv from assess_trial_quality
  - Calculate final research_valid = (quality_status == "valid" AND motion_classification == "valid_tremor")
  - Display tremor_score if research_valid, else "—"
  - Display "Tremor Candidate: X / 100 (not research-valid)" when rejected
  - Status message priority: invalid → low_quality → motion rejection → valid tremor
  - Show all debug metrics
