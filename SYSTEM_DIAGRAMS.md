# MotionBloom - System Diagrams & Visual Overview

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        MOTIONBLOOM SYSTEM                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INPUT LAYER                                                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Webcam (640×480, ~30 FPS)                                   │  │
│  └────────────────────┬────────────────────────────────────────┘  │
│                       │                                            │
│  VISION LAYER                                                      │
│  ┌────────────────────▼────────────────────────────────────────┐  │
│  │                  TremorTracker                               │  │
│  │  (Background Thread - Threading.Thread)                     │  │
│  │                                                              │  │
│  │  ├─ MediaPipe Hands                                         │  │
│  │  │  └─ 21 landmarks/hand (x, y, z, visibility)             │  │
│  │  │                                                          │  │
│  │  ├─ MediaPipe Pose (every 3rd frame)                       │  │
│  │  │  └─ 33 body landmarks (posture validation)              │  │
│  │  │                                                          │  │
│  │  └─ Circular Deque Buffering                               │  │
│  │     └─ 4096 samples (~136 sec @ 30 FPS)                    │  │
│  └────────────────────┬────────────────────────────────────────┘  │
│                       │                                            │
│  SIGNAL PROCESSING LAYER                                           │
│  ┌────────────────────▼────────────────────────────────────────┐  │
│  │          Signal Processing Pipeline (signal.py)             │  │
│  │                                                              │  │
│  │  [1] Window Extraction (4 seconds)                          │  │
│  │  ├─► t, x, y arrays                                         │  │
│  │                                                              │  │
│  │  [2] Resampling to Uniform Grid (30 Hz)                     │  │
│  │  ├─► Linear interpolation                                   │  │
│  │  ├─► Minimum 16 samples required                            │  │
│  │                                                              │  │
│  │  [3] Signal Conditioning                                    │  │
│  │  ├─► Detrend (linear regression)                            │  │
│  │  ├─► High-pass Filter (0.5 Hz, order 2)                    │  │
│  │  ├─► Band-pass Filter (3–12 Hz, order 4)                   │  │
│  │  ├─► Hanning Window                                         │  │
│  │                                                              │  │
│  │  [4] Spectral Analysis (Welch PSD)                          │  │
│  │  ├─► Hanning window, 50% overlap                            │  │
│  │  ├─► Frequency range: 0–15 Hz                               │  │
│  │                                                              │  │
│  │  [5] Metrics Extraction                                     │  │
│  │  ├─► Peak frequency (argmax in band)                        │  │
│  │  ├─► Power metrics (peak, band, total)                      │  │
│  │  ├─► SNR (dB)                                               │  │
│  │  ├─► Regularity (spectral entropy)                          │  │
│  │  ├─► Peak sharpness                                         │  │
│  │  ├─► RMS amplitude                                          │  │
│  │                                                              │  │
│  │  [6] Voluntary Motion Rejection                             │  │
│  │  ├─► Detect 0.3–2.5 Hz power surge                          │  │
│  │  ├─► Suppress score if dominant                             │  │
│  │                                                              │  │
│  │  [7] Adaptive Baseline                                      │  │
│  │  ├─► Track historical RMS amplitudes                        │  │
│  │  ├─► Compute personal threshold                             │  │
│  │                                                              │  │
│  │  [8] Classification (Frequency-based)                       │  │
│  │  ├─► 3–5 Hz   → Parkinsonian-like                           │  │
│  │  ├─► 5–8 Hz   → Essential-like                              │  │
│  │  ├─► 8–12 Hz  → Enhanced Physiological                      │  │
│  │                                                              │  │
│  │  [9] Scoring Algorithm                                      │  │
│  │  ├─► 40% band power contribution                            │  │
│  │  ├─► 30% SNR contribution                                   │  │
│  │  ├─► 20% sharpness contribution                             │  │
│  │  ├─► 10% regularity contribution                            │  │
│  │  └─► Output: 0–100 score                                    │  │
│  └────────────────────┬────────────────────────────────────────┘  │
│                       │                                            │
│  OUTPUT LAYER                                                      │
│  ┌────────────────────▼────────────────────────────────────────┐  │
│  │         TremorMetrics + UI Display (app.py)                 │  │
│  │                                                              │  │
│  │  Real-Time Display:                                         │  │
│  │  ├─ Tremor Score (0–100, color-coded)                       │  │
│  │  ├─ Peak Frequency (Hz)                                     │  │
│  │  ├─ Band Power (dB)                                         │  │
│  │  ├─ SNR (dB)                                                │  │
│  │  ├─ Classification (Parkinsonian/Essential/Physiological)   │  │
│  │  ├─ Sharpness & Regularity                                  │  │
│  │  ├─ PSD Plot                                                │  │
│  │  ├─ Spectrogram                                             │  │
│  │  ├─ History Chart                                           │  │
│  │  └─ Pass/Fail Indicator                                     │  │
│  │                                                              │  │
│  │  Optional: CSV Export                                       │  │
│  │  └─ Session data with all metrics                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Threading Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THREADING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MAIN THREAD (Tkinter UI)                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Tkinter event loop (mainloop)                              │  │
│  │ • UI updates (labels, plots)                                 │  │
│  │ • User input (button clicks, dropdown selections)            │  │
│  │ • Periodic signal processing call (every 200ms)              │  │
│  │ • Non-blocking metrics retrieval from tracker                │  │
│  └───────────────────────┬────────────────────────────────────┬─┘  │
│                          │                                    │    │
│                    (lock access)                   (lock access)  │
│                          │                                    │    │
│  TRACKER THREAD (Background, daemon=True)                     │    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Continuous camera capture (VideoCapture.read)              │  │
│  │ • MediaPipe Hands detection (every frame)                    │  │
│  │ • MediaPipe Pose detection (every 3rd frame)                 │  │
│  │ • Landmark extraction & storage in deque                     │  │
│  │ • Thread-safe updates (locks for frame/pose)                 │  │
│  │ • ~30 FPS operation (camera-dependent)                       │  │
│  │                                                              │  │
│  │ Synchronization:                                             │  │
│  │ ├─ self._frame_lock (latest RGB frame)                       │  │
│  │ ├─ self._pose_lock (latest pose snapshot)                    │  │
│  │ └─ self.samples (thread-safe deque)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Data Flow:                                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Tracker Thread                                              │  │
│  │   └─► Deque: [(t1, x1, y1, vis1), ..., (tN, xN, yN, visN)] │  │
│  │       (auto-limited to 4096 items)                          │  │
│  │                                                              │  │
│  │ Main Thread (every 200ms)                                   │  │
│  │   └─► tracker.snapshot(4.0 seconds)                         │  │
│  │       └─► Signal processing                                 │  │
│  │           └─► Update UI display                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Signal Processing Pipeline (Detailed)

```
┌────────────────────────────────────────────────────────────────┐
│               SIGNAL PROCESSING PIPELINE                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUT: Raw Hand Trajectory                                   │
│  ├─ Source: Circular deque (TremorTracker.samples)            │
│  ├─ Format: [(time, x_norm, y_norm, visibility), ...]        │
│  ├─ Duration: Last 4 seconds                                  │
│  └─ Irregularity: Timestamps may have jitter                  │
│                                                                │
│  ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 1] WINDOWING & EXTRACTION                          │ │
│  │ • Extract 4-second window from deque                      │ │
│  │ • Minimum 16 samples required for analysis                │ │
│  │ • Output: t[], x[], y[] numpy arrays                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 2] RESAMPLING TO UNIFORM GRID                      │ │
│  │ • Input: Irregular timestamps (±jitter from MediaPipe)   │ │
│  │ • Method: Linear interpolation onto uniform 30 Hz grid   │ │
│  │ • Output: t_uniform[], x_resampled[], y_resampled[]      │ │
│  │ • Minimum length: 16 samples → Maximum duration: 4 sec   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 3] SIGNAL CONDITIONING                             │ │
│  │                                                           │ │
│  │  3a. DETRENDING                                          │ │
│  │  ├─ Method: Linear regression (polyfit, order 1)         │ │
│  │  ├─ Purpose: Remove slow baseline shift                  │ │
│  │  └─ Output: Detrended x[], y[]                           │ │
│  │                                                           │ │
│  │  3b. HIGH-PASS FILTER (Butterworth, order 2)             │ │
│  │  ├─ Cutoff: 0.5 Hz                                       │ │
│  │  ├─ Purpose: Remove DC offset & very low drift           │ │
│  │  ├─ Design: Cached via @lru_cache                        │ │
│  │  └─ Output: High-passed x[], y[]                         │ │
│  │                                                           │ │
│  │  3c. BAND-PASS FILTER (Butterworth, order 4)             │ │
│  │  ├─ Band: 3–12 Hz (tremor band)                          │ │
│  │ │ ├─ Design: Cached via @lru_cache                        │ │
│  │  ├─ Method: Second-Order Sections (SOS) + sosfiltfilt    │ │
│  │  ├─ Purpose: Isolate tremor oscillations                 │ │
│  │  └─ Output: Filtered x_f[], y_f[]                        │ │
│  │                                                           │ │
│  │  3d. HANNING WINDOW (optional, for spectral)             │ │
│  │  ├─ Window: np.hanning(n)                                │ │
│  │  ├─ Purpose: Reduce spectral leakage in FFT              │ │
│  │  └─ Output: Windowed x_w[], y_w[]                        │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 4] SPECTRAL ANALYSIS (Welch PSD)                   │ │
│  │                                                           │ │
│  │  • Input: Filtered x_f[], y_f[]                          │ │
│  │  • Method: scipy.signal.welch                            │ │
│  │  • Parameters:                                            │ │
│  │    - Window: Hanning (default)                           │ │
│  │    - Segment length: max(64, fs*2) samples              │ │
│  │    - Overlap: 50% (default)                              │ │
│  │    - Detrending: None (already detrended)                │ │
│  │  • Computation:                                          │ │
│  │    - fxx, pxx = welch(x_f, fs=30)                        │ │
│  │    - _, pyy = welch(y_f, fs=30)                          │ │
│  │    - psd = pxx + pyy  (combined magnitude)               │ │
│  │                                                           │ │
│  │  • Output:                                               │ │
│  │    - fxx: Frequency array [0, 0.25, 0.5, ..., 15] Hz    │ │
│  │    - psd: Power array (matching length)                  │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 5] FEATURE EXTRACTION (13 Metrics)                 │ │
│  │                                                           │ │
│  │  5a. FREQUENCY DOMAIN METRICS                            │ │
│  │  ├─ Peak Frequency: argmax(psd) in 3–12 Hz band          │ │
│  │  ├─ Peak Power: psd[peak_idx] (dB scale)                 │ │
│  │  ├─ Band Power: integral(psd) from 3–12 Hz               │ │
│  │  ├─ Total Power: integral(psd) from 0.5–fs/2             │ │
│  │  ├─ Band Ratio: band_power / total_power                 │ │
│  │  ├─ SNR (dB): 10*log10(peak_power / median_off_band)     │ │
│  │  └─ Peak Sharpness: peak / mean(±1 Hz window)            │ │
│  │                                                           │ │
│  │  5b. TIME DOMAIN METRICS                                 │ │
│  │  ├─ RMS Amplitude: sqrt(mean(mag²))                      │ │
│  │  ├─ RMS in mm: RMS * (hand_width_mm / hand_ref_pixels)   │ │
│  │  └─ Peak-to-peak: max(mag) - min(mag)                    │ │
│  │                                                           │ │
│  │  5c. QUALITY METRICS                                     │ │
│  │  ├─ Regularity: 1 - (entropy / max_entropy) [0..1]       │ │
│  │  │  └─ Shannon entropy of normalized PSD in band          │ │
│  │  └─ Duration: t[-1] - t[0] (seconds)                     │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 6] CLASSIFICATION (Frequency-based)                │ │
│  │                                                           │ │
│  │  IF 3.0 ≤ peak_hz < 5.0:                                 │ │
│  │      label = "Parkinsonian-like (rest)"                  │ │
│  │      code = "parkinsonian"                               │ │
│  │                                                           │ │
│  │  ELSE IF 5.0 ≤ peak_hz < 8.0:                            │ │
│  │      label = "Essential-like (postural/kinetic)"         │ │
│  │      code = "essential"                                  │ │
│  │                                                           │ │
│  │  ELSE IF 8.0 ≤ peak_hz ≤ 12.0:                           │ │
│  │      label = "Enhanced physiological"                    │ │
│  │      code = "physiological"                              │ │
│  │                                                           │ │
│  │  ELSE:                                                    │ │
│  │      label = "Out of band" / "Unclassified"              │ │
│  │      code = "other" / "none"                             │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 7] VOLUNTARY MOTION REJECTION                      │ │
│  │                                                           │ │
│  │  • Check low-frequency power: 0.3–2.5 Hz band            │ │
│  │  • If low_power > 0.5 * band_power:                       │ │
│  │      Classification: "Voluntary motion detected"          │ │
│  │      Action: Suppress tremor score or skip update         │ │
│  │  • Purpose: Prevent false positives from arm raises       │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 8] ADAPTIVE BASELINE LEARNING                      │ │
│  │                                                           │ │
│  │  • Update running average of RMS amplitudes               │ │
│  │  • Window size: Last 60 samples (~12 seconds)             │ │
│  │  • Compute personal baseline:                             │ │
│  │      baseline_rms = mean(history)                         │ │
│  │  • Compute threshold:                                     │ │
│  │      threshold = baseline + 8 * std(history)              │ │
│  │  • Decision: PASS if rms_amp < threshold                  │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ [STEP 9] TREMOR SCORING (0–100)                          │ │
│  │                                                           │ │
│  │  score = (                                                │ │
│  │    0.40 * power_score +     // 40% band power             │ │
│  │    0.30 * snr_score +       // 30% SNR                    │ │
│  │    0.20 * sharpness_score + // 20% peak sharpness        │ │
│  │    0.10 * regularity_score  // 10% spectral regularity    │ │
│  │  )                                                         │ │
│  │                                                           │ │
│  │  Clipped to [0, 100]                                      │ │
│  │                                                           │ │
│  │  Interpretation:                                          │ │
│  │  ├─ 0–20:   Minimal tremor                                │ │
│  │  ├─ 21–50:  Mild to moderate                              │ │
│  │  ├─ 51–80:  Moderate to severe                            │ │
│  │  └─ 81–100: Severe tremor                                 │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          ▼                                     │
│  OUTPUT: TremorMetrics Dataclass                              │
│  ├─ fs: 30.0 (Hz)                                            │
│  ├─ samples: 120 (count)                                     │
│  ├─ duration: 4.0 (sec)                                      │
│  ├─ peak_hz: 6.3 (Hz)                                        │
│  ├─ peak_power: 0.045 (linear)                               │
│  ├─ band_power: 0.087 (linear)                               │
│  ├─ band_ratio: 0.73 (unitless)                              │
│  ├─ rms_amp: 0.012 (normalized)                              │
│  ├─ rms_amp_mm: 4.2 (mm)                                     │
│  ├─ snr_db: 8.3 (dB)                                         │
│  ├─ regularity: 0.68 (unitless, 0..1)                        │
│  ├─ peak_sharpness: 2.1 (ratio)                              │
│  ├─ class_label: "Essential-like (postural/kinetic)"         │
│  ├─ class_code: "essential"                                  │
│  ├─ score: 45 (0..100)                                       │
│  └─ passes: True (adaptive baseline check)                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Tremor Classification Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TREMOR CLASSIFICATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┬──────────────────┬──────────────────────┐    │
│  │  PARKINSONIAN    │   ESSENTIAL      │   PHYSIOLOGICAL      │    │
│  │     (Rest)       │   (Postural)     │   (Normal/Light)     │    │
│  ├──────────────────┼──────────────────┼──────────────────────┤    │
│  │  3–5 Hz          │  5–8 Hz          │  8–12 Hz             │    │
│  ├──────────────────┼──────────────────┼──────────────────────┤    │
│  │  Code:           │  Code:           │  Code:               │    │
│  │  "parkinsonian"  │  "essential"     │  "physiological"     │    │
│  ├──────────────────┼──────────────────┼──────────────────────┤    │
│  │  Typical:        │  Typical:        │  Typical:            │    │
│  │  • Resting hand  │  • Holding       │  • Light hand hold   │    │
│  │  • At rest       │  • Against       │  • Normal muscle     │    │
│  │  • Low motion    │    gravity       │    activity          │    │
│  │                  │  • With action   │  • Stress/anxiety    │    │
│  ├──────────────────┼──────────────────┼──────────────────────┤    │
│  │  Medical         │  Medical         │  Medical             │    │
│  │  Association:    │  Association:    │  Association:        │    │
│  │  • Parkinson's   │  • Essential     │  • Normal variation  │    │
│  │    disease       │    tremor        │  • Stress tremor     │    │
│  │  • Other        │  • Family hx      │  • Hyperthyroidism   │    │
│  │    neurological  │                  │  • Caffeine/drugs    │    │
│  │    conditions    │                  │                      │    │
│  └──────────────────┴──────────────────┴──────────────────────┘    │
│                                                                     │
│  DISCLAIMER: Classification is INFORMATIONAL ONLY                  │
│  ⚠️  NOT a medical diagnosis                                        │
│  ⚠️  NOT a diagnostic tool                                          │
│  ⚠️  Consult healthcare providers for clinical decisions            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure: TremorMetrics

```python
@dataclass
class TremorMetrics:
    # METADATA
    fs: float = 30.0                    # Sampling rate (Hz)
    samples: int = 120                  # Number of samples analyzed
    duration: float = 4.0               # Analysis window (seconds)
    
    # FREQUENCY DOMAIN FEATURES (from Welch PSD)
    peak_hz: float = 6.3                # Dominant tremor frequency
    peak_power: float = 0.045           # Power at peak frequency
    band_power: float = 0.087           # Integrated power (3–12 Hz band)
    total_power: float = 0.119          # Total power (0.5–fs/2 Hz)
    band_ratio: float = 0.731           # band_power / total_power
    
    # AMPLITUDE FEATURES
    rms_amp: float = 0.0120             # RMS amplitude (normalized)
    rms_amp_mm: float = 4.2             # RMS amplitude (millimeters)
    peak_to_peak_mm: float = 9.8        # Peak-to-peak amplitude (mm)
    
    # QUALITY METRICS
    snr_db: float = 8.3                 # Signal-to-noise ratio (dB)
    regularity: float = 0.682           # Spectral concentration (0–1)
    peak_sharpness: float = 2.14        # Peak / mean (±1 Hz)
    
    # CLASSIFICATION
    class_label: str = "Essential-like" # Human-readable classification
    class_code: str = "essential"       # Short code
    
    # OVERALL SCORE
    score: int = 45                     # Tremor severity (0–100)
```

---

## Frequency Response: Filter Chains

```
Band-Pass (3–12 Hz, 4th order Butterworth)
──────────────────────────────────────────

     Magnitude Response (dB)
     │
     │     ┌─────────────────────┐
     │    /                       \
   0 │───/─────────────────────────\───
     │  /                           \
  -6 │ /                             \
     │/                               \
-12 │                                 ────────
     │
     └─────────────────────────────────────► Frequency (Hz)
       0     3      6      9     12     15
            ├───── passband─────┤


High-Pass (0.5 Hz, 2nd order Butterworth)
─────────────────────────────────────────

     Magnitude Response (dB)
     │
     │                               ──────
     │                            ───
   0 │────────────────────────────
     │         \
  -6 │          \
     │           \
-12 │            ────────
     │
     └─────────────────────────────► Frequency (Hz)
       0.1  0.5   1     2     5     10
            ↑ cutoff


Combined Pipeline: Detrend → High-Pass → Band-Pass
──────────────────────────────────────────────────

     │ Original Signal
     │ ├─ DC offset    ──→ High-Pass removes
     │ ├─ Slow drift   ──→ Detrending removes
     │ ├─ Low freq     ──→ High-Pass removes
     │ ├─ 3–12 Hz      ──→ Band-Pass passes (tremor band)
     │ └─ High freq    ──→ Band-Pass removes
     │
     └─► Filtered Signal: Only tremor oscillations (3–12 Hz)
```

---

**Last Updated:** May 21, 2026
