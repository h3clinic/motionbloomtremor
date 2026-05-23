# MotionBloom Analysis Summary

## Executive Summary

**MotionBloom** is a **real-time hand tremor detection system** that analyzes webcam video to measure and classify tremor characteristics. The system uses Google's lightweight MediaPipe models for vision and applies sophisticated signal processing to extract tremor metrics.

---

## Current Models & CV Pipeline

### **Vision Models** (Pre-trained, included in mediapipe v0.10.18)

#### **1. MediaPipe Hands**
- **Model Type:** Hand landmark detector
- **Complexity:** 0 (lightweight)
- **Outputs:** 21 3D landmarks per hand
- **Frequency:** ~30 FPS
- **Use Case:** Tracking hand position/tremor

#### **2. MediaPipe Pose** 
- **Model Type:** Full-body pose estimator
- **Complexity:** 1
- **Outputs:** 33 body landmarks
- **Frequency:** Every 3rd frame (~10 FPS, optimized)
- **Use Case:** Posture validation, voluntary motion detection

**No custom ML models.** Both use Google's pre-trained weights bundled with the mediapipe package.

---

## Signal Processing Pipeline (13-Stage)

### **Abbreviated Flow**
```
Webcam Video → MediaPipe Hand Detection → Landmark Buffering → 
Resampling (30 Hz uniform) → Signal Conditioning (detrend, high-pass, band-pass) → 
Welch PSD → Feature Extraction (13 metrics) → Voluntary Motion Rejection → 
Adaptive Baseline → Tremor Scoring (0–100) → UI Display & Logging
```

### **Core Processing Steps**

| Stage | Function | Algorithm | Purpose |
|-------|----------|-----------|---------|
| 1 | **Video Capture** | OpenCV | 640×480, ~30 FPS |
| 2 | **Hand Detection** | MediaPipe Hands | 21 landmarks/hand |
| 3 | **Pose Tracking** | MediaPipe Pose | Body landmarks |
| 4 | **Buffering** | Circular deque | 4096 samples (~136 sec) |
| 5 | **Grip Detection** | Geometric ratio | 0–1 curl strength |
| 6 | **Resampling** | Linear interpolation | Uniform 30 Hz grid |
| 7 | **Conditioning** | Butterworth filters | Detrend + high-pass + band-pass |
| 8 | **Spectral Analysis** | Welch PSD | Frequency domain power |
| 9 | **Metrics** | Feature extraction | 13 tremor characteristics |
| 10 | **Motion Rejection** | Low-freq detection | Suppress arm raises |
| 11 | **Baseline Learning** | Running average | Personalized threshold |
| 12 | **Scoring** | Weighted combination | 0–100 severity |
| 13 | **UI/Logging** | Tkinter + CSV | Display & export |

---

## Tremor Metrics (13 Extracted)

### **Frequency Domain**
| Metric | Range | Computed Via |
|--------|-------|--------------|
| **peak_hz** | 1–15 Hz | Welch PSD argmax |
| **peak_power** | dB | Welch PSD value at peak |
| **band_power** | dB | Integrated PSD (3–12 Hz) |
| **band_ratio** | 0–1 | band_power / total_power |
| **snr_db** | dB | peak_power / median(off-band) |
| **peak_sharpness** | ratio | peak / mean(±1 Hz window) |

### **Time Domain**
| Metric | Range | Computed Via |
|--------|-------|--------------|
| **rms_amp** | normalized | RMS of bandpassed signal |
| **rms_amp_mm** | mm | RMS × (hand_width_mm / hand_ref_pixels) |
| **peak_to_peak_mm** | mm | Envelope peak-to-peak |

### **Quality Metrics**
| Metric | Range | Computed Via |
|--------|-------|--------------|
| **regularity** | 0–1 | 1 - (spectral entropy / max entropy) |
| **duration** | seconds | Analysis window length |
| **samples** | count | # of samples processed |
| **fs** | Hz | Sampling frequency |

### **Classification**
| Metric | Values | Method |
|--------|--------|--------|
| **class_label** | 3 options | Frequency-based |
| **class_code** | 3 options | Short form |
| **score** | 0–100 | Weighted algorithm |

---

## Tremor Classification (Current)

### **3 Types (Frequency-Based)**

```
PARKINSONIAN-LIKE          ESSENTIAL-LIKE            PHYSIOLOGICAL
(Rest Tremor)              (Postural/Kinetic)        (Enhanced Normal)
     3–5 Hz                    5–8 Hz                   8–12 Hz
   Parkinsonian              Essential Tremor          Stress/Anxiety
   Disease-like              -like                      -induced
```

**Each class has:**
- Human-readable label
- Short code (parkinsonian, essential, physiological)
- Frequency range
- Clinical association (informational only, not diagnostic)

---

## Scoring Algorithm (0–100)

### **Weighted Feature Combination**
```python
score = (
    0.40 * (band_power / max_reference) * 100 +
    0.30 * (snr_db / 30) * 100 +
    0.20 * (peak_sharpness) * 100 +
    0.10 * (regularity) * 100
)
```

### **Interpretation**
- **0–20:** Minimal tremor
- **21–50:** Mild to moderate
- **51–80:** Moderate to severe
- **81–100:** Severe tremor

---

## Key Innovations

### **1. Voluntary Motion Rejection**
- Detects arm raises, waves (0.3–2.5 Hz low-frequency power)
- Suppresses tremor score when voluntary motion detected
- Prevents false positives from user gestures

### **2. Adaptive Baseline Learning**
- Tracks user's historical tremor scores
- Computes personal baseline = running average
- Pass threshold = baseline + 8 standard deviations
- Adjusts for individual differences (not everyone tremors equally)

### **3. Spectral Regularity**
- Uses normalized Shannon entropy
- Measures how "concentrated" the tremor is
- Physiological tremor → more regular (high entropy)
- Pathological tremor → more regular (low entropy in band)

### **4. Peak Sharpness Metric**
- Ratio of peak power to surrounding noise
- High sharpness = clear tremor frequency
- Low sharpness = broad, noisy signal

### **5. Real-Time Spectral Analysis**
- Welch PSD for robust frequency estimation
- No need for peak-picking algorithms
- Updates every 200 ms

---

## Architecture Overview

### **Modules & Lines of Code**

| Module | Lines | Purpose |
|--------|-------|---------|
| `app.py` | 1318 | Main UI (Tkinter interface) |
| `tracker.py` | 316 | MediaPipe integration & threading |
| `signal.py` | 375 | Signal processing & metrics |
| `exercises.py` | 296 | Exercise framework |
| `tremor_app.py` | 480 | Standalone detector |

**Total:** ~2800 lines of production code

### **Threading Model**
- **Main Thread:** Tkinter UI (blocked during video updates)
- **Tracker Thread:** Video capture + MediaPipe (background)
- **Analysis Thread:** Signal processing (main thread, 200ms interval)
- **Synchronization:** Thread-safe deques, locks for pose/frame access

---

## Data Flow

```
Webcam (640×480, ~30 FPS)
    ↓
TremorTracker (background thread)
    ├─→ MediaPipe Hands (21 landmarks)
    ├─→ MediaPipe Pose (33 landmarks, every 3 frames)
    └─→ Store: (time, x, y, visibility) in circular deque
    
Every 200 ms (main thread):
    ↓
Signal Processing
    ├─→ Extract 4-second window from deque
    ├─→ Resampling → Filtering → Welch PSD
    ├─→ Compute 13 metrics
    ├─→ Voluntary motion rejection
    ├─→ Update adaptive baseline
    ├─→ Compute score & classification
    
    ↓
UI Update
    ├─→ Display metrics (frequency, power, SNR, etc.)
    ├─→ Plot PSD
    ├─→ Show spectrogram
    ├─→ Update tremor history
    
    ↓
Optional: CSV Export (user-initiated)
    └─→ Session summary with all metrics
```

---

## Dependencies

### **Core**
```
mediapipe==0.10.18      # Hand + Pose detection (Google)
opencv-python>=4.8.0    # Video capture
numpy<2                 # Numerical arrays
scipy>=1.10             # Signal processing (filters, PSD)
matplotlib>=3.7         # Visualization
Pillow>=10.0            # Image handling
ffpyplayer>=4.5.0       # Video playback (optional)
```

### **Built-in**
- `tkinter` — GUI framework
- `threading` — Background processing
- `csv` — Data export
- `pathlib` — File operations
- `collections.deque` — Circular buffering

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Frame Rate** | ~30 FPS (camera dependent) |
| **Metrics Update** | ~200 ms |
| **Latency (capture→score)** | ~500 ms |
| **Memory Usage** | ~150–300 MB |
| **CPU Usage** | 20–40% (single core, M1 Mac) |
| **GPU Required** | No |
| **Cloud Required** | No |
| **Privacy** | 100% local |

---

## UI/UX Features

### **Main Display**
- ✅ Live camera feed with hand skeleton overlay
- ✅ Real-time tremor score (0–100, color-coded)
- ✅ Key metrics: peak frequency, power, SNR, sharpness
- ✅ Tremor classification (Parkinsonian / Essential / Physiological)

### **Visualization**
- ✅ Live Power Spectral Density (PSD) plot
- ✅ Spectrogram (time-frequency heatmap)
- ✅ Tremor score history (line chart, 600 points)
- ✅ Adaptive baseline threshold indicator

### **Interaction**
- ✅ Landmark selection (Index, Middle, Thumb, Wrist)
- ✅ Exercise guided workflow (5 exercises, 3-stage flow)
- ✅ Real-time guidance messages
- ✅ CSV session export
- ✅ Focus video player (seek, play/pause, volume)

---

## Known Limitations

### **Technical**
- Requires visible hand + face (pose validation)
- Good lighting needed for reliable detection
- Webcam resolution affects amplitude estimation
- No GPU optimization (CPU-only)

### **Clinical**
⚠️ **NOT a medical device**
- Not FDA-approved or clinically validated
- Classification is informational only
- Should not be used for diagnosis
- Consult healthcare providers for medical decisions

---

## Deployment Options

### **1. Source Installation**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python motionbloom_run.py
```

### **2. Pre-built Binaries**
- macOS: Universal binary (.app)
- Windows: MSIX installer
- Both via GitHub Releases

### **3. Web/Docker**
- Not currently supported
- All processing requires local webcam access

---

## Project Status

- **Current Version:** Beta
- **License:** MIT (open source)
- **Maintenance:** Active development
- **Last Update:** May 2026
- **GitHub:** https://github.com/h3clinic/motionbloomtremor

---

## Summary: What's Running Today

### **Vision Components**
✅ **MediaPipe Hands** — Hand landmark detection (21 points, 30 FPS)
✅ **MediaPipe Pose** — Body posture (33 points, 10 FPS effective)

### **Processing Components**
✅ **Butterworth Filters** — Bandpass 3–12 Hz (4th order)
✅ **Welch PSD** — Spectral density estimation
✅ **Voluntary Motion Rejection** — Low-frequency suppression
✅ **Adaptive Baseline** — Personalized thresholding
✅ **Tremor Scoring** — 0–100 weighted algorithm

### **Output**
✅ **13 Tremor Metrics** — Frequency, power, SNR, sharpness, regularity
✅ **3 Classifications** — Parkinsonian, Essential, Physiological
✅ **Real-Time Visualization** — PSD, spectrogram, history
✅ **CSV Export** — Session data logging

---

**Created:** May 21, 2026
**Status:** Complete Analysis & Documentation
