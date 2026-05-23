# MotionBloom Tremor Detector - Project Architecture

## Overview

**MotionBloom** is a real-time hand tremor detection application that uses webcam input and computer vision to analyze tremor characteristics. The system runs **100% locally** with no data leaving the user's machine.

**Project Type:** Desktop Application (Python + Tkinter)
**Current Status:** Beta
**Python Version:** 3.10–3.12

---

## Core Components

### 1. **Entry Points**
- **`motionbloom_run.py`** - Main entry point for the full application with UI
- **`tremor_app.py`** - Standalone tremor detector with Tkinter interface
- Both launch a desktop GUI application with real-time video processing

### 2. **Main Modules** (`motionbloom/`)

#### **`app.py`** - Main Application UI (1318 lines)
- **Tkinter-based** desktop interface
- **Color Scheme:** Modern light theme (off-white background `#f6f7f9`) with red accent (`#ef4444`)
- **Layout:** Card-based, conversational UI
- **Features:**
  - Real-time video feed display
  - Live tremor score (0–100)
  - Interactive metrics visualization
  - Exercise guided workflow
  - Personal adaptive baseline threshold management
  - CSV export for session history
  - Focus video player (with seek, play/pause, volume control)

#### **`tracker.py`** - Webcam & Hand/Pose Tracking (316 lines)
- **Vision Framework:** MediaPipe
- **Models Used:**
  - `MediaPipe Hands` (model_complexity=0, lightweight)
  - `MediaPipe Pose` (runs every 3rd frame for efficiency)
- **Key Functionality:**
  - Webcam capture (640×480)
  - Hand landmark extraction (21 3D points per hand)
  - Pose landmark tracking
  - Hand grip detection
  - Frame buffering (30-second history)
  - Threading-based concurrent video processing
  - Thread-safe data access with locks

#### **`signal.py`** - Signal Processing Engine (375 lines)
- **Tremor Analysis Pipeline:**
  - Detrending (linear regression removal)
  - Highpass filtering (remove drift, cutoff=0.5 Hz)
  - Bandpass filtering (tremor band: 3–12 Hz, Butterworth order 4)
  - Hanning window application
  - Uniform resampling for irregular samples
  - **Welch PSD** (Power Spectral Density) analysis
  - Spectral feature extraction

- **Metrics Computed** (TremorMetrics class):
  - `peak_hz` - Dominant tremor frequency
  - `peak_power` - PSD at dominant frequency
  - `band_power` - Integrated power in 3–12 Hz band
  - `band_ratio` - Tremor band / total power ratio
  - `rms_amp` / `rms_amp_mm` - Amplitude (normalized & millimeters)
  - `snr_db` - Signal-to-noise ratio
  - `regularity` - Spectral concentration (0–1)
  - `peak_sharpness` - Peak prominence within ±1 Hz window
  - `score` - Tremor severity (0–100)

- **Tremor Classification:**
  - **Parkinsonian-like:** 3–5 Hz (rest tremor)
  - **Essential-like:** 5–8 Hz (postural/kinetic)
  - **Enhanced physiological:** 8–12 Hz (light tremor)

#### **`exercises.py`** - Guided Exercise Framework (296 lines)
- **Exercise State Machine:** `IDLE → PREPARE → HOLD → DONE`
- **Key Features:**
  - Pose verification (touch nose, scratch head, hold steady, etc.)
  - Geometric constraints using landmark positions
  - Real-time guidance messages
  - Quality feedback (0–1 confidence)
  - Tremor sampling during HOLD phase
  - Exercise session tracking

#### **`video_gate.py`** - Video Integration
- Built-in focus video player
- Playback controls (seek, play/pause, mute, volume)
- Subtitle/caption support
- Integration with tremor monitoring

---

## Computer Vision Pipeline

### **Step-by-Step Flow**

```
1. WEBCAM INPUT
   └─> 640×480 RGB frame @ ~30 FPS

2. HAND TRACKING (MediaPipe Hands)
   └─> 21 landmarks per hand (3D coordinates)
   └─> Returns: fingertip, palm, joints, etc.

3. POSE TRACKING (MediaPipe Pose, every 3rd frame)
   └─> 33 landmarks (head, shoulders, elbows, wrists, etc.)
   └─> Used for: posture validation, voluntary motion detection

4. LANDMARK EXTRACTION
   └─> Selected landmark (e.g., index fingertip landmark #8)
   └─> Trajectory: [time, x_norm, y_norm, visibility]
   └─> Buffer: Deque of 4096 samples (~30 seconds @ 30 FPS)

5. UNIFORM RESAMPLING
   └─> Interpolate irregular MediaPipe timestamps
   └─> Grid at fixed sampling rate (e.g., 30 Hz)

6. SIGNAL CONDITIONING
   ├─> High-pass filter (remove drift)
   ├─> Detrend (linear regression)
   └─> Band-pass filter (3–12 Hz tremor band)

7. SPECTRAL ANALYSIS (Welch PSD)
   └─> Compute power across 1–15 Hz
   └─> Extract: peak frequency, band power, SNR, sharpness

8. VOLUNTARY MOTION REJECTION
   └─> Detect arm raises (0.3–2.5 Hz power surge)
   └─> Suppress tremor score when voluntary motion detected

9. TREMOR SCORING
   └─> Combine: band_power, peak_hz, regularity, SNR
   └─> Output: 0–100 tremor score
   └─> Classification: Parkinsonian / Essential / Physiological

10. ADAPTIVE BASELINE
    └─> Track user's running average
    └─> Pass threshold ≈ avg + 8
```

---

## Models & Weights

### **MediaPipe Models** (Pre-trained, included in `mediapipe==0.10.18`)

1. **MediaPipe Hands**
   - Model: Lightweight detector + hand landmark model
   - Complexity: 0 (fastest)
   - Output: 21 3D landmarks per hand
   - License: Google Mediapipe (Apache 2.0)

2. **MediaPipe Pose**
   - Model: Full-body pose estimation
   - Landmarks: 33 body points
   - Used for: Posture validation, arm geometry
   - License: Google Mediapipe (Apache 2.0)

**Note:** No custom ML models. All detection uses Google's pre-trained, lightweight models bundled with the `mediapipe` package.

---

## Key Algorithms

### **1. Tremor Extraction**
- **Method:** Butterworth bandpass (3–12 Hz)
- **Order:** 4
- **Purpose:** Isolate tremor oscillations from other movements

### **2. Spectral Density Estimation**
- **Method:** Welch PSD
- **Window:** Hanning
- **Purpose:** Frequency-domain analysis
- **Output:** Dominant frequency, power distribution

### **3. Voluntary Motion Rejection**
- **Detection:** Low-frequency power surge (0.3–2.5 Hz)
- **Logic:** If low-band power dominates, suppress tremor score
- **Purpose:** Prevent arm raises/waves from inflating tremor metric

### **4. Adaptive Baseline Learning**
- **Input:** Historical tremor scores
- **Model:** Running average
- **Pass Threshold:** avg + 8 standard deviations
- **Purpose:** Personalized detection (adjusts to individual baseline)

### **5. Grip Detection**
- **Method:** Ratio of fingertip-to-MCP distances
- **Output:** 0–1 grip strength
- **Purpose:** Detect object holding vs. open hand

---

## Dependencies

### Core Libraries
```
mediapipe==0.10.18          # Hand & pose detection
opencv-python>=4.8.0        # Video capture, image processing
numpy<2                      # Numerical arrays
scipy>=1.10                  # Signal processing (Butterworth, Welch)
matplotlib>=3.7              # Visualization, embedded charts
Pillow>=10.0                 # Image handling
ffpyplayer>=4.5.0            # Video playback (optional)
```

### Python Features
- **Threading:** Background video capture & processing
- **Tkinter:** GUI framework (built-in)
- **CSV:** Session logging
- **Pathlib:** File operations

---

## Configuration & Constants

### **Default Settings** (app.py / tracker.py)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Window Duration | 4.0 seconds | Analysis window for spectral analysis |
| Video FPS | ~30 | Camera frame rate |
| Analysis Interval | 200 ms | Metrics recalculation frequency |
| History Max | 600 samples | Max stored metrics points |
| Tremor Band | 3–12 Hz | Primary detection range |
| Camera Resolution | 640×480 | Input size |
| Hand Landmark | Index fingertip (#8) | Default tracked point |
| Pose Update Rate | Every 3 frames | Efficiency optimization |

---

## Data Flow Summary

```
Webcam
  ↓
[TremorTracker] — MediaPipe Hands/Pose
  ↓ (landmarks + visibility)
[Signal Processing] — bandpass, Welch PSD
  ↓ (metrics)
[App UI] — display + logging
  ↓
User feedback + CSV export
```

---

## Performance & Optimization

1. **MediaPipe Hands:** Lightweight model (`model_complexity=0`)
2. **Pose:** Sampled (every 3 frames) to reduce overhead
3. **Buffering:** Circular deque (automatic memory limit)
4. **Threading:** Async video capture + main UI thread separation
5. **Filters:** Cached Butterworth coefficients (LRU cache)

---

## Privacy & Security

- ✅ **100% Local:** No video transmission
- ✅ **No Cloud Sync:** All processing on-device
- ✅ **No Data Collection:** Optional CSV export only
- ✅ **Open Source:** MIT License

---

## Limitations & Disclaimers

⚠️ **Technical Demo — NOT a Medical Device**
- Not FDA-approved or clinically validated
- Should not be used for medical diagnosis
- Reference only; consult healthcare providers for clinical decisions

---

## File Tree

```
motionbloomtremor/
├── motionbloom/
│   ├── __init__.py
│   ├── app.py              # Main UI (Tkinter)
│   ├── tracker.py          # MediaPipe integration
│   ├── signal.py           # Signal processing & metrics
│   ├── exercises.py        # Exercise framework
│   └── video_gate.py       # Video player
├── motionbloom_run.py      # Entry point (full app)
├── tremor_app.py           # Standalone detector
├── requirements.txt        # Dependencies
├── README.md               # User guide
├── PRIVACY.md              # Privacy policy
├── LICENSE                 # MIT
├── index.html              # (web-related, unused)
└── packaging/              # Build & distribution configs
```

---

## Deployment

### **Source Installation**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python motionbloom_run.py
```

### **Pre-built Binaries**
- macOS & Windows: GitHub Releases (PyInstaller)
- Microsoft Store: MSIX submission
- Apple: Notarization required

---

**Last Updated:** May 21, 2026
**Version:** Beta
**License:** MIT
