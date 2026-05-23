# MotionBloom - Quick Reference Guide

## Current System Summary

### **Project Name**
MotionBloom Tremor Detector

### **Purpose**
Real-time hand tremor analysis via webcam (100% local, no cloud).

### **Status**
Beta, MIT License

---

## Models Currently Running

### **Computer Vision Models**

| Model | Framework | Version | Purpose | Frequency |
|-------|-----------|---------|---------|-----------|
| **MediaPipe Hands** | Google MediaPipe | 0.10.18 | Hand landmark detection (21 points) | ~30 FPS |
| **MediaPipe Pose** | Google MediaPipe | 0.10.18 | Body pose estimation (33 points) | Every 3 frames (~10 FPS) |

**Model Complexity:** Both lightweight (model_complexity=0 for Hands, complexity=1 for Pose)

---

## Signal Processing Pipeline (Current)

### **Flow**
```
Raw Hand Trajectory (x, y, z)
    ↓ [Resampling to 30 Hz uniform grid]
    ↓ [Detrending + High-pass 0.5 Hz + Band-pass 3–12 Hz]
    ↓ [Welch PSD Analysis]
    ↓ [Extract 13 metrics: peak freq, power, SNR, sharpness, etc.]
    ↓ [Voluntary motion rejection]
    ↓ [Adaptive baseline learning]
    ↓ [Score 0–100]
```

### **Key Algorithms**
- **Filtering:** Butterworth IIR (4th order band-pass, 3–12 Hz)
- **Spectral Density:** Welch PSD (Hanning window)
- **Feature Extraction:** 13 tremor metrics
- **Classification:** 3 tremor types (frequency-based)
- **Baseline:** Running average + adaptive threshold

---

## Tremor Metrics (13 Total)

| Metric | Range | Meaning |
|--------|-------|---------|
| **peak_hz** | 1–15 Hz | Dominant tremor frequency |
| **peak_power** | dB | Power at peak frequency |
| **band_power** | dB | Integrated power (3–12 Hz) |
| **band_ratio** | 0–1 | Tremor band / total power |
| **rms_amp** | normalized | RMS amplitude (normalized coords) |
| **rms_amp_mm** | mm | RMS amplitude (millimeters) |
| **snr_db** | dB | Signal-to-noise ratio |
| **regularity** | 0–1 | Spectral concentration (entropy-based) |
| **peak_sharpness** | ratio | Peak prominence within ±1 Hz |
| **class_label** | string | Parkinsonian / Essential / Physiological |
| **class_code** | string | parkinsonian / essential / physiological |
| **score** | 0–100 | Tremor severity |
| **duration** | seconds | Analysis window (e.g., 4 sec) |

---

## Tremor Classification (Current)

### **3 Classes Based on Frequency**

| Class | Frequency | Typical Cause | Icon |
|-------|-----------|---------------|------|
| **Parkinsonian-like** | 3–5 Hz | Rest tremor (Parkinson's disease) | 🟢 |
| **Essential-like** | 5–8 Hz | Postural/kinetic (Essential tremor) | 🟡 |
| **Enhanced Physiological** | 8–12 Hz | Normal, stress/excitement-induced | 🔵 |

**Note:** Classification is **frequency-based only** — not a medical diagnosis.

---

## Scoring Algorithm (0–100)

### **Weighted Features**
```python
score = (
    0.40 * power_contribution +      # Band power
    0.30 * snr_contribution +         # Signal quality
    0.20 * sharpness_contribution +   # Peak prominence
    0.10 * regularity_contribution    # Regularity
)
```

### **Interpretation**
- **0–20:** Minimal tremor
- **21–50:** Mild to moderate
- **51–80:** Moderate to severe
- **81–100:** Severe tremor

---

## Key Features

### **Detection**
✅ Live hand tracking (MediaPipe Hands 21 landmarks)
✅ Tremor frequency estimation (1–15 Hz)
✅ Band-power analysis (3–12 Hz tremor band)
✅ SNR & spectral quality metrics
✅ Peak sharpness (regularity)

### **Motion Rejection**
✅ Voluntary motion detection (0.3–2.5 Hz low-freq suppression)
✅ Arm raise rejection
✅ Wave/gesture filtering

### **Personalization**
✅ Adaptive baseline learning (running average)
✅ Personal pass threshold (avg + 8σ)
✅ Adjusts to individual baseline

### **UI/UX**
✅ Real-time metrics display
✅ Live PSD visualization
✅ Spectrogram heatmap
✅ Tremor history chart
✅ Exercise guided workflow
✅ CSV session export

---

## Dependencies & Libraries

```
Python:          3.10–3.12
mediapipe:       0.10.18       # Hand + Pose detection
opencv-python:   >=4.8.0       # Video capture
numpy:           <2            # Numerical arrays
scipy:           >=1.10        # Signal processing (Butterworth, Welch)
matplotlib:      >=3.7         # Visualization
Pillow:          >=10.0        # Image handling
ffpyplayer:      >=4.5.0       # Video playback (optional)
```

---

## Input/Output

### **Input**
- Webcam video stream (640×480, 30 FPS)
- User hand positioning
- Landmark selection (Index, Middle, Thumb, Wrist)

### **Output**
- **Real-time:** 13 tremor metrics, 0–100 score, classification
- **Storage:** CSV export with session history
- **Display:** Plots, metrics, status indicators

---

## Performance Specs

| Metric | Value |
|--------|-------|
| Frame Rate | ~30 FPS |
| Latency | <200 ms (metrics update) |
| Memory | ~150–300 MB |
| CPU Usage | ~20–40% (single core, M1 Mac) |
| GPU | Not required |
| Privacy | 100% local (no cloud) |

---

## File Structure

```
motionbloomtremor/
├── motionbloom/
│   ├── app.py              # Main UI (Tkinter, 1318 lines)
│   ├── tracker.py          # MediaPipe integration (316 lines)
│   ├── signal.py           # Signal processing (375 lines)
│   ├── exercises.py        # Exercise framework (296 lines)
│   └── video_gate.py       # Video player
├── motionbloom_run.py      # Entry point
├── tremor_app.py           # Standalone detector (480 lines)
├── requirements.txt        # Dependencies
├── README.md               # User guide
├── PRIVACY.md              # Privacy policy
├── PROJECT_ARCHITECTURE.md # Full architecture doc (this repo)
├── CV_PIPELINE_DETAILED.md # Detailed CV pipeline
└── QUICK_REFERENCE.md      # This file
```

---

## Quick Start

### **From Source**
```bash
cd motionbloomtremor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python motionbloom_run.py
```

### **Pre-built**
- Download macOS/Windows binary from GitHub Releases
- Unzip and run — no Python needed

---

## Troubleshooting

### **"No hand detected"**
- Ensure hand is visible to camera
- Good lighting required
- MediaPipe needs clear hand shape

### **"Low confidence" tremor score**
- Hand may be partially out of frame
- Visibility < 0.5 (landmark confidence)
- Try different landmark (Index → Thumb)

### **High false positives (arm raises counted as tremor)**
- Voluntary motion rejection should suppress arm raises
- If not: may need manual threshold tuning

### **Low FPS / lag**
- Check CPU usage
- Reduce camera resolution (640×480 is default)
- Disable MediaPipe Pose (every 3rd frame already)

---

## Limitations & Disclaimers

⚠️ **NOT a medical device**
- Technical demo only
- Not FDA-approved
- Not for clinical diagnosis
- Do not use for medical decisions

---

## Contact & Links

- **Repository:** https://github.com/h3clinic/motionbloomtremor
- **License:** MIT
- **Status:** Beta (active development)

---

**Last Updated:** May 21, 2026
