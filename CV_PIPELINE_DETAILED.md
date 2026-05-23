# MotionBloom - CV Pipeline & Signal Processing

## Overview

The MotionBloom tremor detection system uses a **multi-stage computer vision and signal processing pipeline** to extract tremor characteristics from hand movements captured via webcam.

---

## Stage 1: Video Capture & Preprocessing

### **Input**
- **Source:** Webcam (OpenCV VideoCapture)
- **Resolution:** 640×480 pixels
- **Frame Rate:** ~30 FPS (native camera rate)
- **Color Space:** RGB (converted from BGR by OpenCV)

### **Processing** (tracker.py)
```python
# In TremorTracker._run()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame: (480, 640, 3) uint8 BGR array
```

**Output:** Raw BGR frames at ~30 FPS

---

## Stage 2: Hand Detection (MediaPipe Hands)

### **Model Details**
- **Framework:** Google MediaPipe (hand_landmark.tflite)
- **Model Complexity:** 0 (lightweight, ~4 MB)
- **Architecture:** Two-stage detector
  1. **Palm Detection:** Detects hand bounding boxes
  2. **Hand Landmark Model:** Extracts 21 3D landmarks

### **Landmarks** (21 points per hand)
```
0 — Wrist
1–4 — Thumb (MCP, PIP, DIP, tip)
5–8 — Index finger (MCP, PIP, DIP, tip)
9–12 — Middle finger (MCP, PIP, DIP, tip)
13–16 — Ring finger (MCP, PIP, DIP, tip)
17–20 — Pinky (MCP, PIP, DIP, tip)
```

### **Implementation** (tracker.py)
```python
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # Lightweight
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

results = mp_hands.process(frame)
for hand in results.multi_hand_landmarks:
    for landmark in hand.landmark:
        x, y, z = landmark.x, landmark.y, landmark.z  # Normalized [0..1]
        confidence = landmark.visibility
```

### **Output**
- **Per Hand:** 21 (x, y, z) coordinates (normalized to image dimensions)
- **Confidence:** Visibility score [0..1] for each landmark
- **Frequency:** ~30 times per second

---

## Stage 3: Pose Tracking (MediaPipe Pose)

### **Model Details**
- **Framework:** Google MediaPipe (pose_landmarker.tflite)
- **Landmarks:** 33 body points (head, arms, torso, legs)
- **Frequency:** Every 3rd frame (optimize for real-time performance)

### **Key Landmarks for MotionBloom**
- **Nose:** Head reference
- **L/R Ear:** Head width estimation
- **L/R Shoulder:** Scale reference
- **L/R Wrist:** Arm geometry
- **L/R Elbow:** Arm validation

### **Implementation** (tracker.py)
```python
POSE_EVERY = 3  # Update every 3 frames

if frame_idx % POSE_EVERY == 0:
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    pose_results = mp_pose.process(frame)
    # Extract PoseSnapshot (nose, ears, shoulders, wrists, elbows)
```

### **Output**
```python
@dataclass
class PoseSnapshot:
    nose: tuple[float, float] | None
    l_ear: tuple[float, float] | None
    r_ear: tuple[float, float] | None
    l_wrist: tuple[float, float] | None
    r_wrist: tuple[float, float] | None
    l_elbow: tuple[float, float] | None
    r_elbow: tuple[float, float] | None
    l_shoulder: tuple[float, float] | None
    r_shoulder: tuple[float, float] | None
    visibility: float
```

---

## Stage 4: Landmark Tracking & Buffering

### **Selected Landmark**
User chooses from:
```python
LANDMARK_CHOICES = {
    "Index fingertip": 8,       # Most commonly used
    "Middle fingertip": 12,
    "Thumb tip": 4,
    "Wrist": 0,
}
```

### **Data Buffering** (tracker.py)
```python
class TremorTracker:
    def __init__(self):
        self.samples: deque = deque(maxlen=4096)
        # Circular buffer: [time, x, y, visibility]
        # Capacity: 4096 samples ≈ 136 seconds @ 30 FPS

    def _run(self):
        # Continuous capture loop
        t_now = time.time()
        landmark = hand.landmark[self.landmark_idx]
        
        # Store: (timestamp, x_norm, y_norm, visibility)
        self.samples.append((t_now, landmark.x, landmark.y, landmark.visibility))
```

### **Output**
- **Storage:** Deque of `(time, x, y, visibility)` tuples
- **Max Capacity:** 4096 samples (~136 seconds @ 30 FPS)
- **Time Window:** Default 4 seconds for tremor analysis

---

## Stage 5: Grip Detection

### **Algorithm**
```python
def estimate_grip_strength(landmarks: List[Landmark]) -> float:
    # Ratio of fingertip-to-MCP distances vs palm size
    # 1.0 = fingers fully curled (gripping)
    # 0.0 = hand fully open
    
    palm_size = distance(wrist, middle_mcp)
    avg_curl = mean([
        1 - distance(fingertip, mcp) / palm_size
        for fingertip, mcp in finger_pairs
    ])
    return clamp(avg_curl, 0, 1)
```

### **Purpose**
- Detect object holding vs. open hand
- Used in exercise verification (e.g., "keep hand open")

### **Output**
- Grip strength: [0..1] float
- Updated at ~30 FPS

---

## Stage 6: Resampling to Uniform Grid

### **Problem**
- MediaPipe timestamps may be irregular
- Spectral analysis requires uniform sampling

### **Solution** (signal.py)
```python
def resample_uniform(t: np.ndarray, x: np.ndarray, y: np.ndarray, fs: float):
    """Linear-interpolate irregular samples onto uniform grid."""
    if t.size < 2:
        return None
    
    dur = t[-1] - t[0]
    n = int(dur * fs)  # fs ≈ 30 Hz
    
    tu = t[0] + np.arange(n) / fs  # Uniform time grid
    xu = np.interp(tu, t, x)        # Linear interpolation
    yu = np.interp(tu, t, y)
    
    return tu, xu, yu
```

### **Input**
- Raw timestamps (may have jitter/drops)
- x, y normalized coordinates [0..1]

### **Output**
- Uniform time grid at ~30 Hz
- Interpolated x, y coordinates
- Minimum length: 16 samples

---

## Stage 7: Signal Conditioning

### **7.1 Detrending**
```python
def detrend(arr: np.ndarray) -> np.ndarray:
    """Remove linear trend (drift) via linear regression."""
    n = arr.size
    x = np.arange(n, dtype=np.float64)
    m, b = np.polyfit(x, arr, 1)  # Fit: arr ≈ m*x + b
    return arr - (m * x + b)
```

**Purpose:** Remove slow baseline shifts (e.g., hand drifting across frame)

### **7.2 High-Pass Filtering**
```python
def highpass(x: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """Butterworth high-pass, 2nd order, cutoff=0.5 Hz."""
    sos = butter(2, cutoff / (0.5 * fs), btype='highpass', output='sos')
    return sosfiltfilt(sos, x)  # Zero-phase (forward-backward)
```

**Purpose:** Remove very low-frequency drift (< 0.5 Hz)

### **7.3 Band-Pass Filtering**
```python
def bandpass(x: np.ndarray, fs: float,
             low: float = 3.0,
             high: float = 12.0) -> np.ndarray:
    """Butterworth band-pass, 4th order, 3–12 Hz."""
    sos = butter(4, [low, high] / (0.5 * fs), btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)  # Zero-phase
```

**Purpose:** Isolate tremor oscillations (3–12 Hz band)

### **7.4 Windowing**
```python
def hann_window(arr: np.ndarray) -> np.ndarray:
    """Apply Hanning window for spectral analysis."""
    return arr * np.hanning(arr.size)
```

**Purpose:** Reduce spectral leakage in FFT

### **Combined Conditioning Pipeline**
```python
xf = bandpass(highpass(detrend(xu), fs), fs)
yf = bandpass(highpass(detrend(yu), fs), fs)
```

---

## Stage 8: Spectral Analysis (Welch PSD)

### **Welch Power Spectral Density**
```python
from scipy.signal import welch

# Separate x, y axes
nperseg = int(min(n, max(64, fs * 2)))  # Segment size ~4 seconds
fxx, pxx = welch(xf, fs=fs, nperseg=nperseg, detrend=False)
_, pyy = welch(yf, fs=fs, nperseg=nperseg, detrend=False)

# Combined magnitude
psd = pxx + pyy  # Power is additive for independent axes
```

### **Parameters**
- **Window:** Hanning (default)
- **Segments:** Overlapping 50%
- **Frequency Range:** 0–15 Hz

### **Output**
```
fxx: array [0.0, 0.25, 0.5, ..., 15.0] Hz
psd: Power at each frequency
```

---

## Stage 9: Tremor Metrics Extraction

### **Class: TremorMetrics**
```python
@dataclass
class TremorMetrics:
    # Sampling info
    fs: float               # Sampling rate (Hz)
    samples: int            # Number of samples
    duration: float         # Time window (seconds)
    
    # Frequency domain
    peak_hz: float          # Dominant tremor frequency
    peak_power: float       # Power at peak
    band_power: float       # Integrated 3–12 Hz power
    total_power: float      # Integrated 0.5–fs/2 power
    band_ratio: float       # band_power / total_power
    
    # Amplitude
    rms_amp: float          # RMS of bandpassed signal
    rms_amp_mm: float       # RMS in millimeters
    peak_to_peak_mm: float  # Peak-to-peak in mm
    
    # Quality metrics
    snr_db: float           # Signal-to-noise ratio (dB)
    regularity: float       # Spectral concentration [0..1]
    peak_sharpness: float   # Peak prominence
    
    # Classification
    class_label: str        # "Parkinsonian-like", "Essential-like", etc.
    class_code: str         # Short code: "parkinsonian", "essential"
    score: int              # Tremor severity: 0–100
```

### **Metric Computation** (signal.py)

#### **1. Peak Frequency**
```python
# Find max power in tremor band (3–12 Hz)
band_mask = (fxx >= TREMOR_BAND[0]) & (fxx <= TREMOR_BAND[1])
peak_idx = np.argmax(psd[band_mask])
peak_hz = fxx[band_mask][peak_idx]
peak_power = psd[peak_idx]
```

#### **2. Band Power**
```python
df = fxx[1] - fxx[0]  # Frequency resolution
band_power = psd[band_mask].sum() * df
total_power = psd.sum() * df
band_ratio = band_power / total_power
```

#### **3. SNR (Signal-to-Noise Ratio)**
```python
# Peak vs median excluding ±0.75 Hz around peak
off_peak = band_mask & ~((fxx >= peak_hz - 0.75) & (fxx <= peak_hz + 0.75))
if np.any(off_peak):
    snr_db = 10 * np.log10(peak_power / np.median(psd[off_peak]))
```

#### **4. Peak Sharpness**
```python
# Peak vs mean within ±1 Hz window
around = (fxx >= peak_hz - 1.0) & (fxx <= peak_hz + 1.0)
around_offpeak = around & (np.arange(fxx.size) != peak_idx)
mean_near = psd[around_offpeak].mean()
sharpness = peak_power / mean_near
```

#### **5. Regularity (Spectral Entropy)**
```python
# Normalize PSD to probability distribution
band_psd = psd[band_mask]
p = band_psd / band_psd.sum()

# Shannon entropy: H = -sum(p * log(p))
entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
max_entropy = np.log(p.size)

# Regularity: 1.0 = perfectly regular, 0.0 = flat
regularity = 1.0 - entropy / max_entropy
```

#### **6. RMS Amplitude**
```python
mag = np.sqrt(xf**2 + yf**2)
rms_amp = np.sqrt(np.mean(mag**2))

# Convert to mm if hand size known
if hand_ref_pixels is not None:
    pixel_to_mm = hand_width_mm / hand_ref_pixels
    rms_amp_mm = rms_amp * pixel_to_mm
```

#### **7. Tremor Classification**
```python
FREQ_CLASSES = [
    (3.0, 5.0, "Parkinsonian-like (rest)", "parkinsonian"),
    (5.0, 8.0, "Essential-like (postural/kinetic)", "essential"),
    (8.0, 12.0, "Enhanced physiological", "physiological"),
]

def estimate_class(peak_hz: float) -> tuple[str, str]:
    for lo, hi, label, code in FREQ_CLASSES:
        if lo <= peak_hz < hi:
            return label, code
```

---

## Stage 10: Voluntary Motion Rejection

### **Problem**
- Arm raises, waving detected as tremor
- Low-frequency power (0.3–2.5 Hz) dominates during voluntary motion

### **Solution** (signal.py)
```python
def suppress_for_voluntary_motion(metrics: TremorMetrics,
                                  low_power: float) -> int:
    """Penalize tremor score if low-frequency power is high."""
    if band_power > 0:
        low_ratio = low_power / band_power
        if low_ratio > threshold:
            # Suppress score
            return adjusted_score
```

### **Implementation**
- **Detection:** If 0.3–2.5 Hz power > threshold, classify as voluntary motion
- **Action:** Reduce tremor score or skip metrics update
- **Purpose:** Improve specificity (reduce false positives)

---

## Stage 11: Adaptive Baseline Learning

### **Algorithm** (signal.py / app.py)
```python
class AdaptiveBaseline:
    def __init__(self, window_size: int = 60):
        self.history = deque(maxlen=window_size)
        self.baseline_rms = 0.0
    
    def update(self, rms_amp: float):
        self.history.append(rms_amp)
        self.baseline_rms = np.mean(self.history)
    
    def get_threshold(self, std_mult: float = 8.0) -> float:
        """Pass threshold ≈ baseline + 8 σ"""
        if len(self.history) < 10:
            return float('inf')  # Warmup
        std = np.std(self.history)
        return self.baseline_rms + std_mult * std
    
    def passes(self, rms_amp: float) -> bool:
        """User's tremor within personal baseline?"""
        return rms_amp < self.get_threshold()
```

### **Purpose**
- Learn individual baseline (not all tremors equal)
- Personalized pass/fail threshold
- Improves sensitivity for low-tremor users

---

## Stage 12: Tremor Scoring (0–100)

### **Scoring Algorithm**
```python
def compute_tremor_score(metrics: TremorMetrics) -> int:
    """Combine multiple features into 0–100 severity score."""
    
    # Band power contribution (40%)
    power_score = min(100, (band_ratio / max_ratio) * 100)
    
    # SNR contribution (30%)
    snr_score = min(100, max(0, snr_db) * 5)
    
    # Peak sharpness contribution (20%)
    sharpness_score = min(100, sharpness * 100)
    
    # Regularity contribution (10%)
    regularity_score = regularity * 100
    
    # Weighted combination
    score = int(0.40 * power_score +
                0.30 * snr_score +
                0.20 * sharpness_score +
                0.10 * regularity_score)
    
    return max(0, min(100, score))
```

### **Output**
- **Range:** 0–100
- **Interpretation:**
  - 0–20: Minimal tremor
  - 21–50: Mild to moderate
  - 51–80: Moderate to severe
  - 81–100: Severe tremor

---

## Stage 13: UI Display & Logging

### **Real-Time Metrics Display** (app.py)
```
┌─ TREMOR SCORE: 45 ─────┐
│ Frequency: 6.3 Hz       │
│ Power: -20.5 dB         │
│ SNR: 8.2 dB             │
│ Type: Essential-like    │
│ Status: ✓ PASS          │
└─────────────────────────┘
```

### **Visualization**
- **Live PSD Plot:** Power vs. frequency
- **Spectrogram:** Time-frequency heatmap
- **Tremor History:** Score over time

### **Data Export**
```python
# CSV format
timestamp, tremor_score, peak_hz, peak_power, band_power, snr_db, regularity, class, hand_ref_mm
```

---

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Webcam Input (640×480, 30 FPS)                    │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: MediaPipe Hands (21 landmarks/hand)                │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MediaPipe Pose (every 3rd frame)                   │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: Landmark Selection & Buffering (4096 samples)      │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: Grip Detection (0–1 curl ratio)                    │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 6: Resampling to Uniform Grid (30 Hz)                 │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 7: Signal Conditioning                                │
│  - Detrend  - High-pass (0.5 Hz)  - Band-pass (3–12 Hz)    │
│  - Hanning window                                           │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 8: Welch PSD (Power Spectral Density)                 │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 9: Tremor Metrics Extraction                          │
│  - Peak frequency  - Band power  - SNR  - Sharpness        │
│  - Regularity  - RMS amplitude  - Classification           │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 10: Voluntary Motion Rejection                        │
│  - Detect low-freq power surge → suppress score            │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 11: Adaptive Baseline Learning                        │
│  - Update baseline  - Compute pass threshold                │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 12: Tremor Scoring (0–100)                            │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 13: UI Display & CSV Logging                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Current CV Pipeline & Model

### **Vision Models**
1. **MediaPipe Hands** (model_complexity=0)
   - Lightweight hand landmark detector
   - 21 landmarks per hand
   - ~30 FPS on CPU

2. **MediaPipe Pose** (model_complexity=1)
   - Full-body pose estimation
   - 33 landmarks
   - Sampled every 3 frames for efficiency

### **Signal Processing**
- **Filtering:** Butterworth high-pass (0.5 Hz) + band-pass (3–12 Hz)
- **Spectral Analysis:** Welch PSD
- **Feature Extraction:** 13 tremor metrics (frequency, power, SNR, sharpness, regularity)
- **Classification:** 3 tremor types (Parkinsonian, Essential, Physiological)
- **Scoring:** 0–100 weighted combination of spectral features
- **Personalization:** Adaptive baseline threshold

### **Key Innovations**
✓ Voluntary motion rejection (low-frequency suppression)
✓ Personalized adaptive baseline
✓ Real-time spectral analysis
✓ 100% local processing (no cloud)

---

**Last Updated:** May 21, 2026
**Status:** Active Development
