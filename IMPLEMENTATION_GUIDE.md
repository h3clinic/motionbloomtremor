# Implementation Guide: SOTA Optimizations for MotionBloom

## Quick Reference: Priority Changes

Based on research investigation (see `RESEARCH_INVESTIGATION.md`), these are the **high-priority, high-impact** changes ranked by effort/benefit ratio.

---

## 🚀 Quick Wins (Implement First)

### **1. Reduce Window Size (30 min)**
**Impact:** ⭐⭐⭐⭐⭐ (Critical)  
**Effort:** ⭐ (Trivial)  
**Research:** Sun et al. (2023) — 1.28s windows achieve >92% accuracy

```python
# File: motionbloom/app.py
# Current: WINDOW_SECONDS = 4.0
# Change to:
WINDOW_SECONDS = 1.5  # Matches SOTA research
```

**Why:** 4-second windows allow too much voluntary motion contamination. Research shows 1.28-1.5s is optimal for action tremor isolation.

---

### **2. Narrow Primary Frequency Band (15 min)**
**Impact:** ⭐⭐⭐⭐ (High)  
**Effort:** ⭐ (Trivial)  
**Research:** di Biase (2025) — 3-7 Hz matches 90.6% UPDRS accuracy

```python
# File: motionbloom/signal.py
# Current: TREMOR_BAND = (3.0, 12.0)
# Change to:
PRIMARY_BAND = (3.0, 7.0)    # Main pathological tremor
SECONDARY_BAND = (7.0, 12.0) # Enhanced physiological

# Update classification
FREQ_CLASSES = [
    (3.0, 5.0, "Parkinsonian-like (rest)", "parkinsonian"),
    (5.0, 7.0, "Essential-like (postural/kinetic)", "essential"),
    (7.0, 10.0, "Enhanced physiological (mild)", "physiological_mild"),
    (10.0, 12.0, "Enhanced physiological (high)", "physiological_high"),
]
```

**Why:** 3-7 Hz is the primary pathological tremor band. Wider bands (7-12 Hz) capture normal physiological tremor, reducing specificity.

---

### **3. Add Smoothness Metric (45 min)**
**Impact:** ⭐⭐⭐⭐ (High)  
**Effort:** ⭐⭐ (Easy)  
**Research:** Rupprechter et al. (2021) — Smoothness correlates with clinical impairment

```python
# File: motionbloom/signal.py
# Add to compute_metrics()

def compute_jerk(x, y, fs):
    """Compute jerk (derivative of acceleration) for smoothness analysis."""
    if len(x) < 4:
        return None
    
    # Velocity (first derivative)
    vx = np.gradient(x, 1/fs)
    vy = np.gradient(y, 1/fs)
    
    # Acceleration (second derivative)
    ax = np.gradient(vx, 1/fs)
    ay = np.gradient(vy, 1/fs)
    
    # Jerk (third derivative)
    jx = np.gradient(ax, 1/fs)
    jy = np.gradient(ay, 1/fs)
    
    # Jerk magnitude
    jerk_mag = np.sqrt(jx**2 + jy**2)
    
    # RMS jerk (lower = smoother movement)
    rms_jerk = float(np.sqrt(np.mean(jerk_mag**2)))
    
    return rms_jerk

# Update TremorMetrics dataclass
@dataclass
class TremorMetrics:
    # ... existing fields ...
    rms_jerk: float = 0.0        # Movement smoothness (lower = smoother)
    speed_rms: float = 0.0       # RMS velocity

# In compute_metrics(), add:
jerk = compute_jerk(xu, yu, fs)
speed = np.sqrt(np.gradient(xu, 1/fs)**2 + np.gradient(yu, 1/fs)**2)
speed_rms = float(np.sqrt(np.mean(speed**2)))
```

**Why:** Smoothness (jerk) distinguishes tremor from other movement disorders and correlates with clinical severity scales.

---

## 🏗️ Medium-Priority (Next Week)

### **4. Implement Sliding Window (2-3 hours)**
**Impact:** ⭐⭐⭐⭐⭐ (Critical)  
**Effort:** ⭐⭐⭐ (Moderate)  
**Research:** Sun et al. (2023) — Overlapping windows prevent temporal artifacts

```python
# File: motionbloom/signal.py
# New class

class SlidingWindowAnalyzer:
    """Analyze tremor using overlapping sliding windows."""
    
    def __init__(self, window_sec=1.5, slide_sec=0.5, fs=30):
        self.window_sec = window_sec
        self.slide_sec = slide_sec
        self.fs = fs
        self.window_size = int(window_sec * fs)
        self.slide_size = int(slide_sec * fs)
        self.buffer = deque(maxlen=int(window_sec * fs * 2))
        self.last_analysis_idx = 0
    
    def add_sample(self, t, x, y, visibility):
        """Add sample to buffer and check if window ready."""
        self.buffer.append((t, x, y, visibility))
        
        # Check if we can slide
        if len(self.buffer) >= self.window_size + self.slide_size:
            # Extract sliding window
            window_data = list(self.buffer)[-self.window_size:]
            return window_data
        
        return None
    
    def process_window(self, window_data):
        """Extract metrics from window."""
        t_arr = np.array([s[0] for s in window_data])
        x_arr = np.array([s[1] for s in window_data])
        y_arr = np.array([s[2] for s in window_data])
        
        # Resample and compute metrics
        result = resample_uniform(t_arr, x_arr, y_arr, self.fs)
        if result is None:
            return None
        
        tu, xu, yu = result
        metrics = compute_metrics(xu, yu, self.fs)
        return metrics
```

**Why:** Prevents "cliff edge" effects where sudden changes at window boundaries create artifacts. Provides smoother, more stable metrics.

---

### **5. Add Numba JIT Compilation (1-2 hours)**
**Impact:** ⭐⭐⭐ (Medium)  
**Effort:** ⭐⭐ (Easy)  
**Research:** Edge computing best practices — 10-30x speedup for loops

```python
# File: motionbloom/signal.py
# Add at top
from numba import jit

# Compile performance-critical functions
@jit(nopython=True)
def detrend_numba(arr):
    """Numba-compiled detrending."""
    n = len(arr)
    if n < 2:
        return arr.copy()
    x = np.arange(n, dtype=np.float64)
    # Manual linear fit (faster than polyfit in Numba)
    x_mean = np.mean(x)
    arr_mean = np.mean(arr)
    numerator = np.sum((x - x_mean) * (arr - arr_mean))
    denominator = np.sum((x - x_mean) ** 2)
    m = numerator / denominator
    b = arr_mean - m * x_mean
    return arr - (m * x + b)

@jit(nopython=True)
def compute_rms_numba(x, y):
    """Fast RMS computation."""
    mag = np.sqrt(x * x + y * y)
    return np.sqrt(np.mean(mag * mag))
```

**Installation:**
```bash
pip install numba
```

**Why:** Python loops add ~10-50 µs overhead per iteration. Numba compiles to native machine code, achieving C-level performance.

---

## ⚙️ Advanced (Next Month)

### **6. Three-Thread Architecture (4-6 hours)**
**Impact:** ⭐⭐⭐⭐ (High)  
**Effort:** ⭐⭐⭐⭐ (Complex)  
**Research:** Edge computing — Prevents frame drops, stable 30 FPS

```python
# File: motionbloom/tracker.py
# New architecture

import queue
from threading import Thread, Event

class OptimizedTremorTracker:
    """Three-thread architecture for stable real-time performance."""
    
    def __init__(self):
        # Queues for inter-thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.coord_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue(maxsize=5)
        
        self.stop_event = Event()
        
        # Three worker threads
        self.capture_thread = None
        self.inference_thread = None
        self.analytics_thread = None
    
    def start(self, cap):
        """Start all three threads."""
        self.cap = cap
        
        self.capture_thread = Thread(target=self._capture_worker, daemon=True)
        self.inference_thread = Thread(target=self._inference_worker, daemon=True)
        self.analytics_thread = Thread(target=self._analytics_worker, daemon=True)
        
        self.capture_thread.start()
        self.inference_thread.start()
        self.analytics_thread.start()
    
    def _capture_worker(self):
        """Thread 1: Pure frame capture — no processing."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                try:
                    # Non-blocking put (drop if queue full)
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass  # Drop frame if processing backlogged
    
    def _inference_worker(self):
        """Thread 2: MediaPipe inference only."""
        mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # MediaPipe processing
            results = mp_hands.process(frame)
            
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    landmark = hand.landmark[self.landmark_idx]
                    coords = {
                        'time': time.time(),
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': landmark.visibility
                    }
                    self.coord_queue.put(coords)
    
    def _analytics_worker(self):
        """Thread 3: Signal processing & metrics."""
        analyzer = SlidingWindowAnalyzer()
        
        while not self.stop_event.is_set():
            try:
                coords = self.coord_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Add to sliding window
            window_data = analyzer.add_sample(
                coords['time'], 
                coords['x'], 
                coords['y'], 
                coords['visibility']
            )
            
            if window_data is not None:
                # Compute metrics
                metrics = analyzer.process_window(window_data)
                if metrics is not None:
                    self.results_queue.put(metrics)
```

**Why:** Decouples capture, inference, and analytics. Prevents signal processing from blocking frame capture, ensuring stable 30 FPS.

---

### **7. Unscented Kalman Filter (6-8 hours)**
**Impact:** ⭐⭐⭐⭐ (High)  
**Effort:** ⭐⭐⭐⭐⭐ (Complex)  
**Research:** Ghobadi (2026) — Real-time tracking with uncertainty quantification

```python
# File: motionbloom/signal.py
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

class RealTimeHandTracker:
    """Real-time hand tracking with UKF."""
    
    def __init__(self, fs=30):
        self.fs = fs
        self.dt = 1.0 / fs
        
        # State: [x, y, vx, vy, ax, ay]
        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(
            dim_x=6,
            dim_z=2,
            dt=self.dt,
            fx=self._motion_model,
            hx=self._measurement_model,
            points=points
        )
        
        # Initial state
        self.ukf.x = np.array([0.5, 0.5, 0., 0., 0., 0.])
        
        # Process noise
        self.ukf.Q = np.eye(6) * 0.001
        
        # Measurement noise
        self.ukf.R = np.eye(2) * 0.01
        
        self.last_time = None
    
    def _motion_model(self, x, dt):
        """Constant acceleration model."""
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        return F @ x
    
    def _measurement_model(self, x):
        """Observe position only."""
        return x[:2]
    
    def update(self, x_obs, y_obs, timestamp, confidence):
        """Real-time update with non-uniform timestamps."""
        # Compute actual dt
        if self.last_time is not None:
            dt = timestamp - self.last_time
        else:
            dt = self.dt
        
        # Predict step
        self.ukf.predict(dt=dt)
        
        # Adjust measurement noise based on confidence
        self.ukf.R = np.eye(2) * (0.01 / max(confidence, 0.1))
        
        # Update step
        self.ukf.update([x_obs, y_obs])
        
        self.last_time = timestamp
        
        # Return filtered state
        return {
            'x': float(self.ukf.x[0]),
            'y': float(self.ukf.x[1]),
            'vx': float(self.ukf.x[2]),
            'vy': float(self.ukf.x[3]),
            'ax': float(self.ukf.x[4]),
            'ay': float(self.ukf.x[5]),
            'uncertainty': float(np.trace(self.ukf.P[:2, :2]))
        }
```

**Installation:**
```bash
pip install filterpy
```

**Why:** Handles variable frame rates, provides velocity/acceleration estimates, quantifies uncertainty. No batch windows needed — truly real-time.

---

## 📊 Expected Performance Improvements

### **After Quick Wins (Week 1)**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Window size | 4.0 sec | 1.5 sec | 2.67x faster response |
| Action tremor accuracy | ~70% | ~85% | +15% (estimated) |
| Frequency specificity | Low | High | Better clinical correlation |
| Smoothness tracking | None | RMS jerk | New clinical metric |

### **After Medium Priority (Week 2-3)**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| CPU usage | 30-40% | 20-30% | 25% reduction |
| Frame stability | Variable | Stable 30 FPS | No drops |
| Temporal artifacts | Present | Eliminated | Sliding windows |
| Loop performance | 10-50 µs | 0.5 µs | 20-100x faster |

### **After Advanced (Month 2)**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Latency | ~500 ms | ~100 ms | 5x faster |
| Dropped frame handling | Artifacts | Predictive fill | Robust |
| Uncertainty | None | Confidence intervals | New feature |
| Real-time processing | Batch (4s) | True real-time | Streaming |

---

## 🎯 Implementation Priority Matrix

```
High Impact, Low Effort (DO FIRST):
┌─────────────────────────────────────┐
│ 1. Reduce window size (30 min)     │ ⭐⭐⭐⭐⭐
│ 2. Narrow frequency band (15 min)  │ ⭐⭐⭐⭐
│ 3. Add smoothness metric (45 min)  │ ⭐⭐⭐⭐
└─────────────────────────────────────┘

High Impact, Medium Effort (DO NEXT):
┌─────────────────────────────────────┐
│ 4. Sliding window (2-3 hours)      │ ⭐⭐⭐⭐⭐
│ 5. Numba JIT (1-2 hours)           │ ⭐⭐⭐
└─────────────────────────────────────┘

High Impact, High Effort (DO LATER):
┌─────────────────────────────────────┐
│ 6. Three-thread arch (4-6 hours)   │ ⭐⭐⭐⭐
│ 7. UKF tracking (6-8 hours)        │ ⭐⭐⭐⭐
└─────────────────────────────────────┘
```

---

## 🧪 Testing Strategy

### **After Each Change**
1. **Capture test video** (30 seconds, known tremor)
2. **Run both versions** (old vs. new)
3. **Compare metrics:**
   - Tremor score stability
   - Voluntary motion rejection
   - Processing time
   - CPU usage

### **Validation Data**
- **Resting tremor:** Hand stationary, fingers visible
- **Action tremor:** Finger-to-nose task
- **Voluntary motion:** Large arm movements (should suppress score)
- **No tremor:** Healthy control (should score <20)

---

## 📚 Dependencies to Add

```bash
# Quick wins (none needed)

# Medium priority
pip install numba

# Advanced
pip install filterpy
```

---

## 🚨 Breaking Changes to Consider

### **Window Size Reduction**
- UI history plots may need adjustment (shorter retention)
- Exercise hold times may need recalibration
- CSV export format unchanged

### **Frequency Band Changes**
- Classification labels change (4 classes instead of 3)
- Score distribution may shift (narrower band = potentially higher scores)
- Consider backward compatibility flag

### **Sliding Window**
- Metrics update more frequently (every 0.5s vs. 4s)
- UI update rate may need throttling
- Memory usage slightly higher (overlapping windows)

---

## 📖 Further Reading

See `RESEARCH_INVESTIGATION.md` for:
- Complete paper summaries
- Detailed algorithm explanations
- Performance benchmarks
- Clinical validation requirements

---

**Last Updated:** May 21, 2026  
**Status:** Ready for implementation
