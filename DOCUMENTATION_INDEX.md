# MotionBloom Documentation Index

## 📚 Complete Documentation Suite

This directory contains comprehensive documentation about the **MotionBloom Tremor Detector** project, including architecture, CV pipeline, signal processing, and quick reference guides.

---

## 📖 Documentation Files

### **1. QUICK_REFERENCE.md** ⭐ START HERE
**Purpose:** Quick overview of the entire system
**Contents:**
- Current models & CV pipeline
- 13 tremor metrics overview
- Tremor classification system
- Scoring algorithm (0–100)
- Key features & dependencies
- Quick start instructions
- Troubleshooting

**Best for:** Getting oriented, understanding what's running now

---

### **2. ANALYSIS_SUMMARY.md** 📊 HIGH-LEVEL OVERVIEW
**Purpose:** Executive summary of the project
**Contents:**
- Project overview & status
- Vision models (MediaPipe Hands, MediaPipe Pose)
- 13-stage signal processing pipeline
- All 13 tremor metrics explained
- Tremor classification types
- Scoring algorithm details
- Key innovations
- Architecture overview
- Performance specs
- Deployment options

**Best for:** Understanding the complete system architecture and decisions

---

### **3. CV_PIPELINE_DETAILED.md** 🔬 DEEP DIVE
**Purpose:** Detailed breakdown of computer vision pipeline
**Contents:**
- 13-stage detailed pipeline (with code samples)
  - Stage 1: Video capture & preprocessing
  - Stage 2: Hand detection (MediaPipe Hands)
  - Stage 3: Pose tracking (MediaPipe Pose)
  - Stage 4: Landmark tracking & buffering
  - Stage 5: Grip detection
  - Stage 6: Resampling to uniform grid
  - Stage 7: Signal conditioning (detrending, filtering)
  - Stage 8: Spectral analysis (Welch PSD)
  - Stage 9: Metrics extraction (13 features)
  - Stage 10: Voluntary motion rejection
  - Stage 11: Adaptive baseline learning
  - Stage 12: Tremor scoring (0–100)
  - Stage 13: UI display & logging
- Complete pipeline diagram
- Algorithm details with Python code snippets
- Mathematical formulas for each metric

**Best for:** Understanding the signal processing in depth

---

### **4. PROJECT_ARCHITECTURE.md** 🏗️ SYSTEM DESIGN
**Purpose:** Project structure and architecture overview
**Contents:**
- Project overview & status
- Core components breakdown
  - Entry points
  - Main modules (app.py, tracker.py, signal.py, exercises.py)
- Complete CV pipeline
- Models & weights information
- Key algorithms
- Dependencies
- Configuration & constants
- Data flow summary
- Performance & optimization
- Privacy & security
- File tree
- Deployment options

**Best for:** Understanding how the code is organized and how components interact

---

### **5. SYSTEM_DIAGRAMS.md** 📐 VISUAL REFERENCE
**Purpose:** Visual diagrams and flowcharts
**Contents:**
- Architecture diagram (7-layer system)
- Threading model diagram
- Signal processing pipeline (9-step flowchart with code)
- Tremor classification matrix
- Data structure: TremorMetrics
- Filter frequency response curves

**Best for:** Visual learners, understanding system flow at a glance

---

### **6. README.md** 📝 USER GUIDE
**Purpose:** Original project README (user-focused)
**Contents:**
- Features overview
- Quick start instructions
- Downloads
- Privacy statement
- How it works (high-level)
- Disclaimer
- License info

**Best for:** End users, understanding features

---

### **7. PRIVACY.md** 🔒 PRIVACY POLICY
**Purpose:** Privacy and data handling information
**Contents:**
- Data collection statement
- Privacy guarantees
- Local processing confirmation
- No cloud/transmission

**Best for:** Privacy-conscious users

---

### **8. RESEARCH_INVESTIGATION.md** 🔬 RESEARCH ANALYSIS
**Purpose:** Investigation of SOTA research papers and optimization strategies
**Contents:**
- EVM (Eulerian Video Magnification) analysis
- Action/kinetic tremor decoupling (Sun et al. 2023)
- Markerless clinical tracking (Rupprechter et al. 2021)
- Edge computing optimization strategies
- UKF (Unscented Kalman Filter) for real-time tracking
- Three-thread architecture design
- Numba JIT compilation strategies
- Comparison: MotionBloom vs. SOTA
- Implementation roadmap (4 phases)

**Best for:** Understanding research gaps, optimization opportunities

---

### **9. IMPLEMENTATION_GUIDE.md** 🛠️ PRACTICAL GUIDE
**Purpose:** Step-by-step implementation guide for SOTA optimizations
**Contents:**
- Quick wins (30-45 min each)
  - Reduce window size (4s → 1.5s)
  - Narrow frequency band (3-7 Hz primary)
  - Add smoothness/jerk metric
- Medium-priority improvements (2-3 hours each)
  - Sliding window with overlap
  - Numba JIT compilation
- Advanced features (4-8 hours each)
  - Three-thread architecture
  - UKF real-time tracking
- Priority matrix
- Testing strategy
- Expected performance improvements

**Best for:** Implementing research-backed improvements

---

### **10. ROI_TREMOR_ARCHITECTURE.md** 🎯 ROI TREMOR SIGNAL DESIGN
**Purpose:** Test-first ROI optical-flow tremor architecture notes
**Contents:**
- MediaPipe as coarse ROI anchor only
- Sparse LK optical flow as primary tremor signal
- Median/MAD motion aggregation
- Background/global motion subtraction
- Synthetic validation cases and no-UI-wiring constraint

**Best for:** Understanding the new ROI tremor subsystem before app integration

---

## 🎯 Reading Guide by Role

### **For Project Managers / Stakeholders**
1. Start: **QUICK_REFERENCE.md** (5 min read)
2. Deep dive: **ANALYSIS_SUMMARY.md** (15 min read)
3. Visual: **SYSTEM_DIAGRAMS.md** (10 min read)

### **For Software Engineers**
1. Start: **QUICK_REFERENCE.md** (5 min)
2. Architecture: **PROJECT_ARCHITECTURE.md** (20 min)
3. Pipeline details: **CV_PIPELINE_DETAILED.md** (30 min)
4. Reference: **SYSTEM_DIAGRAMS.md** (10 min)
5. Optimizations: **IMPLEMENTATION_GUIDE.md** (15 min)

### **For ML/Signal Processing Engineers**
1. Start: **CV_PIPELINE_DETAILED.md** (45 min)
2. Research: **RESEARCH_INVESTIGATION.md** (30 min)
3. Implementation: **IMPLEMENTATION_GUIDE.md** (20 min)
4. Deep dive: **SYSTEM_DIAGRAMS.md** (20 min)
5. Code reference: Review `motionbloom/signal.py` (30 min)

### **For Users / Clinicians**
1. Start: **README.md** (5 min)
2. Features: **QUICK_REFERENCE.md** (10 min)
3. Privacy: **PRIVACY.md** (5 min)

---

## 🔑 Key Findings Summary

### **Current Models**
- ✅ **MediaPipe Hands** (v0.10.18) — Hand landmark detection
- ✅ **MediaPipe Pose** (v0.10.18) — Body pose estimation
- ✅ **No custom ML models** — Uses pre-trained Google models

### **Signal Processing**
- ✅ **13 tremor metrics** extracted per analysis
- ✅ **Butterworth filters** (bandpass 3–12 Hz, highpass 0.5 Hz)
- ✅ **Welch PSD** for spectral analysis
- ✅ **Adaptive baseline** for personalization
- ✅ **Voluntary motion rejection** for accuracy

### **Tremor Classification**
- ✅ **3 frequency-based classes:**
  - Parkinsonian-like (3–5 Hz)
  - Essential-like (5–8 Hz)
  - Enhanced Physiological (8–12 Hz)

### **Performance**
- ✅ **~30 FPS** video capture
- ✅ **200 ms** metrics update latency
- ✅ **100% local** processing (no cloud)
- ✅ **CPU-only** (no GPU required)
- ✅ **~150–300 MB** memory usage

---

## 📋 Quick Fact Sheet

| Aspect | Value |
|--------|-------|
| **Project Name** | MotionBloom Tremor Detector |
| **Language** | Python 3.10–3.12 |
| **Framework** | Tkinter (GUI) |
| **Vision Library** | MediaPipe v0.10.18 |
| **Signal Processing** | SciPy + NumPy |
| **Status** | Beta, MIT License |
| **Privacy** | 100% local, no cloud |
| **Model Type** | Pre-trained (Google) |
| **Tremor Band** | 3–12 Hz |
| **Metrics Count** | 13 per analysis |
| **Tremor Classes** | 3 (Parkinsonian, Essential, Physiological) |
| **Scoring Range** | 0–100 |
| **Update Rate** | ~5 Hz (200 ms) |
| **Detection Range** | 1–15 Hz |

---

## 📂 File Organization

```
motionbloomtremor/
├── Documentation (NEW)
│   ├── QUICK_REFERENCE.md              ⭐ Start here
│   ├── ANALYSIS_SUMMARY.md             📊 Executive summary
│   ├── CV_PIPELINE_DETAILED.md         🔬 Technical deep-dive
│   ├── PROJECT_ARCHITECTURE.md         🏗️ System design
│   ├── SYSTEM_DIAGRAMS.md              📐 Visual reference
│   ├── RESEARCH_INVESTIGATION.md       🔬 SOTA research analysis
│   ├── IMPLEMENTATION_GUIDE.md         🛠️ Practical optimization guide
│   ├── DOCUMENTATION_INDEX.md          📚 This file
│   ├── README.md                       📝 User guide
│   └── PRIVACY.md                      🔒 Privacy info
│
├── Source Code
│   ├── motionbloom/
│   │   ├── app.py                      (1318 lines)
│   │   ├── tracker.py                  (316 lines)
│   │   ├── signal.py                   (375 lines)
│   │   ├── exercises.py                (296 lines)
│   │   └── video_gate.py
│   ├── motionbloom_run.py
│   ├── tremor_app.py                   (480 lines)
│   └── requirements.txt
│
└── Build & Config
    └── packaging/
```

---

## 🚀 Getting Started

### **Step 1: Understand the System**
Read: `QUICK_REFERENCE.md` (5 minutes)

### **Step 2: Deep Dive**
Choose your path based on role (see "Reading Guide" above)

### **Step 3: Review Code**
- Core pipeline: `motionbloom/signal.py` (375 lines)
- Vision integration: `motionbloom/tracker.py` (316 lines)
- UI/main loop: `motionbloom/app.py` (1318 lines)

### **Step 4: Run the System**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python motionbloom_run.py
```

---

## 🔗 Key Sections by Topic

### **Vision & Detection**
- 📖 **QUICK_REFERENCE.md** → "Current Models & CV Pipeline"
- 🔬 **CV_PIPELINE_DETAILED.md** → "Stage 2–5" (Detection & Tracking)
- 📐 **SYSTEM_DIAGRAMS.md** → "Architecture Diagram"

### **Signal Processing**
- 🔬 **CV_PIPELINE_DETAILED.md** → "Stage 6–9" (Processing & Analysis)
- 📐 **SYSTEM_DIAGRAMS.md** → "Signal Processing Pipeline"
- 🏗️ **PROJECT_ARCHITECTURE.md** → "Key Algorithms"

### **Metrics & Classification**
- 📊 **ANALYSIS_SUMMARY.md** → "Tremor Metrics" & "Classification"
- 📖 **QUICK_REFERENCE.md** → "Tremor Metrics" & "Classification"
- 📐 **SYSTEM_DIAGRAMS.md** → "Tremor Classification Matrix"

### **Voluntary Motion & Baseline**
- 🔬 **CV_PIPELINE_DETAILED.md** → "Stage 10–11"
- 🏗️ **PROJECT_ARCHITECTURE.md** → "Key Innovations"

### **Scoring**
- 📖 **QUICK_REFERENCE.md** → "Scoring Algorithm"
- 🔬 **CV_PIPELINE_DETAILED.md** → "Stage 12"

---

## ✅ Documentation Checklist

- [x] **Architecture overview** (PROJECT_ARCHITECTURE.md)
- [x] **CV pipeline details** (CV_PIPELINE_DETAILED.md)
- [x] **System diagrams** (SYSTEM_DIAGRAMS.md)
- [x] **Quick reference** (QUICK_REFERENCE.md)
- [x] **Executive summary** (ANALYSIS_SUMMARY.md)
- [x] **Tremor metrics explanation** (Multiple files)
- [x] **Classification details** (Multiple files)
- [x] **Scoring algorithm** (Multiple files)
- [x] **Model information** (All files)
- [x] **Dependencies list** (PROJECT_ARCHITECTURE.md)
- [x] **Performance specs** (ANALYSIS_SUMMARY.md)
- [x] **Deployment guide** (PROJECT_ARCHITECTURE.md)
- [x] **Research investigation** (RESEARCH_INVESTIGATION.md)
- [x] **SOTA optimization strategies** (RESEARCH_INVESTIGATION.md)
- [x] **Implementation guide** (IMPLEMENTATION_GUIDE.md)

---

## 📞 Support & Links

- **Repository:** https://github.com/h3clinic/motionbloomtremor
- **License:** MIT
- **Status:** Beta (Active Development)
- **Last Updated:** May 21, 2026

---

## 💡 Pro Tips

1. **Visual learners:** Start with `SYSTEM_DIAGRAMS.md`
2. **Code reviewers:** Read `PROJECT_ARCHITECTURE.md` first
3. **Quick overview:** Use `QUICK_REFERENCE.md`
4. **Technical depth:** Dive into `CV_PIPELINE_DETAILED.md`
5. **Presentation:** Use diagrams from `SYSTEM_DIAGRAMS.md`

---

**Total Documentation:** 2129 lines across 8 files → **3456 lines across 10 files**
**Created:** May 21, 2026
**Status:** Complete ✅ (includes SOTA research & implementation guide)
