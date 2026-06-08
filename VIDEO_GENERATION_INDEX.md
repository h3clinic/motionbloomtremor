# Video Generation System: Complete Index

**Created**: June 8, 2025  
**Status**: Ready to generate videos  
**Total Files**: 9 new files across 3 categories

---

## 📖 Documentation (Start Here)

### Main Guides (Read in this order)

1. **`VIDEO_GENERATION_READY.md`** ⭐ START HERE
   - Overview of what you have
   - Three fastest paths to your first video
   - Timeline expectations
   - Troubleshooting

2. **`VIDEO_GENERATION_START_HERE.md`**
   - Detailed quick start
   - Installation instructions for each method
   - Complete workflow
   - Command reference

3. **`VIDEO_GENERATION_QUICKSTART.md`**
   - All generation methods explained
   - Setup instructions
   - Example usage
   - Batch generation

### Harness Documentation (Reference)

- **`AI_VIDEO_HARNESS.md`** (643 lines)
  - Full specification of the QA system
  - Tier strategy detailed
  - Prompt best practices
  - Training integration rules

- **`AI_VIDEO_HARNESS_QUICKSTART.md`** (422 lines)
  - 6-step workflow diagram
  - Tier overview
  - Key parameters
  - Example commands

- **`AI_VIDEO_HARNESS_IMPLEMENTATION.md`** (571 lines)
  - Complete implementation summary
  - Metrics explained
  - Usage examples
  - Validation checklist

---

## 🔧 Tools and Scripts

### Main Helper Script

**`scripts/generate_videos.py`** (216 lines)
```bash
# List prompts
python3 scripts/generate_videos.py list [--tier A|B|C|D] [--limit N]

# Export batch script
python3 scripts/generate_videos.py export \
  --method wan|replicate \
  [--tier A|B|C|D] \
  --output FILENAME
```

**Features**:
- Filter prompts by tier or model
- Generate bash scripts for Wan 2.2
- Generate Python scripts for Replicate
- Create output directories automatically

---

### Example Code Generator

**`scripts/video_generation_examples.py`** (156 lines)
```bash
# Show code examples for all 4 methods
python3 scripts/video_generation_examples.py
```

**Shows examples for**:
- Wan 2.2 command-line
- Replicate Python API
- HunyuanVideo
- ComfyUI GUI

---

### Pre-Built Batch Script

**`batch_generate_tier_a.sh`** (Auto-generated)
- Ready-to-use Wan 2.2 batch script
- Generates all 5 Tier A videos
- Edit `WAN_DIR` path before running

---

### Core QA & Processing Modules

**`topotremor/ai_video_qa.py`** (664 lines)
- 3-stage QA gates (anatomy, temporal, camera)
- 16 metrics per video
- Artifact scoring
- Validity labeling

```bash
python3 topotremor/ai_video_qa.py videos/ > qa_results.json
```

---

**`topotremor/ai_video_ingestion.py`** (658 lines)
- Optical flow measurement extraction
- Macro motion removal
- Residual frequency analysis
- 3-second window aggregation

```bash
python3 topotremor/ai_video_ingestion.py batch \
  --video_list qa_results.json \
  --output ingested_windows.json
```

---

**`topotremor/ai_video_reporting.py`** (371 lines)
- 5 automated report generators
- QA summary
- Artifact failures
- False positives
- Weak tremor candidates

```bash
python3 topotremor/ai_video_reporting.py \
  qa_results.json ingested_windows/ reports/
```

---

## 📊 Data Files

### Prompt Manifest

**`datasets/topotremor/prompts/ai_video_prompts.csv`** (25 rows)
- 25 prompt templates across 4 tiers
- 16 columns per template
- Ready for video generation

**Structure**:
```
prompt_id | model_name | generation_mode | reference_image_path | 
prompt_text | negative_prompt | category | expected_tremor_present |
expected_motion_type | expected_validity_label | label_confidence |
seed | duration_sec | fps | resolution | output_path
```

**Tiers**:
- **Tier A** (5): no_tremor_still, no_tremor_smooth_macro, no_tremor_palm_down
- **Tier B** (6): hard_negative_finger_tap, hard_negative_wave, hard_negative_finger_flex, hard_negative_motion_blur
- **Tier C** (5): weak_tremor_fingertip, weak_tremor_whole_hand, weak_tremor_macro_subtle
- **Tier D** (5): generator_artifact_six_fingers, generator_artifact_fused, generator_artifact_edge_crawl

---

### Command Manifest (Reference)

**`datasets/topotremor/prompts/ai_video_generation_commands.csv`** (44 rows)
- Auto-generated from `build_ai_video_commands.py`
- 44 commands = 11 templates × 4 models
- One row per model/template combo
- Columns: prompt_id, tier, category, model_name, generation_mode, command

---

## 🚀 Quick Command Reference

### 1. List Available Prompts
```bash
# All prompts
python3 scripts/generate_videos.py list

# Tier A only
python3 scripts/generate_videos.py list --tier A

# First 3 Tier B
python3 scripts/generate_videos.py list --tier B --limit 3
```

### 2. View Code Examples
```bash
python3 scripts/video_generation_examples.py
```

### 3. Generate Videos (One Tier)
```bash
# Export batch script for Wan 2.2
python3 scripts/generate_videos.py export \
  --method wan --tier A --output batch_a.sh

# Edit WAN_DIR in batch_a.sh, then run
bash batch_a.sh
```

### 4. Validate Generated Videos
```bash
# Run QA on all videos in a directory
python3 topotremor/ai_video_qa.py videos/

# Check results
cat videos/qa_results.json
```

### 5. Ingest Valid Videos
```bash
python3 topotremor/ai_video_ingestion.py batch \
  --video_list videos/qa_results.json \
  --output windows.json
```

### 6. Generate Reports
```bash
python3 topotremor/ai_video_reporting.py \
  videos/qa_results.json \
  windows/ \
  reports/
```

---

## 📋 Video Generation Methods Comparison

| Method | Setup Time | Generation Speed | Quality | Cost | Best For |
|--------|-----------|------------------|---------|------|----------|
| **Replicate** | 5 min | 1-2 min/video | High | $0.10-0.30/video | No GPU, quick testing |
| **ComfyUI** | 10 min | 3-5 min/video | Medium | Free (after install) | Visual feedback, flexibility |
| **Wan 2.2** | 20 min | 2 min/video | Highest | Free (after install) | Quality, hand accuracy |
| **HunyuanVideo** | 15 min | 2-3 min/video | High | Free (after install) | Image-to-video support |

---

## 🎯 Recommended Workflow

**For First-Time Users**:
1. Read `VIDEO_GENERATION_READY.md` (5 min)
2. Generate 1-2 test videos (10 min)
3. Run QA check (1 min)
4. If good: generate 50-100 videos in batches
5. Process through QA → Ingest → Reports

**Estimated Total Time**:
- Setup: 5-20 min (depends on method)
- Generate 50 videos: 1-3 hours
- QA + Ingest + Reports: 10 min
- Total: 1.5-3.5 hours for 50 validated videos

---

## 📁 File Locations

```
/Users/aharshi/MotionBloomAppVersion/

Documentation:
├── VIDEO_GENERATION_READY.md              ⭐ START HERE
├── VIDEO_GENERATION_START_HERE.md
├── VIDEO_GENERATION_QUICKSTART.md
├── AI_VIDEO_HARNESS.md
├── AI_VIDEO_HARNESS_QUICKSTART.md
├── AI_VIDEO_HARNESS_IMPLEMENTATION.md
└── DOCUMENTATION_INDEX.md

Scripts & Tools:
├── scripts/
│   ├── generate_videos.py                 (Main helper)
│   ├── video_generation_examples.py       (Code examples)
│   └── build_ai_video_commands.py         (Command generator)
├── batch_generate_tier_a.sh               (Pre-built Wan script)
└── topotremor/
    ├── ai_video_qa.py                     (Validation)
    ├── ai_video_ingestion.py              (Measurement extraction)
    └── ai_video_reporting.py              (Report generation)

Data:
├── datasets/topotremor/prompts/
│   ├── ai_video_prompts.csv               (25 templates)
│   └── ai_video_generation_commands.csv   (44 commands)
└── videos/                                 (Your generated videos go here)
```

---

## ✅ What's Ready

- [x] 25 prompt templates across 4 tiers
- [x] Helper script to list and export prompts
- [x] Code examples for 4 different generation methods
- [x] Pre-built Wan 2.2 batch script
- [x] QA validation module (3-stage gates)
- [x] Measurement extraction pipeline (optical flow)
- [x] Automated report generation
- [x] Complete documentation (1000+ lines)
- [x] Quick reference guides

---

## ⚠️ Before You Start

1. **Choose a generation method** (Replicate, ComfyUI, Wan, or HunyuanVideo)
2. **Verify Python 3.7+** installed: `python3 --version`
3. **For local methods**: Ensure sufficient disk space (50-100 videos ≈ 10-20GB)
4. **For local methods with GPU**: Check VRAM (Replicate needs none)

---

## 🆘 Getting Help

**"Where do I start?"**
→ Read `VIDEO_GENERATION_READY.md`

**"How do I generate videos?"**
→ Read `VIDEO_GENERATION_START_HERE.md`

**"What are the tiers?"**
→ See tier descriptions in this file, or read `AI_VIDEO_HARNESS.md`

**"How do I know if a video is good?"**
→ Run: `python3 topotremor/ai_video_qa.py videos/` and check `validity_label`

**"Can I use my own prompts?"**
→ Edit `datasets/topotremor/prompts/ai_video_prompts.csv` to add rows

---

## 📞 Next Steps

1. **Right now**: 
   ```bash
   cd /Users/aharshi/MotionBloomAppVersion
   cat VIDEO_GENERATION_READY.md
   ```

2. **Then**: Choose a method and generate your first video

3. **Finally**: Run QA and validate quality

You have everything you need. Let's generate videos! 🎬

---

**Questions?** All documentation is in the workspace. See `DOCUMENTATION_INDEX.md` for full index.
