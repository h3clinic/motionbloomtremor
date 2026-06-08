# AI Video Generation: Ready to Launch

**Status**: ✅ All systems ready for video generation  
**Date**: June 8, 2025  
**Next Step**: Choose a generation method and start creating videos

---

## What You Have Now

### 📋 Prompt Manifest
- **File**: `datasets/topotremor/prompts/ai_video_prompts.csv`
- **Prompts**: 25 templates across 4 tiers
- **Tiers**:
  - **Tier A** (5 prompts): Strict no-tremor (confidence 0.80-0.88)
  - **Tier B** (6 prompts): Hard negatives (tapping, waving, blur)
  - **Tier C** (5 prompts): Weak tremor-like (confidence 0.25-0.40)
  - **Tier D** (5 prompts): Artifacts (six fingers, melting, edge crawl)
- **Coverage**: All variations of hand positions, motions, and distortions

### 🔧 Helper Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `scripts/generate_videos.py` | List & export prompts | `python3 scripts/generate_videos.py list --tier A` |
| `scripts/video_generation_examples.py` | Show code examples | `python3 scripts/video_generation_examples.py` |
| `batch_generate_tier_a.sh` | Wan 2.2 batch script | `bash batch_generate_tier_a.sh` (after editing path) |
| `topotremor/ai_video_qa.py` | Validate generated videos | `python3 topotremor/ai_video_qa.py videos/` |
| `topotremor/ai_video_ingestion.py` | Extract measurements | `python3 topotremor/ai_video_ingestion.py batch ...` |
| `topotremor/ai_video_reporting.py` | Generate reports | `python3 topotremor/ai_video_reporting.py ...` |

### 📚 Documentation

| Document | Content |
|----------|---------|
| `VIDEO_GENERATION_START_HERE.md` | **← Start here** |
| `VIDEO_GENERATION_QUICKSTART.md` | Installation & usage guides |
| `AI_VIDEO_HARNESS.md` | Complete system specification |
| `AI_VIDEO_HARNESS_QUICKSTART.md` | 6-step workflow |
| `DOCUMENTATION_INDEX.md` | Full index of all docs |

---

## Three Fastest Paths to Your First Video

### 🚀 Path 1: Replicate (Easiest, 5 minutes)
```bash
# 1. Sign up at https://replicate.com/
# 2. Get API token

export REPLICATE_API_TOKEN="your_token_here"
pip install replicate

# 3. Create a Python script with one of our prompts
python3 scripts/video_generation_examples.py
# Copy the "Replicate (Python) Example" code

# 4. Run it
python3 my_replicate_example.py
```

**Pros**: No installation, instant results  
**Cons**: ~$0.10-0.30 per video (50 videos ≈ $5-15)

---

### 🎨 Path 2: ComfyUI (Beginner-Friendly, 10 minutes)
```bash
# 1. Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI && pip install -r requirements.txt

# 2. Run the UI
python main.py
# Opens at http://localhost:8188

# 3. In the UI:
# - Load a video generation model
# - Copy prompt text from scripts/video_generation_examples.py
# - Generate!
```

**Pros**: GUI, visual feedback, flexible  
**Cons**: Slower generation, requires GPU

---

### 💪 Path 3: Wan 2.2 (Best Quality, 15 minutes)
```bash
# 1. Install Wan 2.2
git clone https://github.com/MaaS2/Wan.git
cd Wan && pip install -r requirements.txt
# Download model weights (follow Wan README)

# 2. Use our batch script
cd /Users/aharshi/MotionBloomAppVersion
# Edit batch_generate_tier_a.sh to set correct WAN_DIR path
bash batch_generate_tier_a.sh

# 3. Videos appear in videos/no_tremor_still/
```

**Pros**: Best hand quality, few artifacts  
**Cons**: Requires GPU with 24GB+ VRAM, slowest generation

---

## Recommended Starting Point

**Choose based on your setup:**

| Your Setup | Recommended | Reason |
|------------|-------------|--------|
| No GPU | **Replicate** | Instant, no setup |
| GPU with <12GB VRAM | **Replicate** | Can't run models locally |
| GPU with 12-24GB VRAM | **ComfyUI** | Good balance |
| GPU with 24GB+ VRAM | **Wan 2.2** | Best quality |
| Want GUI | **ComfyUI** | Easiest to use |
| Want best quality | **Wan 2.2** | Fewest artifacts |

---

## Your First Video (Step-by-Step)

### Step 1: View Tier A Prompts
```bash
cd /Users/aharshi/MotionBloomAppVersion
python3 scripts/generate_videos.py list --tier A
```

Output shows:
- `TIER_A_1` through `TIER_A_5`
- Category: `no_tremor_still`, `no_tremor_smooth_macro`, `no_tremor_palm_down`
- All marked as `expected_tremor_present: 0`

### Step 2: Generate 1 Test Video

**Using Replicate** (easiest):
```python
import replicate
import os

os.environ['REPLICATE_API_TOKEN'] = 'your_token'

# Use Tier A_1 prompt
prompt = "One realistic human hand palm facing camera with five clearly visible fingers. Fixed white background. Neutral skin tone. No motion. Still pose. Natural skin texture. No jewelry no gloves no text."
negative = "Extra fingers six fingers fused fingers missing fingers malformed hand distorted hand warped skin changing finger count duplicate hand text watermark gloves jewelry camera shake hand rotation"

output = replicate.run(
    "lumalabs/lumix:5616461203ac5c74c3d43854797d94c7595bcedf11499ccee2a628a6e8cd3f1e",
    input={"prompt": prompt, "negative_prompt": negative, "duration": 3}
)

print(f"Video: {output}")
```

**Using ComfyUI**: Paste prompt in UI, generate  
**Using Wan 2.2**: Run command from `scripts/video_generation_examples.py`

### Step 3: Validate with QA
```bash
mkdir -p test_videos
# Copy your generated video to test_videos/

python3 topotremor/ai_video_qa.py test_videos/
cat test_videos/qa_results.json
```

Check for:
- `"validity_label": "VALID"` or `"WEAK"` (good)
- `"artifact_score": < 0.5` (good)
- `"visible_hand_ratio": > 0.6` (good)
- `"finger_count": >= 4` (good)

### Step 4: If Passed, Generate More
```bash
# Generate all Tier A videos (5 total)
bash batch_generate_tier_a.sh

# Or use helper to generate Tier B
python3 scripts/generate_videos.py export \
  --method wan --tier B --output batch_tier_b.sh

bash batch_tier_b.sh
```

---

## Full Pipeline After Video Generation

Once you have 10-20 validated videos:

```bash
# 1. Run QA on all
python3 topotremor/ai_video_qa.py videos/

# 2. Ingest valid ones
python3 topotremor/ai_video_ingestion.py batch \
  --video_list videos/qa_results.json

# 3. Generate reports
python3 topotremor/ai_video_reporting.py \
  videos/qa_results.json \
  ingested_windows/ \
  reports/

# 4. Use in training
# Import windows with label_confidence as sample weights
```

---

## Key Files You'll Use

```
/Users/aharshi/MotionBloomAppVersion/
├── datasets/topotremor/prompts/
│   ├── ai_video_prompts.csv           ← 25 prompt templates
│   └── ai_video_generation_commands.csv (read-only reference)
├── videos/                              ← Your generated videos go here
├── scripts/
│   ├── generate_videos.py              ← Main helper tool
│   ├── video_generation_examples.py    ← Code examples
│   └── build_ai_video_commands.py      (read-only)
├── topotremor/
│   ├── ai_video_qa.py                  ← Validate videos
│   ├── ai_video_ingestion.py           ← Extract measurements
│   └── ai_video_reporting.py           ← Generate reports
├── VIDEO_GENERATION_START_HERE.md      ← Quick ref
├── batch_generate_tier_a.sh            ← Pre-built Wan script
└── (other docs)
```

---

## Expected Timeline

| Step | Method | Time | Notes |
|------|--------|------|-------|
| Setup | Replicate | 5 min | API key only |
| Setup | ComfyUI | 10 min | Install + models |
| Setup | Wan 2.2 | 20 min | Install + weights |
| Gen 1 video | Any | 1-5 min | Test quality |
| QA check | All | < 1 min | Run on video |
| Gen 50 videos | Replicate | ~45 min | ~1 min per video + queue |
| Gen 50 videos | ComfyUI | ~150 min | ~3 min per video |
| Gen 50 videos | Wan 2.2 | ~100 min | ~2 min per video |
| Batch QA | All | ~2 min | Parallel processing |
| Ingest | All | ~5 min | Per-window processing |
| Reports | All | < 1 min | JSON generation |

---

## Troubleshooting Quick Links

**"Module not found"**: Check you're in the right directory (`/Users/aharshi/MotionBloomAppVersion`)

**"Python version error"**: Use `python3` not `python`

**"API token invalid"**: `export REPLICATE_API_TOKEN="..."` and restart terminal

**"Out of memory"**: Use Replicate (cloud-based), or reduce duration/resolution

**"Videos have artifacts"**: Check negative prompt includes "extra fingers", try image-to-video mode

**"QA rejects videos"**: Check artifact_score and finger_count, may need better prompts

---

## Next Action

**Right now, do this**:

```bash
cd /Users/aharshi/MotionBloomAppVersion

# Option A: See what prompts are available
python3 scripts/generate_videos.py list --tier A

# Option B: See code examples for your chosen method
python3 scripts/video_generation_examples.py

# Option C: Jump straight to generating (if you already have a method set up)
# [Follow path from section above]
```

---

## Questions?

**Q: Which tier should I generate first?**  
A: **Tier A** (no-tremor) - easiest, highest confidence

**Q: How many videos total?**  
A: Start with 50-100 across all tiers. Scale after validating quality.

**Q: Will AI videos make my model worse?**  
A: No - our weak label system (confidence 0.25-0.4) and sample weighting prevent harm. Tier A videos are confidence 0.80-0.88.

**Q: Can I use images instead of generating videos?**  
A: Yes - HunyuanVideo and Wan 2.2 support image-to-video mode (use `reference_image_path` from manifest).

**Q: How do I know if a video is good?**  
A: Run QA (`topotremor/ai_video_qa.py`) - look for `validity_label: VALID` and `artifact_score < 0.5`.

---

## You're Ready 🚀

Everything is set up. Pick a method above and generate your first video. You've got this!

For detailed instructions, see: `VIDEO_GENERATION_START_HERE.md`
