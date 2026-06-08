# Video Generation: Start Here

**You now have:**
- ✅ 25 prompt templates across 4 tiers
- ✅ Helper script to generate commands
- ✅ QA system ready to validate videos
- ✅ Full measurement pipeline

---

## Quick Start (Choose One Path)

### Path 1: Use Replicate (Easiest, No Installation)

**No setup needed. Cloud-based video generation.**

**Cost**: ~$0.10-0.30 per 3-sec video

**Steps**:

1. **Sign up** at https://replicate.com/

2. **Get API token**:
   ```bash
   export REPLICATE_API_TOKEN="your_token_here"
   ```

3. **Install replicate client**:
   ```bash
   pip install replicate
   ```

4. **Generate videos** using example commands from manifest:
   ```bash
   # View Tier A prompts
   python3 scripts/generate_videos.py list --tier A --limit 3
   ```

5. **Copy one prompt and adapt**:
   ```python
   import replicate
   
   output = replicate.run(
       "lumalabs/lumix:5616...",  # Model ID from Replicate
       input={
           "prompt": "One realistic human hand palm facing camera with five clearly visible fingers...",
           "negative_prompt": "Extra fingers six fingers fused fingers...",
           "duration": 3
       }
   )
   print(output)  # Video URL
   ```

---

### Path 2: Install Wan 2.2 (Best Quality)

**High-quality hand rendering, fewer artifacts, image-to-video support.**

**Requirements**: GPU with 24GB+ VRAM (or 12GB with optimizations)

**Installation**:

```bash
# 1. Clone Wan repository
git clone https://github.com/MaaS2/Wan.git
cd Wan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights
# (Follow instructions in Wan README)

# 4. Copy batch script
cp /Users/aharshi/MotionBloomAppVersion/batch_generate_tier_a.sh .

# 5. Edit to point to your videos directory
# Replace: /path/to/Wan → your actual Wan path
# Replace: videos/ → /path/to/output/videos/

# 6. Run batch generation
bash batch_generate_tier_a.sh
```

---

### Path 3: Use ComfyUI (GUI-Based)

**Node-based interface, flexible, beginner-friendly.**

**Installation**:
```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install -r requirements.txt

# Run UI
python main.py
# Opens at http://localhost:8188
```

Then use the prompt text from `ai_video_prompts.csv` directly in ComfyUI.

---

## Test Video Generation

**Before generating 50+ videos, test with 1-2 videos first:**

```bash
# Example: Generate 1 Tier A video
python3 scripts/generate_videos.py list --tier A --limit 1

# Copy the prompt text and run via your chosen method
# (Replicate, Wan, or ComfyUI)
```

---

## Validate Generated Video

Once you have a video, run our QA system:

```bash
# Create test directory
mkdir -p test_videos
# Copy your generated video there

# Run QA
python3 topotremor/ai_video_qa.py test_videos/

# Check output
cat test_videos/qa_results.json
```

**Good QA signs**:
- `validity_label: "VALID"` or `"WEAK"`
- `artifact_score < 0.5`
- `visible_hand_ratio > 0.6`
- `finger_count >= 4` (at least 4 fingers visible)

---

## Generate Multiple Videos

Once you've tested 1-2 videos successfully:

### Option A: Wan 2.2 Batch Script

```bash
cd /path/to/Wan

# Generate ALL Tier A videos (5 videos)
bash /Users/aharshi/MotionBloomAppVersion/batch_generate_tier_a.sh

# Or generate Tier B (hard negatives)
python3 /Users/aharshi/MotionBloomAppVersion/scripts/generate_videos.py \
  export --tier B --method wan --output batch_tier_b.sh

bash batch_tier_b.sh
```

### Option B: Python Helper Script

```bash
cd /Users/aharshi/MotionBloomAppVersion

# List all available prompts by tier
python3 scripts/generate_videos.py list                    # All
python3 scripts/generate_videos.py list --tier A           # Tier A only
python3 scripts/generate_videos.py list --tier B --limit 3 # First 3 Tier B

# Export for your chosen method
python3 scripts/generate_videos.py export \
  --method wan \
  --tier A \
  --output my_batch_script.sh

python3 scripts/generate_videos.py export \
  --method replicate \
  --limit 5 \
  --output my_replicate_batch.py
```

---

## Next Steps After Generation

### 1. Run QA on All Generated Videos

```bash
# Assuming videos are in videos/
python3 topotremor/ai_video_qa.py videos/ > qa_results.json

# Check summary
python3 -c "
import json
with open('qa_results.json') as f:
    results = json.load(f)
    valid = sum(1 for r in results if r['validity_label'] in ['VALID', 'WEAK'])
    invalid = sum(1 for r in results if r['validity_label'] in ['INVALID', 'GENERATOR_ARTIFACT'])
    print(f'Valid/Weak: {valid}, Invalid/Artifacts: {invalid}')
"
```

### 2. Ingest Valid Videos

```bash
# Create manifest of valid videos
python3 topotremor/ai_video_ingestion.py batch \
  --video_list qa_results.json \
  --output ingested_windows.json
```

### 3. Generate Reports

```bash
python3 topotremor/ai_video_reporting.py \
  qa_results.json \
  ingested_windows/ \
  reports/

# View reports
ls -lh reports/
```

### 4. Prepare for Training

```bash
# Weak label windows ready for training
# With sample weights: window['label_confidence']
```

---

## Complete Workflow Summary

```
1. Choose generation method
   ↓
2. Generate test video(s) (1-2)
   ↓
3. Run QA check
   ↓
4. If passed: Generate batch (10-50 videos)
   ↓
5. Run QA on batch
   ↓
6. Ingest valid videos
   ↓
7. Generate reports
   ↓
8. Use in training with weak labels
```

---

## Command Reference

**List commands**:
```bash
python3 scripts/generate_videos.py list [--tier A|B|C|D] [--model MODEL] [--limit N]
```

**Export batch script**:
```bash
python3 scripts/generate_videos.py export \
  --method wan|replicate \
  [--tier A|B|C|D] \
  [--limit N] \
  --output FILENAME
```

**Run QA**:
```bash
python3 topotremor/ai_video_qa.py VIDEO_DIR/ [--output OUTPUT_JSON]
```

**Ingest videos**:
```bash
python3 topotremor/ai_video_ingestion.py batch \
  --video_list QA_RESULTS.json \
  --output OUTPUT.json
```

**Generate reports**:
```bash
python3 topotremor/ai_video_reporting.py \
  QA_RESULTS.json \
  INGESTED_VIDEOS/ \
  OUTPUT_DIR/
```

---

## Troubleshooting

### "Model not found"
- **Replicate**: API key not set or invalid
- **Wan**: Model weights not downloaded
- **ComfyUI**: Model not installed via ComfyUI manager

### "Out of memory"
- Reduce video duration (try 2 sec instead of 3)
- Reduce resolution (768 instead of 1024)
- Use a smaller model variant
- Use cloud service (Replicate)

### "Quality looks bad / artifacts in videos"
- More specific negative prompt
- Use image-to-video instead of text-to-video
- Try different seed values
- Check that generation parameters match prompts

### "QA rejects videos"
- Check `artifact_score` (should be < 0.5)
- Check `finger_count` (should be ≈5)
- Check `visible_hand_ratio` (should be > 0.6)
- Review rejected videos manually

---

## Resources

**Our Files**:
- Prompts: `datasets/topotremor/prompts/ai_video_prompts.csv`
- Helper script: `scripts/generate_videos.py`
- This guide: `VIDEO_GENERATION_QUICKSTART.md`

**External**:
- Wan 2.2: https://github.com/MaaS2/Wan
- HunyuanVideo: https://github.com/tencent/HunyuanVideo
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- Replicate: https://replicate.com/

**Documentation**:
- Harness overview: `AI_VIDEO_HARNESS.md`
- Full workflow: `AI_VIDEO_HARNESS_QUICKSTART.md`
- Implementation: `AI_VIDEO_HARNESS_IMPLEMENTATION.md`

---

## Need Help?

**Question**: Which method should I use?  
**Answer**: 
- **New to video generation?** → Use Replicate (easiest)
- **Have GPU?** → Install Wan 2.2 (best quality)
- **Want GUI?** → Use ComfyUI

**Question**: How many videos do I need?  
**Answer**: Start with 50-100 across all 4 tiers (12-13 per tier). Scale up after validating quality.

**Question**: How long does generation take?  
**Answer**: 
- Replicate: ~30-60 sec per video (cloud)
- Wan 2.2: ~60-120 sec per video (depends on hardware)
- ComfyUI: ~2-5 min per video (depends on settings)

---

**Ready to start?** Choose your method above and follow the steps. Start with 1-2 test videos!
