# AI Video Generation Quick Start Guide

**Status**: Ready to generate videos  
**Generated**: June 8, 2026  
**Command Manifest**: `datasets/topotremor/prompts/ai_video_generation_commands.csv`

---

## Overview

You have **44 ready-to-use generation commands** across **4 tiers** and **4 models**:

| Tier | Model | Commands | Purpose |
|------|-------|----------|---------|
| **A** | Wan 2.2 | 4 | Strict no-tremor (high confidence) |
| **B** | Wan 2.2 | 3 | Hard negatives (tapping, waving) |
| **C** | Wan 2.2 | 2 | Weak tremor-like |
| **D** | Wan 2.2 | 2 | Artifacts (six fingers, merging) |
| **All** | HunyuanVideo, LTX-Video, ComfyUI | 27 | Alternative models |

---

## Video Generation Options

### Option 1: Use Open-Source Models Locally

You can use free, open-source video generation models:

**Recommended: Wan 2.2**
- Best quality hand rendering
- Fewer anatomical artifacts
- Stable output
- GitHub: https://github.com/MaaS2/Wan

**Installation**:
```bash
# Clone the repository
git clone https://github.com/MaaS2/Wan.git
cd Wan

# Install dependencies
pip install -r requirements.txt

# Download model weights
# (Follow instructions at https://github.com/MaaS2/Wan)
```

**Usage**:
```bash
python -m inference.generate \
  --prompt "Close-up video of one realistic human hand..." \
  --negative_prompt "extra fingers, six fingers, merged fingers..." \
  --duration 3 \
  --fps 30 \
  --output /path/to/output.mp4
```

---

### Option 2: Use HunyuanVideo

**GitHub**: https://github.com/tencent/HunyuanVideo

**Installation**:
```bash
git clone https://github.com/tencent/HunyuanVideo.git
cd HunyuanVideo
pip install -r requirements.txt
```

---

### Option 3: Use ComfyUI (GUI-Based)

**ComfyUI** is a node-based interface for video generation (easier for beginners).

**Installation**:
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Run the UI
python main.py
```

Then use the `ComfyUI workflow` JSON from our manifest.

---

### Option 4: Use Cloud Services (Paid)

If you don't want to install locally, you can use paid cloud services:

- **Runway ML**: https://runwayml.com/
- **Eleven Labs**: https://elevenlabs.io/
- **Leonardo.ai**: https://leonardo.ai/

These have web interfaces and handle video generation in the cloud.

---

## Getting Started: Simple Path

### Step 1: View Available Commands

```bash
cd /Users/aharshi/MotionBloomAppVersion

# View first few commands
head -5 datasets/topotremor/prompts/ai_video_generation_commands.csv

# Count by tier
awk -F',' 'NR>1 {print $2}' datasets/topotremor/prompts/ai_video_generation_commands.csv | sort | uniq -c
```

### Step 2: Extract a Specific Command

To get a readable command for Wan 2.2 text-to-video:

```bash
# Extract Tier A no-tremor command
awk -F',' 'NR==2 {print $NF}' datasets/topotremor/prompts/ai_video_generation_commands.csv
```

This will show you the exact command to copy and modify.

### Step 3: Create Output Directory

```bash
mkdir -p videos/{no_tremor_still,hard_negative_finger_tap,weak_tremor_fingertip,generator_artifact_six_fingers}
```

---

## Simple Video Generation (No Installation Required)

### Use Replicate (Easiest)

If you want to generate videos without installing anything locally, use **Replicate**:

1. **Sign up** at https://replicate.com/
2. **Generate API token** and set env variable:
   ```bash
   export REPLICATE_API_TOKEN="your_token_here"
   ```

3. **Use Replicate Python client**:
   ```bash
   pip install replicate
   ```

4. **Generate a video** (example):
   ```python
   import replicate

   output = replicate.run(
       "lumalabs/lumix:5616461203ac5c74c3d43854797d94c7595bcedf11499ccee2a628a6e8cd3f1e",
       input={
           "prompt": "Close-up of one realistic human hand, palm facing camera, five fingers clearly visible, fixed white background, completely still, no motion, no tremor.",
           "negative_prompt": "extra fingers, six fingers, fused fingers, malformed hand, warped skin, camera shake, tremor, shaking",
           "duration": 3,
           "fps": 30
       }
   )
   print(output)  # Returns URL to video
   ```

---

## Step-by-Step: Generate Your First Video

### Using Wan 2.2 (Locally)

**1. Install Wan 2.2**:
```bash
cd /tmp
git clone https://github.com/MaaS2/Wan.git
cd Wan
pip install -e .
```

**2. Download model weights** (follow Wan README)

**3. Create a simple prompt file** (`prompt.txt`):
```
Close-up video of one realistic human hand, palm facing camera, 
five fingers clearly visible, fixed white background, neutral skin tone, 
completely still pose, natural skin texture, no jewelry no gloves no text, 
no motion, no tremor.
```

**4. Generate video**:
```bash
cd /path/to/Wan

python -m inference.generate \
  --prompt "$(cat prompt.txt)" \
  --negative_prompt "extra fingers, six fingers, fused fingers, missing fingers, malformed hand, distorted hand, warped skin, changing finger count, duplicate hand, text, watermark, gloves, jewelry, camera shake, tremor, shaking" \
  --duration 3 \
  --fps 30 \
  --output /Users/aharshi/MotionBloomAppVersion/videos/no_tremor_still/001.mp4
```

**5. Check the output**:
```bash
ls -lh /Users/aharshi/MotionBloomAppVersion/videos/no_tremor_still/001.mp4
```

---

## Batch Generation Script

Create a Python script to generate multiple videos:

```python
# batch_generate.py
import subprocess
import csv
from pathlib import Path

# Load manifest
manifest_path = "datasets/topotremor/prompts/ai_video_prompts.csv"
output_base = Path("videos")

with open(manifest_path) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 5:  # Limit to first 5 for testing
            break
        
        prompt = row['prompt_text']
        neg_prompt = row['negative_prompt']
        category = row['category']
        duration = row['duration_sec']
        fps = row['fps']
        
        output_dir = output_base / category
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{i:03d}.mp4"
        
        # Build command (adjust for your installed model)
        cmd = [
            "python", "-m", "inference.generate",
            "--prompt", prompt,
            "--negative_prompt", neg_prompt,
            "--duration", str(int(float(duration))),
            "--fps", str(int(float(fps))),
            "--output", str(output_path)
        ]
        
        print(f"Generating {output_path}...")
        # subprocess.run(cmd, cwd="/path/to/Wan")
        print(f"  Command: {' '.join(cmd)}")
```

---

## What to Do Next

### If You Have Wan 2.2 Installed:
1. Generate 5-10 test videos across different tiers
2. Run QA on them: `python topotremor/ai_video_qa.py videos/`
3. Check validity_label output

### If You Don't Have Wan 2.2:
1. Use Replicate (easiest, cloud-based)
2. Or install ComfyUI (GUI-friendly)
3. Or use an online service (Runway ML, etc.)

### To Monitor Generation Progress:
```bash
# Watch generated videos
ls -lh videos/
du -sh videos/

# Count videos by tier
find videos -name "*.mp4" | awk -F/ '{print $2}' | sort | uniq -c
```

---

## Next: QA Check

Once you have generated videos, run:

```bash
python topotremor/ai_video_qa.py videos/ > qa_results.json

# Preview results
head -20 qa_results.json
```

This will check each video for:
- Finger count (should be 5)
- Hand stability
- Temporal consistency
- Camera motion
- Overall validity

---

## Troubleshooting

### "Model not found"
- Download model weights first (Wan README has instructions)
- Or use Replicate (cloud-based, no local setup needed)

### "Out of memory"
- Reduce duration or fps
- Use a smaller model variant
- Run on a machine with more VRAM

### "Quality is low / Many artifacts"
- Check negative prompt (more specific = better)
- Use image-to-video instead of text-to-video
- Adjust prompt keywords (avoid vague terms)

---

## Resources

**Wan 2.2**: https://github.com/MaaS2/Wan  
**HunyuanVideo**: https://github.com/tencent/HunyuanVideo  
**ComfyUI**: https://github.com/comfyanonymous/ComfyUI  
**Replicate**: https://replicate.com/  

**Our prompts**: `datasets/topotremor/prompts/ai_video_prompts.csv`  
**Our commands**: `datasets/topotremor/prompts/ai_video_generation_commands.csv`

---

## Summary

**Start here**:
1. Pick a video generation method (Replicate easiest, Wan 2.2 best quality)
2. Generate 5-10 test videos
3. Run QA: `python topotremor/ai_video_qa.py videos/`
4. Check results

**Ready to ingest**:
5. Once QA passes, run ingestion: `python topotremor/ai_video_ingestion.py batch ...`
6. Generate reports: `python topotremor/ai_video_reporting.py ...`
7. Use in training with sample weights

**Questions?** Check `AI_VIDEO_HARNESS_QUICKSTART.md` for full workflow.
