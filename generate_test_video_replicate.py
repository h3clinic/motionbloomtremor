#!/usr/bin/env python3
"""
Generate 1 test video using Replicate API (easiest method)

Requirements:
1. pip install replicate
2. export REPLICATE_API_TOKEN="your_api_token_here"

Get token from: https://replicate.com/account/api-tokens
"""

import os
import sys
import urllib.request
from pathlib import Path

# Check for API token
api_token = os.environ.get('REPLICATE_API_TOKEN')
if not api_token:
    print("❌ ERROR: REPLICATE_API_TOKEN not set")
    print("\nSet it with:")
    print("  export REPLICATE_API_TOKEN='your_token_here'")
    print("\nGet your token from: https://replicate.com/account/api-tokens")
    sys.exit(1)

try:
    import replicate
except ImportError:
    print("❌ ERROR: replicate module not installed")
    print("\nInstall with:")
    print("  pip install replicate")
    sys.exit(1)

# Tier A_1 prompt
prompt = """One realistic human hand palm facing camera with five clearly visible fingers. Fixed white background. Neutral skin tone. No motion. Still pose. Natural skin texture. No jewelry no gloves no text."""

negative_prompt = """Extra fingers six fingers fused fingers missing fingers malformed hand distorted hand warped skin changing finger count duplicate hand text watermark gloves jewelry camera shake hand rotation"""

print("=" * 80)
print("Generating test video #1 (Tier A: no_tremor_still)")
print("=" * 80)
print(f"\nPrompt: {prompt[:60]}...")
print(f"Negative: {negative_prompt[:60]}...")
print("\nGenerating... this may take 1-3 minutes\n")

try:
    # Use Luma AI model via Replicate
    output = replicate.run(
        "lumalabs/lumix:5616461203ac5c74c3d43854797d94c7595bcedf11499ccee2a628a6e8cd3f1e",
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": 3,
            "fps": 30,
            "aspect_ratio": "1:1"
        }
    )
    
    print(f"✅ Video generated!")
    print(f"\nOutput URL: {output}")
    
    # Create output directory
    output_dir = Path("videos/no_tremor_still")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the video
    output_path = output_dir / "test_001.mp4"
    print(f"\nDownloading to: {output_path}")
    urllib.request.urlretrieve(output, str(output_path))
    
    print(f"✅ Saved: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Check video quality: open {output_path}")
    print(f"2. Run QA: python3 topotremor/ai_video_qa.py videos/")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("- Check API token is valid: export REPLICATE_API_TOKEN='...'")
    print("- Check internet connection")
    print("- Check Replicate service status: https://status.replicate.com/")
    sys.exit(1)
