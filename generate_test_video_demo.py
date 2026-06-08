#!/usr/bin/env python3
"""
Generate 1 test video - DEMO MODE
Creates a mock video file for testing the QA pipeline

This is useful for testing without API credentials or GPU.
Run: python3 generate_test_video_demo.py
"""

import sys
from pathlib import Path
import subprocess

print("=" * 80)
print("Generating test video #1 (DEMO MODE - Mock MP4 file)")
print("=" * 80)

# Create output directory
output_dir = Path("videos/no_tremor_still")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "test_001.mp4"

print(f"\nCreating mock video file: {output_path}")

# Create a simple test video using ffmpeg (if available)
# This creates a 3-second black video - good for testing pipeline
try:
    cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", "color=c=black:s=1024x768:d=3",
        "-pix_fmt", "yuv420p",
        "-y",  # Overwrite
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print(f"✅ Mock video created: {output_path}")
        print(f"\nFile size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"\nNext steps:")
        print(f"1. Run QA: python3 topotremor/ai_video_qa.py videos/")
        print(f"2. Check results: cat videos/qa_results.json | python3 -m json.tool | head -50")
    else:
        print(f"⚠️  ffmpeg not available or failed")
        print(f"\nInstead, try:")
        print(f"1. Install ffmpeg: brew install ffmpeg")
        print(f"2. Or use Replicate: python3 generate_test_video_replicate.py")
        sys.exit(1)

except FileNotFoundError:
    print("❌ ffmpeg not found")
    print("\nInstall with:")
    print("  brew install ffmpeg")
    print("\nOr use Replicate instead:")
    print("  python3 generate_test_video_replicate.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
