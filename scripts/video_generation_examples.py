#!/usr/bin/env python3
"""
Simple example: Generate videos from our prompt manifest
Shows how to use each of the 4 supported methods
"""

import csv
import json
import subprocess
from pathlib import Path
from typing import Optional


def load_prompts(manifest_path: str = "datasets/topotremor/prompts/ai_video_prompts.csv"):
    """Load prompts from manifest."""
    prompts = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            prompts.append(row)
    return prompts


def example_wan_command(prompt: dict):
    """Show example Wan 2.2 command."""
    print("\n=== Wan 2.2 Example ===")
    print("Install Wan 2.2 first:")
    print("  git clone https://github.com/MaaS2/Wan.git")
    print("  cd Wan && pip install -r requirements.txt")
    print("\nThen run:")
    
    cmd = (
        f"cd /path/to/Wan && "
        f"python -m inference.generate "
        f'--prompt "{prompt["prompt_text"]}" '
        f'--negative_prompt "{prompt["negative_prompt"]}" '
        f"--duration {int(float(prompt['duration_sec']))} "
        f"--fps {int(float(prompt['fps']))} "
        f"--output {prompt['output_path']}"
    )
    print(cmd)


def example_replicate_python(prompt: dict):
    """Show example Replicate Python code."""
    print("\n=== Replicate (Python) Example ===")
    print("Install: pip install replicate")
    print("Set API token: export REPLICATE_API_TOKEN='...'")
    print("\nCode:")
    
    code = f"""
import replicate

# Generate video
output = replicate.run(
    "lumalabs/lumix:5616461203ac5c74c3d43854797d94c7595bcedf11499ccee2a628a6e8cd3f1e",
    input={{
        "prompt": "{prompt['prompt_text']}",
        "negative_prompt": "{prompt['negative_prompt']}",
        "duration": {int(float(prompt['duration_sec']))},
        "fps": {int(float(prompt['fps']))}
    }}
)

print(f"Video URL: {{output}}")

# Download the video
import urllib.request
urllib.request.urlretrieve(output, "{prompt['output_path']}")
print(f"Saved to: {prompt['output_path']}")
"""
    print(code)


def example_hunyuan_command(prompt: dict):
    """Show example HunyuanVideo command."""
    print("\n=== HunyuanVideo Example ===")
    print("Install HunyuanVideo first:")
    print("  git clone https://github.com/tencent/HunyuanVideo.git")
    print("  cd HunyuanVideo && pip install -r requirements.txt")
    print("\nThen run:")
    
    if prompt.get('generation_mode') == 'IMAGE_TO_VIDEO':
        # Use image-to-video if reference image provided
        cmd = (
            f"python inference.py "
            f"--prompt '{prompt['prompt_text']}' "
            f"--neg_prompt '{prompt['negative_prompt']}' "
            f"--reference_image {prompt.get('reference_image_path', 'hand.jpg')} "
            f"--output {prompt['output_path']}"
        )
    else:
        # Use text-to-video
        cmd = (
            f"python inference.py "
            f"--prompt '{prompt['prompt_text']}' "
            f"--neg_prompt '{prompt['negative_prompt']}' "
            f"--output {prompt['output_path']}"
        )
    
    print(cmd)


def example_comfyui(prompt: dict):
    """Show example ComfyUI workflow."""
    print("\n=== ComfyUI Example ===")
    print("Install ComfyUI:")
    print("  git clone https://github.com/comfyanonymous/ComfyUI.git")
    print("  cd ComfyUI && pip install -r requirements.txt")
    print("  python main.py  # Opens UI at http://localhost:8188")
    
    print("\nIn ComfyUI UI:")
    print("1. Load text-to-video model")
    print("2. Paste prompt text:")
    print(f"   {prompt['prompt_text'][:100]}...")
    print("3. Paste negative prompt:")
    print(f"   {prompt['negative_prompt'][:100]}...")
    print(f"4. Set duration: {int(float(prompt['duration_sec']))} seconds")
    print(f"5. Set fps: {int(float(prompt['fps']))}")
    print("6. Queue and generate")


def main():
    """Run examples."""
    # Load first Tier A prompt
    prompts = load_prompts()
    tier_a = [p for p in prompts if p['prompt_id'].startswith('TIER_A')]
    
    if not tier_a:
        print("No Tier A prompts found")
        return
    
    prompt = tier_a[0]  # First Tier A prompt
    
    print("=" * 70)
    print("AI Video Generation Examples")
    print("=" * 70)
    print(f"\nUsing prompt: {prompt['prompt_id']}")
    print(f"Category: {prompt['category']}")
    print(f"Duration: {prompt['duration_sec']} sec")
    print(f"FPS: {prompt['fps']}")
    
    # Show all examples
    example_wan_command(prompt)
    example_replicate_python(prompt)
    example_hunyuan_command(prompt)
    example_comfyui(prompt)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Choose your preferred method above
2. Generate 1-2 test videos
3. Run QA check:
   python3 topotremor/ai_video_qa.py videos/
4. If passed, generate more videos
5. Ingest and prepare for training

Quick reference:
  - List prompts: python3 scripts/generate_videos.py list --tier A
  - Export batch script: python3 scripts/generate_videos.py export --method wan --tier A --output batch.sh
  - Check QA results: cat qa_results.json
    """)


if __name__ == '__main__':
    main()
