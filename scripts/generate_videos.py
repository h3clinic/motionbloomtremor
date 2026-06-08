#!/usr/bin/env python3
"""
AI Video Generation Helper
Simplifies the process of generating videos from our prompt manifest.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
import argparse


def load_manifest(manifest_path: str) -> list[dict]:
    """Load the prompt manifest CSV."""
    prompts = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def filter_by_tier(prompts: list[dict], tier: str) -> list[dict]:
    """Filter prompts by tier (A, B, C, D)."""
    # Check prompt_id for tier prefix (e.g., TIER_A_1)
    return [p for p in prompts if p.get('prompt_id', '').startswith(f'TIER_{tier.upper()}')]


def filter_by_model(prompts: list[dict], model: str) -> list[dict]:
    """Filter prompts by model."""
    return [p for p in prompts if p.get('model_name', '').lower() == model.lower()]


def create_output_dirs(prompts: list[dict], base_path: str = "videos"):
    """Create output directories for each category."""
    categories = set(p.get('category', 'unknown') for p in prompts)
    base = Path(base_path)
    base.mkdir(exist_ok=True)
    
    for category in categories:
        (base / category).mkdir(exist_ok=True, parents=True)
    
    return base


def show_prompts(prompts: list[dict], limit: Optional[int] = None):
    """Display available prompts in a table format."""
    if limit:
        prompts = prompts[:limit]
    
    print(f"\n{'ID':<15} {'Tier':<6} {'Category':<40} {'Tremor':<8}")
    print("-" * 75)
    
    for p in prompts:
        pid = p.get('prompt_id', 'unknown')
        category = p.get('category', 'unknown')
        tremor = p.get('expected_tremor_present', 'unknown')
        
        # Extract tier from prompt_id (TIER_A_1 -> A)
        tier = 'N/A'
        if pid.startswith('TIER_'):
            tier = pid.split('_')[1]
        
        print(f"{pid:<15} {tier:<6} {category:<40} {str(tremor):<8}")
    
    print(f"\nTotal: {len(prompts)} prompts")


def generate_wan_command(prompt: dict, output_path: str) -> str:
    """Generate a Wan 2.2 command for a prompt."""
    prompt_text = prompt.get('prompt_text', '')
    negative_prompt = prompt.get('negative_prompt', '')
    duration = int(float(prompt.get('duration_sec', '3')))
    fps = int(float(prompt.get('fps', '30')))
    
    # Build the command
    cmd = (
        f"python -m inference.generate "
        f'--prompt "{prompt_text}" '
        f'--negative_prompt "{negative_prompt}" '
        f"--duration {duration} "
        f"--fps {fps} "
        f"--output {output_path}"
    )
    
    return cmd


def generate_replicate_command(prompt: dict, output_path: str) -> str:
    """Generate a Python script for Replicate API."""
    prompt_text = prompt.get('prompt_text', '')
    negative_prompt = prompt.get('negative_prompt', '')
    duration = int(float(prompt.get('duration_sec', '3')))
    
    script = f'''
import replicate
import json

output = replicate.run(
    "lumalabs/lumix:5616461203ac5c74c3d43854797d94c7595bcedf11499ccee2a628a6e8cd3f1e",
    input={{
        "prompt": "{prompt_text}",
        "negative_prompt": "{negative_prompt}",
        "duration": {duration}
    }}
)

# Save output
with open("{output_path}.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Video saved to: {{output}}")
'''
    return script.strip()


def export_command_script(prompts: list[dict], output_file: str, method: str = "wan"):
    """Export a batch script for video generation."""
    
    if method == "wan":
        script = "#!/bin/bash\n# Batch video generation using Wan 2.2\n\n"
        script += "WAN_DIR=/path/to/Wan\ncd $WAN_DIR\n\n"
        
        for i, prompt in enumerate(prompts):
            category = prompt.get('category', 'unknown')
            output_dir = f"videos/{category}"
            output_path = f"{output_dir}/{i:03d}.mp4"
            
            cmd = generate_wan_command(prompt, output_path)
            script += f"# Prompt {i}: {prompt.get('prompt_id', 'unknown')}\n"
            script += f"{cmd}\n\n"
    
    elif method == "replicate":
        script = "#!/usr/bin/env python3\n# Batch video generation using Replicate API\n\n"
        script += "import os\nimport json\nfrom pathlib import Path\nfrom dotenv import load_dotenv\n"
        script += "load_dotenv()\n\n"
        
        for i, prompt in enumerate(prompts):
            category = prompt.get('category', 'unknown')
            output_path = f"videos/{category}/{i:03d}"
            
            cmd = generate_replicate_command(prompt, output_path)
            script += f"\n# Prompt {i}: {prompt.get('prompt_id', 'unknown')}\n"
            script += cmd + "\n"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    with open(output_file, 'w') as f:
        f.write(script)
    
    Path(output_file).chmod(0o755)
    print(f"Saved script to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Video Generation Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all prompts
  %(prog)s list
  
  # List Tier A prompts only
  %(prog)s list --tier A
  
  # Show first 5 Tier B prompts
  %(prog)s list --tier B --limit 5
  
  # Export batch script for Wan 2.2
  %(prog)s export --method wan --tier A --output batch_tier_a.sh
  
  # Export Python script for Replicate
  %(prog)s export --method replicate --limit 10 --output batch_replicate.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available prompts')
    list_parser.add_argument('--tier', choices=['A', 'B', 'C', 'D'], 
                            help='Filter by tier')
    list_parser.add_argument('--model', help='Filter by model')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export batch generation script')
    export_parser.add_argument('--method', choices=['wan', 'replicate'], 
                              default='wan', help='Generation method')
    export_parser.add_argument('--tier', choices=['A', 'B', 'C', 'D'], 
                              help='Filter by tier')
    export_parser.add_argument('--limit', type=int, help='Limit number of prompts')
    export_parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Load manifest
    manifest_path = "datasets/topotremor/prompts/ai_video_prompts.csv"
    try:
        prompts = load_manifest(manifest_path)
    except FileNotFoundError:
        print(f"Error: Manifest not found at {manifest_path}")
        return 1
    
    if args.command == 'list':
        # Filter by tier
        if args.tier:
            prompts = filter_by_tier(prompts, args.tier)
        
        # Filter by model
        if args.model:
            prompts = filter_by_model(prompts, args.model)
        
        show_prompts(prompts, limit=args.limit)
    
    elif args.command == 'export':
        # Filter by tier
        if args.tier:
            prompts = filter_by_tier(prompts, args.tier)
        
        # Apply limit
        if args.limit:
            prompts = prompts[:args.limit]
        
        # Create output directories
        create_output_dirs(prompts)
        
        # Export script
        export_command_script(prompts, args.output, method=args.method)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
