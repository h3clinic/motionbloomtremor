"""
render_validation.py – Validates rendered output meets training quality standards.

Checks:
1. No wireframe/triangle artifacts (edge detection on render)
2. Hand silhouette is connected (single main component)
3. Hand occupies 25-90% of frame height
4. Shading has actual depth variation (not flat)
5. Not a debug/viewport render

Usage:
    python handharness/render_validation.py --dir handharness/output/pose_grid
    python handharness/render_validation.py --file output.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required. pip install Pillow")
    sys.exit(1)


def validate_render(image_path: str) -> dict:
    """
    Validate a single rendered image meets training quality.
    Returns dict with pass/fail and details.
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    results = {
        "file": str(image_path),
        "resolution": f"{w}x{h}",
        "checks": {},
        "passed": True,
    }
    
    # ─── Check 1: No wireframe artifacts ──────────────────────────────────
    # Wireframe shows as thin black lines. Check if too many near-black pixels
    # form thin linear patterns
    gray = np.mean(arr[:, :, :3], axis=2)
    black_mask = gray < 15  # near-black pixels
    black_ratio = black_mask.sum() / (h * w)
    
    # Check for thin black lines (horizontal/vertical edge patterns)
    # Compute edge density in black regions
    if black_ratio > 0.01:  # some black exists
        # Check if black pixels form thin lines (wireframe indicator)
        # Count isolated black pixels surrounded by lighter ones
        padded = np.pad(black_mask, 1, mode='constant', constant_values=False)
        neighbors = (
            padded[:-2, 1:-1].astype(int) + padded[2:, 1:-1].astype(int) +
            padded[1:-1, :-2].astype(int) + padded[1:-1, 2:].astype(int)
        )
        # Wireframe: black pixel with 2 black neighbors (line-like)
        line_pixels = (black_mask & (neighbors == 2)).sum()
        line_ratio = line_pixels / max(1, black_mask.sum())
        wireframe_detected = line_ratio > 0.3 and black_ratio > 0.02
    else:
        wireframe_detected = False
        line_ratio = 0.0
    
    results["checks"]["no_wireframe"] = {
        "passed": not wireframe_detected,
        "black_ratio": round(float(black_ratio), 4),
        "line_ratio": round(float(line_ratio), 4),
    }
    if wireframe_detected:
        results["passed"] = False
    
    # ─── Check 2: Hand silhouette connected ───────────────────────────────
    # The hand should be a single connected region (not scattered fragments)
    # Use alpha channel for RGBA images, otherwise threshold on brightness
    if arr.shape[2] == 4:
        foreground = arr[:, :, 3] > 128
    else:
        foreground = gray > 20
    
    fg_ratio = foreground.sum() / (h * w)
    
    # Simple connectivity: check that foreground forms one main blob
    from scipy import ndimage
    labeled, num_features = ndimage.label(foreground)
    if num_features > 0:
        component_sizes = ndimage.sum(foreground, labeled, range(1, num_features + 1))
        largest = max(component_sizes)
        connectivity = largest / max(1, foreground.sum())
    else:
        connectivity = 0.0
    
    silhouette_ok = connectivity > 0.7  # 70% of foreground in one component
    results["checks"]["silhouette_connected"] = {
        "passed": silhouette_ok,
        "connectivity": round(float(connectivity), 3),
        "num_components": int(num_features),
    }
    if not silhouette_ok:
        results["passed"] = False
    
    # ─── Check 3: Frame coverage ─────────────────────────────────────────
    # Hand should take 25-90% of frame height
    if foreground.any():
        rows_with_fg = np.any(foreground, axis=1)
        fg_rows = np.where(rows_with_fg)[0]
        height_coverage = (fg_rows[-1] - fg_rows[0] + 1) / h
    else:
        height_coverage = 0.0
    
    coverage_ok = 0.25 <= height_coverage <= 0.90
    results["checks"]["frame_coverage"] = {
        "passed": coverage_ok,
        "height_ratio": round(float(height_coverage), 3),
    }
    if not coverage_ok:
        results["passed"] = False
    
    # ─── Check 4: Shading variation (not flat) ───────────────────────────
    # A properly lit/shaded hand has gradients. A flat render has uniform color.
    # IMPORTANT: Only evaluate foreground (non-transparent) pixels
    if foreground.any():
        fg_values = gray[foreground]
        shading_std = float(np.std(fg_values))
        shading_range = float(np.percentile(fg_values, 95) - np.percentile(fg_values, 5))
    else:
        shading_std = 0.0
        shading_range = 0.0
    
    # Good shading: std > 4 and range > 12 (on foreground-only pixels)
    # Lower thresholds account for compact poses (fist) and even studio lighting
    shading_ok = shading_std > 4 and shading_range > 12
    results["checks"]["shading_depth"] = {
        "passed": shading_ok,
        "std": round(shading_std, 2),
        "range_95_5": round(shading_range, 2),
    }
    if not shading_ok:
        results["passed"] = False
    
    # ─── Check 5: Not debug/viewport render ──────────────────────────────
    # Debug renders often have uniform gray backgrounds with wireframe overlay
    # or have very specific flat color fills
    bg_mask = ~foreground
    if bg_mask.any():
        bg_values = gray[bg_mask]
        bg_std = float(np.std(bg_values))
        bg_uniform = bg_std < 5  # background is uniform (expected for good render)
    else:
        bg_uniform = True
    
    # The debug indicator is: uniform gray bg (128±30) + wireframe
    bg_mean = float(np.mean(gray[bg_mask])) if bg_mask.any() else 0
    is_debug_bg = 90 < bg_mean < 170 and bg_uniform and wireframe_detected
    
    results["checks"]["not_debug_render"] = {
        "passed": not is_debug_bg,
        "bg_mean": round(bg_mean, 1),
        "bg_uniform": bg_uniform,
    }
    if is_debug_bg:
        results["passed"] = False
    
    return results


def validate_directory(dir_path: str) -> dict:
    """Validate all PNG renders in a directory."""
    dir_p = Path(dir_path)
    pngs = sorted(dir_p.glob("*.png"))
    
    if not pngs:
        return {"error": "No PNG files found", "passed": False}
    
    report = {
        "directory": str(dir_path),
        "total_files": len(pngs),
        "passed": 0,
        "failed": 0,
        "failures": [],
        "all_passed": True,
    }
    
    for png in pngs:
        result = validate_render(str(png))
        if result["passed"]:
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["all_passed"] = False
            failed_checks = [k for k, v in result["checks"].items() if not v["passed"]]
            report["failures"].append({
                "file": png.name,
                "failed_checks": failed_checks,
            })
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate render quality for training")
    parser.add_argument("--dir", help="Directory of renders to validate")
    parser.add_argument("--file", help="Single file to validate")
    parser.add_argument("--output", help="Write JSON report to file")
    args = parser.parse_args()

    if args.file:
        result = validate_render(args.file)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            failed = [k for k, v in result["checks"].items() if not v["passed"]]
            print(f"\n✗ FAILED: {', '.join(failed)}")
            sys.exit(1)
        else:
            print("\n✓ PASSED all render quality checks")
    elif args.dir:
        report = validate_directory(args.dir)
        print(json.dumps(report, indent=2))
        
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(f"\nReport saved: {args.output}")
        
        print(f"\nResults: {report['passed']}/{report['total_files']} passed")
        if not report["all_passed"]:
            print("✗ Some renders failed validation")
            for f in report["failures"][:5]:
                print(f"  {f['file']}: {', '.join(f['failed_checks'])}")
            sys.exit(1)
        else:
            print("✓ All renders passed validation")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
