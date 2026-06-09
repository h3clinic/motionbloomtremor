#!/usr/bin/env python3
"""Generate MotionBloom logos, mascots, and UI icons with Google's image models.

Uses the modern `google-genai` SDK. Tries Imagen first, then falls back to the
Gemini image model (a.k.a. "nano banana"). Reads prompts from
`motionbloom/assets/duolingo/assets_manifest.json` and writes PNGs next to it.

Setup:
    pip install google-genai pillow
    export GEMINI_API_KEY=<your-key>      # or GOOGLE_API_KEY

Run:
    python scripts/generate_gemini_assets.py            # generate missing only
    python scripts/generate_gemini_assets.py --force    # regenerate everything
    python scripts/generate_gemini_assets.py --only bloom_idle motionbloom_logo
    python scripts/generate_gemini_assets.py --dry-run  # validate, no API calls

Exit codes:
    0 success   1 missing key/SDK   2 API error   3 file/manifest error
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "motionbloom" / "assets" / "duolingo"
MANIFEST = ASSETS_DIR / "assets_manifest.json"

# Imagen first (best quality), then the Gemini image model as fallback.
IMAGEN_MODELS = ["imagen-3.0-generate-002", "imagen-3.0-generate-001"]
GEMINI_IMAGE_MODELS = [
    os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image-preview"),
    "gemini-2.0-flash-preview-image-generation",
]


def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 256, 256


def save_png(raw: bytes, output_path: Path, size: tuple[int, int]) -> bool:
    """Normalize raw image bytes to a square RGBA PNG at the target size."""
    try:
        from PIL import Image
    except ImportError:
        output_path.write_bytes(raw)  # best effort
        return True
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path, "PNG")
        return True
    except Exception as exc:
        print(f"  ! could not post-process image: {exc}", file=sys.stderr)
        return False


def gen_with_imagen(client, prompt: str) -> bytes | None:
    from google.genai import types

    for model in IMAGEN_MODELS:
        try:
            result = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1),
            )
            imgs = getattr(result, "generated_images", None) or []
            if imgs:
                img = imgs[0].image
                data = getattr(img, "image_bytes", None)
                if data:
                    return data
        except Exception as exc:
            print(f"  imagen[{model}] unavailable: {exc}", file=sys.stderr)
    return None


def gen_with_gemini_image(client, prompt: str) -> bytes | None:
    from google.genai import types

    for model in GEMINI_IMAGE_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                ),
            )
            for cand in response.candidates or []:
                for part in cand.content.parts or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return inline.data
        except Exception as exc:
            print(f"  gemini-image[{model}] unavailable: {exc}", file=sys.stderr)
    return None


def generate_one(client, prompt: str, output_path: Path,
                 size: tuple[int, int]) -> bool:
    raw = gen_with_imagen(client, prompt)
    if raw is None:
        raw = gen_with_gemini_image(client, prompt)
    if raw is None:
        print(f"  x no image returned for {output_path.name}", file=sys.stderr)
        return False
    return save_png(raw, output_path, size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate manifest/key without calling the API")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate assets even if they already exist")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only generate these asset names")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found: {MANIFEST}", file=sys.stderr)
        return 3
    try:
        manifest = json.loads(MANIFEST.read_text())
    except Exception as exc:
        print(f"ERROR: bad manifest JSON: {exc}", file=sys.stderr)
        return 3

    assets = manifest.get("assets", [])
    if args.only:
        wanted = set(args.only)
        assets = [a for a in assets if a.get("name") in wanted]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()
    print(f"Assets dir : {ASSETS_DIR}")
    print(f"Assets     : {len(assets)} in manifest")
    print(f"API key    : {'present' if api_key else 'MISSING'}")
    print()

    if args.dry_run:
        for a in assets:
            name = a.get("name", "???")
            path = ASSETS_DIR / f"{name}.png"
            print(f"  - {name}.png (exists: {path.exists()})")
        print("\n[dry-run] manifest valid")
        return 0

    if not api_key:
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY)", file=sys.stderr)
        return 1

    try:
        from google import genai
    except ImportError:
        print("ERROR: pip install google-genai", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)

    generated = skipped = failed = 0
    for a in assets:
        name = a.get("name")
        prompt = a.get("prompt", "")
        if not name or not prompt:
            skipped += 1
            continue
        out = ASSETS_DIR / f"{name}.png"
        if out.exists() and not args.force:
            print(f"skip  {name}.png (exists)")
            skipped += 1
            continue
        size = parse_size(a.get("size", "256x256"))
        print(f"gen   {name}.png ...")
        if generate_one(client, prompt, out, size):
            print(f"  ok  -> {out.name}")
            generated += 1
        else:
            failed += 1

    print(f"\nSummary: {generated} generated, {skipped} skipped, {failed} failed")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
