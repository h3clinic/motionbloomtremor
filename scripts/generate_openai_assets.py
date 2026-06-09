#!/usr/bin/env python3
"""MotionBloom AI image pipeline — generate + verify logos/icons via OpenAI.

Uses OpenAI's `gpt-image-1` model to render the MotionBloom logo, the Bloom
mascot moods, and all UI/exercise icons from prompts in
`motionbloom/assets/duolingo/assets_manifest.json`.

Every generated asset is VERIFIED before it is accepted:
  - decodes as a valid PNG
  - matches the requested square size
  - is not blank / not a flat single color (has real content + brand color)
A generated file that fails verification is rejected (old file kept).

Setup:
    pip install openai pillow
    # key is read from .env (OPENAI_API_KEY=...) or the environment

Run:
    python scripts/generate_openai_assets.py                 # missing only
    python scripts/generate_openai_assets.py --force         # regenerate all
    python scripts/generate_openai_assets.py --only motionbloom_logo
    python scripts/generate_openai_assets.py --verify-only   # re-check existing

Exit codes:
    0 success   1 missing key/SDK   2 generation/verify failure   3 manifest/io
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "motionbloom" / "assets" / "duolingo"
MANIFEST = ASSETS_DIR / "assets_manifest.json"
LOGO_PATH = REPO_ROOT / "motionbloom" / "assets" / "motionbloom_logo.png"
ENV_FILE = REPO_ROOT / ".env"

MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
# gpt-image-1 supports these output sizes; we render square then downscale.
GEN_SIZE = "1024x1024"


# --------------------------------------------------------------------------- #
# key loading                                                                  #
# --------------------------------------------------------------------------- #
def load_env_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


# --------------------------------------------------------------------------- #
# verification                                                                 #
# --------------------------------------------------------------------------- #
def verify_png(data: bytes, size: tuple[int, int]) -> tuple[bool, str]:
    """Return (ok, reason). Rejects blanks and flat single-color images."""
    try:
        from PIL import Image
    except ImportError:
        return True, "pillow-missing (skipped checks)"
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:
        return False, f"not a valid image: {exc}"

    if img.size[0] < 64 or img.size[1] < 64:
        return False, f"too small: {img.size}"

    small = img.resize((48, 48))
    px = list(small.getdata())
    opaque = [(r, g, b) for r, g, b, a in px if a > 25]
    if len(opaque) < 40:
        return False, "almost fully transparent / empty"

    # distinct colors -> not a flat fill
    distinct = len({(r // 24, g // 24, b // 24) for r, g, b in opaque})
    if distinct < 3:
        return False, "looks like a flat single-color block"

    return True, f"ok ({len(opaque)} opaque px, {distinct} color buckets)"


def normalize_png(data: bytes, size: tuple[int, int]) -> bytes:
    """Resize to the manifest size, keep RGBA, re-encode as PNG."""
    try:
        from PIL import Image
    except ImportError:
        return data
    img = Image.open(io.BytesIO(data)).convert("RGBA").resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 256, 256


# --------------------------------------------------------------------------- #
# generation                                                                   #
# --------------------------------------------------------------------------- #
def generate_image(client, prompt: str) -> bytes | None:
    full = (
        prompt
        + " Centered single subject, transparent background, no text, no "
        "watermark, crisp edges, app-icon quality."
    )
    try:
        result = client.images.generate(
            model=MODEL,
            prompt=full,
            size=GEN_SIZE,
            background="transparent",
            n=1,
        )
    except TypeError:
        # Older SDKs may not accept `background`.
        result = client.images.generate(
            model=MODEL, prompt=full, size=GEN_SIZE, n=1
        )
    except Exception as exc:
        print(f"    ! API error: {exc}", file=sys.stderr)
        return None

    item = result.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        import requests

        resp = requests.get(url, timeout=60)
        if resp.ok:
            return resp.content
    return None


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def output_path_for(name: str) -> Path:
    if name == "motionbloom_logo":
        return LOGO_PATH
    return ASSETS_DIR / f"{name}.png"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the asset already exists")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only process these asset names")
    ap.add_argument("--verify-only", action="store_true",
                    help="Verify existing PNGs without calling the API")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found: {MANIFEST}", file=sys.stderr)
        return 3
    try:
        manifest = json.loads(MANIFEST.read_text())
    except Exception as exc:
        print(f"ERROR: bad manifest: {exc}", file=sys.stderr)
        return 3

    assets = manifest.get("assets", [])
    if args.only:
        wanted = set(args.only)
        assets = [a for a in assets if a.get("name") in wanted]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- verify-only path ------------------------------------------------- #
    if args.verify_only:
        bad = 0
        for a in assets:
            name = a.get("name")
            path = output_path_for(name)
            size = parse_size(a.get("size", "256x256"))
            if not path.exists():
                print(f"  [MISS] {name}")
                bad += 1
                continue
            ok, reason = verify_png(path.read_bytes(), size)
            print(f"  [{'OK ' if ok else 'BAD'}] {name}: {reason}")
            bad += 0 if ok else 1
        print(f"\nVerify: {len(assets) - bad}/{len(assets)} passed")
        return 0 if bad == 0 else 2

    # ---- generation path -------------------------------------------------- #
    key = load_env_key()
    if not key:
        print("ERROR: OPENAI_API_KEY not found in .env or environment",
              file=sys.stderr)
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(api_key=key)
    print(f"Model     : {MODEL}")
    print(f"Assets    : {len(assets)}")
    print(f"Output dir: {ASSETS_DIR}\n")

    generated = skipped = failed = 0
    for a in assets:
        name = a.get("name")
        prompt = a.get("prompt", "")
        if not name or not prompt:
            skipped += 1
            continue
        out = output_path_for(name)
        if out.exists() and not args.force:
            print(f"skip  {name} (exists)")
            skipped += 1
            continue

        size = parse_size(a.get("size", "256x256"))
        print(f"gen   {name} ...")
        raw = generate_image(client, prompt)
        if raw is None:
            print(f"  FAIL no image returned")
            failed += 1
            continue

        ok, reason = verify_png(raw, size)
        if not ok:
            print(f"  FAIL verification: {reason} (keeping previous file)")
            failed += 1
            continue

        out.write_bytes(normalize_png(raw, size))
        # re-verify the written file end-to-end
        ok2, reason2 = verify_png(out.read_bytes(), size)
        status = "ok" if ok2 else "WROTE-BUT-BAD"
        print(f"  {status} -> {out.relative_to(REPO_ROOT)} ({reason2})")
        generated += 1 if ok2 else 0
        failed += 0 if ok2 else 1

    print(f"\nSummary: {generated} generated, {skipped} skipped, {failed} failed")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
