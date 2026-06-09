from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
VIDEO_OUT = OUTPUT / "video_pipeline"
FRAMES_DIR = VIDEO_OUT / "frames"
CUTOUTS_DIR = VIDEO_OUT / "cutouts"
MASKS_DIR = VIDEO_OUT / "masks"
MODEL_DIR = OUTPUT / "model"


def ensure_dirs() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    CUTOUTS_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def clear_dir(dir_path: Path) -> None:
    for p in dir_path.glob("*"):
        if p.is_file():
            p.unlink()


def extract_frames(video_path: Path, count: int = 14, margin_ratio: float = 0.08) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames")

    start = int(total_frames * margin_ratio)
    end = int(total_frames * (1.0 - margin_ratio))
    if end <= start:
        start = 0
        end = total_frames - 1

    indices = np.linspace(start, end, count).astype(np.int32)
    out_paths: list[Path] = []

    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        path = FRAMES_DIR / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(path), frame)
        out_paths.append(path)

    cap.release()
    if len(out_paths) < 8:
        raise RuntimeError("Too few valid frames extracted from video")

    return out_paths


def largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def build_masks(frame_paths: list[Path], target: int = 320, use_rembg: bool = False) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    for frame_path in frame_paths:
        print(f"Processing frame: {frame_path.name}", flush=True)
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if use_rembg:
            with frame_path.open("rb") as f:
                rgba_bytes = remove(f.read())
            decoded = cv2.imdecode(np.frombuffer(rgba_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                rgba = np.array(Image.open(frame_path).convert("RGBA"))
            else:
                if decoded.ndim == 3 and decoded.shape[2] == 4:
                    rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
                elif decoded.ndim == 3 and decoded.shape[2] == 3:
                    rgb2 = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    alpha = np.full((rgb2.shape[0], rgb2.shape[1], 1), 255, dtype=np.uint8)
                    rgba = np.concatenate([rgb2, alpha], axis=2)
                else:
                    rgba = np.array(Image.open(frame_path).convert("RGBA"))
            alpha = rgba[:, :, 3]
            mask = (alpha > 24).astype(np.uint8)
            rgb = rgba[:, :, :3]
        else:
            h0, w0 = bgr.shape[:2]
            scale = min(1.0, 700.0 / max(h0, w0))
            if scale < 1.0:
                sw = max(1, int(round(w0 * scale)))
                sh = max(1, int(round(h0 * scale)))
                bgr_small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            else:
                bgr_small = bgr

            hs, ws = bgr_small.shape[:2]
            rect = (
                int(ws * 0.12),
                int(hs * 0.08),
                int(ws * 0.76),
                int(hs * 0.84),
            )
            gc_mask = np.zeros((hs, ws), np.uint8)
            bg = np.zeros((1, 65), np.float64)
            fg = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr_small, gc_mask, rect, bg, fg, 3, cv2.GC_INIT_WITH_RECT)
            mask_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            if (hs, ws) != (h0, w0):
                mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask_small
        mask = largest_component(mask)

        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        crop = mask[y0 : y1 + 1, x0 : x1 + 1]
        h, w = crop.shape
        scale = (target * 0.78) / max(h, w)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((target, target), dtype=np.uint8)
        ox = (target - nw) // 2
        oy = (target - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized

        kernel = np.ones((3, 3), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel, iterations=2)
        canvas = cv2.GaussianBlur(canvas.astype(np.float32), (0, 0), 0.9)
        canvas = (canvas > 0.33).astype(np.uint8)

        cutout = np.zeros((target, target, 4), dtype=np.uint8)
        rgb_resized = cv2.resize(rgb[y0 : y1 + 1, x0 : x1 + 1], (nw, nh), interpolation=cv2.INTER_LINEAR)
        cutout[oy : oy + nh, ox : ox + nw, :3] = rgb_resized
        cutout[:, :, 3] = (canvas * 255).astype(np.uint8)

        idx = len(masks) + 1
        Image.fromarray(cutout).save(CUTOUTS_DIR / f"view_{idx:03d}.png")
        Image.fromarray((canvas * 255).astype(np.uint8)).save(MASKS_DIR / f"view_{idx:03d}_mask.png")

        masks.append(canvas)

    if len(masks) < 10:
        raise RuntimeError("Foreground extraction failed for too many frames")

    return masks


def carve_visual_hull(masks: list[np.ndarray], grid_n: int = 78, keep_ratio: float = 0.66) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)
    xg, yg, zg = np.meshgrid(coords, coords, coords, indexing="ij")

    votes = np.zeros(xg.shape, dtype=np.uint16)
    valid = np.ones(xg.shape, dtype=bool)

    n = len(masks)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float32)

    h, w = masks[0].shape

    for i, angle in enumerate(angles):
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        xr = xg * c + zg * s
        yr = yg

        u = ((xr + 1.0) * 0.5 * (w - 1)).astype(np.int32)
        v = ((1.0 - (yr + 1.0) * 0.5) * (h - 1)).astype(np.int32)

        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid &= inside

        u2 = np.clip(u, 0, w - 1)
        v2 = np.clip(v, 0, h - 1)
        in_mask = masks[i][v2, u2] > 0

        votes += (inside & in_mask).astype(np.uint16)

    need = max(3, int(math.ceil(n * keep_ratio)))
    occupied = valid & (votes >= need)

    points = np.column_stack((xg[occupied], yg[occupied], zg[occupied])).astype(np.float32)
    if len(points) < 3000:
        raise RuntimeError("Too few points reconstructed. Video may not contain a stable full orbit.")

    return points


def colorize_points(points: np.ndarray) -> np.ndarray:
    # Soft skin-like neutral color; can be replaced with true reprojection color later.
    cols = np.zeros((len(points), 3), dtype=np.uint8)
    cols[:, 0] = 212
    cols[:, 1] = 182
    cols[:, 2] = 160
    return cols


def write_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main() -> None:
    ensure_dirs()
    clear_dir(FRAMES_DIR)
    clear_dir(CUTOUTS_DIR)
    clear_dir(MASKS_DIR)

    source_env = os.environ.get("VIDEO_SOURCE", "").strip()
    if source_env:
        video_path = Path(source_env).expanduser()
    else:
        candidates = sorted(
            Path.home().joinpath("Downloads").glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("No .mp4 files found in Downloads; set VIDEO_SOURCE explicitly")
        video_path = candidates[0]
    if not video_path.exists():
        raise RuntimeError(f"Latest expected video not found: {video_path}")

    frame_count = int(os.environ.get("VIDEO_FRAME_COUNT", "14"))
    mask_size = int(os.environ.get("VIDEO_MASK_SIZE", "320"))
    grid_n = int(os.environ.get("VIDEO_GRID_N", "78"))
    keep_ratio = float(os.environ.get("VIDEO_KEEP_RATIO", "0.66"))
    use_rembg = os.environ.get("VIDEO_USE_REMBG", "0") == "1"

    frame_paths = extract_frames(video_path, count=frame_count)
    masks = build_masks(frame_paths, target=mask_size, use_rembg=use_rembg)
    points = carve_visual_hull(masks, grid_n=grid_n, keep_ratio=keep_ratio)

    # Keep viewer performance smooth.
    if len(points) > 95000:
        idx = np.random.choice(len(points), size=95000, replace=False)
        points = points[idx]

    colors = colorize_points(points)
    out_path = MODEL_DIR / "hand_pointcloud.ply"
    write_ply(points, colors, out_path)

    print(f"Video: {video_path}")
    print(f"Frames used: {len(frame_paths)}")
    print(f"Masks kept: {len(masks)}")
    print(f"Grid: {grid_n}")
    print(f"Point cloud: {out_path}")
    print(f"Points: {len(points)}")


if __name__ == "__main__":
    main()
