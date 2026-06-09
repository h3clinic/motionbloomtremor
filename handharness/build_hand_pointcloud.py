from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
CUTOUT_DIR = OUTPUT_DIR / "cutouts"
MASK_DIR = OUTPUT_DIR / "masks"
MODEL_DIR = OUTPUT_DIR / "model"


def ensure_dirs() -> None:
    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def remove_backgrounds(image_paths: list[Path]) -> list[Path]:
    out_paths: list[Path] = []
    for img_path in image_paths:
        with img_path.open("rb") as f:
            data = f.read()
        rgba = remove(data)
        out_path = CUTOUT_DIR / f"{img_path.stem}.png"
        out_path.write_bytes(rgba)
        out_paths.append(out_path)
    return out_paths


def largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def load_cutouts(cutout_paths: list[Path], size: int = 700) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    for path in cutout_paths:
        rgba = np.array(Image.open(path).convert("RGBA"))
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        mask = (alpha > 24).astype(np.uint8)
        mask = largest_component(mask)

        ys, xs = np.where(mask > 0)
        if len(xs) < 16:
            raise RuntimeError(f"No foreground found in {path.name}")

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        crop_rgb = rgb[y0 : y1 + 1, x0 : x1 + 1]

        h, w = crop_mask.shape
        scale = (size * 0.8) / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas_mask = np.zeros((size, size), dtype=np.uint8)
        canvas_rgb = np.zeros((size, size, 3), dtype=np.uint8)

        ox = (size - new_w) // 2
        oy = (size - new_h) // 2

        canvas_mask[oy : oy + new_h, ox : ox + new_w] = resized_mask
        canvas_rgb[oy : oy + new_h, ox : ox + new_w] = resized_rgb

        kernel = np.ones((3, 3), np.uint8)
        canvas_mask = cv2.morphologyEx(canvas_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        canvas_mask = cv2.GaussianBlur(canvas_mask.astype(np.float32), (0, 0), 0.8)
        canvas_mask = (canvas_mask > 0.33).astype(np.uint8)

        masked_rgb = canvas_rgb.copy()
        masked_rgb[canvas_mask == 0] = 0

        Image.fromarray((canvas_mask * 255).astype(np.uint8)).save(MASK_DIR / f"{path.stem}_mask.png")

        masks.append(canvas_mask)
        colors.append(masked_rgb)

    return masks, colors


def carve_point_cloud(masks: list[np.ndarray], colors: list[np.ndarray], grid_n: int = 110) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)

    angles = [0.0, math.pi * 0.5, math.pi, math.pi * 1.5]

    points: list[list[float]] = []
    point_colors: list[list[int]] = []

    for x in coords:
        for y in coords:
            for z in coords:
                keep = True
                sample_colors = []

                for vi, angle in enumerate(angles):
                    c = math.cos(angle)
                    s = math.sin(angle)

                    xr = x * c + z * s
                    yr = y

                    w = masks[vi].shape[1]
                    h = masks[vi].shape[0]

                    u = int(((xr + 1.0) * 0.5) * (w - 1))
                    v = int(((1.0 - (yr + 1.0) * 0.5)) * (h - 1))

                    if u < 0 or v < 0 or u >= w or v >= h:
                        keep = False
                        break

                    if masks[vi][v, u] == 0:
                        keep = False
                        break

                    px = colors[vi][v, u]
                    if int(px[0]) + int(px[1]) + int(px[2]) > 0:
                        sample_colors.append(px)

                if keep:
                    points.append([float(x), float(y), float(z)])
                    if sample_colors:
                        mean = np.mean(np.array(sample_colors), axis=0)
                        point_colors.append([int(mean[0]), int(mean[1]), int(mean[2])])
                    else:
                        point_colors.append([215, 215, 215])

    if len(points) < 3000:
        raise RuntimeError("Too few carved points. Input views may not align as side views.")

    pts = np.array(points, dtype=np.float32)
    cols = np.array(point_colors, dtype=np.uint8)

    keep_idx = np.random.choice(len(pts), size=min(70000, len(pts)), replace=False)
    pts = pts[keep_idx]
    cols = cols[keep_idx]

    return pts, cols


def write_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="ascii") as f:
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
    images = sorted(INPUT_DIR.glob("view_*.jpg"))[:4]
    if len(images) < 4:
        raise RuntimeError("Need 4 inputs named view_1.jpg .. view_4.jpg")

    cutouts = remove_backgrounds(images)
    masks, colors = load_cutouts(cutouts)
    points, point_colors = carve_point_cloud(masks, colors)
    out_path = MODEL_DIR / "hand_pointcloud.ply"
    write_ply(points, point_colors, out_path)

    print(f"Point cloud written: {out_path}")
    print(f"Points: {len(points)}")


if __name__ == "__main__":
    main()
