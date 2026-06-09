from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from skimage.measure import marching_cubes

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
    cutout_paths: list[Path] = []
    for img_path in image_paths:
        with img_path.open("rb") as f:
            data = f.read()
        out = remove(data)
        out_path = CUTOUT_DIR / f"{img_path.stem}.png"
        out_path.write_bytes(out)
        cutout_paths.append(out_path)
    return cutout_paths


def largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest).astype(np.uint8)
    return out


def load_and_normalize_masks(cutout_paths: list[Path], size: int = 640) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for path in cutout_paths:
        rgba = np.array(Image.open(path).convert("RGBA"))
        alpha = rgba[:, :, 3]
        mask = (alpha > 24).astype(np.uint8)

        mask = largest_component(mask)
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            raise RuntimeError(f"Foreground not detected in {path.name}")

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = mask[y0 : y1 + 1, x0 : x1 + 1]

        h, w = crop.shape
        scale = (size * 0.78) / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((size, size), dtype=np.uint8)
        ox = (size - new_w) // 2
        oy = (size - new_h) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = resized

        kernel = np.ones((3, 3), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel, iterations=2)
        canvas = cv2.GaussianBlur(canvas.astype(np.float32), (0, 0), 0.8)
        canvas = (canvas > 0.33).astype(np.uint8)

        out_path = MASK_DIR / f"{path.stem}_mask.png"
        Image.fromarray((canvas * 255).astype(np.uint8)).save(out_path)
        masks.append(canvas)

    return masks


def carve_voxels(masks: list[np.ndarray], grid_n: int = 100) -> np.ndarray:
    grid = np.zeros((grid_n, grid_n, grid_n), dtype=np.uint8)

    # Coordinate system: x (left-right), y (down-up), z (front-back)
    coords = np.linspace(-1.0, 1.0, grid_n)

    # Assume four views around vertical axis at 0, 90, 180, 270 degrees.
    angles = [0.0, math.pi * 0.5, math.pi, math.pi * 1.5]

    yy, xx = np.meshgrid(np.arange(grid_n), np.arange(grid_n), indexing="ij")

    for ix, x in enumerate(coords):
        for iy, y in enumerate(coords):
            for iz, z in enumerate(coords):
                keep = True
                for vi, angle in enumerate(angles):
                    c = math.cos(angle)
                    s = math.sin(angle)

                    xr = x * c + z * s
                    yr = y

                    u = int(((xr + 1.0) * 0.5) * (masks[vi].shape[1] - 1))
                    v = int(((1.0 - (yr + 1.0) * 0.5)) * (masks[vi].shape[0] - 1))

                    if u < 0 or v < 0 or u >= masks[vi].shape[1] or v >= masks[vi].shape[0]:
                        keep = False
                        break
                    if masks[vi][v, u] == 0:
                        keep = False
                        break

                if keep:
                    grid[ix, iy, iz] = 1

    if grid.sum() < 1000:
        raise RuntimeError("Voxel carving produced too few occupied cells. Views likely not aligned side views.")

    # Smooth occupancy field for cleaner surfaces.
    blurred = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 0.8)
    return blurred


def build_mesh(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _normals, _values = marching_cubes(field, level=0.5)

    # Normalize to approximately [-1, 1]
    min_v = verts.min(axis=0)
    max_v = verts.max(axis=0)
    center = (min_v + max_v) * 0.5
    scale = np.max(max_v - min_v)
    verts = (verts - center) / scale * 2.0

    # Reorder from array axes (x,y,z) to 3D model axes (x,z,y) for nicer orientation.
    verts = verts[:, [0, 2, 1]]

    return verts.astype(np.float32), faces.astype(np.int32)


def make_texture(cutout_paths: list[Path]) -> str:
    tex_img = Image.open(cutout_paths[0]).convert("RGBA")
    tex_img = tex_img.resize((1024, 1024), Image.Resampling.LANCZOS)
    tex_name = "hand_texture.png"
    tex_img.save(MODEL_DIR / tex_name)
    return tex_name


def write_obj_mtl(verts: np.ndarray, faces: np.ndarray, tex_name: str) -> None:
    obj_path = MODEL_DIR / "hand_model.obj"
    mtl_path = MODEL_DIR / "hand_model.mtl"

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)

    with obj_path.open("w", encoding="ascii") as f:
        f.write("mtllib hand_model.mtl\n")
        f.write("o HandModel\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Simple planar UV projection from x/z.
        for v in verts:
            u = (v[0] - mins[0]) / span[0]
            vv = (v[2] - mins[2]) / span[2]
            f.write(f"vt {u:.6f} {vv:.6f}\n")

        f.write("usemtl HandMat\n")
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    with mtl_path.open("w", encoding="ascii") as f:
        f.write("newmtl HandMat\n")
        f.write("Ka 0.150000 0.150000 0.150000\n")
        f.write("Kd 0.900000 0.900000 0.900000\n")
        f.write("Ks 0.080000 0.080000 0.080000\n")
        f.write("Ns 40.000000\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {tex_name}\n")


def main() -> None:
    ensure_dirs()
    images = sorted(INPUT_DIR.glob("view_*.jpg"))
    if len(images) < 4:
        raise RuntimeError("Need 4 input images named view_1.jpg ... view_4.jpg")
    images = images[:4]

    cutouts = remove_backgrounds(images)
    masks = load_and_normalize_masks(cutouts)
    field = carve_voxels(masks)
    verts, faces = build_mesh(field)
    tex_name = make_texture(cutouts)
    write_obj_mtl(verts, faces, tex_name)

    print("Done")
    print(f"Cutouts: {CUTOUT_DIR}")
    print(f"Masks:   {MASK_DIR}")
    print(f"Model:   {MODEL_DIR / 'hand_model.obj'}")


if __name__ == "__main__":
    main()
