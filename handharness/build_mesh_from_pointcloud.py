from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def read_ascii_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    if "end_header" not in lines:
        raise RuntimeError("Invalid PLY")
    start = lines.index("end_header") + 1
    pts = []
    for ln in lines[start:]:
        sp = ln.split()
        if len(sp) < 3:
            continue
        pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
    a = np.asarray(pts, dtype=np.float64)
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < 50:
        raise RuntimeError("Too few points")
    return a


def trim_outliers(points: np.ndarray, q_lo: float = 0.02, q_hi: float = 0.98) -> np.ndarray:
    lo = np.quantile(points, q_lo, axis=0)
    hi = np.quantile(points, q_hi, axis=0)
    keep = np.logical_and(points >= lo, points <= hi).all(axis=1)
    out = points[keep]
    return out if out.shape[0] > 100 else points


def to_field(points: np.ndarray, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = points.copy()
    mn = p.min(axis=0)
    mx = p.max(axis=0)
    span = np.maximum(mx - mn, 1e-8)
    p = (p - mn) / span
    p = np.clip(p, 0.0, 1.0)

    grid = np.zeros((n, n, n), dtype=np.float32)
    idx = np.clip((p * (n - 1)).astype(np.int32), 0, n - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    # Dilate by splatting to local 3x3x3 neighborhood
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                src_x0 = max(0, -dx)
                src_x1 = n - max(0, dx)
                src_y0 = max(0, -dy)
                src_y1 = n - max(0, dy)
                src_z0 = max(0, -dz)
                src_z1 = n - max(0, dz)
                dst_x0 = max(0, dx)
                dst_x1 = n - max(0, -dx)
                dst_y0 = max(0, dy)
                dst_y1 = n - max(0, -dy)
                dst_z0 = max(0, dz)
                dst_z1 = n - max(0, -dz)
                grid[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = np.maximum(
                    grid[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1],
                    grid[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1] * 0.8,
                )

    field = gaussian_filter(grid, sigma=1.8)
    return field, mn, mx


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("o HandSurface\n")
        for v in verts:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    inp = root / "output/model/hand_colmap_sparse_ascii.ply"
    out = root / "output/model/hand_surface.obj"

    points = read_ascii_ply_xyz(inp)
    points = trim_outliers(points)
    field, mn, mx = to_field(points, n=120)

    level = max(0.06, float(np.percentile(field[field > 0], 35)))
    verts, faces, _normals, _vals = marching_cubes(field, level=level)

    verts = verts / 119.0
    verts = verts * (mx - mn) + mn

    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.max(verts.max(axis=0) - verts.min(axis=0))
    if scale > 1e-8:
        verts = verts / scale * 2.2

    write_obj(out, verts.astype(np.float32), faces.astype(np.int32))
    print(f"mesh_vertices={len(verts)}")
    print(f"mesh_faces={len(faces)}")
    print(f"mesh_obj={out}")


if __name__ == "__main__":
    main()
