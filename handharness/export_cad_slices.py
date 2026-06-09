from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    vertex_count = 0
    header_end = -1

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i + 1
            break

    if header_end < 0 or vertex_count <= 0:
        raise RuntimeError(f"Invalid or unsupported PLY file: {path}")

    pts = []
    cols = []
    for line in lines[header_end : header_end + vertex_count]:
        sp = line.strip().split()
        if len(sp) < 6:
            continue
        x, y, z = float(sp[0]), float(sp[1]), float(sp[2])
        r, g, b = int(sp[3]), int(sp[4]), int(sp[5])
        pts.append((x, y, z))
        cols.append((r, g, b))

    if not pts:
        raise RuntimeError("No points found in PLY")

    return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.uint8)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def sor_filter(points: np.ndarray, k: int = 30, std_ratio: float = 2.0) -> np.ndarray:
    if len(points) < k + 1:
        return np.ones(len(points), dtype=bool)

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_neighbor_dist = dists[:, 1:].mean(axis=1)

    mu = mean_neighbor_dist.mean()
    sigma = mean_neighbor_dist.std()
    threshold = mu + std_ratio * sigma

    return mean_neighbor_dist <= threshold


def pca_main_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    if not np.isfinite(centered).all():
        raise RuntimeError("Point cloud contains non-finite values")

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    basis_u = vt[1]
    basis_v = vt[2]

    axis = axis / max(np.linalg.norm(axis), 1e-12)
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-12)
    basis_v = basis_v / max(np.linalg.norm(basis_v), 1e-12)

    return center, axis, np.column_stack((basis_u, basis_v))


def export_slices(
    points_mm: np.ndarray,
    out_dir: Path,
    interval_mm: float,
    half_window_mm: float,
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in out_dir.glob("slice_*.csv"):
        old_csv.unlink()

    center, axis, basis = pca_main_axis(points_mm)

    centered = points_mm - center
    t = np.einsum("ij,j->i", centered, axis)

    t_min = float(np.min(t))
    t_max = float(np.max(t))

    slice_count = 0
    non_empty_count = 0
    cur = t_min

    while cur <= t_max + 1e-9:
        mask = np.abs(t - cur) <= half_window_mm
        pts = points_mm[mask]

        if len(pts) > 0:
            rel = pts - center
            xy = np.einsum("ij,jk->ik", rel, basis)

            csv_path = out_dir / f"slice_{slice_count:03d}.csv"
            with csv_path.open("w", encoding="ascii") as f:
                f.write("x_mm,y_mm\n")
                for p in xy:
                    f.write(f"{p[0]:.6f},{p[1]:.6f}\n")

            non_empty_count += 1

        slice_count += 1
        cur += interval_mm

    meta = out_dir / "_slices_meta.txt"
    with meta.open("w", encoding="ascii") as f:
        f.write("Slices exported from filtered COLMAP sparse cloud\n")
        f.write(f"interval_mm={interval_mm}\n")
        f.write(f"half_window_mm={half_window_mm}\n")
        f.write(f"axis_vector={axis[0]:.6f},{axis[1]:.6f},{axis[2]:.6f}\n")
        f.write(
            f"center_mm={center[0]:.6f},{center[1]:.6f},{center[2]:.6f}\n"
        )
        f.write(f"slice_bins={slice_count}\n")
        f.write(f"non_empty_slices={non_empty_count}\n")

    return slice_count, non_empty_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter sparse PLY and export CAD slice CSVs")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/model/hand_colmap_sparse_ascii.ply"),
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("output/model/hand_colmap_sparse_clean.ply"),
    )
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=Path("output/cad_slices"),
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    parser.add_argument("--interval-mm", type=float, default=15.0)
    parser.add_argument("--window-mm", type=float, default=2.0)
    parser.add_argument(
        "--scale-mm-per-unit",
        type=float,
        default=0.0,
        help="Scale factor to convert COLMAP units to mm. If <= 0, auto-scale from arm length.",
    )
    parser.add_argument(
        "--assume-arm-length-mm",
        type=float,
        default=260.0,
        help="Used only when --scale-mm-per-unit <= 0.",
    )

    args = parser.parse_args()

    points, colors = read_ascii_ply(args.input)
    keep_mask = sor_filter(points, k=args.k, std_ratio=args.std_ratio)
    points_f = points[keep_mask]
    colors_f = colors[keep_mask]

    args.filtered_output.parent.mkdir(parents=True, exist_ok=True)
    write_ascii_ply(args.filtered_output, points_f, colors_f)

    c0, axis0, _basis0 = pca_main_axis(points_f)
    t0 = np.einsum("ij,j->i", points_f - c0, axis0)
    major_extent_units = max(1e-9, float(np.max(t0) - np.min(t0)))

    if args.scale_mm_per_unit > 0:
        scale_mm_per_unit = args.scale_mm_per_unit
    else:
        scale_mm_per_unit = args.assume_arm_length_mm / major_extent_units

    points_mm = points_f * scale_mm_per_unit
    bins, non_empty = export_slices(
        points_mm,
        args.slices_dir,
        interval_mm=args.interval_mm,
        half_window_mm=args.window_mm,
    )

    print(f"Input points: {len(points)}")
    print(f"Filtered points: {len(points_f)}")
    print(f"Filtered PLY: {args.filtered_output}")
    print(f"Slice dir: {args.slices_dir}")
    print(f"Scale mm/unit: {scale_mm_per_unit:.6f}")
    print(f"Slice bins: {bins}")
    print(f"Non-empty slices: {non_empty}")


if __name__ == "__main__":
    main()
