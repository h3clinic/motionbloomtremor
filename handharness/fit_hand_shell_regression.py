#!/usr/bin/env python3
"""Fit a smooth hand shell from a sparse point cloud using axis-aligned regression.

This keeps everything point-based (no mesh). It outputs:
1) A regressed shell point cloud (PLY)
2) A centerline/radius profile CSV for CAD workflows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


def read_ascii_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    try:
        start = lines.index("end_header") + 1
    except ValueError as exc:
        raise RuntimeError(f"Invalid PLY (missing end_header): {path}") from exc

    pts = []
    for ln in lines[start:]:
        sp = ln.split()
        if len(sp) < 3:
            continue
        try:
            pts.append([float(sp[0]), float(sp[1]), float(sp[2])])
        except ValueError:
            continue
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"No XYZ points loaded from {path}")
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        raise RuntimeError(f"All points were non-finite in {path}")
    return arr


def write_ascii_ply_xyz(path: Path, pts: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} 220 235 255\n")


def sor_filter(points: np.ndarray, k: int = 24, std_ratio: float = 2.0) -> np.ndarray:
    if cKDTree is None or len(points) <= (k + 1):
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    mu = float(mean_d.mean())
    sigma = float(mean_d.std())
    thresh = mu + std_ratio * sigma
    keep = mean_d <= thresh
    return points[keep]


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    helper = x if abs(np.dot(axis, x)) < 0.9 else y
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v


def fit_shell(points: np.ndarray, poly_deg: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    centered = points - c

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis = axis / np.linalg.norm(axis)
    u, v = orthonormal_basis_from_axis(axis)

    t = np.einsum("ij,j->i", centered, axis)
    ou = np.einsum("ij,j->i", centered, u)
    ov = np.einsum("ij,j->i", centered, v)

    deg = int(max(1, min(poly_deg, 5)))
    pu = np.polyfit(t, ou, deg=deg)
    pv = np.polyfit(t, ov, deg=deg)
    ou_fit = np.polyval(pu, t)
    ov_fit = np.polyval(pv, t)

    residual_r = np.sqrt((ou - ou_fit) ** 2 + (ov - ov_fit) ** 2)

    t_min, t_max = float(t.min()), float(t.max())
    edges = np.linspace(t_min, t_max, int(max(12, bins)) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.digitize(t, edges) - 1

    rb = []
    tb = []
    for bi, tc in enumerate(centers):
        m = bidx == bi
        if np.count_nonzero(m) < 6:
            continue
        rb.append(float(np.percentile(residual_r[m], 70.0)))
        tb.append(float(tc))

    if len(tb) < 4:
        tb = t.tolist()
        rb = residual_r.tolist()

    tr = np.asarray(tb, dtype=np.float64)
    rr = np.asarray(rb, dtype=np.float64)
    pr = np.polyfit(tr, rr, deg=min(5, max(1, len(tr) - 1)))

    return c, axis, u, v, pr, pu, pv, np.array([t_min, t_max], dtype=np.float64)


def generate_shell(
    c: np.ndarray,
    axis: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pr: np.ndarray,
    pu: np.ndarray,
    pv: np.ndarray,
    t_range: np.ndarray,
    n_t: int,
    n_theta: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_min, t_max = float(t_range[0]), float(t_range[1])
    ts = np.linspace(t_min, t_max, int(max(24, n_t)))
    th = np.linspace(0.0, 2.0 * np.pi, int(max(12, n_theta)), endpoint=False)

    shell = []
    profile = []
    for tv in ts:
        cu = float(np.polyval(pu, tv))
        cv = float(np.polyval(pv, tv))
        r = float(max(1e-5, np.polyval(pr, tv)))
        center = c + tv * axis + cu * u + cv * v
        profile.append([tv, center[0], center[1], center[2], r])
        for ang in th:
            p = center + r * (np.cos(ang) * u + np.sin(ang) * v)
            shell.append([p[0], p[1], p[2]])

    return np.asarray(shell, dtype=np.float64), np.asarray(profile, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a hand shell regression from sparse points")
    ap.add_argument("--input", required=True, help="Input ASCII PLY (x y z required)")
    ap.add_argument("--shell-output", default="output/model/hand_shell_regressed.ply")
    ap.add_argument("--profile-output", default="output/model/hand_shell_profile.csv")
    ap.add_argument("--poly-deg", type=int, default=3)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--nt", type=int, default=140, help="Samples along main axis")
    ap.add_argument("--ntheta", type=int, default=64, help="Angular samples per section")
    ap.add_argument("--sor-k", type=int, default=24)
    ap.add_argument("--sor-std", type=float, default=2.0)
    args = ap.parse_args()

    in_path = Path(args.input)
    shell_path = Path(args.shell_output)
    profile_path = Path(args.profile_output)
    shell_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    pts = read_ascii_ply_xyz(in_path)
    pts_f = sor_filter(pts, k=int(args.sor_k), std_ratio=float(args.sor_std))

    c, axis, u, v, pr, pu, pv, t_range = fit_shell(pts_f, poly_deg=int(args.poly_deg), bins=int(args.bins))
    shell_pts, profile = generate_shell(
        c, axis, u, v, pr, pu, pv, t_range, n_t=int(args.nt), n_theta=int(args.ntheta)
    )

    write_ascii_ply_xyz(shell_path, shell_pts)
    np.savetxt(
        profile_path,
        profile,
        delimiter=",",
        header="t_axis,center_x,center_y,center_z,radius",
        comments="",
        fmt="%.8f",
    )

    print(f"input_points={len(pts)}")
    print(f"filtered_points={len(pts_f)}")
    print(f"shell_points={len(shell_pts)}")
    print(f"shell_output={shell_path}")
    print(f"profile_output={profile_path}")


if __name__ == "__main__":
    main()
