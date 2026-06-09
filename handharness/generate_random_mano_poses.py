"""Generate random hand positions from the MANO parametric model.

Samples random poses across:
  - PCA pose space (finger articulation)
  - Shape betas (hand proportions)
  - Global wrist orientation

Outputs OBJ meshes + a manifest JSON with all parameters for each pose.

Usage:
    python generate_random_mano_poses.py --count 50 --output-dir output/random_poses
    python generate_random_mano_poses.py --count 200 --seed 123 --include-left

Requires: pip install torch smplx numpy
MANO model files must be in input/mano/ (MANO_RIGHT.pkl, MANO_LEFT.pkl)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# PCA pose coefficient ranges for realistic hand poses
# Derived from empirical observation of natural hand movements
# First few components have largest effect on overall pose
PCA_RANGES = {
    0: (-2.5, 2.5),   # Major curl: negative=open, positive=fist
    1: (-2.0, 2.0),   # Finger splay / convergence
    2: (-1.5, 1.5),   # Thumb opposition
    3: (-1.0, 1.0),   # Ring/pinky coupling
    4: (-1.5, 1.5),   # Index independence
    5: (-1.0, 1.0),   # Middle flex
    6: (-0.8, 0.8),   # Fine thumb rotation
    7: (-0.6, 0.6),   # Subtle finger spacing
    8: (-0.5, 0.5),   # Higher-order articulation
    9: (-0.4, 0.4),
    10: (-0.3, 0.3),
    11: (-0.3, 0.3),
}

# Named pose archetypes to bias sampling toward natural positions
POSE_ARCHETYPES = {
    "relaxed": {"center": [0.5, -0.2, 0.1, 0.0, 0.15, -0.05], "spread": 0.4},
    "fist": {"center": [2.2, 1.5, 0.9, 0.4, 1.3, 0.8], "spread": 0.3},
    "open_flat": {"center": [-1.2, -0.8, -0.4, -0.2, -0.5, -0.2], "spread": 0.3},
    "pinch": {"center": [0.7, 0.8, 0.2, 0.1, 0.9, 0.4], "spread": 0.4},
    "pointing": {"center": [-0.8, 0.5, -0.3, 0.6, -0.8, 0.3], "spread": 0.4},
    "claw": {"center": [1.5, 0.3, 0.8, 0.5, 0.7, 0.4], "spread": 0.5},
    "hook": {"center": [1.8, 0.8, 0.5, 0.3, 1.0, 0.5], "spread": 0.3},
    "spread_wide": {"center": [-0.5, -1.5, -0.3, -0.8, -0.3, -0.6], "spread": 0.4},
    "thumb_up": {"center": [1.5, 0.8, -0.8, 0.3, 1.2, 0.6], "spread": 0.4},
    "cup": {"center": [1.0, 0.3, 0.5, 0.2, 0.6, 0.3], "spread": 0.3},
}


def sample_random_pose(
    num_pca: int = 12,
    archetype: str | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a random PCA pose vector.

    Args:
        num_pca: Number of PCA components.
        archetype: If given, bias toward this named pose.
        rng: Random generator.

    Returns:
        PCA coefficients array of shape (num_pca,).
    """
    if rng is None:
        rng = np.random.default_rng()

    pose = np.zeros(num_pca, dtype=np.float32)

    if archetype and archetype in POSE_ARCHETYPES:
        arch = POSE_ARCHETYPES[archetype]
        center = np.array(arch["center"], dtype=np.float32)
        spread = arch["spread"]

        for i in range(min(len(center), num_pca)):
            pose[i] = center[i] + rng.normal(0, spread)
        for i in range(len(center), num_pca):
            lo, hi = PCA_RANGES.get(i, (-0.3, 0.3))
            pose[i] = rng.uniform(lo * 0.3, hi * 0.3)
    else:
        # Pure random sampling from PCA ranges
        for i in range(num_pca):
            lo, hi = PCA_RANGES.get(i, (-0.3, 0.3))
            pose[i] = rng.uniform(lo, hi)

    return pose


def sample_random_shape(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample random shape betas (hand proportions).

    Returns 10 beta values. Most variation is in first 3-4 components.
    """
    if rng is None:
        rng = np.random.default_rng()

    betas = np.zeros(10, dtype=np.float32)
    # First few betas control major shape (size, finger length ratio)
    betas[0] = rng.uniform(-1.5, 1.5)  # Overall size
    betas[1] = rng.uniform(-1.0, 1.0)  # Finger length
    betas[2] = rng.uniform(-0.8, 0.8)  # Palm width
    betas[3] = rng.uniform(-0.5, 0.5)
    # Higher betas have less impact
    for i in range(4, 10):
        betas[i] = rng.uniform(-0.3, 0.3)

    return betas


def sample_random_orientation(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample random global wrist orientation (axis-angle).

    Constrains to natural wrist orientations (not fully inverted).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Axis-angle: rotation around each axis
    orient = np.zeros(3, dtype=np.float32)
    orient[0] = rng.uniform(-0.8, 0.8)   # Pitch (flex/extend)
    orient[1] = rng.uniform(-0.5, 0.5)   # Yaw (radial/ulnar deviation)
    orient[2] = rng.uniform(-1.2, 1.2)   # Roll (pronation/supination)

    return orient


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write mesh as Wavefront OBJ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("o MANOHand\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")


def generate_single_pose(
    model,
    pose_pca: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
) -> np.ndarray:
    """Run MANO forward pass and return normalized vertices."""
    import torch

    hand_pose = torch.tensor(pose_pca, dtype=torch.float32).unsqueeze(0)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
    transl = torch.zeros((1, 3), dtype=torch.float32)

    out = model(
        betas=betas_t,
        global_orient=orient_t,
        hand_pose=hand_pose,
        transl=transl,
        return_verts=True,
    )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)

    # Normalize: center at origin, scale to fit ~2.2 units
    center = verts.mean(axis=0)
    verts = verts - center
    span = np.max(verts.max(axis=0) - verts.min(axis=0))
    if span > 1e-8:
        verts = verts / span * 2.2

    return verts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random hand positions from MANO model"
    )
    parser.add_argument("--count", type=int, default=50,
                        help="Number of random poses to generate")
    parser.add_argument("--output-dir", default="output/random_mano_poses",
                        help="Output directory for OBJ files + manifest")
    parser.add_argument("--mano-dir", default="input/mano",
                        help="Folder containing MANO_RIGHT.pkl")
    parser.add_argument("--side", choices=["right", "left"], default="right")
    parser.add_argument("--include-left", action="store_true",
                        help="Also generate left-hand versions")
    parser.add_argument("--num-pca-comps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use-archetypes", action="store_true",
                        help="Bias 60%% of poses toward named archetypes")
    args = parser.parse_args()

    # --- Validate dependencies ---
    try:
        import torch
        from smplx.body_models import MANO
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch smplx")
        sys.exit(1)

    # --- Find MANO model files ---
    mano_dir = Path(args.mano_dir)
    needed = "MANO_RIGHT.pkl" if args.side == "right" else "MANO_LEFT.pkl"
    candidate_dirs = [mano_dir, mano_dir / "mano", mano_dir / "models"]
    model_dir = None
    for d in candidate_dirs:
        if (d / needed).exists():
            model_dir = d
            break
    if model_dir is None:
        print(f"ERROR: MANO model not found. Place {needed} in {mano_dir}/")
        print("Download from: https://mano.is.tue.mpg.de/")
        sys.exit(1)

    # --- Load MANO ---
    print(f"Loading MANO model from {model_dir}...")
    is_rhand = args.side == "right"
    model = MANO(
        model_path=str(model_dir),
        is_rhand=is_rhand,
        use_pca=True,
        num_pca_comps=args.num_pca_comps,
        flat_hand_mean=False,
        batch_size=1,
    )
    faces = model.faces.astype(np.int32)
    print(f"  Loaded: {model.num_betas} betas, {args.num_pca_comps} PCA comps, "
          f"{faces.shape[0]} faces")

    # --- Generate random poses ---
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archetype_names = list(POSE_ARCHETYPES.keys())
    manifest = []

    print(f"\nGenerating {args.count} random poses (seed={args.seed})...")

    for i in range(args.count):
        # Decide if this pose is archetype-biased or pure random
        archetype = None
        if args.use_archetypes and rng.random() < 0.6:
            archetype = rng.choice(archetype_names)

        # Sample parameters
        pose_pca = sample_random_pose(args.num_pca_comps, archetype, rng)
        betas = sample_random_shape(rng)
        orient = sample_random_orientation(rng)

        # Generate mesh
        verts = generate_single_pose(model, pose_pca, betas, orient)

        # Save OBJ
        pose_id = f"mano_{args.side}_{i:04d}"
        obj_path = output_dir / f"{pose_id}.obj"
        write_obj(obj_path, verts, faces)

        # Record metadata
        record = {
            "pose_id": pose_id,
            "index": i,
            "side": args.side,
            "archetype": archetype,
            "pca_coefficients": pose_pca.tolist(),
            "betas": betas.tolist(),
            "global_orient": orient.tolist(),
            "num_vertices": len(verts),
            "num_faces": len(faces),
            "obj_file": f"{pose_id}.obj",
        }
        manifest.append(record)

        if (i + 1) % 10 == 0 or i == 0:
            tag = f" [{archetype}]" if archetype else ""
            print(f"  [{i+1:4d}/{args.count}] {pose_id}{tag}")

    # --- Save manifest ---
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "count": len(manifest),
            "side": args.side,
            "seed": args.seed,
            "num_pca_comps": args.num_pca_comps,
            "poses": manifest,
        }, f, indent=2)

    print(f"\nDone! Generated {len(manifest)} poses.")
    print(f"  OBJs: {output_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  PCA range used: {args.num_pca_comps} components")
    print(f"  Shape betas: 10 dimensions")
    print(f"  Orientation: 3-axis (pitch/yaw/roll)")

    if args.use_archetypes:
        arch_counts = {}
        for r in manifest:
            a = r["archetype"] or "pure_random"
            arch_counts[a] = arch_counts.get(a, 0) + 1
        print(f"\n  Archetype distribution:")
        for name, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")


if __name__ == "__main__":
    main()
