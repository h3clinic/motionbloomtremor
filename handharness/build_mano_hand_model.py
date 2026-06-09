from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pose_preset(name: str, n: int) -> np.ndarray:
    # MANO PCA hand pose coefficients. These presets are simple seeds
    # to generate useful starting hand shapes.
    presets = {
        "relaxed": np.array([0.5, -0.2, 0.1, 0.0, 0.15, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "fist": np.array([2.2, 1.5, 0.9, 0.4, 1.3, 0.8, 0.2, 0.0, 0.1, -0.1, 0.05, 0.0]),
        "open": np.array([-1.2, -0.8, -0.4, -0.2, -0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pinch": np.array([0.7, 0.8, 0.2, 0.1, 0.9, 0.4, 0.0, 0.0, -0.2, 0.15, 0.0, 0.0]),
    }
    p = presets.get(name, presets["relaxed"])
    if n <= len(p):
        return p[:n].copy()
    out = np.zeros((n,), dtype=np.float32)
    out[: len(p)] = p
    return out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("o MANOHand\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3D hand model from MANO")
    parser.add_argument("--mano-dir", default="input/mano", help="Folder containing MANO_LEFT.pkl / MANO_RIGHT.pkl")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--pose", choices=["relaxed", "fist", "open", "pinch"], default="relaxed")
    parser.add_argument("--num-pca-comps", type=int, default=12)
    parser.add_argument("--shape-scale", type=float, default=0.0, help="Apply same value to all 10 betas")
    parser.add_argument("--output", default="output/model/hand_mano.obj")
    args = parser.parse_args()

    try:
        import torch
        from smplx.body_models import MANO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install with: pip install torch smplx"
        ) from exc

    mano_dir = Path(args.mano_dir)
    if not mano_dir.exists():
        raise RuntimeError(
            f"MANO directory not found: {mano_dir}. Place MANO model files there first."
        )

    is_rhand = args.side == "right"

    # Handle common MANO layouts:
    # - input/mano/MANO_RIGHT.pkl
    # - input/mano/mano/MANO_RIGHT.pkl
    candidate_dirs = [mano_dir, mano_dir / "mano", mano_dir / "models", mano_dir / "MANO"]
    model_dir = None
    needed = "MANO_RIGHT.pkl" if is_rhand else "MANO_LEFT.pkl"
    for d in candidate_dirs:
        if (d / needed).exists():
            model_dir = d
            break
    if model_dir is None:
        raise RuntimeError(
            "Could not find MANO model files. Expected one of:\n"
            f"- {mano_dir / 'MANO_RIGHT.pkl'}\n"
            f"- {mano_dir / 'MANO_LEFT.pkl'}\n"
            f"- {(mano_dir / 'mano' / 'MANO_RIGHT.pkl')}\n"
            f"- {(mano_dir / 'mano' / 'MANO_LEFT.pkl')}"
        )

    model = MANO(
        model_path=str(model_dir),
        is_rhand=is_rhand,
        use_pca=True,
        num_pca_comps=int(args.num_pca_comps),
        flat_hand_mean=False,
        batch_size=1,
    )

    hand_pose_np = _pose_preset(args.pose, int(args.num_pca_comps)).astype(np.float32)
    betas_np = np.full((10,), float(args.shape_scale), dtype=np.float32)

    hand_pose = torch.tensor(hand_pose_np, dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(betas_np, dtype=torch.float32).unsqueeze(0)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    transl = torch.zeros((1, 3), dtype=torch.float32)

    out = model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
        return_verts=True,
    )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)

    # Normalize for viewer-friendly scale and center.
    center = verts.mean(axis=0)
    verts = verts - center
    span = np.max(verts.max(axis=0) - verts.min(axis=0))
    if span > 1e-8:
        verts = verts / span * 2.2

    output_path = Path(args.output)
    write_obj(output_path, verts, faces)

    print(f"MANO model generated: {output_path}")
    print(f"Side: {args.side}")
    print(f"Pose preset: {args.pose}")
    print(f"Vertices: {len(verts)}")
    print(f"Faces: {len(faces)}")


if __name__ == "__main__":
    main()
