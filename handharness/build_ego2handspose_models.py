from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("o Ego2HandsPoseHand\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")


def infer_side_from_name(path: Path) -> str:
    s = path.stem.lower()
    if "left" in s or s.endswith("_l") or "_l_" in s:
        return "left"
    return "right"


def load_mano_param_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.npy"))
    out: list[Path] = []
    for p in files:
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception:
            continue
        flat = np.asarray(arr).reshape(-1)
        # Ego2HandsPose/3DHandsForAll MANO param format: 51 values
        # [tx,ty,tz, rx,ry,rz, 45 joint rotations]
        if flat.size == 51 and np.isfinite(flat).all():
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ego2HandsPose MANO parameter files (.npy, 51 values) to OBJ hand meshes"
    )
    parser.add_argument("--dataset-dir", required=True, help="Root of downloaded Ego2HandsPose split (train or test)")
    parser.add_argument("--mano-dir", default="input/mano", help="Folder containing MANO_RIGHT.pkl / MANO_LEFT.pkl")
    parser.add_argument("--output-dir", default="output/model/ego2handspose_models")
    parser.add_argument("--max-models", type=int, default=200)
    args = parser.parse_args()

    try:
        import torch
        from smplx.body_models import MANO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies. Install with: pip install torch smplx") from exc

    ds_root = Path(args.dataset_dir)
    if not ds_root.exists():
        raise RuntimeError(f"Dataset path does not exist: {ds_root}")

    mano_root = Path(args.mano_dir)
    candidate_dirs = [mano_root, mano_root / "mano", mano_root / "models", mano_root / "MANO"]
    model_dir = None
    for d in candidate_dirs:
        if (d / "MANO_RIGHT.pkl").exists() and (d / "MANO_LEFT.pkl").exists():
            model_dir = d
            break
    if model_dir is None:
        raise RuntimeError(
            "Could not find MANO files. Expected MANO_RIGHT.pkl and MANO_LEFT.pkl under input/mano or input/mano/mano"
        )

    right_model = MANO(model_path=str(model_dir), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1)
    left_model = MANO(model_path=str(model_dir), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1)

    param_files = load_mano_param_files(ds_root)
    if not param_files:
        raise RuntimeError(
            f"No MANO parameter files found under {ds_root}. Expected .npy files with 51 values."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_models = max(1, int(args.max_models))
    count = 0

    for p in param_files:
        if count >= max_models:
            break
        vec = np.load(p, allow_pickle=False).reshape(-1).astype(np.float32)

        transl = torch.tensor(vec[0:3], dtype=torch.float32).unsqueeze(0)
        global_orient = torch.tensor(vec[3:6], dtype=torch.float32).unsqueeze(0)
        hand_pose = torch.tensor(vec[6:51], dtype=torch.float32).unsqueeze(0)
        betas = torch.zeros((1, 10), dtype=torch.float32)

        side = infer_side_from_name(p)
        model = right_model if side == "right" else left_model

        out = model(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
            return_verts=True,
        )

        verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
        # Normalize each sample for browser display
        center = verts.mean(axis=0)
        verts = verts - center
        span = np.max(verts.max(axis=0) - verts.min(axis=0))
        if span > 1e-8:
            verts = verts / span * 2.2

        stem = p.stem
        obj_name = f"{count:05d}_{side}_{stem}.obj"
        write_obj(out_dir / obj_name, verts, model.faces.astype(np.int32))
        count += 1

    print(f"Converted models: {count}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
