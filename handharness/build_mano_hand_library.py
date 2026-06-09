from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple MANO hand OBJ files")
    parser.add_argument("--mano-dir", default="input/mano")
    parser.add_argument("--out-dir", default="output/model/mano_library")
    parser.add_argument("--sides", default="right,left", help="Comma-separated: right,left")
    parser.add_argument("--poses", default="relaxed,open,fist,pinch", help="Comma-separated presets")
    parser.add_argument("--num-pca-comps", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    poses = [p.strip() for p in args.poses.split(",") if p.strip()]

    for side in sides:
        for pose in poses:
            out_path = out_dir / f"hand_mano_{side}_{pose}.obj"
            cmd = [
                "python3",
                "build_mano_hand_model.py",
                "--mano-dir",
                args.mano_dir,
                "--side",
                side,
                "--pose",
                pose,
                "--num-pca-comps",
                str(args.num_pca_comps),
                "--output",
                str(out_path),
            ]
            print("RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print("Done")
    print(f"Library folder: {out_dir}")


if __name__ == "__main__":
    main()
