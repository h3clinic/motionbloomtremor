# Single-Model Hand Workflow

This workspace is now set up for a one-model pipeline:

- Use one clean rigged hand asset (`.glb` or `.fbx`) as the only source model.
- Generate all pose variations via scripting.
- Do not create multiple base hand meshes.

## 1) Keep One Base Model

Recommended location:

- `input/hand_base/hand.glb`

If you currently have only STL, keep using the STL for viewer/export capture,
but for rig-driven pose generation in Blender, use one rigged hand model.

## 2) Generate Poses in Blender (Python)

Script:

- `scripts/blender_single_model_pose_batch.py`

Example:

```bash
blender -b --python scripts/blender_single_model_pose_batch.py -- \
  --input /absolute/path/to/hand.glb \
  --output /absolute/path/to/output/single_model_poses \
  --count 1000 \
  --seed 42 \
  --export-glb 1 \
  --render-png 1
```

Outputs:

- `output/single_model_poses/glb/*.glb` (posed mesh exports)
- `output/single_model_poses/png/*.png` (optional pose renders)

## 3) Multiview Tremor Capture (Synchronized)

For the current web-based model capture pipeline:

```bash
npm run export:tremor-multiview -- --steps=120 --fps=60 --hz=8 --amp=1.0
```

This creates a 32-view dome capture per tremor state:

- 16 equatorial cameras
- 8 upper cameras (+45°)
- 8 lower cameras (-45°)

## 4) Regular Multiview Pose Export

```bash
npm run export:pose-views
```

This regenerates the sidebar angle images from the same hand model.

## Notes

- The browser viewer uses capture mode (`?capture=1`) when exporting images,
  so no sidebar/HUD artifacts are included in generated datasets.
- Always relaunch the site after updates.
