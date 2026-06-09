# One Pose: MediaPipe Peace Sign -> Rigged Hand Mesh

This is a single-pose pipeline:

1. Capture one peace-sign hand pose with MediaPipe.
2. Convert that pose to finger/wrist bone rotations in Blender.
3. Export one posed mesh render using one base rigged hand model.

## Rig Configuration File

Create or edit [rig_map.json](rig_map.json) at repo root to match your exact armature names.

Current template:

```json
{
  "armature_name": "HandArmature",
  "wrist": "Wrist",
  "fingers": {
    "thumb": ["Thumb_01", "Thumb_02", "Thumb_03"],
    "index": ["Index_01", "Index_02", "Index_03"],
    "middle": ["Middle_01", "Middle_02", "Middle_03"],
    "ring": ["Ring_01", "Ring_02", "Ring_03"],
    "pinky": ["Pinky_01", "Pinky_02", "Pinky_03"]
  }
}
```

The Blender script now reads this map via --rig-map and falls back to auto-detection only if entries are missing.

For the attached XR hand model in this workspace, rig map is already populated for:

- model: input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb
- armature: Do_HandRigged
- wrist: radius_ulna

## Install Python packages

```bash
python3 -m pip install --upgrade pip
python3 -m pip install mediapipe opencv-python
```

## Step 1: Capture one peace pose

Webcam mode:

```bash
python3 scripts/mediapipe_capture_one_peace.py \
  --output output/mediapipe/peace_pose.json \
  --annotated output/mediapipe/peace_annotated.png \
  --require-peace 1
```

Image mode:

```bash
python3 scripts/mediapipe_capture_one_peace.py \
  --image input/peace.jpg \
  --output output/mediapipe/peace_pose.json \
  --annotated output/mediapipe/peace_annotated.png \
  --require-peace 1
```

If you need a quick dry run without camera/image detection, generate a synthetic peace pose JSON:

```bash
npm run capture:peace-dummy
```

## Step 2: Wrap one rigged mesh in Blender

Use one rigged source hand model (GLB or FBX):

```bash
blender -b --python scripts/blender_apply_one_mediapipe_pose.py -- \
  --input /absolute/path/to/your_rigged_hand.glb \
  --pose-json output/mediapipe/peace_pose.json \
  --rig-map rig_map.json \
  --output-glb output/mediapipe/peace_wrapped.glb \
  --output-png output/mediapipe/peace_wrapped.png
```

For dry run with synthetic landmarks:

```bash
blender -b --python scripts/blender_apply_one_mediapipe_pose.py -- \
  --input /absolute/path/to/your_rigged_hand.glb \
  --pose-json output/mediapipe/peace_pose_dummy.json \
  --rig-map rig_map.json \
  --output-glb output/mediapipe/peace_wrapped_dummy.glb \
  --output-png output/mediapipe/peace_wrapped_dummy.png
```

## macOS Blender path fallback

Check Blender path detection:

```bash
npm run blender:check
```

One-command dry run (requires HAND_MODEL env var):

```bash
HAND_MODEL=/absolute/path/to/your_rigged_hand.glb npm run wrap:peace-dummy
```

One-command dry run with the attached model already in repo:

```bash
npm run wrap:peace-dummy:attached
```

Notes:

- The script searches armature bones by keyword groups (thumb/index/middle/ring/pinky/wrist).
- If your rig naming is unusual, pass --armature-name or rename bones to common names.
- This is deterministic on one mesh, so geometry remains consistent across poses.
