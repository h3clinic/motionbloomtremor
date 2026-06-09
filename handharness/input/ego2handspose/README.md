Ego2HandsPose dataset source (paper: arXiv:2206.04927)

Official links are provided in 3DHandsForAll:
- train: https://app.box.com/s/y5jx4g8cbqikcsgb2jx5g3ymqx71updj
- test:  https://app.box.com/s/wwze8ksixxinixkiq79eb84kulhdb8zm

1) Download either split and extract under this folder, for example:
- input/ego2handspose/train/
- input/ego2handspose/test/

2) Ensure MANO model files are available (licensed separately):
- input/mano/MANO_RIGHT.pkl
- input/mano/MANO_LEFT.pkl

3) Convert MANO parameter files to OBJ models:

python3 build_ego2handspose_models.py \
  --dataset-dir input/ego2handspose/test \
  --mano-dir input/mano \
  --output-dir output/model/ego2handspose_models \
  --max-models 200

4) Open output folder to inspect generated OBJs:
- output/model/ego2handspose_models

Notes:
- Converter expects Ego2HandsPose-style .npy MANO vectors with 51 values:
  [tx,ty,tz, rx,ry,rz, 45 joint rotations]
- If your extracted dataset uses different file names/structure, adjust --dataset-dir.
