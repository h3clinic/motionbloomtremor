MANO dataset setup (licensed)

1) Request/download MANO from the official source under its license.
2) Place MANO model files in this folder, for example:
   - input/mano/MANO_RIGHT.pkl
   - input/mano/MANO_LEFT.pkl

Then generate a hand model:

python3 build_mano_hand_model.py --mano-dir input/mano --side right --pose relaxed --output output/model/hand_mano.obj

Open viewer:

http://127.0.0.1:4173/output/model/mano_viewer.html

Citation (provided by MANO authors):

@article{MANO:SIGGRAPHASIA:2017,
  title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
  volume = {36},
  number = {6},
  series = {245:1--245:17},
  month = nov,
  year = {2017},
  month_numeric = {11}
}
