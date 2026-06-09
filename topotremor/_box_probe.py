"""One-shot probe: open the camera, run detect_hand_box on N frames, print the
box size as a % of the frame. No stdout piping, hard frame cap so it can't hang.
Run: ../.venv/bin/python _box_probe.py
"""
import time
import cv2
import numpy as np
from topo_bridge import detect_hand_box, MOTION_DIFF_THRESH

N = 120  # hard cap on frames so this always terminates
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ok, frame = cap.read()
if not ok:
    print("camera read failed")
    raise SystemExit(1)

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
boxes = []
t0 = time.time()
for i in range(N):
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, mm = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    box = detect_hand_box(frame, mm)
    if box:
        boxes.append(box)
    prev_gray = gray
    if time.time() - t0 > 8:
        break

cap.release()
H, W = frame.shape[:2]
if not boxes:
    print(f"frames probed, NO box detected (frame {W}x{H})")
else:
    a = np.array(boxes, float)
    print(f"frame {W}x{H} | boxes {len(boxes)}")
    print("width%%  mean %.0f max %.0f" % ((a[:, 2] / W * 100).mean(), (a[:, 2] / W * 100).max()))
    print("height%% mean %.0f max %.0f" % ((a[:, 3] / H * 100).mean(), (a[:, 3] / H * 100).max()))
    print("area%%   mean %.0f max %.0f" % (((a[:, 2] * a[:, 3]) / (W * H) * 100).mean(),
                                           ((a[:, 2] * a[:, 3]) / (W * H) * 100).max()))
    print("last box (x,y,w,h):", boxes[-1])
