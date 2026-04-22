"""Exercise definitions and verification using pose + hand landmarks.

Each exercise returns a Verification with:
  - ok: bool (posture currently satisfies the requirement)
  - message: human-readable guidance
  - quality: 0..1 how well the pose matches (for visual feedback)

State machine drives a 3-stage flow per exercise:
  IDLE → PREPARE (show instruction, require pose match) → HOLD (countdown,
  sample tremor metrics) → DONE (show result).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .tracker import PoseSnapshot


# ---------- data types ------------------------------------------------------
@dataclass
class Verification:
    ok: bool
    message: str
    quality: float = 0.0


@dataclass
class Exercise:
    key: str
    name: str
    description: str
    prepare_secs: float
    hold_secs: float
    verify: Callable[[PoseSnapshot | None, tuple[float, float] | None], Verification]


# ---------- geometry helpers ------------------------------------------------
def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _nearest_wrist(pose: PoseSnapshot, target) -> tuple | None:
    cands = [w for w in (pose.l_wrist, pose.r_wrist) if w is not None]
    if not cands:
        return None
    return min(cands, key=lambda w: _dist(w, target))


# ---------- verifiers -------------------------------------------------------
def verify_touch_nose(pose: PoseSnapshot | None,
                      tip: tuple[float, float] | None) -> Verification:
    if pose is None or pose.nose is None:
        return Verification(False, "Make sure your face is visible", 0.0)
    if tip is None:
        return Verification(False, "Show your hand to the camera", 0.0)

    # Reference scale: inter-shoulder distance (fallback to face width)
    if pose.l_shoulder and pose.r_shoulder:
        scale = _dist(pose.l_shoulder, pose.r_shoulder)
    elif pose.l_ear and pose.r_ear:
        scale = _dist(pose.l_ear, pose.r_ear) * 2.0
    else:
        scale = 0.2  # normalized-image fallback
    scale = max(scale, 0.05)

    d = _dist(tip, pose.nose)
    norm = d / scale
    # Within 25% of inter-shoulder distance ≈ touching / near nose
    if norm < 0.18:
        return Verification(True, "Holding at nose — keep still", 1.0)
    if norm < 0.35:
        return Verification(False, "Move fingertip closer to your nose",
                            1.0 - (norm - 0.18) / 0.17)
    return Verification(False, "Bring your index fingertip to your nose", 0.0)


def verify_scratch_head(pose: PoseSnapshot | None,
                        tip: tuple[float, float] | None) -> Verification:
    if pose is None:
        return Verification(False, "Face the camera so your head is visible", 0.0)
    if tip is None:
        return Verification(False, "Show your hand to the camera", 0.0)

    # Head top estimated as nose raised by ~inter-ear distance
    if pose.nose is None or pose.l_ear is None or pose.r_ear is None:
        return Verification(False, "Keep your head & ears in frame", 0.0)
    ear_mid = _mid(pose.l_ear, pose.r_ear)
    head_width = _dist(pose.l_ear, pose.r_ear)
    head_top = (ear_mid[0], ear_mid[1] - head_width * 0.7)

    # Hand must be above both ears (y smaller) and near the head midline
    above = tip[1] < min(pose.l_ear[1], pose.r_ear[1]) + head_width * 0.1
    lateral = abs(tip[0] - ear_mid[0]) / max(0.01, head_width)

    d = _dist(tip, head_top)
    norm = d / max(0.05, head_width)

    if above and lateral < 1.2 and norm < 1.6:
        q = 1.0 - min(1.0, max(0.0, (norm - 0.4) / 1.2))
        return Verification(True, "On top of head — hold steady", q)
    if above and lateral < 1.8:
        return Verification(False, "Move hand closer to the top of your head",
                            0.5)
    return Verification(False, "Raise your hand to the top of your head", 0.0)


def verify_hold_object(pose: PoseSnapshot | None,
                       tip: tuple[float, float] | None,
                       grip: float | None = None) -> Verification:
    if tip is None or pose is None:
        return Verification(False, "Show your hand to the camera", 0.0)

    # Require hand clearly extended out from body and roughly at chest/eye height.
    # Use nose & shoulders as vertical reference.
    nose = pose.nose
    l_sh, r_sh = pose.l_shoulder, pose.r_shoulder
    if not (nose and l_sh and r_sh):
        return Verification(False, "Face the camera, hold object out in front",
                            0.0)

    shoulder_mid = _mid(l_sh, r_sh)
    chest_y = (nose[1] + shoulder_mid[1]) / 2
    # Hand y should be between eye level (nose) and slightly below shoulders
    y_ok = (nose[1] - 0.05) <= tip[1] <= (shoulder_mid[1] + 0.10)

    # Hand should be extended (wrist fairly far from shoulder in x or z-proxy
    # via 2D distance)
    wrist = _nearest_wrist(pose, tip) or tip
    sh_width = _dist(l_sh, r_sh)
    extended = _dist(wrist, shoulder_mid) / max(0.05, sh_width) > 0.9

    # Grip/object check — fingers should be curled as if gripping.
    gripping = grip is None or grip >= 0.45

    if y_ok and extended and gripping:
        return Verification(True,
                            "Arm outstretched, gripping object — hold steady",
                            1.0)
    if y_ok and extended and not gripping:
        return Verification(False,
                            "Close your fingers around an object (cup, pen…)",
                            0.6)
    if y_ok:
        return Verification(False,
                            "Extend your arm further out in front", 0.5)
    if extended:
        return Verification(False,
                            "Raise the object to around chest/eye level", 0.5)
    return Verification(False,
                        "Hold an object out in front at chest height", 0.0)


# ---------- catalogue -------------------------------------------------------
EXERCISES: list[Exercise] = [
    Exercise(
        key="nose",
        name="Touch Your Nose",
        description=(
            "Slowly bring your index fingertip to the tip of your nose "
            "and hold it there. Useful for detecting intention tremor."
        ),
        prepare_secs=3.0,
        hold_secs=6.0,
        verify=verify_touch_nose,
    ),
    Exercise(
        key="head",
        name="Scratch Your Head",
        description=(
            "Raise your hand to the top of your head as if scratching, "
            "then hold still. Useful for kinetic/postural tremor."
        ),
        prepare_secs=3.0,
        hold_secs=6.0,
        verify=verify_scratch_head,
    ),
    Exercise(
        key="hold",
        name="Hold Object",
        description=(
            "Hold an object (e.g. a cup) out in front of you at chest "
            "height with your arm extended. Useful for postural tremor."
        ),
        prepare_secs=3.0,
        hold_secs=8.0,
        verify=verify_hold_object,
    ),
]


# ---------- session state machine ------------------------------------------
class Stage(str, Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    HOLD = "hold"
    DONE = "done"


@dataclass
class ExerciseSession:
    exercise: Exercise
    stage: Stage = Stage.IDLE
    stage_start: float = 0.0
    hold_samples: list[float] = field(default_factory=list)   # tremor score samples
    hold_peaks: list[float] = field(default_factory=list)     # peak Hz
    hold_amps: list[float] = field(default_factory=list)      # amp mm
    result_summary: str = ""
    prepare_ready_since: float | None = None

    def start(self) -> None:
        self.stage = Stage.PREPARE
        self.stage_start = time.time()
        self.hold_samples.clear()
        self.hold_peaks.clear()
        self.hold_amps.clear()
        self.prepare_ready_since = None
        self.result_summary = ""

    def cancel(self) -> None:
        self.stage = Stage.IDLE
        self.result_summary = ""

    def elapsed(self) -> float:
        return time.time() - self.stage_start

    def update(self, pose: PoseSnapshot | None,
               tip: tuple[float, float] | None,
               score: float | None, peak_hz: float | None,
               amp_mm: float | None,
               grip: float | None = None) -> Verification:
        if self.exercise.key == "hold":
            v = verify_hold_object(pose, tip, grip=grip)
        else:
            v = self.exercise.verify(pose, tip)
        now = time.time()

        if self.stage == Stage.PREPARE:
            if v.ok:
                if self.prepare_ready_since is None:
                    self.prepare_ready_since = now
                # Once the user has been in pose for the prepare duration,
                # move to HOLD.
                if now - self.prepare_ready_since >= self.exercise.prepare_secs:
                    self.stage = Stage.HOLD
                    self.stage_start = now
            else:
                self.prepare_ready_since = None

        elif self.stage == Stage.HOLD:
            if v.ok and score is not None:
                self.hold_samples.append(float(score))
                if peak_hz is not None:
                    self.hold_peaks.append(float(peak_hz))
                if amp_mm is not None:
                    self.hold_amps.append(float(amp_mm))
            if now - self.stage_start >= self.exercise.hold_secs:
                self.stage = Stage.DONE
                self._summarize()

        return v

    def _summarize(self) -> None:
        if not self.hold_samples:
            self.result_summary = (
                f"{self.exercise.name}: pose not held long enough. "
                "Please try again."
            )
            return
        s = np.array(self.hold_samples)
        mean_s = float(np.mean(s))
        peak = float(np.mean(self.hold_peaks)) if self.hold_peaks else 0.0
        amp = float(np.mean(self.hold_amps)) if self.hold_amps else 0.0
        if mean_s < 15:
            verdict = "No significant tremor"
        elif mean_s < 40:
            verdict = "Mild tremor"
        elif mean_s < 70:
            verdict = "Moderate tremor"
        else:
            verdict = "Strong tremor"
        self.result_summary = (
            f"{self.exercise.name}: {verdict}. "
            f"Mean score {mean_s:.0f}/100, peak ≈ {peak:.1f} Hz, "
            f"amp ≈ {amp:.1f} mm, samples={len(s)}."
        )
