"""Session-level tremor score reports for MotionBloom.

Tracks each video playback session, collecting tremor scores while the video
plays, and persists per-session averages to disk so they show up in the
Reports tab across launches.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional


REPORTS_DIR = Path.home() / ".motionbloom"
SESSIONS_FILE = REPORTS_DIR / "video_sessions.json"


@dataclass
class SessionRecord:
    id: str
    started_at: str  # ISO timestamp
    ended_at: Optional[str] = None
    video_path: str = ""
    video_name: str = ""
    samples: int = 0
    avg_score: Optional[float] = None
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    duration_sec: float = 0.0
    recording_path: str = ""  # local path to recorded webcam session (mp4)

    def to_dict(self) -> dict:
        return asdict(self)


class SessionReportStore:
    """Tracks live video sessions and persists their score averages."""

    def __init__(self, path: Path = SESSIONS_FILE) -> None:
        self.path = path
        self._active: Optional[SessionRecord] = None
        self._active_scores: list[int] = []
        self._active_start_t: float = 0.0
        self._records: list[SessionRecord] = []
        self._load()

    # ------------------------------------------------------------- I/O
    def _load(self) -> None:
        try:
            if self.path.exists():
                with self.path.open("r") as f:
                    raw = json.load(f)
                # Filter keys to those known by the dataclass so older
                # JSON files (without newer fields) keep working.
                known = set(SessionRecord.__dataclass_fields__.keys())
                self._records = [
                    SessionRecord(**{k: v for k, v in r.items() if k in known})
                    for r in raw
                ]
        except Exception as exc:
            print(f"[REPORTS] Failed to load sessions: {exc}", flush=True)
            self._records = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = [r.to_dict() for r in self._records]
            with self.path.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            print(f"[REPORTS] Failed to save sessions: {exc}", flush=True)

    # --------------------------------------------------- session control
    def start_session(self, video_path: str) -> SessionRecord:
        # End any prior session first.
        self.end_session()
        rec = SessionRecord(
            id=str(uuid.uuid4())[:8],
            started_at=datetime.now().isoformat(timespec="seconds"),
            video_path=str(video_path),
            video_name=Path(video_path).name if video_path else "",
        )
        self._active = rec
        self._active_scores = []
        self._active_start_t = time.time()
        print(f"[REPORTS] Session started: {rec.id} ({rec.video_name})", flush=True)
        return rec

    def add_score(self, score: Optional[float]) -> None:
        if self._active is None or score is None:
            return
        try:
            self._active_scores.append(int(score))
        except Exception:
            pass

    def end_session(self) -> Optional[SessionRecord]:
        if self._active is None:
            return None
        rec = self._active
        rec.ended_at = datetime.now().isoformat(timespec="seconds")
        rec.duration_sec = round(time.time() - self._active_start_t, 2)
        if self._active_scores:
            rec.samples = len(self._active_scores)
            rec.avg_score = round(sum(self._active_scores) / len(self._active_scores), 1)
            rec.min_score = int(min(self._active_scores))
            rec.max_score = int(max(self._active_scores))
        self._active = None
        self._active_scores = []
        # Always persist the session so every video the user plays appears
        # in the Reports tab, even if no scores were captured (e.g. the
        # camera was off or the hand was out of view the whole time).
        self._records.append(rec)
        self._save()
        print(f"[REPORTS] Session ended: {rec.id} avg={rec.avg_score} samples={rec.samples}", flush=True)
        return rec

    # ---------------------------------------------------------- queries
    def all_records(self) -> list[SessionRecord]:
        return list(self._records)

    def overall_average(self) -> Optional[float]:
        scored = [r.avg_score for r in self._records if r.avg_score is not None]
        if not scored:
            return None
        return round(sum(scored) / len(scored), 1)

    def best_score(self) -> Optional[float]:
        scored = [r.avg_score for r in self._records if r.avg_score is not None]
        return min(scored) if scored else None  # lower = less tremor = "best"

    def worst_score(self) -> Optional[float]:
        scored = [r.avg_score for r in self._records if r.avg_score is not None]
        return max(scored) if scored else None

    def clear(self) -> None:
        self._records = []
        self._save()
        print("[REPORTS] All sessions cleared", flush=True)
