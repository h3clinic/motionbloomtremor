"""Local media playback backends for MotionBloom.

This module keeps user-facing video playback separate from camera/vision
analysis. VLC is preferred for real media playback with audio, seeking, and a
stable media clock. OpenCV is retained only as a degraded preview fallback.
"""

from __future__ import annotations

import sys
import time
import os
import ctypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ._cv_lock import CV_LOCK


class LocalVideoPlayer(ABC):
    backend_name = "unknown"
    audio_supported = False
    warning: str | None = None

    def attach(self, widget: Any) -> None:
        self.widget = widget

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def play(self) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @abstractmethod
    def seek_ratio(self, ratio: float) -> None: ...

    @abstractmethod
    def get_time_ms(self) -> int: ...

    @abstractmethod
    def get_duration_ms(self) -> int: ...

    @abstractmethod
    def get_position_ratio(self) -> float: ...

    @abstractmethod
    def is_playing(self) -> bool: ...

    @abstractmethod
    def get_volume(self) -> int: ...

    @abstractmethod
    def set_volume(self, value: int) -> None: ...

    @abstractmethod
    def is_muted(self) -> bool: ...

    @abstractmethod
    def set_muted(self, value: bool) -> None: ...

    @abstractmethod
    def has_audio_stream(self) -> bool: ...

    @abstractmethod
    def debug_state(self) -> dict: ...

    def render_frame(self) -> np.ndarray | None:
        return None

    def log_debug_state(self, label: str = "") -> None:
        state = self.debug_state()
        parts = ["VIDEO_AUDIO_STATE"]
        if label:
            parts.append(f"label={label}")
        for key, value in state.items():
            text = str(value).replace(" ", "_")
            parts.append(f"{key}={text}")
        print(" ".join(parts), flush=True)


class VLCVideoPlayer(LocalVideoPlayer):
    backend_name = "VLC"
    audio_supported = True

    def __init__(self, widget: Any | None = None) -> None:
        self._prepare_vlc_environment()
        try:
            import vlc  # type: ignore
        except Exception as exc:
            raise RuntimeError("python-vlc is not installed") from exc
        self.vlc = vlc
        # Use minimal VLC options to avoid macOS crashes
        vlc_args = ["--quiet", "--no-video-title-show"]
        if sys.platform == "darwin":
            vlc_args.append("--no-xlib")
        print(f"[VLC] Creating VLC instance with args: {vlc_args}", flush=True)
        self.instance = vlc.Instance(*vlc_args)
        print(f"[VLC] Creating media player...", flush=True)
        self.player = self.instance.media_player_new()
        print(f"[VLC] Player created successfully", flush=True)
        self.widget = None
        self.path: str | None = None
        self.media = None
        # Store widget but do NOT attach in constructor - wait for load/play
        if widget is not None:
            print(f"[VLC] Widget provided, will attach during play()", flush=True)
            self.widget = widget

    def _safe(self, fn, default="unknown"):
        """Safely call a VLC API function, returning default if it fails."""
        try:
            result = fn()
            return result if result is not None else default
        except Exception as exc:
            return f"unknown:{type(exc).__name__}"

    def _prepare_vlc_environment(self) -> None:
        if sys.platform != "darwin":
            return
        app_root = Path("/Applications/VLC.app/Contents/MacOS")
        lib_path = app_root / "lib" / "libvlc.dylib"
        core_path = app_root / "lib" / "libvlccore.dylib"
        plugin_path = app_root / "plugins"
        if core_path.exists():
            try:
                ctypes.CDLL(str(core_path), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
        if lib_path.exists():
            os.environ.setdefault("PYTHON_VLC_LIB_PATH", str(lib_path))
        if plugin_path.exists():
            os.environ.setdefault("PYTHON_VLC_MODULE_PATH", str(plugin_path))
            os.environ.setdefault("VLC_PLUGIN_PATH", str(plugin_path))

    def attach(self, widget: Any) -> None:
        self.widget = widget
        try:
            widget.update_idletasks()
            if not widget.winfo_ismapped():
                print("[VLC] Widget not yet mapped, skipping attach", flush=True)
                return
            # Ensure window is fully realized before getting handle
            widget.update()
            handle = widget.winfo_id()
            if not handle:
                print("[VLC] Warning: widget has no window ID yet", flush=True)
                return
            if sys.platform.startswith("win"):
                self.player.set_hwnd(handle)
            elif sys.platform == "darwin":
                # macOS set_nsobject can segfault if called at wrong time
                print(f"[VLC] Attaching to macOS window handle: {handle}", flush=True)
                self.player.set_nsobject(handle)
                print("[VLC] Successfully attached to window", flush=True)
                # Give macOS time to process the window attachment
                import time
                time.sleep(0.05)
            else:
                self.player.set_xwindow(handle)
        except Exception as exc:
            print(f"[VLC] Failed to attach to widget: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    def load(self, path: str) -> None:
        print(f"[VLC] load() called with: {path}", flush=True)
        self.stop()
        media_path = str(Path(path).expanduser())
        print(f"[VLC] Creating media object...", flush=True)
        media = self.instance.media_new(media_path)
        print(f"[VLC] Setting media on player...", flush=True)
        self.player.set_media(media)
        self.media = media
        self.path = media_path
        print(f"[VLC] Configuring audio...", flush=True)
        self.set_muted(False)
        self.set_volume(100)
        print(f"[VLC] Parsing media metadata...", flush=True)
        try:
            media.parse_with_options(self.vlc.MediaParseFlag.local, 1000)
        except Exception as e:
            print(f"[VLC] parse_with_options failed: {e}", flush=True)
            try:
                media.parse()
            except Exception as e2:
                print(f"[VLC] parse also failed: {e2}", flush=True)
        # Do NOT attach window during load - wait until play() is called
        print(f"[VLC] load() completed successfully", flush=True)

    def play(self) -> None:
        print(f"[VLC] play() called", flush=True)
        if self.widget is not None:
            print(f"[VLC] Attaching to widget before playback...", flush=True)
            self.attach(self.widget)
        print(f"[VLC] Setting audio config...", flush=True)
        self.set_muted(False)
        self.set_volume(100)
        print(f"[VLC] Starting playback...", flush=True)
        self.player.play()
        print(f"[VLC] Playback started", flush=True)
        self.set_muted(False)
        self.set_volume(100)

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass

    def cleanup(self) -> None:
        self.stop()
        try:
            self.player.release()
        except Exception:
            pass
        try:
            self.instance.release()
        except Exception:
            pass

    def seek_ratio(self, ratio: float) -> None:
        self.player.set_position(float(min(1.0, max(0.0, ratio))))

    def get_time_ms(self) -> int:
        value = self.player.get_time()
        return int(value) if value and value > 0 else 0

    def get_duration_ms(self) -> int:
        value = self.player.get_length()
        return int(value) if value and value > 0 else 0

    def get_position_ratio(self) -> float:
        value = self.player.get_position()
        if value is None or value < 0:
            duration = self.get_duration_ms()
            return (self.get_time_ms() / duration) if duration > 0 else 0.0
        return float(min(1.0, max(0.0, value)))

    def is_playing(self) -> bool:
        try:
            return bool(self.player.is_playing())
        except Exception:
            return False

    def get_volume(self) -> int:
        try:
            value = self.player.audio_get_volume()
            return int(value) if value is not None and value >= 0 else 0
        except Exception:
            return 0

    def set_volume(self, value: int) -> None:
        try:
            self.player.audio_set_volume(int(max(0, min(100, value))))
        except Exception:
            pass

    def is_muted(self) -> bool:
        try:
            return bool(self.player.audio_get_mute())
        except Exception:
            return True

    def set_muted(self, value: bool) -> None:
        try:
            self.player.audio_set_mute(bool(value))
        except Exception:
            pass

    def _audio_track_count(self) -> int | None:
        try:
            value = self.player.audio_get_track_count()
            return int(value) if value is not None else None
        except Exception:
            return None

    def _audio_track(self) -> int | None:
        try:
            value = self.player.audio_get_track()
            return int(value) if value is not None else None
        except Exception:
            return None

    def has_audio_stream(self) -> bool:
        try:
            count = self._audio_track_count()
            if count is not None and count > 0:
                return True
        except Exception:
            pass
        try:
            if self.media is not None:
                tracks = self.media.tracks_get()
                for track in tracks or []:
                    if getattr(track, "type", None) == self.vlc.TrackType.audio:
                        return True
        except Exception:
            pass
        return False

    def debug_state(self) -> dict:
        return {
            "backend": self.backend_name,
            "audio_supported": self.audio_supported,
            "path": getattr(self, "path", None),
            "duration_ms": self._safe(lambda: self.player.get_length(), 0),
            "time_ms": self._safe(lambda: self.player.get_time(), 0),
            "muted": self._safe(lambda: self.player.audio_get_mute(), "unknown"),
            "volume": self._safe(lambda: self.player.audio_get_volume(), "unknown"),
            "audio_track": self._safe(lambda: self.player.audio_get_track(), "unknown"),
            "audio_track_count": self._safe(lambda: self.player.audio_get_track_count(), "unknown"),
            "has_audio_stream": self._safe(lambda: self.has_audio_stream(), "unknown"),
            "state": self._safe(lambda: str(self.player.get_state()), "unknown"),
        }


class OpenCVFallbackVideoPlayer(LocalVideoPlayer):
    backend_name = "OpenCV preview"
    audio_supported = False
    warning = "Audio is unavailable in fallback playback mode."

    def __init__(self, widget: Any | None = None) -> None:
        self.widget = widget
        self.cap: cv2.VideoCapture | None = None
        self.path: str | None = None
        self.playing = False
        self.duration_ms = 0
        self.position_ms = 0
        self.started_at = 0.0
        self.started_position_ms = 0
        self.last_frame: np.ndarray | None = None

    def load(self, path: str) -> None:
        self.stop()
        media_path = str(Path(path).expanduser())
        with CV_LOCK:
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("OpenCV could not open video")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_ms = int((frames / fps) * 1000) if fps and fps > 0 and frames and frames > 0 else 0
        self.cap = cap
        self.path = media_path
        self.position_ms = 0
        self.started_position_ms = 0
        self.last_frame = None

    def play(self) -> None:
        if self.cap is None:
            return
        self.playing = True
        self.started_at = time.time()
        self.started_position_ms = self.position_ms

    def pause(self) -> None:
        self.position_ms = self.get_time_ms()
        self.playing = False

    def stop(self) -> None:
        self.playing = False
        self.position_ms = 0
        self.started_position_ms = 0
        self.last_frame = None
        if self.cap is not None:
            with CV_LOCK:
                self.cap.release()
            self.cap = None

    def cleanup(self) -> None:
        self.stop()

    def seek_ratio(self, ratio: float) -> None:
        if self.cap is None or self.duration_ms <= 0:
            return
        self.position_ms = int(self.duration_ms * min(1.0, max(0.0, ratio)))
        with CV_LOCK:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.position_ms)
        self.started_at = time.time()
        self.started_position_ms = self.position_ms
        self.last_frame = None

    def get_time_ms(self) -> int:
        if self.playing:
            elapsed_ms = int((time.time() - self.started_at) * 1000)
            value = self.started_position_ms + elapsed_ms
            if self.duration_ms > 0:
                value = min(value, self.duration_ms)
            return max(0, value)
        return max(0, self.position_ms)

    def get_duration_ms(self) -> int:
        return max(0, self.duration_ms)

    def get_position_ratio(self) -> float:
        duration = self.get_duration_ms()
        return (self.get_time_ms() / duration) if duration > 0 else 0.0

    def is_playing(self) -> bool:
        return self.playing

    def get_volume(self) -> int:
        return 0

    def set_volume(self, value: int) -> None:
        return

    def is_muted(self) -> bool:
        return True

    def set_muted(self, value: bool) -> None:
        return

    def has_audio_stream(self) -> bool:
        return False

    def debug_state(self) -> dict:
        return {
            "backend": self.backend_name,
            "audio_supported": self.audio_supported,
            "path": self.path,
            "duration_ms": self.get_duration_ms(),
            "time_ms": self.get_time_ms(),
            "muted": self.is_muted(),
            "volume": self.get_volume(),
            "audio_track": None,
            "audio_track_count": 0,
            "has_audio_stream": False,
            "state": "Playing" if self.playing else "Stopped",
        }

    def render_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        target_ms = self.get_time_ms()
        if self.duration_ms > 0 and target_ms >= self.duration_ms:
            self.pause()
            return self.last_frame
        with CV_LOCK:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return self.last_frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = frame_rgb
        self.position_ms = target_ms
        return frame_rgb


def create_video_player(widget: Any | None = None) -> LocalVideoPlayer:
    # On macOS, embedding VLC into Tkinter via set_nsobject() causes
    # frequent segfaults. Use HybridVideoPlayer (OpenCV frames + VLC audio-only)
    # for stable playback with audio.
    if sys.platform == "darwin":
        try:
            return HybridVideoPlayer(widget)
        except Exception as exc:
            print(f"[VIDEO] HybridVideoPlayer failed: {exc}, falling back to OpenCV", flush=True)
            fallback = OpenCVFallbackVideoPlayer(widget)
            fallback.warning = f"Audio unavailable. VLC could not start: {exc}"
            return fallback
    try:
        return VLCVideoPlayer(widget)
    except Exception as exc:
        fallback = OpenCVFallbackVideoPlayer(widget)
        fallback.warning = f"Audio is unavailable in fallback playback mode. VLC could not start: {exc}"
        return fallback


class HybridVideoPlayer(LocalVideoPlayer):
    """Stable video playback for macOS.
    
    Uses OpenCV to read and render video frames into a Tkinter Canvas/widget
    (avoiding the unstable VLC NSView embedding), while a separate audio-only
    VLC instance handles sound. This avoids macOS segfaults from set_nsobject().
    """
    backend_name = "Hybrid (OpenCV+VLC audio)"
    audio_supported = True

    def __init__(self, widget: Any | None = None) -> None:
        # Initialize VLC for audio only (no video output)
        VLCVideoPlayer._prepare_vlc_environment(self)
        try:
            import vlc  # type: ignore
        except Exception as exc:
            raise RuntimeError("python-vlc is not installed") from exc
        self.vlc = vlc
        # --no-video: VLC won't try to create any video window (no NSView issues)
        # --no-xlib: prevent X11 conflicts on macOS
        vlc_args = ["--quiet", "--no-video", "--no-xlib", "--no-video-title-show"]
        print(f"[HYBRID] Creating audio-only VLC instance: {vlc_args}", flush=True)
        self.instance = vlc.Instance(*vlc_args)
        self.audio_player = self.instance.media_player_new()
        self.media = None
        print(f"[HYBRID] Audio player created", flush=True)
        
        # OpenCV state for video frames
        self.widget = widget
        self.cap: cv2.VideoCapture | None = None
        self.path: str | None = None
        self.playing = False
        self.duration_ms = 0
        self.position_ms = 0
        self.started_at = 0.0
        self.started_position_ms = 0
        self.last_frame: np.ndarray | None = None
        self._fps = 30.0
        self._muted = False
        self._volume = 100

    # Prepare environment helper (same as VLCVideoPlayer)
    _prepare_vlc_environment = VLCVideoPlayer._prepare_vlc_environment

    def attach(self, widget: Any) -> None:
        # No NSView attachment needed - OpenCV renders frames into widget
        self.widget = widget

    def load(self, path: str) -> None:
        print(f"[HYBRID] load() called with: {path}", flush=True)
        self.stop()
        media_path = str(Path(path).expanduser())
        
        # Open video with OpenCV for frames
        with CV_LOCK:
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"OpenCV could not open video: {media_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._fps = fps if fps and fps > 0 else 30.0
        self.duration_ms = int((frames / fps) * 1000) if fps and fps > 0 and frames and frames > 0 else 0
        self.cap = cap
        self.path = media_path
        self.position_ms = 0
        self.started_position_ms = 0
        self.last_frame = None
        print(f"[HYBRID] OpenCV opened: fps={self._fps:.1f} duration_ms={self.duration_ms}", flush=True)
        
        # Load same file into VLC for audio
        try:
            media = self.instance.media_new(media_path)
            self.audio_player.set_media(media)
            self.media = media
            self.set_muted(False)
            self.set_volume(self._volume)
            try:
                media.parse_with_options(self.vlc.MediaParseFlag.local, 1000)
            except Exception:
                try:
                    media.parse()
                except Exception:
                    pass
            print(f"[HYBRID] VLC audio loaded", flush=True)
        except Exception as exc:
            print(f"[HYBRID] VLC audio load failed (continuing video-only): {exc}", flush=True)

    def play(self) -> None:
        print(f"[HYBRID] play() called", flush=True)
        if self.cap is None:
            return
        self.playing = True
        self.started_at = time.time()
        self.started_position_ms = self.position_ms
        try:
            self.audio_player.play()
            self.set_muted(False)
            self.set_volume(self._volume)
            print(f"[HYBRID] Audio playback started", flush=True)
        except Exception as exc:
            print(f"[HYBRID] Audio play failed: {exc}", flush=True)

    def pause(self) -> None:
        self.position_ms = self.get_time_ms()
        self.playing = False
        try:
            self.audio_player.pause()
        except Exception:
            pass

    def stop(self) -> None:
        self.playing = False
        self.position_ms = 0
        self.started_position_ms = 0
        self.last_frame = None
        if self.cap is not None:
            try:
                with CV_LOCK:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            self.audio_player.stop()
        except Exception:
            pass

    def cleanup(self) -> None:
        self.stop()
        try:
            self.audio_player.release()
        except Exception:
            pass
        try:
            self.instance.release()
        except Exception:
            pass

    def seek_ratio(self, ratio: float) -> None:
        if self.cap is None or self.duration_ms <= 0:
            return
        ratio = float(min(1.0, max(0.0, ratio)))
        self.position_ms = int(self.duration_ms * ratio)
        try:
            with CV_LOCK:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, self.position_ms)
        except Exception:
            pass
        self.started_at = time.time()
        self.started_position_ms = self.position_ms
        self.last_frame = None
        try:
            self.audio_player.set_position(ratio)
        except Exception:
            pass

    def get_time_ms(self) -> int:
        if self.playing:
            elapsed_ms = int((time.time() - self.started_at) * 1000)
            value = self.started_position_ms + elapsed_ms
            if self.duration_ms > 0:
                value = min(value, self.duration_ms)
            return max(0, value)
        return max(0, self.position_ms)

    def get_duration_ms(self) -> int:
        return max(0, self.duration_ms)

    def get_position_ratio(self) -> float:
        duration = self.get_duration_ms()
        return (self.get_time_ms() / duration) if duration > 0 else 0.0

    def is_playing(self) -> bool:
        return self.playing

    def get_volume(self) -> int:
        return int(self._volume)

    def set_volume(self, value: int) -> None:
        value = int(max(0, min(100, value)))
        self._volume = value
        try:
            self.audio_player.audio_set_volume(value)
        except Exception:
            pass

    def is_muted(self) -> bool:
        return bool(self._muted)

    def set_muted(self, value: bool) -> None:
        self._muted = bool(value)
        try:
            self.audio_player.audio_set_mute(bool(value))
        except Exception:
            pass

    def has_audio_stream(self) -> bool:
        try:
            count = self.audio_player.audio_get_track_count()
            if count is not None and count > 0:
                return True
        except Exception:
            pass
        try:
            if self.media is not None:
                tracks = self.media.tracks_get()
                for track in tracks or []:
                    if getattr(track, "type", None) == self.vlc.TrackType.audio:
                        return True
        except Exception:
            pass
        return False

    def debug_state(self) -> dict:
        def _safe(fn, default="unknown"):
            try:
                r = fn()
                return r if r is not None else default
            except Exception as exc:
                return f"unknown:{type(exc).__name__}"
        return {
            "backend": self.backend_name,
            "audio_supported": self.audio_supported,
            "path": self.path,
            "duration_ms": self.get_duration_ms(),
            "time_ms": self.get_time_ms(),
            "muted": self.is_muted(),
            "volume": self.get_volume(),
            "audio_track": _safe(lambda: self.audio_player.audio_get_track()),
            "audio_track_count": _safe(lambda: self.audio_player.audio_get_track_count()),
            "has_audio_stream": _safe(lambda: self.has_audio_stream()),
            "state": "Playing" if self.playing else "Stopped",
        }

    def render_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        target_ms = self.get_time_ms()
        if self.duration_ms > 0 and target_ms >= self.duration_ms:
            self.pause()
            return self.last_frame
        try:
            with CV_LOCK:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                ok, frame = self.cap.read()
            if not ok:
                self.pause()
                return self.last_frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame_rgb
            self.position_ms = target_ms
            return frame_rgb
        except Exception as exc:
            print(f"[HYBRID] render_frame error: {exc}", flush=True)
            return self.last_frame
