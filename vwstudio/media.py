"""Video I/O: locate ffmpeg, probe files, read frames with OpenCV, encode with ffmpeg."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

# Keep ffmpeg from flashing a console window when launched from the GUI on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    """Raised when the user cancels a long-running job."""


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffprobe() -> str | None:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    ff = find_ffmpeg()
    if ff:
        cand = os.path.join(os.path.dirname(ff), "ffprobe" + (".exe" if os.name == "nt" else ""))
        if os.path.exists(cand):
            return cand
    return None


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_str: str          # exact rate for ffmpeg, e.g. "30000/1001"
    frame_count: int
    duration: float
    audio_codec: str | None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def _ffprobe_streams(path: str) -> list[dict]:
    probe = find_ffprobe()
    if not probe:
        return []
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW,
        )
        return json.loads(out.stdout or "{}").get("streams", [])
    except Exception:
        return []


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read frames from: {path}")
    # Use the decoded frame's shape: it already has phone rotation applied.
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    fps_str = str(Fraction(fps).limit_denominator(1001))
    # Without ffprobe we can't see the audio track; "unknown" makes the writer map it optionally.
    audio = None if find_ffprobe() else "unknown"
    got_rate = False
    for s in _ffprobe_streams(path):
        if s.get("codec_type") == "video" and not got_rate \
                and s.get("avg_frame_rate", "0/0") not in ("0/0", ""):
            rate = Fraction(s["avg_frame_rate"])
            if rate > 0:
                fps, fps_str, got_rate = float(rate), s["avg_frame_rate"], True
        elif s.get("codec_type") == "audio" and audio is None:
            audio = s.get("codec_name", "unknown")
    if count <= 0:
        count = 1
    return VideoInfo(path, width, height, fps, fps_str, count, count / fps, audio)


class FrameReader:
    """Random-access and sequential frame reading for one video."""

    def __init__(self, info: VideoInfo):
        self.info = info
        self.cap = cv2.VideoCapture(info.path)

    def frame_at(self, t: float) -> np.ndarray | None:
        idx = max(0, min(self.info.frame_count - 1, int(round(t * self.info.fps))))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:  # some files misreport the last frames; step back a little
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 3))
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def sample_frames(info: VideoInfo, count: int = 32, start: float | None = None,
                  end: float | None = None, budget_mb: int = 400) -> tuple[list[float], list[np.ndarray]]:
    """Evenly spaced (times, frames) from [start, end], capped to fit in `budget_mb` of RAM."""
    per_frame = info.width * info.height * 3 / 1e6
    count = max(6, min(count, int(budget_mb / max(per_frame, 0.01))))
    t0 = max(0.0, start or 0.0)
    t1 = min(info.duration, end if end is not None else info.duration)
    if t1 <= t0:
        t0, t1 = 0.0, info.duration
    reader = FrameReader(info)
    times, frames = [], []
    for t in np.linspace(t0, max(t0, t1 - 1.0 / info.fps), count):
        f = reader.frame_at(float(t))
        if f is not None and f.shape[:2] == (info.height, info.width):
            times.append(float(t))
            frames.append(f)
    reader.close()
    return times, frames


class VideoWriter:
    """Pipes BGR frames into ffmpeg (H.264), copying the source audio and optional cover image."""

    _COPYABLE_AUDIO = {".mp4": {"aac", "mp3", "alac", "ac3", "eac3", "opus"},
                       ".mov": {"aac", "mp3", "alac", "pcm_s16le"},
                       ".m4v": {"aac", "mp3"},
                       ".mkv": None}  # None = anything

    def __init__(self, out_path: str, info: VideoInfo, crf: int = 20, preset: str = "medium",
                 keep_audio: bool = True, cover_image: str | None = None):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found. Install it (winget install ffmpeg) "
                               "or run: pip install imageio-ffmpeg")
        self.out_path = out_path
        # libx264 with yuv420p needs even dimensions.
        self.w, self.h = info.width - info.width % 2, info.height - info.height % 2
        ext = os.path.splitext(out_path)[1].lower()

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
               "-r", info.fps_str, "-i", "-"]
        inputs = 1
        audio_in = None
        if keep_audio and info.has_audio:
            cmd += ["-i", info.path]
            audio_in, inputs = inputs, inputs + 1
        cover_in = None
        if cover_image and ext in (".mp4", ".m4v", ".mov", ".mkv"):
            cmd += ["-i", cover_image]
            cover_in = inputs

        cmd += ["-map", "0:v:0"]
        if audio_in is not None:
            cmd += ["-map", f"{audio_in}:a:0?"]
        if cover_in is not None:
            cmd += ["-map", f"{cover_in}:v:0"]
        cmd += ["-c:v:0", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if audio_in is not None:
            allowed = self._COPYABLE_AUDIO.get(ext, set())
            if allowed is None or info.audio_codec in allowed:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        if cover_in is not None:
            cmd += ["-c:v:1", "mjpeg", "-disposition:v:1", "attached_pic"]
        if ext in (".mp4", ".m4v", ".mov"):
            cmd += ["-movflags", "+faststart"]
        cmd.append(out_path)

        self._log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._log, creationflags=_NO_WINDOW)

    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = frame[: self.h, : self.w]
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.close()  # raises with ffmpeg's error message

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        self._log.seek(0)
        err = self._log.read().decode("utf-8", "replace").strip()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed (exit {code}):\n{err[-1500:]}")

    def abort(self):
        try:
            self.proc.kill()
            self.proc.wait()
        finally:
            self._log.close()
            if os.path.exists(self.out_path):
                try:
                    os.remove(self.out_path)
                except OSError:
                    pass
