"""Glue: process a whole video, or one frame, with the chosen removal + watermark."""
from __future__ import annotations

import os
import tempfile
import threading
from typing import Callable, Iterator

import cv2
import numpy as np

from .ai_fill import best_inpainter
from .logo_removal import LogoRegion, LogoRemover
from .media import Cancelled, FrameReader, VideoInfo, VideoWriter
from .smart_fill import LOOKAHEAD, SmartRemover, clean_frame_at
from .thumbnail import ThumbnailSettings, make_thumbnail
from .watermark import WatermarkRenderer, WatermarkSettings


class Processor:
    """Everything that happens to a frame, in order: remove logos, then add your watermark."""

    def __init__(self, info: VideoInfo, regions: list[LogoRegion] | None = None, method: str = "smart",
                 expand: int = 4, watermark: WatermarkSettings | None = None, inpainter=None):
        self.info = info
        self.regions = list(regions or [])
        self.expand = expand
        self.smart = bool(self.regions) and method == "smart"
        self.remover = LogoRemover(self.regions, method, expand) if self.regions and not self.smart else None
        self._inpainter = inpainter
        self.watermark = (WatermarkRenderer(watermark, info.width, info.height)
                          if watermark is not None and watermark.enabled else None)

    @property
    def inpainter(self):
        if self._inpainter is None:
            self._inpainter = best_inpainter()
        return self._inpainter

    def _finish(self, frame: np.ndarray, with_watermark: bool) -> np.ndarray:
        if with_watermark and self.watermark:
            frame = self.watermark.apply(frame)
        return frame

    def process(self, frame: np.ndarray, t: float, with_watermark: bool = True) -> np.ndarray:
        """One frame on its own (preview, thumbnail). Smart mode reads the frames around it."""
        if self.smart:
            idx = int(round(t * self.info.fps))
            cleaned = clean_frame_at(self.info.path, self.regions, self.expand, idx, self.info.fps,
                                     self.info.frame_count, self.inpainter)
            if cleaned is not None:
                frame = cleaned
        elif self.remover:
            frame = self.remover.apply(frame, t)
        return self._finish(frame, with_watermark)

    def frames(self) -> Iterator[np.ndarray]:
        """Every frame of the video, processed, in order."""
        cap = cv2.VideoCapture(self.info.path)
        smart = ahead = None
        if self.smart:
            smart = SmartRemover(self.regions, self.expand, self.info.width, self.info.height,
                                 self.info.fps, self.inpainter)
            ahead = cv2.VideoCapture(self.info.path)   # second reader running ahead for lookahead
        pushed, ahead_done, i = 0, False, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if smart:
                    while not ahead_done and pushed <= i + LOOKAHEAD:
                        ok2, f2 = ahead.read()
                        if not ok2:
                            ahead_done = True
                            break
                        smart.push(pushed, f2)
                        pushed += 1
                    frame = smart.clean(i, frame)
                elif self.remover:
                    frame = self.remover.apply(frame, i / self.info.fps)
                yield self._finish(frame, True)
                i += 1
        finally:
            cap.release()
            if ahead is not None:
                ahead.release()


def render_thumbnail(info: VideoInfo, processor: Processor, t: float, settings: ThumbnailSettings):
    reader = FrameReader(info)
    frame = reader.frame_at(t)
    reader.close()
    if frame is None:
        raise RuntimeError(f"Could not read the frame at {t:.2f}s")
    frame = processor.process(frame, t, with_watermark=settings.include_watermark)
    return make_thumbnail(frame, settings)


def export_video(info: VideoInfo, out_path: str, processor: Processor, crf: int = 20,
                 preset: str = "medium", keep_audio: bool = True, cover=None,
                 progress: Callable[[int, int], None] | None = None,
                 cancel: threading.Event | None = None) -> str:
    """Re-encode the video with `processor` applied to every frame. `cover` is a PIL image or None."""
    cover_path = None
    if cover is not None:
        fd, cover_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cover.convert("RGB").save(cover_path, quality=92)

    tmp_out = out_path + ".part" + os.path.splitext(out_path)[1]
    writer = VideoWriter(tmp_out, info, crf=crf, preset=preset, keep_audio=keep_audio, cover_image=cover_path)
    try:
        for i, frame in enumerate(processor.frames(), 1):
            writer.write(frame)
            if progress and (i % 4 == 0):
                progress(i, info.frame_count)
            if cancel is not None and cancel.is_set():
                writer.abort()
                raise Cancelled()
        writer.close()
        os.replace(tmp_out, out_path)
        if progress:
            progress(info.frame_count, info.frame_count)
        return out_path
    except Cancelled:
        raise
    except Exception:
        if writer.proc.poll() is None:
            writer.abort()
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise
    finally:
        if cover_path and os.path.exists(cover_path):
            os.remove(cover_path)
