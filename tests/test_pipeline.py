"""End-to-end checks on a synthetic clip: a panning textured scene with a static see-through logo."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vwstudio.ai_fill import TeleaInpainter  # noqa: E402
from vwstudio.logo_removal import LogoRegion, detect_logos  # noqa: E402
from vwstudio.media import probe, sample_frames  # noqa: E402
from vwstudio.pipeline import Processor, export_video  # noqa: E402
from vwstudio.thumbnail import ThumbnailSettings, best_frame_time  # noqa: E402
from vwstudio.watermark import WatermarkSettings  # noqa: E402

W, H, N, FPS = 320, 240, 72, 24
LOGO = (240, 170, 50, 40)   # x, y, w, h of the fake logo


def _logo_mask():
    m = np.zeros((H, W), np.float32)
    x, y, w, h = LOGO
    cv2.ellipse(m, (x + w // 2, y + h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (5, 5), 0)[..., None]


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    rng = np.random.default_rng(0)
    world = cv2.GaussianBlur(rng.integers(0, 255, (H + 200, W + 300, 3), dtype=np.uint8), (0, 0), 3)
    world = cv2.normalize(world, None, 30, 220, cv2.NORM_MINMAX)
    alpha = 0.35 * _logo_mask()
    path = str(tmp_path_factory.mktemp("clip") / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    clean = []
    for i in range(N):
        frame = world[60 + i // 2: 60 + i // 2 + H, 40 + 3 * i: 40 + 3 * i + W].copy()   # pan right/down
        clean.append(frame.copy())
        writer.write((frame * (1 - alpha) + 255 * alpha).astype(np.uint8))
    writer.release()
    return path, clean


def _logo_error(frame, reference):
    x, y, w, h = LOGO
    return float(np.abs(frame[y:y + h, x:x + w].astype(np.float32) - reference[y:y + h, x:x + w]).mean())


def test_detects_static_logo(clip):
    path, _ = clip
    info = probe(path)
    assert (info.width, info.height) == (W, H)
    regions = detect_logos(sample_frames(info, 24)[1])
    assert regions, "logo was not detected"
    r = regions[0]
    x, y, w, h = LOGO
    assert r.x <= x + w // 2 <= r.x + r.w and r.y <= y + h // 2 <= r.y + r.h
    assert r.mask is not None and r.polarity == 1


@pytest.mark.parametrize("method", ["inpaint", "smart"])
def test_removal_reduces_logo(clip, method):
    path, clean = clip
    info = probe(path)
    region = LogoRegion(LOGO[0] - 6, LOGO[1] - 6, LOGO[2] + 12, LOGO[3] + 12)
    proc = Processor(info, [region], method, 2, None, inpainter=TeleaInpainter())
    out = list(proc.frames())
    assert len(out) == len(clean)
    original = cv2.VideoCapture(path)
    before, after = [], []
    for i in range(10, N - 10, 6):
        original.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, src = original.read()
        before.append(_logo_error(src, clean[i]))
        after.append(_logo_error(out[i], clean[i]))
    # Fast fill can only paint in from the edges; smart mode should recover most of the real texture.
    limit = {"inpaint": 0.8, "smart": 0.4}[method]
    assert np.mean(after) < limit * np.mean(before), (method, np.mean(before), np.mean(after))
    if method == "smart":
        # Real background from other frames should beat painting in from the edges.
        fast = list(Processor(info, [region], "inpaint", 2, None).frames())
        fast_err = np.mean([_logo_error(fast[i], clean[i]) for i in range(10, N - 10, 6)])
        assert np.mean(after) < fast_err


def test_export_with_watermark_and_cover(clip, tmp_path):
    path, _ = clip
    info = probe(path)
    proc = Processor(info, [], "inpaint", 2, WatermarkSettings(text="@clearmark"))
    from vwstudio.pipeline import render_thumbnail
    thumb = render_thumbnail(info, proc, best_frame_time(info, samples=8),
                             ThumbnailSettings(title="Test", size="Square 1080x1080"))
    assert thumb.size == (1080, 1080)
    out = str(tmp_path / "out.mp4")
    export_video(info, out, proc, crf=28, preset="ultrafast", cover=thumb)
    result = probe(out)
    assert (result.width, result.height) == (W, H)
    assert abs(result.frame_count - N) <= 1
