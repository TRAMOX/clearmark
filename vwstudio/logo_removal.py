"""Find static logos/watermarks in a video and paint them out frame by frame."""
from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

METHODS = {"smart": "Best: real background + AI fill (slower)",
           "inpaint": "Fast fill", "blur": "Blur"}


@dataclass
class LogoRegion:
    x: int
    y: int
    w: int
    h: int
    start: float | None = None      # seconds; None = from the beginning
    end: float | None = None        # seconds; None = until the end
    mask: np.ndarray | None = field(default=None, repr=False)  # h x w uint8, 255 = logo; None = whole box
    polarity: int = 0               # +1 logo is lighter than the picture, -1 darker, 0 unknown
    strength: float = 0.0           # typical opacity of the logo (0..1), set by shape detection

    def active(self, t: float) -> bool:
        return (self.start is None or t >= self.start) and (self.end is None or t <= self.end)

    def clamp(self, width: int, height: int) -> "LogoRegion":
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(width, self.x + self.w), min(height, self.y + self.h)
        mask = self.mask
        if mask is not None:
            mask = mask[y0 - self.y: y1 - self.y, x0 - self.x: x1 - self.x]
        return replace(self, x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0), mask=mask)

    def to_dict(self) -> dict:
        d = {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "start": self.start, "end": self.end,
             "polarity": self.polarity, "strength": self.strength}
        if self.mask is not None:
            d["mask_png"] = base64.b64encode(cv2.imencode(".png", self.mask)[1].tobytes()).decode()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LogoRegion":
        mask = None
        if d.get("mask_png"):
            buf = np.frombuffer(base64.b64decode(d["mask_png"]), np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return cls(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), d.get("start"), d.get("end"),
                   mask, int(d.get("polarity", 0)), float(d.get("strength", 0.0)))


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _fill_holes(bw: np.ndarray) -> np.ndarray:
    h, w = bw.shape
    padded = cv2.copyMakeBorder(bw, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return (padded | cv2.bitwise_not(flood))[1:-1, 1:-1]


def _drop_specks(bw: np.ndarray, rel: float = 0.03, min_px: int = 6) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return bw
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < max(min_px, rel * biggest):
            bw[labels == k] = 0
    return bw


def refine_region(frames: list[np.ndarray], region: LogoRegion) -> LogoRegion:
    """Find the exact shape of a static overlay inside a region's box.

    Each sampled frame's box interior is filled in from its border, giving a rough background.
    A semi-transparent overlay brightens (or darkens) the same pixels by the same fraction in
    every frame, so the per-pixel median of that fraction lights up the logo and averages
    away scene content. Returns the region unchanged (whole box) when nothing is found.
    """
    x, y, w, h = region.x, region.y, region.w, region.h
    if w < 10 or h < 10 or len(frames) < 4:
        return region
    crops = np.stack([f[y:y + h, x:x + w] for f in frames])
    border = max(2, min(w, h) // 20)
    hole = np.zeros((h, w), np.uint8)
    hole[border:-border, border:-border] = 255
    bg = np.stack([cv2.inpaint(c, hole, 5, cv2.INPAINT_TELEA) for c in crops]).astype(np.float32).mean(-1)
    fg = crops.astype(np.float32).mean(-1)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        lighter = np.where(255 - bg > 40, (fg - bg) / (255 - bg), np.nan)   # white-ish logos
        darker = np.where(bg > 40, (bg - fg) / bg, np.nan)                 # dark logos
        maps = {+1: np.nan_to_num(np.nanmedian(lighter, 0)), -1: np.nan_to_num(np.nanmedian(darker, 0))}

    # A logo is one polarity and covers a small part of the box, so pick the map with the
    # sharpest peak over its typical level (the other one mostly carries fill-in errors).
    def contrast(s):
        inside = s[hole > 0]
        return np.percentile(inside, 99.5) - np.percentile(inside, 75)

    polarity = max(maps, key=lambda p: contrast(maps[p]))
    score = maps[polarity]
    score[hole == 0] = 0
    peak = float(np.percentile(score[hole > 0], 99.5))
    if peak < 0.05:
        return region

    bw = (score > max(0.035, 0.3 * peak)).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _disk(2))
    bw = _drop_specks(_fill_holes(bw))
    if not bw.any():
        return region
    strength = float(np.median(score[bw > 0]))
    return replace(region, mask=cv2.dilate(bw, _disk(2)), polarity=polarity, strength=strength)


def tighten(region: LogoRegion, margin: int, width: int, height: int) -> LogoRegion:
    """Shrink a region's box to its mask's bounding box (plus margin)."""
    if region.mask is None or not region.mask.any():
        return region
    ys, xs = np.nonzero(region.mask)
    x0 = max(0, region.x + xs.min() - margin)
    y0 = max(0, region.y + ys.min() - margin)
    x1 = min(width, region.x + xs.max() + 1 + margin)
    y1 = min(height, region.y + ys.max() + 1 + margin)
    full = np.zeros((height, width), np.uint8)
    full[region.y:region.y + region.mask.shape[0], region.x:region.x + region.mask.shape[1]] = region.mask
    return replace(region, x=x0, y=y0, w=x1 - x0, h=y1 - y0, mask=full[y0:y1, x0:x1].copy())


def detect_logos(frames: list[np.ndarray], max_logos: int = 4) -> list[LogoRegion]:
    """Scan whole frames for overlays that stay put while the picture changes underneath.

    Works best when the video has camera motion or several shots; on a locked-off tripod
    shot, static scenery looks just like a logo, so review the results.
    """
    if len(frames) < 6:
        return []
    height, width = frames[0].shape[:2]
    scale = min(1.0, 640 / max(height, width))
    gx, gy = [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if scale < 1:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        gx.append(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy.append(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    gx, gy = np.stack(gx), np.stack(gy)
    mx, my = np.median(gx, 0), np.median(gy, 0)
    mag = np.maximum(np.hypot(mx, my), 1e-3)
    # How strongly each frame agrees with the median edge direction; a logo's edges are
    # there in (nearly) every frame, scenery edges come and go.
    agree = np.percentile((gx * mx + gy * my) / mag, 10, axis=0)
    if agree.max() < 15:
        return []
    bw = (agree > max(10.0, 0.3 * float(agree.max()))).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _disk(4))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    sh, sw = bw.shape
    seeds = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < max(12, 5e-5 * sh * sw) or w > 0.5 * sw or h > 0.5 * sh:
            continue
        seeds.append((float(agree[labels == k].sum()), x, y, w, h))
    seeds.sort(reverse=True)

    boxes: list[list[int]] = []
    for _, x, y, w, h in seeds[: max_logos * 3]:
        # Grow the seed so faint neighbouring parts of the same logo are inside the box.
        grow = int(0.75 * max(w, h)) + 4
        box = [int(round(v / scale)) for v in (x - grow, y - grow, x + w + grow, y + h + grow)]
        box = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        for b in boxes:
            if box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]:
                b[:] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
                break
        else:
            boxes.append(box)

    regions = []
    for x0, y0, x1, y1 in boxes[:max_logos]:
        r = refine_region(frames, LogoRegion(x0, y0, x1 - x0, y1 - y0))
        if r.mask is not None:
            regions.append(tighten(r, 14, width, height))
    return regions


TRACK_RADIUS = 10   # px around a detected shape searched for parts that drift or twinkle


def prepare_region(r: LogoRegion, width: int, height: int, expand: int, pad: int) -> dict:
    """Masks for one region inside a padded box ("roi") around it, in roi coordinates."""
    r = r.clamp(width, height)
    x0, y0 = max(0, r.x - pad), max(0, r.y - pad)
    x1, y1 = min(width, r.x + r.w + pad), min(height, r.y + r.h + pad)
    base = np.zeros((y1 - y0, x1 - x0), np.uint8)
    base[r.y - y0: r.y - y0 + r.h, r.x - x0: r.x - x0 + r.w] = 255 if r.mask is None else r.mask
    p = {"region": r, "box": (x0, y0, x1, y1), "base": base, "expand": expand,
         "mask": cv2.dilate(base, _disk(expand)) if expand else base}
    if r.mask is not None and r.polarity and r.strength > 0:
        around = cv2.dilate(base, _disk(TRACK_RADIUS))
        p["track_hole"] = around
        p["track_ring"] = cv2.subtract(around, base) > 0
    return p


def frame_mask(roi: np.ndarray, p: dict) -> np.ndarray:
    """The area to replace in this frame: the static shape plus logo pixels that drifted a little."""
    if "track_ring" not in p:
        return p["mask"]
    r = p["region"]
    bg = cv2.inpaint(roi, p["track_hole"], 3, cv2.INPAINT_TELEA).astype(np.float32).mean(-1)
    fg = roi.astype(np.float32).mean(-1)
    if r.polarity > 0:
        ratio = (fg - bg) / np.maximum(255 - bg, 30)
    else:
        ratio = (bg - fg) / np.maximum(bg, 30)
    extra = ((ratio > 0.55 * r.strength) & p["track_ring"]).astype(np.uint8) * 255
    extra = cv2.morphologyEx(extra, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if not extra.any():
        return p["mask"]
    extra = _drop_specks(extra, rel=0.0, min_px=8)
    combined = cv2.bitwise_or(p["base"], cv2.dilate(extra, _disk(2)))
    return cv2.dilate(combined, _disk(p["expand"])) if p["expand"] else combined


class LogoRemover:
    """Per-frame removal (fast fill or blur). Masks are prepared once per frame size."""

    def __init__(self, regions: list[LogoRegion], method: str = "inpaint", expand: int = 4):
        self.regions = [r for r in regions if r.w > 1 and r.h > 1]
        self.method = "blur" if method == "blur" else "inpaint"
        self.expand = max(0, int(expand))
        self._prepared: list[dict] = []
        self._size = None

    def _prepare(self, width: int, height: int):
        pad = self.expand + TRACK_RADIUS + 8
        self._prepared = [prepare_region(r, width, height, self.expand, pad) for r in self.regions]
        for p in self._prepared:
            p["soft"] = cv2.GaussianBlur(p["mask"], (0, 0), 2.0).astype(np.float32)[..., None] / 255.0
        self._size = (width, height)

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._prepare(w, h)
        for p in self._prepared:
            if not p["region"].active(t) or not p["mask"].any():
                continue
            x0, y0, x1, y1 = p["box"]
            roi = frame[y0:y1, x0:x1]
            if self.method == "blur":
                r = p["region"]
                filled = cv2.GaussianBlur(roi, (0, 0), max(4.0, 0.15 * max(r.w, r.h)))
                frame[y0:y1, x0:x1] = (roi * (1 - p["soft"]) + filled * p["soft"]).astype(np.uint8)
            else:
                frame[y0:y1, x0:x1] = cv2.inpaint(roi, frame_mask(roi, p), 5, cv2.INPAINT_TELEA)
        return frame
