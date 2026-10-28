"""Video-aware logo removal.

A logo sits still on the screen while the camera moves, so whatever it hides in one frame is
usually visible in a nearby frame. For each frame we line up nearby frames with the current
one (tracking the scenery around the logo), copy the real background across, and match its
brightness. Whatever was never uncovered (a locked-off shot, slow motion) is generated once
per shot by the inpainter and then carried along with the camera, so it doesn't flicker.
"""
from __future__ import annotations

import cv2
import numpy as np

from .logo_removal import LogoRegion, _disk, frame_mask, prepare_region

LOOKAHEAD = 40          # frames before/after the current one used as background sources
MAX_ALIGN_ERR = 9.0     # mean abs difference (0-255) for consecutive frames to count as one shot
MAX_RESIDUAL = 4.0      # same, after removing brightness differences, for borrowing background
_I3 = np.eye(3)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _harmonize(out: np.ndarray, patch: np.ndarray, take: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Shift `patch` brightness smoothly so it meets the trusted pixels around `take` without a seam.

    The difference along the patch border is spread inward with a normalised blur, so a patch
    from a frame with slightly different lighting (or a vignette) blends in like a Poisson fill.
    """
    sigma = 6.0
    ys, xs = np.nonzero(take)
    m = int(3 * sigma) + 4   # work on the patch's neighbourhood only
    y0, y1 = max(0, ys.min() - m), min(take.shape[0], ys.max() + 1 + m)
    x0, x1 = max(0, xs.min() - m), min(take.shape[1], xs.max() + 1 + m)
    t, k = take[y0:y1, x0:x1], known[y0:y1, x0:x1]
    border = (cv2.dilate(t.astype(np.uint8), _disk(3)) > 0) & k & ~t
    result = patch.copy()
    if border.sum() < 10:
        return result
    p = patch[y0:y1, x0:x1].astype(np.float32)
    diff = np.zeros_like(p)
    diff[border] = out[y0:y1, x0:x1][border].astype(np.float32) - p[border]
    num = cv2.GaussianBlur(diff, (0, 0), sigma)
    den = cv2.GaussianBlur(border.astype(np.float32), (0, 0), sigma)[..., None]
    eps = 0.02   # far from any border, fall back to the average shift
    corr = (num + eps * np.median(diff[border], axis=0)) / (den + eps)
    result[y0:y1, x0:x1] = np.clip(p + corr, 0, 255).astype(np.uint8)
    return result


def _sources(idx: int, lo: int, hi: int) -> list[int]:
    """Neighbour frames to try, nearest first, thinning out further away."""
    out = []
    for d in range(1, LOOKAHEAD + 1):
        if d > 3 and d % (2 if d <= 12 else 4):
            continue
        for j in (idx - d, idx + d):
            if lo <= j <= hi:
                out.append(j)
    return out


class _Track:
    """Everything needed to clean one logo region across a video."""

    def __init__(self, region: LogoRegion, width: int, height: int, expand: int):
        margin = int(max(48, max(region.w, region.h)))  # room to find scenery that slid under the logo
        self.p = prepare_region(region, width, height, expand, margin)
        self.region = self.p["region"]
        self.crops: dict[int, np.ndarray] = {}
        self.grays: dict[int, np.ndarray] = {}
        self.holes: dict[int, np.ndarray] = {}
        self.motion: dict[int, np.ndarray | None] = {}   # j -> 3x3 mapping roi(j) coords to roi(j+1)
        self.key = None   # (frame index, cleaned roi, transform from key frame to last cleaned frame)

    # -- buffering
    def push(self, idx: int, frame: np.ndarray):
        x0, y0, x1, y1 = self.p["box"]
        roi = frame[y0:y1, x0:x1].copy()
        self.crops[idx] = roi
        self.grays[idx] = _gray(roi)
        self.holes[idx] = frame_mask(roi, self.p) > 0
        if idx - 1 in self.crops:
            self.motion[idx - 1] = self._estimate(idx - 1, idx)

    def prune(self, before: int):
        for d in (self.crops, self.grays, self.holes):
            for k in [k for k in d if k < before]:
                del d[k]
        for k in [k for k in self.motion if k < before - 1]:
            del self.motion[k]

    # -- motion
    def _estimate(self, a: int, b: int) -> np.ndarray | None:
        """Similarity transform (move/zoom/rotate) of the scenery from frame a to frame b."""
        ga, gb = self.grays[a], self.grays[b]
        excluded = cv2.dilate((self.holes[a] | self.holes[b]).astype(np.uint8), _disk(6)) > 0
        M = None
        pts = cv2.goodFeaturesToTrack(ga, 160, 0.01, 6, mask=(~excluded).astype(np.uint8) * 255)
        if pts is not None and len(pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                A, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5)
                if A is not None and inliers is not None and inliers.sum() >= max(6, 0.3 * ok.sum()):
                    M = np.vstack([A, [0, 0, 1]])
        if M is None:
            M = _I3.copy()   # flat area with nothing to track: try "no motion"
        err = self._align_error(self.crops[a], ~self.holes[a], self.crops[b], ~excluded, M)
        return M if err < MAX_ALIGN_ERR * 1.5 else None   # None = scene cut or unusable

    @staticmethod
    def _align_error(src, src_valid, dst, dst_area, M) -> float:
        h, w = dst.shape[:2]
        warped = cv2.warpAffine(src, M[:2], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(src_valid.astype(np.uint8), M[:2], (w, h), flags=cv2.INTER_NEAREST) > 0
        sel = valid & dst_area
        if sel.sum() < 200:
            return np.inf
        return float(np.abs(warped[sel].astype(np.float32) - dst[sel].astype(np.float32)).mean())

    def _transforms_to(self, idx: int) -> dict[int, np.ndarray]:
        """Transforms mapping each buffered frame's roi onto frame idx's roi (stopping at cuts)."""
        mats, M = {}, _I3
        j = idx - 1
        while j in self.crops and self.motion.get(j) is not None:
            M = M @ self.motion[j]
            mats[j] = M
            j -= 1
        M, j = _I3, idx
        while j + 1 in self.crops and self.motion.get(j) is not None:
            M = M @ np.linalg.inv(self.motion[j])
            mats[j + 1] = M
            j += 1
        return mats

    # -- cleaning
    def clean(self, idx: int, frame: np.ndarray, t: float, inpainter) -> np.ndarray:
        if idx not in self.crops:
            self.push(idx, frame)
        if not self.region.active(t):
            self.key = None
            return frame
        roi, hole = self.crops[idx], self.holes[idx]
        if not hole.any():
            return frame
        h, w = hole.shape
        hole_u8 = hole.astype(np.uint8)
        ring = (cv2.dilate(hole_u8, _disk(10)) > 0) & ~(cv2.dilate(hole_u8, _disk(2)) > 0)
        out = roi.copy()
        need = hole.copy()
        self.sources = np.zeros((h, w), np.uint8)   # 1 = other frame, 2 = carried-over generated, 3 = generated

        # 1) Real background from nearby frames.
        mats = self._transforms_to(idx)
        ys, xs = np.nonzero(hole)
        centre = np.array([xs.mean(), ys.mean(), 1.0])
        lo, hi = min(self.crops), max(self.crops)
        candidates = []
        for j in _sources(idx, lo, hi):
            M = mats.get(j)
            if M is None or np.hypot(*(M @ centre - centre)[:2]) < 1.0:
                continue   # cut in between, or the scenery hasn't moved enough to reveal anything
            valid = cv2.warpAffine((~self.holes[j]).astype(np.uint8), M[:2], (w, h), flags=cv2.INTER_NEAREST) > 0
            cover = int((hole & valid).sum())
            check = ring & valid
            if cover == 0 or check.sum() < 60:
                continue
            warped = cv2.warpAffine(self.crops[j], M[:2], (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            diff = roi[check].astype(np.float32) - warped[check].astype(np.float32)
            # How well the scenery lines up once overall brightness differences are ignored.
            if np.abs(diff - np.median(diff, axis=0)).mean() > MAX_RESIDUAL:
                continue
            candidates.append((cover, -abs(j - idx), warped, valid))
        # Fewest, biggest pieces first: every extra source frame adds a seam.
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        known = ~hole
        for _, _, warped, valid in candidates:
            take = need & valid
            if take.sum() < 25:
                continue   # slivers add seams for little gain; later steps cover them
            out[take] = _harmonize(out, warped, take, known)[take]
            self.sources[take] = 1
            known |= take
            need &= ~take
            if not need.any():
                break

        # 2) Background generated earlier in this shot, moved along with the camera.
        key = self.key if self.key is not None and self.key[0] == idx - 1 else None
        if key is not None and self.motion.get(idx - 1) is not None:
            M = self.motion[idx - 1] @ key[2]
            if need.any():
                warped = cv2.warpAffine(key[1], M[:2], (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                covered = cv2.warpAffine(np.ones((h, w), np.uint8), M[:2], (w, h), flags=cv2.INTER_NEAREST) > 0
                check = ring & covered
                if need[~covered].any() or check.sum() < 60:
                    key = None
                else:
                    diff = roi[check].astype(np.float32) - warped[check].astype(np.float32)
                    if np.abs(diff - np.median(diff, axis=0)).mean() > MAX_RESIDUAL * 1.3:
                        key = None   # scenery changed (something moved through): regenerate
                    else:
                        out[need] = _harmonize(out, warped, need, known)[need]
                        self.sources[need] = 2
                        need[:] = False
            if key is not None:
                self.key = (idx, key[1], M)
        else:
            key = None

        # 3) Generate whatever is still missing, and remember it for the next frames.
        if need.any():
            self.sources[need] = 3
            out = inpainter.inpaint(out, need.astype(np.uint8) * 255)
            self.key = (idx, out.copy(), _I3)
        elif key is None:
            self.key = None

        x0, y0, x1, y1 = self.p["box"]
        frame[y0:y1, x0:x1] = out
        return frame


class SmartRemover:
    """Streaming remover: push() frames ahead of time, then clean() them in order."""

    def __init__(self, regions: list[LogoRegion], expand: int, width: int, height: int,
                 fps: float, inpainter):
        self.tracks = [_Track(r, width, height, max(0, int(expand))) for r in regions if r.w > 1 and r.h > 1]
        self.fps = fps
        self.inpainter = inpainter

    def push(self, idx: int, frame: np.ndarray):
        for tr in self.tracks:
            tr.push(idx, frame)

    def clean(self, idx: int, frame: np.ndarray) -> np.ndarray:
        t = idx / self.fps
        for tr in self.tracks:
            frame = tr.clean(idx, frame, t, self.inpainter)
            tr.prune(idx - LOOKAHEAD)
        return frame


def clean_frame_at(path: str, regions: list[LogoRegion], expand: int, idx: int, fps: float,
                   frame_count: int, inpainter, window: int = 24) -> np.ndarray | None:
    """Clean a single frame (for previews/thumbnails) using the frames around it."""
    start, end = max(0, idx - window), min(frame_count - 1, idx + window)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    remover, centre = None, None
    for i in range(start, end + 1):
        ok, f = cap.read()
        if not ok:
            break
        if remover is None:
            remover = SmartRemover(regions, expand, f.shape[1], f.shape[0], fps, inpainter)
        remover.push(i, f)
        if i == idx:
            centre = f.copy()
    cap.release()
    if remover is None or centre is None:
        return None
    return remover.clean(idx, centre)
