"""AI background generation with the LaMa inpainting model (ONNX, runs offline on the CPU)."""
from __future__ import annotations

import os
import threading
import urllib.request

import cv2
import numpy as np

MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL_SIZE = 208_044_816
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "lama_fp32.onnx")


def model_available(path: str = MODEL_PATH) -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(path) and os.path.getsize(path) == MODEL_SIZE


def download_model(progress=None, cancel: threading.Event | None = None, path: str = MODEL_PATH) -> str:
    """Fetch the model once (~200 MB). `progress(done_bytes, total_bytes)` is called as it goes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or MODEL_SIZE)
        done = 0
        while True:
            if cancel is not None and cancel.is_set():
                out.close()
                os.remove(part)
                from .media import Cancelled
                raise Cancelled()
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if os.path.getsize(part) != MODEL_SIZE:
        os.remove(part)
        raise RuntimeError("The AI model download was incomplete - please try again.")
    os.replace(part, path)
    return path


class LamaInpainter:
    """Fills a masked area with plausible new background. ~1-4 s per call on a CPU."""

    SIZE = 512
    _lock = threading.Lock()
    _session = None

    def __init__(self, path: str = MODEL_PATH):
        with LamaInpainter._lock:  # one shared session: loading takes a few seconds and ~1 GB RAM
            if LamaInpainter._session is None:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Don't let idle worker threads spin between calls: it starves the rest of the export.
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
                LamaInpainter._session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.sess = LamaInpainter._session
        self.in_img, self.in_mask = (i.name for i in self.sess.get_inputs())

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`img` BGR uint8, `mask` uint8 (255 = fill). Returns a copy with the masked pixels replaced.

        Works on a square of context around the mask (about 2.5x its size) so the model sees
        enough surroundings, and keeps the rest of the image untouched.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return img.copy()
        h, w = mask.shape
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 2.5) + 32
        side = max(96, min(side, max(h, w)))
        x0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
        y0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
        x1, y1 = min(w, x0 + side), min(h, y0 + side)

        crop, cmask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        ch, cw = cmask.shape
        # Give the model a slightly larger hole and pre-fill it, so no trace of the logo bleeds
        # into the context when the crop is resized (it would echo it back as a dark/bright shape).
        grown = cv2.dilate(cmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        crop = cv2.inpaint(crop, grown, 3, cv2.INPAINT_TELEA)
        rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (self.SIZE, self.SIZE), interpolation=cv2.INTER_CUBIC)
        m = (cv2.resize(grown, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR) > 0).astype(np.float32)
        with LamaInpainter._lock:
            out = self.sess.run(None, {
                self.in_img: (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
                self.in_mask: m[None, None],
            })[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.resize(out, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)

        result = img.copy()
        sel = cmask > 0
        result[y0:y1, x0:x1][sel] = out[sel]
        return result


class TeleaInpainter:
    """Non-AI fallback when the model isn't installed."""

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def best_inpainter():
    if model_available():
        try:
            return LamaInpainter()
        except Exception:
            pass
    return TeleaInpainter()
