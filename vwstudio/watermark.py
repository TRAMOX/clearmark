"""Render your own text and/or image watermark and blend it onto frames."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

POSITIONS = ["top-left", "top-center", "top-right",
             "center-left", "center", "center-right",
             "bottom-left", "bottom-center", "bottom-right"]


def font_dirs() -> list[str]:
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        local = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        return [os.path.join(windir, "Fonts"), os.path.join(local, "Microsoft", "Windows", "Fonts")]
    if sys.platform == "darwin":
        return ["/System/Library/Fonts", "/Library/Fonts", os.path.join(home, "Library", "Fonts")]
    return ["/usr/share/fonts", "/usr/local/share/fonts", os.path.join(home, ".fonts"),
            os.path.join(home, ".local", "share", "fonts")]


_font_cache: dict[str, str] | None = None


def list_fonts() -> dict[str, str]:
    """Map of font file name (e.g. 'arialbd.ttf') -> full path."""
    global _font_cache
    if _font_cache is None:
        found = {}
        for d in font_dirs():
            for root, _, files in os.walk(d):
                for name in files:
                    if name.lower().endswith((".ttf", ".otf", ".ttc")):
                        found.setdefault(name, os.path.join(root, name))
        _font_cache = dict(sorted(found.items(), key=lambda kv: kv[0].lower()))
    return _font_cache


def load_font(name_or_path: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(6, int(size))
    candidates = [name_or_path] if os.path.isfile(name_or_path or "") else []
    fonts = list_fonts()
    lower = {k.lower(): v for k, v in fonts.items()}
    for name in (name_or_path, "arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf",
                 "Arial Bold.ttf", "Helvetica.ttc"):
        if name and name.lower() in lower:
            candidates.append(lower[name.lower()])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def parse_color(value: str, fallback=(255, 255, 255)) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(value)[:3]
    except (ValueError, AttributeError):
        return fallback


def render_text(text: str, font: ImageFont.FreeTypeFont, color, outline_color=None,
                outline: int = 0, shadow: int = 0, align: str = "center") -> Image.Image:
    """Text (multi-line allowed) on a transparent RGBA canvas cropped to its ink."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=outline,
                                                        align=align, spacing=font.size // 5)
    pad = outline + shadow + 2
    img = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    origin = (pad - left, pad - top)
    if shadow:
        draw.multiline_text((origin[0] + shadow, origin[1] + shadow), text, font=font,
                            fill=(0, 0, 0, 150), stroke_width=outline, stroke_fill=(0, 0, 0, 150),
                            align=align, spacing=font.size // 5)
    draw.multiline_text(origin, text, font=font, fill=tuple(color) + (255,), stroke_width=outline,
                        stroke_fill=tuple(outline_color or (0, 0, 0)) + (255,), align=align,
                        spacing=font.size // 5)
    return img


def place(position: str, item_w: int, item_h: int, frame_w: int, frame_h: int, margin: int) -> tuple[int, int]:
    """Top-left corner for an item at one of POSITIONS ("<row>-<col>", or "center")."""
    parts = position.split("-") if position in POSITIONS else ["bottom", "right"]
    row, col = parts[0], parts[1] if len(parts) > 1 else "center"
    x = {"left": margin, "center": (frame_w - item_w) // 2, "right": frame_w - item_w - margin}[col]
    y = {"top": margin, "center": (frame_h - item_h) // 2, "bottom": frame_h - item_h - margin}[row]
    return x, y


@dataclass
class WatermarkSettings:
    text: str = ""
    font: str = "arialbd.ttf"
    text_size: float = 4.5          # % of the video's shorter side
    text_color: str = "#FFFFFF"
    text_opacity: float = 0.85
    text_outline: bool = True
    text_position: str = "bottom-left"
    logo_path: str = ""
    logo_size: float = 18.0         # logo width as % of the video's shorter side
    logo_opacity: float = 0.9
    logo_position: str = "top-right"
    margin: float = 3.0             # % of the video's shorter side

    @property
    def enabled(self) -> bool:
        return bool(self.text.strip()) or bool(self.logo_path and os.path.isfile(self.logo_path))


class WatermarkRenderer:
    """Pre-renders the watermark for one frame size, then blends it onto frames cheaply."""

    def __init__(self, settings: WatermarkSettings, width: int, height: int):
        self.layers: list[tuple[int, int, np.ndarray, np.ndarray]] = []  # x, y, bgr, alpha
        s = settings
        short = min(width, height)
        margin = int(short * s.margin / 100)
        items = []  # (position, PIL RGBA image)

        if s.logo_path and os.path.isfile(s.logo_path):
            logo = Image.open(s.logo_path).convert("RGBA")
            target_w = max(8, int(short * s.logo_size / 100))
            target_h = max(1, int(logo.height * target_w / logo.width))
            logo = logo.resize((target_w, target_h), Image.LANCZOS)
            items.append((s.logo_position, self._fade(logo, s.logo_opacity)))

        if s.text.strip():
            font = load_font(s.font, short * s.text_size / 100)
            outline = max(1, font.size // 14) if s.text_outline else 0
            img = render_text(s.text.strip(), font, parse_color(s.text_color), outline=outline,
                              shadow=0 if s.text_outline else max(1, font.size // 16))
            items.append((s.text_position, self._fade(img, s.text_opacity)))

        # Logo and text in the same corner: stack them instead of drawing one over the other.
        if len(items) == 2 and items[0][0] == items[1][0]:
            (pos, logo), (_, text) = items
            gap = max(4, short // 100)
            w = max(logo.width, text.width)
            combo = Image.new("RGBA", (w, logo.height + gap + text.height), (0, 0, 0, 0))
            align = pos.split("-")[-1]
            for img, y in ((logo, 0), (text, logo.height + gap)):
                x = {"left": 0, "right": w - img.width}.get(align, (w - img.width) // 2)
                combo.alpha_composite(img, (x, y))
            items = [(pos, combo)]

        for pos, img in items:
            x, y = place(pos, img.width, img.height, width, height, margin)
            self._add_layer(img, x, y, width, height)

    @staticmethod
    def _fade(img: Image.Image, opacity: float) -> Image.Image:
        if opacity >= 0.999:
            return img
        r, g, b, a = img.split()
        a = a.point(lambda v: int(v * max(0.0, min(1.0, opacity))))
        return Image.merge("RGBA", (r, g, b, a))

    def _add_layer(self, img: Image.Image, x: int, y: int, width: int, height: int):
        arr = np.asarray(img, dtype=np.float32)
        # Clip to the frame in case the item is bigger than the video.
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + arr.shape[1]), min(height, y + arr.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        arr = arr[y0 - y: y1 - y, x0 - x: x1 - x]
        bgr = arr[..., 2::-1].copy()
        alpha = arr[..., 3:4] / 255.0
        self.layers.append((x0, y0, bgr * alpha, 1.0 - alpha))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        for x, y, premult, inv_alpha in self.layers:
            h, w = inv_alpha.shape[:2]
            roi = frame[y:y + h, x:x + w]
            frame[y:y + h, x:x + w] = (roi * inv_alpha + premult).astype(np.uint8)
        return frame
