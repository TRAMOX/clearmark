"""Pick a good frame and turn it into a thumbnail with a title."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .media import FrameReader, VideoInfo
from .watermark import load_font, parse_color, render_text

SIZES = {
    "YouTube 1280x720": (1280, 720),
    "Shorts / Reels 1080x1920": (1080, 1920),
    "Square 1080x1080": (1080, 1080),
    "Same as video": None,
}


@dataclass
class ThumbnailSettings:
    title: str = ""
    font: str = "impact.ttf"
    title_size: float = 13.0         # % of the thumbnail's shorter side
    title_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    title_position: str = "bottom"   # top / center / bottom
    title_band: bool = False         # dark band behind the title
    size: str = "YouTube 1280x720"
    fit: str = "blur"                # blur = whole frame on blurred backdrop, crop = fill and crop
    enhance: bool = True
    include_watermark: bool = True


def best_frame_time(info: VideoInfo, samples: int = 40) -> float:
    """Sharp, well-exposed, colourful frame, skipping the first/last 5% (fades, black frames)."""
    reader = FrameReader(info)
    times = np.linspace(info.duration * 0.05, info.duration * 0.95, samples)
    scores = []
    for t in times:
        f = reader.frame_at(float(t))
        if f is None:
            scores.append((0, 0, 0))
            continue
        small = cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        exposure = max(0.0, 1.0 - abs(gray.mean() / 255.0 - 0.5) * 1.6)
        b, g, r = [c.astype(np.float32) for c in cv2.split(small)]
        rg, yb = r - g, 0.5 * (r + g) - b
        colour = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
        scores.append((sharp, exposure, colour))
    reader.close()
    s = np.array(scores, dtype=np.float64)
    s[:, 0] /= max(s[:, 0].max(), 1e-6)
    s[:, 2] /= max(s[:, 2].max(), 1e-6)
    total = 0.5 * s[:, 0] + 0.25 * s[:, 1] + 0.25 * s[:, 2]
    return float(times[int(np.argmax(total))])


def _fit(img: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    tw, th = size
    cover = max(tw / img.width, th / img.height)
    big = img.resize((max(tw, round(img.width * cover)), max(th, round(img.height * cover))), Image.LANCZOS)
    left, top = (big.width - tw) // 2, (big.height - th) // 2
    cropped = big.crop((left, top, left + tw, top + th))
    if mode == "crop" or abs(img.width / img.height - tw / th) < 0.02:
        return cropped
    backdrop = ImageEnhance.Brightness(cropped.filter(ImageFilter.GaussianBlur(max(tw, th) / 40))).enhance(0.6)
    contain = min(tw / img.width, th / img.height)
    fg = img.resize((round(img.width * contain), round(img.height * contain)), Image.LANCZOS)
    backdrop.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
    return backdrop


def _wrap(text: str, font, max_width: int) -> str:
    lines = []
    for para in text.splitlines() or [""]:
        words = para.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            if font.getlength(f"{line} {word}") <= max_width:
                line = f"{line} {word}"
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return "\n".join(lines)


def make_thumbnail(frame_bgr: np.ndarray, s: ThumbnailSettings) -> Image.Image:
    """`frame_bgr` should already have logos removed / your watermark applied as wanted."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    size = SIZES.get(s.size) or (img.width, img.height)
    img = _fit(img, size, s.fit)
    if s.enhance:
        img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.18)).enhance(1.08)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))

    title = s.title.strip()
    if title:
        tw, th = img.size
        short = min(tw, th)
        font = load_font(s.font, short * s.title_size / 100)
        title = _wrap(title.upper() if s.font.lower().startswith("impact") else title, font, int(tw * 0.9))
        outline = max(2, font.size // 12)
        text = render_text(title, font, parse_color(s.title_color), parse_color(s.outline_color, (0, 0, 0)),
                           outline=outline, shadow=max(2, font.size // 18))
        # Shrink if a long title would overflow the frame.
        limit_h = int(th * 0.45)
        if text.width > tw * 0.94 or text.height > limit_h:
            k = min(tw * 0.94 / text.width, limit_h / text.height)
            text = text.resize((max(1, int(text.width * k)), max(1, int(text.height * k))), Image.LANCZOS)
        margin = int(short * 0.05)
        x = (tw - text.width) // 2
        y = {"top": margin, "center": (th - text.height) // 2}.get(s.title_position, th - text.height - margin)

        canvas = img.convert("RGBA")
        if s.title_band:
            band = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            pad = int(short * 0.025)
            ImageDraw.Draw(band).rectangle((0, y - pad, tw, y + text.height + pad), fill=(0, 0, 0, 150))
            canvas = Image.alpha_composite(canvas, band)
        canvas.alpha_composite(text, (x, max(0, y)))
        img = canvas.convert("RGB")
    return img
