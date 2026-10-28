"""Remembered settings (your watermark, thumbnail style, export options, logo regions)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

from .thumbnail import ThumbnailSettings
from .watermark import WatermarkSettings

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")


@dataclass
class ExportSettings:
    quality: str = "High"           # key of QUALITY
    preset: str = "medium"
    keep_audio: bool = True
    embed_cover: bool = True
    save_thumbnail: bool = True


QUALITY = {"Best (bigger file)": 16, "High": 19, "Balanced": 22, "Small file": 26}


@dataclass
class AppSettings:
    watermark: WatermarkSettings = field(default_factory=WatermarkSettings)
    thumbnail: ThumbnailSettings = field(default_factory=ThumbnailSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    removal_method: str = "smart"
    removal_expand: int = 3
    auto_shape: bool = True
    saved_regions: dict = field(default_factory=dict)   # "WxH" -> [LogoRegion.to_dict()]
    version: int = 2


def _merge(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_settings(path: str = SETTINGS_PATH) -> AppSettings:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return AppSettings()
    method = data.get("removal_method", "smart")
    if data.get("version", 1) < 2 and method == "inpaint":
        method = "smart"   # "inpaint" was only the old default; move everyone to the better method once
    try:
        return AppSettings(
            watermark=_merge(WatermarkSettings, data.get("watermark")),
            thumbnail=_merge(ThumbnailSettings, data.get("thumbnail")),
            export=_merge(ExportSettings, data.get("export")),
            removal_method=method,
            removal_expand=int(data.get("removal_expand", 3)),
            auto_shape=bool(data.get("auto_shape", True)),
            saved_regions=data.get("saved_regions") or {},
        )
    except (TypeError, ValueError):
        return AppSettings()


def save_settings(settings: AppSettings, path: str = SETTINGS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(asdict(settings), fh, indent=2)
    os.replace(tmp, path)
