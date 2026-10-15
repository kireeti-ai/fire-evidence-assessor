"""Minimal EXIF extraction. EXIF is frequently missing or forgeable: treat as weak evidence."""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from PIL import Image


@dataclass
class ExifInfo:
    captured_at: Optional[datetime] = None
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    gps: Optional[tuple] = None  # (lat, lon)


def _deg(v, ref) -> Optional[float]:
    try:
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    except Exception:
        return None
    return -d if ref in ("S", "W") else d


def read_exif(data: bytes) -> ExifInfo:
    info = ExifInfo()
    try:
        ex = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return info
    if not ex:
        return info
    info.make, info.model, info.software = ex.get(0x010F), ex.get(0x0110), ex.get(0x0131)
    try:
        sub = ex.get_ifd(0x8769)
        raw = sub.get(0x9003) or ex.get(0x0132)
        if raw:
            info.captured_at = datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S").replace(
                tzinfo=timezone.utc)  # offset unknown: treated as UTC (approximation)
    except Exception:
        pass
    try:
        g = ex.get_ifd(0x8825)
        if g.get(2) and g.get(4):
            lat, lon = _deg(g[2], g.get(1)), _deg(g[4], g.get(3))
            if lat is not None and lon is not None:
                info.gps = (lat, lon)
    except Exception:
        pass
    return info
