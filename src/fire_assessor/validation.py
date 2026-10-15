"""Upload validation: magic bytes, size/pixel limits, decode test, decompression-bomb guard."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from .config import Limits

ImageFile.LOAD_TRUNCATED_IMAGES = False


class ValidationError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass
class ValidatedImage:
    image: Image.Image  # RGB, EXIF-orientation applied, re-decoded (original bytes never forwarded)
    fmt: str
    width: int
    height: int
    raw: bytes


def sniff(data: bytes) -> Optional[str]:
    """Identify the format from magic bytes, ignoring the filename and client content type."""
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[4:8] == b"ftyp" and data[8:12] in (
        b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif",
    ):
        return "HEIC"
    return None


def _register_heif() -> bool:
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def validate_image(data: bytes, limits: Limits = Limits()) -> ValidatedImage:
    if not data:
        raise ValidationError("EMPTY_FILE")
    if len(data) > limits.max_bytes:
        raise ValidationError("FILE_TOO_LARGE")
    fmt = sniff(data)
    if fmt is None:
        raise ValidationError("UNSUPPORTED_TYPE")
    if fmt == "HEIC" and not _register_heif():
        raise ValidationError("HEIC_UNSUPPORTED", "install pillow-heif to decode HEIC")

    prev_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limits.max_pixels
    try:
        try:
            im = Image.open(io.BytesIO(data))
            w, h = im.size  # header only; checked before pixel data is decoded
            if w * h > limits.max_pixels:
                raise ValidationError("TOO_MANY_PIXELS")
            if min(w, h) < limits.min_side:
                raise ValidationError("IMAGE_TOO_SMALL")
            detected = (im.format or "").upper()
            expected = {"HEIC": {"HEIF", "HEIC", "AVIF"}}.get(fmt, {fmt})
            if detected not in expected:  # polyglot / mislabeled content
                raise ValidationError("FORMAT_MISMATCH")
            if getattr(im, "n_frames", 1) > 1 and fmt != "WEBP":
                im.seek(0)
            im.load()
        except ValidationError:
            raise
        except Image.DecompressionBombError as e:
            raise ValidationError("TOO_MANY_PIXELS", str(e)) from e
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as e:
            raise ValidationError("DECODE_FAILED", str(e)) from e
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max

    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    return ValidatedImage(image=im, fmt=fmt, width=im.width, height=im.height, raw=data)
