import io

import pytest
from PIL import Image

from fire_assessor.config import Limits
from fire_assessor.quality import quality_flags
from fire_assessor.validation import ValidationError, sniff, validate_image
from conftest import textured, to_bytes


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_formats_decode(fmt):
    v = validate_image(to_bytes(textured(), fmt))
    assert v.fmt == fmt and v.image.mode == "RGB"


@pytest.mark.parametrize("data,code", [
    (b"", "EMPTY_FILE"),
    (b"hello world, not an image", "UNSUPPORTED_TYPE"),
    (b"\xff\xd8\xff\xe0garbage", "DECODE_FAILED"),
])
def test_rejects_bad_input(data, code):
    with pytest.raises(ValidationError) as e:
        validate_image(data)
    assert e.value.code == code


def test_truncated_file():
    data = to_bytes(textured(size=(640, 480)), "PNG")
    with pytest.raises(ValidationError) as e:
        validate_image(data[: len(data) // 2])
    assert e.value.code == "DECODE_FAILED"


def test_size_and_pixel_limits():
    data = to_bytes(textured())
    with pytest.raises(ValidationError) as e:
        validate_image(data, Limits(max_bytes=100))
    assert e.value.code == "FILE_TOO_LARGE"
    with pytest.raises(ValidationError) as e:
        validate_image(data, Limits(max_pixels=1000))
    assert e.value.code == "TOO_MANY_PIXELS"


def test_polyglot_mismatch():
    # PNG magic bytes but body that decodes as something else: must not pass as valid
    jpg = to_bytes(textured())
    with pytest.raises(ValidationError):
        validate_image(b"\x89PNG\r\n\x1a\n" + jpg)


def test_sniff_ignores_name_and_detects_heic_brand():
    assert sniff(b"\x00\x00\x00\x18ftypheic" + b"\0" * 8) == "HEIC"
    assert sniff(b"MZ\x90\x00") is None


def test_quality_flags():
    assert quality_flags(textured(size=(320, 240))) == []
    assert "TOO_DARK" in quality_flags(Image.new("RGB", (300, 300), (5, 5, 5)))
    assert "NEAR_UNIFORM" in quality_flags(Image.new("RGB", (300, 300), (128, 128, 128)))
    assert "OVEREXPOSED" in quality_flags(Image.new("RGB", (300, 300), (250, 250, 250)))
    assert "TOO_SMALL" in quality_flags(textured(size=(64, 64)))
    import numpy as np
    blur = Image.fromarray(np.tile(np.linspace(0, 255, 300).astype('uint8'), (300, 1))).convert('RGB')  # smooth ramp: std ok, no detail
    assert "EXTREME_BLUR" in quality_flags(blur)
