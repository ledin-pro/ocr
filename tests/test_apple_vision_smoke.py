"""Opt-in real macOS integration. No private assets or network OCR.

Run: OCR_APPLE_VISION_SMOKE=1 python -m pytest tests/test_apple_vision_smoke.py -v
The macOS CI job must opt in; missing dependencies then FAIL, never skip.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from pro.ledin.ocr import RecognizeOptions, recognize


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("OCR_APPLE_VISION_SMOKE") != "1",
    reason="real Apple Vision smoke requires macOS and explicit opt-in",
)
@pytest.mark.parametrize("lang", ["eng+rus", "auto"])
def test_real_accurate_recognition_of_synthetic_image(lang):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 500
    image = Image.new("RGB", (width, height), "white")
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
    draw = ImageDraw.Draw(image)
    draw.text((80, 70), "APPLE VISION LOCAL 12345", font=font, fill="black")
    draw.text((80, 220), "ПРИВЕТ МИР 67890", font=font, fill="black")
    with tempfile.TemporaryDirectory(prefix="ocr_apple_smoke_") as tmp:
        path = Path(tmp) / "synthetic.png"
        image.save(path)
        started = time.monotonic()
        page, = recognize(path, RecognizeOptions(
            engine="apple-vision", lang=lang, preprocess="none", no_cleanup=True
        ))
        print(f"Apple Vision synthetic OCR elapsed: {time.monotonic() - started:.3f}s")
    assert page["source"] == "apple-vision"
    for expected in ("APPLE VISION LOCAL", "12345", "ПРИВЕТ МИР", "67890"):
        assert expected in page["text"]
    assert 0 < page["mean_conf"] <= 100
    assert page["words"]
    for line in page["words"]:
        assert line["granularity"] == "line"
        assert 0 < line["conf"] <= 100
        x, y, w, h = line["bbox"]
        assert 0 <= x < x + w <= width
        assert 0 <= y < y + h <= height
    # Also check location, not just bounds: catches a bottom-left/top-left mixup.
    english = next(line for line in page["words"] if "APPLE" in line["text"])
    russian = next(line for line in page["words"] if "ПРИВЕТ" in line["text"])
    assert english["bbox"][1] < russian["bbox"][1]


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("OCR_APPLE_VISION_SMOKE") != "1",
    reason="real Apple Vision smoke requires macOS and explicit opt-in",
)
def test_real_multipage_raster_pdf_preserves_pages_and_geometry():
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 500
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
    images = []
    expected = ["FIRST PAGE LOCAL 12345", "SECOND PAGE LOCAL 67890"]
    for text in expected:
        image = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(image).text((80, 100), text, font=font, fill="black")
        images.append(image)
    with tempfile.TemporaryDirectory(prefix="ocr_apple_pdf_smoke_") as tmp:
        path = Path(tmp) / "synthetic-multipage.pdf"
        images[0].save(path, save_all=True, append_images=images[1:], resolution=72)
        started = time.monotonic()
        pages = recognize(path, RecognizeOptions(
            engine="apple-vision", lang="eng", dpi=72, preprocess="none", no_cleanup=True
        ))
        print(f"Apple Vision multipage PDF elapsed: {time.monotonic() - started:.3f}s")
    assert [page["n"] for page in pages] == [1, 2]
    for page, text in zip(pages, expected):
        assert page["source"] == "apple-vision"
        assert text in page["text"]
        assert 0 < page["mean_conf"] <= 100
        assert page["words"]
        for line in page["words"]:
            assert line["granularity"] == "line"
            assert 0 < line["conf"] <= 100
            x, y, w, h = line["bbox"]
            assert 0 <= x < x + w <= width
            assert 0 <= y < y + h <= height
