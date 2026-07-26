"""pro.ledin.ocr — layered OCR workhorse.

Public library API (import as ``from pro.ledin import ocr``):

    pages = ocr.recognize("scan.pdf", ocr.RecognizeOptions(engine="tesseract"))
    markdown = ocr.to_markdown(pages, "scan.pdf")

Catch ``ocr.OcrError`` for recoverable failures (missing binaries/packages,
unsupported input, vision-api config). The library never calls ``sys.exit()``.
"""

from __future__ import annotations

from .core import (
    Cache,
    Caps,
    DEFAULT_MIN_CONF,
    DEFAULT_PSM,
    DEFAULT_VISION_PROMPT,
    EXIT_BAD_ARGS,
    EXIT_MISSING_BINARY,
    EXIT_OK,
    EXIT_UNSUPPORTED,
    OcrError,
    RecognizeOptions,
    __version__,
    probe_script_path,
    process_file,
    recognize,
    resolve_vision_prompt,
    to_json,
    to_markdown,
    to_text,
)

__all__ = [
    "Cache",
    "Caps",
    "DEFAULT_MIN_CONF",
    "DEFAULT_PSM",
    "DEFAULT_VISION_PROMPT",
    "EXIT_BAD_ARGS",
    "EXIT_MISSING_BINARY",
    "EXIT_OK",
    "EXIT_UNSUPPORTED",
    "OcrError",
    "RecognizeOptions",
    "__version__",
    "probe_script_path",
    "process_file",
    "recognize",
    "resolve_vision_prompt",
    "to_json",
    "to_markdown",
    "to_text",
]
