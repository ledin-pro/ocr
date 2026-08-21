#!/usr/bin/env python3
"""
core — layered OCR workhorse for the `pro.ledin.ocr` package.

Baseline: Python stdlib + Camelot + pdftoppm + pdftotext + tesseract.
Optional tiers: pytesseract, PyMuPDF, opencv-python, numpy, easyocr, openai.
Install optional tiers on-demand via extras: pip install "pro-ledin-ocr[all]".

Library usage:  from pro.ledin import ocr
                pages = ocr.recognize("scan.pdf", ocr.RecognizeOptions(engine="tesseract"))
                markdown = ocr.to_markdown(pages, "scan.pdf")
                Catch `ocr.OcrError` for recoverable failures (missing
                binaries/packages, unsupported input, vision config).

CLI usage:      ocr INPUT [INPUT ...] [options]   (see `ocr --help`)
"""

from __future__ import annotations

import base64
from collections import Counter
import html as html_lib
import hashlib
import importlib.util
import ipaddress
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn
from urllib.parse import urlparse

# ── exit codes ────────────────────────────────────────────────────────────────
EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_UNSUPPORTED = 3
EXIT_MISSING_BINARY = 4


class OcrError(Exception):
    """Recoverable OCR failure: bad input, missing binaries/packages, or a
    vision configuration/request error.

    CLI: `main()` catches this at the top level, prints `[ocr] ERROR: ...` to
    stderr, and exits with `.code`.
    Library: catch `OcrError` directly — it never calls `sys.exit()`.
    """

    def __init__(self, message: str, code: int = EXIT_BAD_ARGS) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RequirementResult:
    """Side-effect-free selected-engine requirement probe result."""

    engine: str
    available: bool
    code: str
    missing_component: str | None = None
    ocr_extra: str | None = None
    component_type: str | None = None
    first_run_note: str | None = None
    missing_components: tuple[str, ...] = ()
    # "all": every listed component is required.
    # "any": installing any single listed component satisfies the requirement.
    components_relation: str = "all"

    def __post_init__(self) -> None:
        if self.missing_components:
            object.__setattr__(self, "missing_component", self.missing_components[0])
        elif self.missing_component is not None:
            object.__setattr__(self, "missing_components", (self.missing_component,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "available": self.available,
            "code": self.code,
            "missing_component": self.missing_component,
            "missing_components": self.missing_components,
            "components_relation": self.components_relation,
            "ocr_extra": self.ocr_extra,
            "component_type": self.component_type,
            "first_run_note": self.first_run_note,
        }


class OcrRequirementError(OcrError):
    """Missing selected-engine dependency or configuration."""

    def __init__(self, result: RequirementResult) -> None:
        super().__init__(_requirement_message(result), _requirement_exit_code(result))
        self.result = result
        self.engine = result.engine
        self.requirement_code = result.code
        self.stable_code = result.code
        self.missing_component = result.missing_component
        self.missing_components = result.missing_components
        self.components_relation = result.components_relation
        self.ocr_extra = result.ocr_extra
        self.component_type = result.component_type
        self.first_run_note = result.first_run_note

# ── version ───────────────────────────────────────────────────────────────────
__version__ = "0.6.0"
OCR_OUTPUT_SCHEMA_VERSION = 4
TABLE_OUTPUT_VERSION = 1


# ── constants ─────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".heic", ".webp", ".bmp", ".gif"}
PDF_EXTENSION = ".pdf"
DEFAULT_VISION_PROMPT = (
    "Read this page image faithfully. Reproduce all visible text in reading order. "
    "For tables use Markdown table syntax. For charts describe axis labels and key "
    "data values. No commentary."
)

# OSD script → tesseract language code
SCRIPT_TO_LANG: dict[str, str] = {
    "cyrillic":    "rus",
    "latin":       "eng",
    "han":         "chi_sim",
    "arabic":      "ara",
    "devanagari":  "hin",
    "bengali":     "ben",
    "korean":      "kor",
    "japanese":    "jpn",
    "greek":       "ell",
    "hebrew":      "heb",
    "thai":        "tha",
    "georgian":    "kat",
    "armenian":    "hye",
}

# tesseract language code → PaddleOCR language code (primary code only)
TESS_TO_PADDLE_LANG: dict[str, str] = {
    "eng":     "en",
    "rus":     "ru",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn":     "japan",
    "kor":     "korean",
    "ara":     "arabic",
    "hin":     "hi",
    "ben":     "bn",
    "ell":     "el",
    "heb":     "he",
    "tha":     "th",
    "fra":     "fr",
    "deu":     "german",
    "spa":     "es",
    "ita":     "it",
    "por":     "pt",
    "vie":     "vi",
}

DEFAULT_MIN_CONF = 60.0
DEFAULT_PSM = 3
TABLE_FLAVORS = ("auto", "lattice", "stream", "network", "hybrid")
SMALL_WIDTH_THRESHOLD = 1400  # px — upscale if narrower
PADDLEX_OCR_MODULES = (
    "bs4", "einops", "ftfy", "imagesize", "jinja2", "latex2mathml", "lxml",
    "cv2", "openpyxl", "premailer", "pyclipper", "pypdfium2", "bidi", "regex",
    "safetensors", "sklearn", "scipy", "sentencepiece", "shapely", "tiktoken",
    "tokenizers",
)

ENGINE_SETUP_GUIDANCE = (
    "See the pro-ledin-ocr engine setup guide and review platform-specific "
    "requirements before installing dependencies."
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _capability(
    caps: Any | Mapping[str, Any] | None,
    name: str,
    detect: Callable[[], bool],
) -> bool:
    if isinstance(caps, Mapping) and name in caps:
        return bool(caps[name])
    if caps is not None and hasattr(caps, name):
        return bool(getattr(caps, name))
    return detect()


def _unavailable_requirement(
    engine: str,
    *,
    vision_api_key: str = "",
    vision_model: str = "",
    paddle_vl_server_url: str = "",
    paddle_vl_model: str = "",
    **caps_overrides: bool,
) -> RequirementResult:
    """Build the probe result for a component known to be unusable."""
    return probe_engine_requirements(
        engine,
        vision_api_key=vision_api_key,
        vision_model=vision_model,
        paddle_vl_server_url=paddle_vl_server_url,
        paddle_vl_model=paddle_vl_model,
        caps=caps_overrides,
    )


@contextmanager
def _engine_import(
    result: RequirementResult | Callable[[BaseException], RequirementResult],
):
    """Convert a failing engine import into the documented structured error.

    ``find_spec`` only proves that a module *spec* exists. Broken builds,
    namespace-package shells, and failed native extension loads still raise at
    import time, so every engine import site funnels through here to keep the
    ``OcrError``/``OcrRequirementError`` contract intact.
    """
    try:
        yield
    except (ImportError, OSError) as exc:
        # OSError covers native/shared-library load failures that some engine
        # builds raise instead of ImportError.
        resolved = result(exc) if callable(result) else result
        raise OcrRequirementError(resolved) from exc


def _fitz_or_fallback(caps: Any, need: str) -> Any | None:
    """Import PyMuPDF for a PDF need.

    Returns ``None`` when PyMuPDF is unusable but Poppler can still serve the
    need, so callers take their existing fallback path. Raises the structured
    requirement error when no backend remains.
    """
    try:
        import fitz

        return fitz
    except (ImportError, OSError) as exc:
        without_fitz = {
            "has_fitz": False,
            "bin_pdftoppm": _capability(
                caps, "bin_pdftoppm", lambda: shutil.which("pdftoppm") is not None
            ),
            "bin_pdftotext": _capability(
                caps, "bin_pdftotext", lambda: shutil.which("pdftotext") is not None
            ),
        }
        fallback = _probe_pdf_need(without_fitz, need)
        if fallback.available:
            return None
        raise OcrRequirementError(fallback) from exc


def _paddle_import_requirement(exc: BaseException) -> RequirementResult:
    """Attribute a failing PaddleOCR import to the package or the runtime."""
    failed_module = getattr(exc, "name", "") or ""
    root = failed_module.split(".", 1)[0]
    if root == "paddle":
        return _unavailable_requirement(
            "paddleocr", has_paddleocr=True, has_paddle=False
        )
    return _unavailable_requirement(
        "paddleocr", has_paddleocr=False, has_paddle=True
    )


def _paddle_vl_import_requirement(exc: BaseException) -> RequirementResult:
    """Attribute a failing PaddleOCR-VL import to package or runtime."""
    failed_module = getattr(exc, "name", "") or ""
    root = failed_module.split(".", 1)[0]
    if root == "paddle":
        return _unavailable_requirement(
            "paddleocr-vl-mlx", has_paddleocr=True, has_paddle=False,
            paddle_vl_server_url="http://127.0.0.1:8111/",
            paddle_vl_model="configured-model",
        )
    return _unavailable_requirement(
        "paddleocr-vl-mlx", has_paddleocr=False, has_paddle=True,
        paddle_vl_server_url="http://127.0.0.1:8111/",
        paddle_vl_model="configured-model",
    )


def _is_loopback_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return False
        if hostname.casefold() == "localhost":
            return True
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _easyocr_dependency_requirement(component: str) -> RequirementResult:
    return RequirementResult(
        engine="easyocr",
        available=False,
        code="missing_easyocr_dependency",
        missing_component=component,
        ocr_extra="easyocr",
        component_type="python-package",
        first_run_note="First OCR run may download EasyOCR model weights.",
    )


def _easyocr_import_requirement(exc: BaseException) -> RequirementResult:
    """Attribute EasyOCR import failures to a transitive module when known."""
    failed_module = getattr(exc, "name", "") or ""
    root = failed_module.split(".", 1)[0]
    if root and root != "easyocr":
        component = "pillow" if root == "PIL" else root
        return _easyocr_dependency_requirement(component)
    return _unavailable_requirement("easyocr", has_easyocr=False)


def probe_engine_requirements(
    engine: str,
    *,
    vision_api_key: str = "",
    vision_model: str = "",
    paddle_vl_server_url: str = "",
    paddle_vl_model: str = "",
    caps: Any | None = None,
) -> RequirementResult:
    """Inspect selected-engine requirements without imports or mutation.

    ``caps`` may supply already-detected capability attributes for callers that
    already created ``Caps``. Without it, executable and module-spec discovery
    is used; no engine package is imported and no model or network work occurs.
    """
    if engine == "tesseract":
        if not _capability(
            caps, "bin_tesseract", lambda: shutil.which("tesseract") is not None
        ):
            return RequirementResult(
                engine, False, "missing_tesseract_binary", "tesseract", None, "binary"
            )
        return RequirementResult(engine, True, "ok")

    if engine == "easyocr":
        note = "First OCR run may download EasyOCR model weights."
        if not _capability(caps, "has_easyocr", lambda: _module_available("easyocr")):
            return RequirementResult(
                engine, False, "missing_easyocr_package", "easyocr", "easyocr",
                "python-package", note,
            )
        return RequirementResult(engine, True, "ok", first_run_note=note)

    if engine == "paddleocr":
        note = "First OCR run may download PaddleOCR/PaddleX model weights."
        has_paddleocr = _capability(
            caps, "has_paddleocr", lambda: _module_available("paddleocr")
        )
        has_paddle = _capability(
            caps, "has_paddle", lambda: _module_available("paddle")
        )
        if not has_paddleocr and not has_paddle:
            missing_components = ("paddleocr", "paddle")
        elif not has_paddleocr:
            missing_components = ("paddleocr",)
        elif not has_paddle:
            missing_components = ("paddle",)
        else:
            missing_components = ()
        if missing_components:
            missing_component = missing_components[0]
            missing_runtime_only = missing_components == ("paddle",)
            return RequirementResult(
                engine=engine,
                available=False,
                code=(
                    "missing_paddle_runtime"
                    if missing_runtime_only
                    else "missing_paddleocr_package"
                ),
                missing_component=missing_component,
                ocr_extra="paddle",
                component_type=(
                    "python-runtime" if missing_runtime_only else "python-package"
                ),
                first_run_note=note,
                missing_components=missing_components,
            )
        return RequirementResult(engine, True, "ok", first_run_note=note)

    if engine == "paddleocr-vl-mlx":
        note = (
            "PaddleOCR-VL layout models may download on first use; start the "
            "VLM-only backend with 'mlx_vlm.server --host 127.0.0.1 --port 8111'."
        )
        has_paddleocr = _capability(
            caps, "has_paddleocr", lambda: _module_available("paddleocr")
        )
        has_paddle = _capability(
            caps, "has_paddle", lambda: _module_available("paddle")
        )
        has_paddlex_ocr = _capability(
            caps,
            "has_paddlex_ocr",
            lambda: all(_module_available(module) for module in PADDLEX_OCR_MODULES),
        )
        if not has_paddleocr or not has_paddle:
            missing_components = tuple(
                component
                for component, available in (
                    ("paddleocr", has_paddleocr),
                    ("paddle", has_paddle),
                )
                if not available
            )
            runtime_only = missing_components == ("paddle",)
            return RequirementResult(
                engine=engine,
                available=False,
                code=(
                    "missing_paddle_runtime"
                    if runtime_only
                    else "missing_paddleocr_doc_parser"
                ),
                ocr_extra="paddle-vl",
                component_type=(
                    "python-runtime" if runtime_only else "python-package"
                ),
                first_run_note=note,
                missing_components=missing_components,
            )
        if not has_paddlex_ocr:
            return RequirementResult(
                engine=engine,
                available=False,
                code="missing_paddleocr_doc_parser",
                missing_component="paddleocr[doc-parser]",
                ocr_extra="paddle-vl",
                component_type="python-package",
                first_run_note=note,
            )
        if not paddle_vl_server_url.strip():
            return RequirementResult(
                engine, False, "missing_paddle_vl_server_url",
                "paddle_vl_server_url", "paddle-vl", "configuration", note,
            )
        if not _is_loopback_url(paddle_vl_server_url.strip()):
            return RequirementResult(
                engine, False, "unsafe_paddle_vl_server_url",
                "paddle_vl_server_url", "paddle-vl", "configuration", note,
            )
        if not paddle_vl_model.strip():
            return RequirementResult(
                engine, False, "missing_paddle_vl_model",
                "paddle_vl_model", "paddle-vl", "configuration", note,
            )
        return RequirementResult(engine, True, "ok", first_run_note=note)

    if engine == "vision":
        if not _capability(caps, "has_openai", lambda: _module_available("openai")):
            return RequirementResult(
                engine, False, "missing_openai_package", "openai", "vision",
                "python-package",
            )
        if not (vision_api_key or "").strip():
            return RequirementResult(
                engine, False, "missing_vision_api_key", "vision_api_key", "vision",
                "configuration",
            )
        if not (vision_model or "").strip():
            return RequirementResult(
                engine, False, "missing_vision_model", "vision_model", "vision",
                "configuration",
            )
        return RequirementResult(engine, True, "ok")

    return RequirementResult(
        engine, False, "unsupported_engine", engine, None, "engine"
    )


PDF_RENDER_COMPONENTS = ("pdftoppm", "pymupdf")
PDF_TEXT_COMPONENTS = ("pdftotext", "pymupdf")


def _probe_pdf_need(caps: Any | None, need: str) -> RequirementResult:
    """Evaluate one PDF need ('render' or 'text') independently."""
    has_fitz = _capability(caps, "has_fitz", lambda: _module_available("fitz"))
    if need == "render":
        binary, components, code = (
            "pdftoppm", PDF_RENDER_COMPONENTS, "missing_pdf_render_backend",
        )
    else:
        binary, components, code = (
            "pdftotext", PDF_TEXT_COMPONENTS, "missing_pdf_text_backend",
        )
    has_binary = _capability(
        caps, f"bin_{binary}", lambda: shutil.which(binary) is not None
    )
    if has_fitz or has_binary:
        return RequirementResult("pdf", True, "ok")
    return RequirementResult(
        engine="pdf",
        available=False,
        code=code,
        ocr_extra="pdf",
        component_type="binary-or-python-package",
        missing_components=components,
        components_relation="any",
    )


def probe_pdf_requirements(caps: Any | None = None) -> RequirementResult:
    """Inspect PDF render and text-layer requirements without imports.

    Rendering needs Poppler's ``pdftoppm`` or PyMuPDF; text-layer extraction
    needs Poppler's ``pdftotext`` or PyMuPDF. Either alternative satisfies each
    need, so results use ``components_relation="any"``. When both needs are
    unmet, the render result is returned first. Like
    ``probe_engine_requirements()``, this performs discovery only: no imports,
    no subprocess execution, and no mutation.
    """
    render = _probe_pdf_need(caps, "render")
    if not render.available:
        return render
    return _probe_pdf_need(caps, "text")


def _requirement_exit_code(result: RequirementResult) -> int:
    if result.component_type in {"configuration", "engine"}:
        return EXIT_BAD_ARGS
    return EXIT_MISSING_BINARY


def _requirement_message(result: RequirementResult) -> str:
    component = result.missing_component or "unknown"
    components = result.missing_components or (component,)
    if result.code == "unsupported_engine":
        return f"Unsupported OCR engine: {component}"
    if result.component_type == "configuration":
        cli_flag = component.replace("_", "-")
        return f"{component} is required for engine={result.engine} (CLI: --{cli_flag})."
    if result.component_type == "binary":
        return f"Required binary '{component}' was not found. {ENGINE_SETUP_GUIDANCE}"
    extra = (
        f" Install package extra 'pro-ledin-ocr[{result.ocr_extra}]'."
        if result.ocr_extra else ""
    )
    note = f" {result.first_run_note}" if result.first_run_note else ""
    quoted = ", ".join(f"'{item}'" for item in components)
    if result.components_relation == "any":
        return (
            f"Requires any one of {quoted}; none was found.{extra} "
            f"{ENGINE_SETUP_GUIDANCE}{note}"
        )
    if len(components) > 1:
        return (
            f"Required Python components {quoted} were not found.{extra} "
            f"{ENGINE_SETUP_GUIDANCE}{note}"
        )
    runtime = " runtime" if result.component_type == "python-runtime" else " package"
    return (
        f"Required Python{runtime} '{component}' was not found.{extra} "
        f"{ENGINE_SETUP_GUIDANCE}{note}"
    )


def _raise_requirement(result: RequirementResult) -> None:
    if not result.available:
        raise OcrRequirementError(result)


# ── capability detection ──────────────────────────────────────────────────────

class Caps:
    """Detect available binaries and Python libraries once at startup."""

    def __init__(self, verbose: bool = False, *, report: bool | None = None):
        self.verbose = verbose
        # Detection is spec-based for every optional dependency so `Caps` and
        # `probe_engine_requirements()` agree; import failures are converted to
        # structured errors at the actual import sites.
        self.has_pytesseract = _module_available("pytesseract")
        self.has_fitz = _module_available("fitz")          # PyMuPDF
        self.has_cv2 = _module_available("cv2")            # opencv-python
        self.has_numpy = _module_available("numpy")
        self.has_easyocr = _module_available("easyocr")
        self.has_openai = _module_available("openai")
        self.has_pil = _module_available("PIL")            # Pillow
        self.has_paddleocr = _module_available("paddleocr")
        self.has_paddle = _module_available("paddle")
        self.has_paddlex_ocr = all(
            _module_available(module) for module in PADDLEX_OCR_MODULES
        )

        self.bin_pdftoppm = shutil.which("pdftoppm")
        self.bin_pdftotext = shutil.which("pdftotext")
        self.bin_pdfinfo = shutil.which("pdfinfo")
        self.bin_pdffonts = shutil.which("pdffonts")
        self.bin_pdfimages = shutil.which("pdfimages")
        self.bin_tesseract = shutil.which("tesseract")
        self.bin_ocrmypdf = shutil.which("ocrmypdf")

        # The capability dump is a CLI diagnostic; library callers opt in
        # explicitly so `verbose` progress logging alone stays quiet.
        if report:
            self._report()

    def _report(self):
        lines = ["[caps] Available capabilities:"]
        for attr, val in sorted(self.__dict__.items()):
            if attr.startswith("has_") or attr.startswith("bin_"):
                status = "OK" if val else "MISSING"
                lines.append(f"  {attr:<20} {status}  ({val if val and attr.startswith('bin_') else ''})")
        print("\n".join(lines), file=sys.stderr, flush=True)

    def require_render(self):
        _raise_requirement(_probe_pdf_need(self, "render"))

    def require_ocr(self):
        _raise_requirement(probe_engine_requirements("tesseract", caps=self))

    def require_paddleocr(self):
        _raise_requirement(probe_engine_requirements("paddleocr", caps=self))

    def require_paddleocr_vl(
        self, server_url: str, model: str
    ) -> None:
        _raise_requirement(probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url=server_url,
            paddle_vl_model=model,
            caps=self,
        ))

    def require_pdftotext(self):
        _raise_requirement(_probe_pdf_need(self, "text"))


# ── utilities ─────────────────────────────────────────────────────────────────

def _fatal(msg: str, code: int = EXIT_BAD_ARGS) -> NoReturn:
    """Raise OcrError(msg, code).

    This used to call sys.exit() directly, which made ocr.py unsafe to import
    as a library (any failure anywhere would kill the whole host process).
    The CLI entry point now converts OcrError to the equivalent stderr
    message + exit code at the top level; library callers catch it normally.
    """
    raise OcrError(msg, code)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[ocr] {msg}", file=sys.stderr, flush=True)


def _nonspace_byte_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text).encode("utf-8"))


_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)
_LETTER_RUN_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_NUMERIC_TOKEN_RE = re.compile(r"^[-+]?\d[\d.,:/%\u2013\u2014-]*$", re.UNICODE)


def text_readability(text: str) -> dict[str, float]:
    """Score how much extracted text looks like real language, not glyph soup.

    Returns letter_digit_ratio, word_score, and replacement_ratio computed over
    the decoded characters. A broken CMap (embedded text with no ToUnicode)
    yields symbol/punctuation runs that score low even when the raw character
    count is high.
    """
    nonspace = re.sub(r"\s+", "", text)
    if not nonspace:
        return {
            "letter_digit_ratio": 0.0,
            "word_score": 0.0,
            "replacement_ratio": 0.0,
        }

    letters_digits = sum(
        1 for ch in nonspace if _LETTER_RE.match(ch) or ch.isdigit()
    )
    replacements = sum(
        1 for ch in nonspace if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\t")
    )

    # A token counts as content when it has a run of >=3 letters (a real word)
    # or is a bare numeric/measurement value (e.g. "4.53", "3.92-5.08", "62").
    # This keeps dense lab tables readable while rejecting punctuation soup.
    tokens = _WORD_TOKEN_RE.findall(text)
    word_like = sum(
        1
        for token in tokens
        if _LETTER_RUN_RE.search(token) or _NUMERIC_TOKEN_RE.match(token)
    )

    return {
        "letter_digit_ratio": round(letters_digits / len(nonspace), 4),
        "word_score": round(word_like / len(tokens), 4) if tokens else 0.0,
        "replacement_ratio": round(replacements / len(nonspace), 4),
    }


def _run(cmd: list[str], capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        check=check,
        text=True,
    )


def _parse_page_range(spec: str, total: int) -> list[int]:
    """Parse '1-3,5,7' into [1, 2, 3, 5, 7] (1-indexed, clamped to total)."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= total]


def _resolve_page_range(spec: str, total: int, max_pages: int) -> list[int] | None:
    if max_pages < 0:
        _fatal("max_pages must be zero or greater", EXIT_BAD_ARGS)
    pages = _parse_page_range(spec, total) if spec else None
    if max_pages:
        if pages is not None:
            pages = pages[:max_pages]
        else:
            pages = list(range(1, min(total, max_pages) + 1))
    if pages == []:
        _fatal("page selection matched no pages", EXIT_BAD_ARGS)
    return pages


def _sha1_key(
    path: str,
    engine: str,
    dpi: str,
    preprocess: str,
    lang: str,
    page_selection: str,
    engine_context: str = "",
) -> str:
    stat = os.stat(path)
    raw = (
        f"{os.path.abspath(path)}|{stat.st_mtime}|{stat.st_size}|{engine}|{dpi}|"
        f"{preprocess}|{lang}|pages={page_selection}|schema={OCR_OUTPUT_SCHEMA_VERSION}"
    )
    if engine_context:
        raw += f"|engine_context={engine_context}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ── input classification ──────────────────────────────────────────────────────

def classify_input(path: str) -> str:
    """Return 'pdf', 'image', or 'unsupported'."""
    ext = Path(path).suffix.lower()
    if ext == PDF_EXTENSION:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unsupported"


# ── PDF probing ───────────────────────────────────────────────────────────────

MIN_TEXT_CHARS = 30
HIGH_SMASK_COUNT = 3
MIN_LETTER_DIGIT_RATIO = 0.55
MIN_WORD_SCORE = 0.5
MAX_REPLACEMENT_RATIO = 0.05


def decide_probe(signals: dict[str, Any]) -> dict[str, Any]:
    """Apply the canonical text-layer decision to collected probe signals."""
    encrypted = bool(signals.get("encrypted"))
    median = int(signals.get("median_chars") or 0)
    total_fonts = signals.get("total_fonts")
    non_unicode_fonts = signals.get("non_unicode_fonts")
    smask_count = signals.get("smask_count")
    readability = signals.get("readability")

    suspected_rasterized = bool(
        total_fonts is not None
        and (total_fonts == 0 or (non_unicode_fonts or 0) > 0)
    )
    all_fonts_non_unicode = bool(
        total_fonts and (non_unicode_fonts or 0) == total_fonts
    )
    high_image_coverage = bool(
        smask_count is not None and smask_count >= HIGH_SMASK_COUNT
    )

    # A text layer is trustworthy only if it is both dense enough and readable.
    # Readability is scored when sample text is available; if scoring could not
    # run (no text extracted), fall back to the density-only decision so we do
    # not reject legitimate text on missing signals.
    readable = True
    unreadable_reason = None
    if readability is not None:
        letters = readability.get("letter_digit_ratio", 0.0)
        words = readability.get("word_score", 0.0)
        repl = readability.get("replacement_ratio", 0.0)
        readable = (
            letters >= MIN_LETTER_DIGIT_RATIO
            and words >= MIN_WORD_SCORE
            and repl <= MAX_REPLACEMENT_RATIO
        )
        if not readable:
            unreadable_reason = (
                f"unreadable text layer (letters={letters}, words={words}, "
                f"repl={repl})"
            )

    text_layer_rejected = False
    if encrypted:
        status = "blocked"
        has_text_layer = False
        needs_ocr = False
        reason_parts = ["PDF is encrypted; decrypt it before processing"]
    elif median >= MIN_TEXT_CHARS and readable and not (
        suspected_rasterized and high_image_coverage
    ):
        status = "ready"
        has_text_layer = True
        needs_ocr = False
        reason_parts = [f"median {median} chars/page; usable text layer"]
    else:
        status = "ready"
        has_text_layer = False
        needs_ocr = True
        if median >= MIN_TEXT_CHARS and not readable:
            text_layer_rejected = True
            reason_parts = [unreadable_reason]
        else:
            reason_parts = [
                f"median {median} non-space chars/page (threshold: {MIN_TEXT_CHARS})"
            ]

    if text_layer_rejected and all_fonts_non_unicode:
        reason_parts.append("all fonts non-Unicode (no ToUnicode map)")
    if suspected_rasterized:
        if total_fonts == 0:
            reason_parts.append("no fonts found")
        else:
            reason_parts.append(
                f"{non_unicode_fonts}/{total_fonts} fonts non-Unicode"
            )
    if high_image_coverage:
        reason_parts.append(f"{smask_count} image masks detected")

    font_coverage = None
    if total_fonts:
        font_coverage = round(
            (total_fonts - (non_unicode_fonts or 0)) / total_fonts,
            4,
        )

    return {
        **signals,
        "status": status,
        "has_text_layer": has_text_layer,
        "needs_ocr": needs_ocr,
        "text_layer_rejected": text_layer_rejected,
        "suspected_rasterized_text": suspected_rasterized,
        "high_image_coverage": high_image_coverage,
        "font_unicode_coverage": font_coverage,
        "reason": "; ".join(reason_parts),
    }


def probe_input(path: str, caps: Caps, verbose: bool = False) -> dict[str, Any]:
    input_type = classify_input(path)
    absolute_path = os.path.abspath(path)
    if input_type == "unsupported":
        _fatal(f"Unsupported file type: {path}", EXIT_UNSUPPORTED)
    if input_type == "image":
        return {
            "path": absolute_path,
            "input_type": "image",
            "status": "ready",
            "pages": 1,
            "needs_ocr": True,
            "reason": "image input requires OCR",
            "median_chars": 0,
            "per_page_chars": [0],
            "has_text_layer": False,
            "text_layer_rejected": False,
            "suspected_rasterized_text": False,
            "encrypted": False,
            "high_image_coverage": False,
            "font_unicode_coverage": None,
            "total_fonts": None,
            "non_unicode_fonts": None,
            "image_count": None,
            "smask_count": None,
        }
    return probe_pdf(path, caps, verbose)


def probe_pdf(path: str, caps: Caps, verbose: bool = False) -> dict[str, Any]:
    """Collect PDF signals and apply the canonical probe decision."""
    pages = 1
    encrypted = False
    per_page_chars: list[int] = []
    per_page_text: list[str] = []

    if caps.has_fitz:
        fitz = _fitz_or_fallback(caps, "text")
        if fitz is not None:
            doc = fitz.open(path)
            pages = len(doc)
            encrypted = bool(doc.needs_pass)
            if not encrypted:
                for page in doc:
                    text = page.get_text()
                    per_page_text.append(text)
                    per_page_chars.append(_nonspace_byte_count(text))
            doc.close()

    if caps.bin_pdfinfo:
        try:
            for line in _run([caps.bin_pdfinfo, path]).stdout.splitlines():
                lower = line.lower()
                if lower.startswith("pages:"):
                    pages = int(line.split(":", 1)[1].strip())
                elif lower.startswith("encrypted:"):
                    encrypted = line.split(":", 1)[1].strip().lower().startswith("yes")
        except Exception:
            pass

    if not encrypted and not per_page_chars:
        caps.require_pdftotext()
        for page_number in range(1, pages + 1):
            try:
                result = _run([
                    caps.bin_pdftotext,
                    "-layout",
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    path,
                    "-",
                ])
                page_text = result.stdout
                count = _nonspace_byte_count(page_text)
            except Exception:
                page_text = ""
                count = 0
            per_page_text.append(page_text)
            per_page_chars.append(count)

    counts = sorted(per_page_chars or [0])
    median = counts[len(counts) // 2]

    # Score readability on the densest sample pages so a page-1 header does not
    # dominate; skip when no text was extracted (leaves the density-only path).
    readability = None
    ranked = sorted(
        range(len(per_page_text)),
        key=lambda i: per_page_chars[i],
        reverse=True,
    )
    sample = "\n".join(per_page_text[i] for i in ranked[:3]).strip()
    if sample:
        readability = text_readability(sample)

    total_fonts = None
    non_unicode_fonts = None
    if caps.bin_pdffonts and not encrypted:
        try:
            rows = [
                line.split()
                for line in _run([caps.bin_pdffonts, path]).stdout.splitlines()[2:]
                if line.strip()
            ]
            total_fonts = len(rows)
            non_unicode_fonts = sum(
                1 for row in rows if len(row) > 5 and row[5].lower() == "no"
            )
        except Exception:
            pass

    image_count = None
    smask_count = None
    if caps.bin_pdfimages and not encrypted:
        try:
            rows = [
                line.split()
                for line in _run([caps.bin_pdfimages, "-list", path]).stdout.splitlines()[2:]
                if line.strip()
            ]
            image_count = len(rows)
            smask_count = sum(
                1 for row in rows if len(row) > 2 and row[2].lower() == "smask"
            )
        except Exception:
            pass

    result = decide_probe({
        "path": os.path.abspath(path),
        "input_type": "pdf",
        "pages": pages,
        "encrypted": encrypted,
        "per_page_chars": per_page_chars,
        "median_chars": median,
        "total_fonts": total_fonts,
        "non_unicode_fonts": non_unicode_fonts,
        "image_count": image_count,
        "smask_count": smask_count,
        "readability": readability,
    })
    _log(f"probe: {result['reason']}", verbose)
    return result


# ── text-layer extraction ─────────────────────────────────────────────────────

def extract_text_layer(path: str, pages: list[int] | None, caps: Caps) -> list[str]:
    """Extract embedded text from a PDF that has a real text layer."""
    if caps.has_fitz:
        fitz = _fitz_or_fallback(caps, "text")
        if fitz is not None:
            return _extract_fitz(path, pages, fitz)
    caps.require_pdftotext()
    return _extract_pdftotext(path, pages, caps)


def _extract_fitz(path: str, pages: list[int] | None, fitz: Any) -> list[str]:
    doc = fitz.open(path)
    result = []
    for i, page in enumerate(doc):
        if pages and (i + 1) not in pages:
            continue
        result.append(page.get_text("text"))
    doc.close()
    return result


def _extract_pdftotext(path: str, pages: list[int] | None, caps: Caps) -> list[str]:
    # Extract all pages then split
    r = _run([caps.bin_pdftotext, "-layout", path, "-"])
    # pdftotext uses form-feed (\x0c) as page separator
    all_pages = r.stdout.split("\x0c")
    if pages:
        return [all_pages[p - 1] for p in pages if p <= len(all_pages)]
    return [p for p in all_pages if p.strip()]


# ── text-PDF table extraction ─────────────────────────────────────────────────

def _clean_table_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _render_pipe_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(_markdown_cell(cell) for cell in padded[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in padded[1:]]
    return "\n".join([header, separator, *body])


def _render_html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = ["<table>", "  <thead>", "    <tr>"]
    for cell in padded[0]:
        value = html_lib.escape(cell).replace("\n", "<br>")
        lines.append(f"      <th>{value}</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in padded[1:]:
        lines.append("    <tr>")
        for cell in row:
            value = html_lib.escape(cell).replace("\n", "<br>")
            lines.append(f"      <td>{value}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def _camelot_has_spans(table: Any) -> bool:
    for row in getattr(table, "cells", ()) or ():
        for cell in row:
            if getattr(cell, "hspan", False) or getattr(cell, "vspan", False):
                return True
    return False


def _camelot_rows(table: Any) -> list[list[str]]:
    frame = getattr(table, "df", None)
    if frame is None:
        return []
    values = frame.values.tolist() if hasattr(frame, "values") else frame
    return [[_clean_table_cell(cell) for cell in row] for row in values]


def _camelot_bbox(table: Any) -> list[float] | None:
    raw = getattr(table, "bbox", None) or getattr(table, "_bbox", None)
    if not raw or len(raw) != 4:
        return None
    return [float(value) for value in raw]


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_camelot_bbox(raw: list[float], page_height: float) -> list[float]:
    x0, y0, x1, y1 = raw
    left, right = sorted((x0, x1))
    top = page_height - max(y0, y1)
    bottom = page_height - min(y0, y1)
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def _bbox_intersection(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(first: list[float], second: list[float]) -> float:
    intersection = _bbox_intersection(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def _layout_fitz(path: str, pages: list[int] | None, fitz: Any) -> dict[int, dict[str, Any]]:
    doc = fitz.open(path)
    result: dict[int, dict[str, Any]] = {}
    try:
        for index, page in enumerate(doc):
            page_number = index + 1
            if pages and page_number not in pages:
                continue
            blocks = []
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    bbox = line.get("bbox")
                    if text and bbox and len(bbox) == 4:
                        blocks.append({"bbox": [float(value) for value in bbox], "text": text})
            result[page_number] = {
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "blocks": blocks,
            }
    finally:
        doc.close()
    return result


def _layout_pdftotext(path: str, pages: list[int] | None, caps: Caps) -> dict[int, dict[str, Any]]:
    caps.require_pdftotext()
    xml = _run([caps.bin_pdftotext, "-bbox-layout", path, "-"]).stdout
    root = ET.fromstring(xml)
    result: dict[int, dict[str, Any]] = {}
    page_number = 0
    for page in (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "page"):
        page_number += 1
        if pages and page_number not in pages:
            continue
        blocks = []
        for line in (element for element in page.iter() if element.tag.rsplit("}", 1)[-1] == "line"):
            words = [
                (word.text or "").strip()
                for word in line.iter()
                if word.tag.rsplit("}", 1)[-1] == "word" and (word.text or "").strip()
            ]
            if not words:
                continue
            try:
                bbox = [
                    float(line.attrib["xMin"]),
                    float(line.attrib["yMin"]),
                    float(line.attrib["xMax"]),
                    float(line.attrib["yMax"]),
                ]
            except (KeyError, ValueError):
                continue
            blocks.append({"bbox": bbox, "text": " ".join(words)})
        result[page_number] = {
            "width": float(page.attrib.get("width", 0)),
            "height": float(page.attrib.get("height", 0)),
            "blocks": blocks,
        }
    return result


def extract_text_layer_layout(
    path: str,
    pages: list[int] | None,
    caps: Caps,
) -> dict[int, dict[str, Any]]:
    if caps.has_fitz:
        fitz = _fitz_or_fallback(caps, "text layout")
        if fitz is not None:
            return _layout_fitz(path, pages, fitz)
    return _layout_pdftotext(path, pages, caps)


def extract_text_pdf_tables(
    path: str,
    pages: list[int] | None,
    flavor: str = "auto",
    verbose: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    if flavor not in TABLE_FLAVORS:
        _fatal(f"unsupported Camelot table flavor: {flavor}", EXIT_BAD_ARGS)
    try:
        import camelot
    except ImportError as exc:
        raise OcrError(
            "Camelot is required for text-PDF table extraction; reinstall pro-ledin-ocr"
        ) from exc

    page_spec = ",".join(str(page) for page in pages) if pages else "all"
    def read(selected_pages: str, selected_flavor: str) -> list[Any]:
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter("ignore", UserWarning)
            return list(camelot.read_pdf(
                path,
                pages=selected_pages,
                flavor=selected_flavor,
            ))

    tables = read(page_spec, flavor)
    if flavor == "auto" and pages:
        stream_pages = [
            page
            for page in pages
            if not any(
                int(getattr(table, "page", 1)) == page
                and str(getattr(table, "flavor", "")) == "stream"
                for table in tables
            )
        ]
        if stream_pages:
            tables.extend(read(",".join(str(page) for page in stream_pages), "stream"))
    result: dict[int, list[dict[str, Any]]] = {}
    for table in tables:
        rows = _camelot_rows(table)
        raw_bbox = _camelot_bbox(table)
        if not rows or raw_bbox is None:
            continue
        report = dict(getattr(table, "parsing_report", {}) or {})
        page_number = int(getattr(table, "page", report.get("page", 1)))
        complex_structure = _camelot_has_spans(table)
        issues = ["table_spans_flattened"] if complex_structure else []
        rendered = _render_html_table(rows) if complex_structure else _render_pipe_table(rows)
        result.setdefault(page_number, []).append({
            "extractor": "camelot",
            "flavor": str(getattr(table, "flavor", flavor)),
            "page": page_number,
            "raw_bbox": raw_bbox,
            "rows": rows,
            "format": "html" if complex_structure else "markdown",
            "rendered": rendered,
            "accuracy": _metric_float(report.get("accuracy")),
            "whitespace": _metric_float(report.get("whitespace")),
            "confidence": _metric_float(report.get("confidence")),
            "order": report.get("order"),
            "issues": issues,
            "accepted": False,
        })
    _log(f"camelot: detected {sum(len(items) for items in result.values())} table(s)", verbose)
    return result


def _normalized_tokens(text: str) -> list[str]:
    return [token.casefold().replace(",", ".") for token in re.findall(r"[\w]+(?:[.,]\d+)?", text)]


def _numeric_tokens(text: str) -> list[str]:
    return [
        token.replace(",", ".").replace("–", "-")
        for token in re.findall(r"(?<!\w)[<>±]?\d+(?:[.,]\d+)?(?:[-–]\d+(?:[.,]\d+)?)?", text)
    ]


def _counter_coverage(source: list[str], extracted: list[str]) -> float:
    if not source:
        return 1.0
    source_counts = Counter(source)
    extracted_counts = Counter(extracted)
    matched = sum(min(count, extracted_counts[token]) for token, count in source_counts.items())
    return matched / sum(source_counts.values())


def _blocks_in_bbox(blocks: list[dict[str, Any]], bbox: list[float]) -> list[dict[str, Any]]:
    selected = []
    for block in blocks:
        block_bbox = block["bbox"]
        intersection = _bbox_intersection(block_bbox, bbox)
        if intersection and intersection / max(_bbox_area(block_bbox), 1.0) >= 0.25:
            selected.append(block)
    return selected


def _validate_table(table: dict[str, Any], blocks: list[dict[str, Any]]) -> list[str]:
    issues = list(table.get("issues", []))
    rows = table["rows"]
    width = max((len(row) for row in rows), default=0)
    nonempty_rows = sum(any(cell.strip() for cell in row) for row in rows)
    nonempty_columns = sum(
        any(column < len(row) and row[column].strip() for row in rows)
        for column in range(width)
    )
    if nonempty_rows < 2 or nonempty_columns < 2:
        issues.append("table_too_small")
    accuracy = table.get("accuracy")
    confidence = table.get("confidence")
    whitespace = table.get("whitespace")
    if accuracy is not None and accuracy < 80:
        issues.append("table_parse_quality_low")
    if (confidence is not None and whitespace is not None
            and confidence < 0.4 and whitespace > 60):
        issues.append("table_parse_quality_low")

    region_text = "\n".join(block["text"] for block in _blocks_in_bbox(blocks, table["bbox"]))
    table_text = "\n".join("\t".join(row) for row in rows)
    if region_text:
        text_coverage = _counter_coverage(_normalized_tokens(region_text), _normalized_tokens(table_text))
        numeric_coverage = _counter_coverage(_numeric_tokens(region_text), _numeric_tokens(table_text))
        table["text_coverage"] = round(text_coverage, 4)
        table["numeric_coverage"] = round(numeric_coverage, 4)
        if text_coverage < 0.55:
            issues.append("table_text_coverage_low")
        if numeric_coverage < 0.9:
            issues.append("table_numeric_coverage_low")
    else:
        issues.append("table_region_text_missing")
    return issues


def _deduplicate_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    ranked = sorted(
        tables,
        key=lambda table: (
            bool(table.get("accepted")),
            table.get("confidence") or 0,
            table.get("accuracy") or 0,
            -(table.get("whitespace") if table.get("whitespace") is not None else 100),
        ),
        reverse=True,
    )
    for candidate in ranked:
        if any(
            _bbox_iou(candidate["bbox"], existing["bbox"]) >= 0.8
            or _bbox_intersection(candidate["bbox"], existing["bbox"])
            / max(_bbox_area(existing["bbox"]), 1.0)
            >= 0.9
            or _bbox_intersection(candidate["bbox"], existing["bbox"])
            / max(_bbox_area(candidate["bbox"]), 1.0)
            >= 0.9
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda table: (table["bbox"][1], table["bbox"][0]))


def _compose_page_markdown(
    blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> str:
    accepted = [table for table in tables if table.get("accepted")]
    remaining_blocks = []
    for block in blocks:
        if any(
            _bbox_intersection(block["bbox"], table["bbox"])
            / max(_bbox_area(block["bbox"]), 1.0)
            >= 0.25
            for table in accepted
        ):
            continue
        remaining_blocks.append(block)
    items = [
        (block["bbox"][1], block["bbox"][0], "text", block["text"])
        for block in remaining_blocks
    ]
    items.extend(
        (table["bbox"][1], table["bbox"][0], "table", table["rendered"])
        for table in accepted
    )
    sections = []
    text_lines = []
    for _, _, kind, text in sorted(items):
        text = text.strip()
        if not text:
            continue
        if kind == "table":
            if text_lines:
                sections.append("\n".join(text_lines))
                text_lines = []
            sections.append(text)
        else:
            text_lines.append(text)
    if text_lines:
        sections.append("\n".join(text_lines))
    return "\n\n".join(sections)


# ── page rendering ────────────────────────────────────────────────────────────

def auto_dpi(path: str, caps: Caps) -> int:
    """Choose DPI based on page dimensions."""
    width_pt = 595.0  # default A4
    try:
        if caps.has_fitz:
            import fitz

            doc = fitz.open(path)
            rect = doc[0].rect
            width_pt = rect.width
            doc.close()
        elif caps.bin_pdfinfo:
            r = _run([caps.bin_pdfinfo, path])
            for line in r.stdout.splitlines():
                if "page size" in line.lower():
                    # "Page size: 595.32 x 841.92 pts"
                    nums = re.findall(r"[\d.]+", line)
                    if nums:
                        width_pt = float(nums[0])
                    break
    except Exception:
        pass

    if width_pt > 1000:   # wide slide canvas (1920 pt)
        return 150
    if width_pt > 700:    # large page
        return 200
    return 300             # A4 / Letter


def render_pages(path: str, dpi: int, pages: list[int] | None,
                 tmpdir: str, caps: Caps, verbose: bool = False) -> list[tuple[int, str]]:
    """
    Render PDF pages to PNG files in tmpdir.
    Returns [(page_number, png_path), …].
    Uses PyMuPDF if available, else pdftoppm.
    """
    if caps.has_fitz:
        fitz = _fitz_or_fallback(caps, "render")
        if fitz is not None:
            return _render_fitz(path, dpi, pages, tmpdir, verbose, fitz)
    caps.require_render()
    return _render_pdftoppm(path, dpi, pages, tmpdir, caps, verbose)


def _render_fitz(path: str, dpi: int, pages: list[int] | None,
                 tmpdir: str, verbose: bool, fitz: Any) -> list[tuple[int, str]]:
    doc = fitz.open(path)
    results = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc):
        pnum = i + 1
        if pages and pnum not in pages:
            continue
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(tmpdir, f"page_{pnum:04d}.png")
        pix.save(out_path)
        _log(f"rendered page {pnum} → {pix.width}×{pix.height} px @ {dpi} DPI", verbose)
        results.append((pnum, out_path))
    doc.close()
    return results


def _render_pdftoppm(path: str, dpi: int, pages: list[int] | None,
                     tmpdir: str, caps: Caps, verbose: bool) -> list[tuple[int, str]]:
    prefix = os.path.join(tmpdir, "page")
    cmd = [caps.bin_pdftoppm, "-png", "-r", str(dpi)]
    if pages:
        cmd += ["-f", str(min(pages)), "-l", str(max(pages))]
    cmd += [path, prefix]
    _run(cmd, capture=False)

    # pdftoppm writes page-NNNN.png
    rendered = sorted(Path(tmpdir).glob("page-*.png"))
    results: list[tuple[int, str]] = []
    for png in rendered:
        # extract page number from filename "page-0001.png"
        stem = png.stem  # "page-0001"
        num_str = stem.split("-")[-1]
        pnum = int(num_str)
        if pages and pnum not in pages:
            continue
        _log(f"rendered page {pnum} → {png}", verbose)
        results.append((pnum, str(png)))
    return results


# ── language detection ────────────────────────────────────────────────────────

def detect_lang(img_path: str, caps: Caps, verbose: bool = False) -> str:
    """
    Run tesseract OSD on img_path to detect script, map to language code.
    Always appends +eng. Falls back to 'eng' on any failure.
    """
    if not caps.bin_tesseract:
        return "eng"
    try:
        r = subprocess.run(
            [caps.bin_tesseract, img_path, "stdout", "--psm", "0", "-l", "osd"],
            capture_output=True, text=True, timeout=30,
        )
        output = r.stdout + r.stderr
        script = None
        conf = 0.0
        for line in output.splitlines():
            if "script:" in line.lower() and "confidence" not in line.lower():
                script = line.split(":", 1)[1].strip().lower()
            if "script confidence:" in line.lower():
                try:
                    conf = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        _log(f"OSD detected script={script!r} confidence={conf}", verbose)

        if script and conf >= 1.0 and script in SCRIPT_TO_LANG:
            lang_code = SCRIPT_TO_LANG[script]
            if lang_code == "eng":
                return "eng"
            return f"{lang_code}+eng"
    except Exception as e:
        _log(f"OSD failed: {e} — falling back to eng", verbose)

    return "eng"


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_path: str, level: str, caps: Caps, tmpdir: str,
               verbose: bool = False) -> str:
    """
    Apply image preprocessing before OCR.
    Returns path to processed image (may be same as input for level=none).
    Requires PIL for basic; opencv-python+numpy for enhanced/full.
    Gracefully falls back to basic if cv2 missing.
    """
    if level == "none":
        return img_path

    out_path = os.path.join(tmpdir, "pp_" + os.path.basename(img_path))

    if level == "basic" or (level in ("enhanced", "full") and not (caps.has_cv2 and caps.has_numpy)):
        if not caps.has_pil:
            _log("Pillow not available — skipping preprocessing", verbose)
            return img_path
        if level in ("enhanced", "full") and not caps.has_cv2:
            _log("opencv-python not available — falling back to basic preprocessing", verbose)
        return _preprocess_basic(img_path, out_path, verbose)

    if level in ("enhanced", "full"):
        return _preprocess_opencv(img_path, out_path, level, verbose)

    return img_path


def _preprocess_basic(img_path: str, out_path: str, verbose: bool) -> str:
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except (ImportError, OSError) as exc:
        _log(
            f"Pillow import failed ({type(exc).__name__}) — skipping preprocessing",
            verbose,
        )
        return img_path
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < SMALL_WIDTH_THRESHOLD:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
        _log(f"upscaled {w}×{h} → {w*2}×{h*2}", verbose)
    gray = img.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(1.5)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    sharpened.save(out_path)
    return out_path


def _preprocess_opencv(img_path: str, out_path: str, level: str, verbose: bool) -> str:
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        component = getattr(exc, "name", "") or type(exc).__name__
        _log(
            f"enhanced preprocessing dependency {component!r} failed to load — "
            "falling back to basic preprocessing",
            verbose,
        )
        return _preprocess_basic(img_path, out_path, verbose)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < SMALL_WIDTH_THRESHOLD:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
        _log(f"upscaled {w}×{h} → {w*2}×{h*2}", verbose)

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Deskew for 'full'
    if level == "full":
        denoised = _deskew(denoised, verbose, cv2, np)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    cv2.imwrite(out_path, thresh)
    return out_path


def _deskew(gray: Any, verbose: bool, cv2: Any, np: Any) -> Any:
    inv = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inv, 50, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        _log(f"deskew: angle {angle:.2f}° < 0.5° — skipped", verbose)
        return gray
    _log(f"deskew: correcting {angle:.2f}°", verbose)
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def resolve_preprocess(level: str, probe: dict | None, caps: Caps,
                       input_type: str = "pdf") -> str:
    """
    Resolve 'auto' preprocessing level.
    - Standalone images: default none (already rasterized at native quality)
    - PDF renders that need OCR: enhanced if cv2 available, else basic
    - PDF with real text layer: basic (shouldn't matter, fast path skips OCR)
    """
    if level != "auto":
        return level
    if input_type == "image":
        return "none"  # don't degrade already-rasterized images
    if probe and probe.get("needs_ocr"):
        return "enhanced" if (caps.has_cv2 and caps.has_numpy) else "basic"
    return "basic"


# ── OCR engines ───────────────────────────────────────────────────────────────

def ocr_tesseract(img_path: str, lang: str, psm: int,
                  caps: Caps, verbose: bool = False) -> tuple[str, float, list[dict]]:
    """
    Run tesseract OCR. Returns (text, mean_conf, words).
    Prefers pytesseract for TSV confidence data; falls back to tesseract CLI.
    """
    caps.require_ocr()

    if caps.has_pytesseract:
        try:
            import pytesseract
        except (ImportError, OSError) as exc:
            _log(
                f"pytesseract import failed ({type(exc).__name__}) — "
                "falling back to tesseract CLI",
                verbose,
            )
        else:
            return _ocr_pytesseract(
                img_path, lang, psm, verbose, pytesseract
            )
    return _ocr_tesseract_cli(img_path, lang, psm, caps, verbose)


def _ocr_pytesseract(img_path: str, lang: str, psm: int,
                     verbose: bool, pytesseract: Any) -> tuple[str, float, list[dict]]:
    config = f"--oem 3 --psm {psm}"
    _log(f"pytesseract lang={lang} config={config}", verbose)

    # Pass img_path directly to avoid PIL round-trip issues with large images
    data = pytesseract.image_to_data(img_path, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    words = []
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf == -1 or not word.strip():
            continue
        words.append({
            "text": word,
            "conf": conf,
            "bbox": [data["left"][i], data["top"][i],
                     data["width"][i], data["height"][i]],
        })
        confidences.append(conf)

    full_text = pytesseract.image_to_string(img_path, lang=lang, config=config)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    _log(f"pytesseract: {len(words)} words, mean_conf={mean_conf:.1f}", verbose)

    # Safety retry with rus+eng when lang was auto-detected as eng-only:
    # OSD may have failed (sparse page, logo-heavy), leaving us with bad lang.
    # Always try rus+eng when we used plain "eng" — it's cheap and catches mixed docs.
    if ("rus" not in lang) and lang == "eng":
        _log("retrying with rus+eng (zero words or garbled non-ASCII)", verbose)
        fallback = "rus+eng"
        data2 = pytesseract.image_to_data(img_path, lang=fallback, config=config,
                                          output_type=pytesseract.Output.DICT)
        words2, confs2 = [], []
        for i, w in enumerate(data2["text"]):
            c = int(data2["conf"][i])
            if c == -1 or not w.strip():
                continue
            words2.append({"text": w, "conf": c,
                           "bbox": [data2["left"][i], data2["top"][i],
                                    data2["width"][i], data2["height"][i]]})
            confs2.append(c)
        mean_conf2 = sum(confs2) / len(confs2) if confs2 else 0.0
        # Use rus+eng if it got more words, or similar words with better confidence
        if len(words2) > len(words) or (words2 and mean_conf2 > mean_conf + 5):
            full_text2 = pytesseract.image_to_string(img_path, lang=fallback, config=config)
            _log(f"retry rus+eng: {len(words2)} words, mean_conf={mean_conf2:.1f}", verbose)
            return full_text2, mean_conf2, words2

    return full_text, mean_conf, words


def _ocr_tesseract_cli(img_path: str, lang: str, psm: int,
                       caps: Caps, verbose: bool) -> tuple[str, float, list[dict]]:
    _log(f"tesseract CLI lang={lang} psm={psm}", verbose)
    config = f"--oem 3 --psm {psm}"

    # TSV for confidence
    tsv_result = subprocess.run(
        [caps.bin_tesseract, img_path, "stdout", "-l", lang,
         "--oem", "3", "--psm", str(psm), "tsv"],
        capture_output=True, text=True,
    )
    words: list[dict] = []
    confidences: list[float] = []
    if tsv_result.returncode == 0:
        lines = tsv_result.stdout.splitlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 12:
                    continue
                word_text = parts[11].strip()
                try:
                    conf = float(parts[10])
                except ValueError:
                    continue
                if conf == -1 or not word_text:
                    continue
                try:
                    bbox = [int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])]
                except ValueError:
                    bbox = [0, 0, 0, 0]
                words.append({"text": word_text, "conf": int(conf), "bbox": bbox})
                confidences.append(conf)

    # Plain text
    txt_result = subprocess.run(
        [caps.bin_tesseract, img_path, "stdout", "-l", lang,
         "--oem", "3", "--psm", str(psm)],
        capture_output=True, text=True,
    )
    if txt_result.returncode != 0:
        _fatal(
            f"tesseract text extraction failed (exit {txt_result.returncode})",
            EXIT_BAD_ARGS,
        )
    full_text = txt_result.stdout
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    _log(f"tesseract CLI: {len(words)} words, mean_conf={mean_conf:.1f}", verbose)
    return full_text, mean_conf, words


def ocr_easyocr(img_path: str, caps: Caps, verbose: bool = False) -> tuple[str, float, list[dict]]:
    _raise_requirement(probe_engine_requirements("easyocr", caps=caps))
    with _engine_import(_easyocr_import_requirement):
        import easyocr
    with _engine_import(_easyocr_dependency_requirement("numpy")):
        import numpy as np
    with _engine_import(_easyocr_dependency_requirement("pillow")):
        from PIL import Image
    _log("loading easyocr reader (ru+en)...", verbose)
    reader = easyocr.Reader(["ru", "en"], gpu=False)
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    result = reader.readtext(arr, detail=1, paragraph=False)
    words = []
    texts = []
    confs = []
    for bbox, text, conf in result:
        if text.strip():
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            words.append({
                "text": text,
                "conf": int(conf * 100),
                "bbox": [int(min(x_coords)), int(min(y_coords)),
                         int(max(x_coords) - min(x_coords)),
                         int(max(y_coords) - min(y_coords))],
            })
            texts.append(text)
            confs.append(conf)
    full_text = "\n".join(texts)
    mean_conf = (sum(confs) / len(confs) * 100) if confs else 0.0
    _log(f"easyocr: {len(words)} words, mean_conf={mean_conf:.1f}", verbose)
    return full_text, mean_conf, words


# ── PaddleOCR (opt-in, 3.x) ──────────────────────────────────────────────────

_PADDLE_CACHE: dict[str, Any] = {}
_PADDLE_VL_CACHE: dict[tuple[str, str], Any] = {}


def resolve_paddle_lang(lang: str) -> str:
    """Map a tesseract lang spec (possibly 'rus+eng') to a PaddleOCR code.
    Takes the primary code before '+', maps via TESS_TO_PADDLE_LANG, default 'en'.
    """
    primary = (lang or "").split("+", 1)[0].strip().lower()
    if primary in ("", "auto"):
        return "en"
    return TESS_TO_PADDLE_LANG.get(primary, "en")


def _poly_bbox(poly: Any) -> list[int]:
    """[[x,y],...] → [min_x, min_y, w, h]."""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _parse_paddle_result(result: Any) -> tuple[str, float, list[dict]]:
    """
    Parse PaddleOCR 3.x predict() output into (text, mean_conf, words).
    Each result item exposes rec_texts, rec_scores, rec_polys (or dt_polys).
    Words are sorted top-to-bottom, left-to-right for reading order.
    Pure helper — accepts any object/dict with those fields (testable via stub).
    """
    words: list[dict] = []
    scores: list[float] = []

    for item in result:
        def _get(name: str, default: Any = None) -> Any:
            if isinstance(item, dict):
                return item.get(name, default)
            return getattr(item, name, default)

        texts = _get("rec_texts") or []
        confs = _get("rec_scores") or []
        polys = _get("rec_polys")
        if polys is None:
            polys = _get("dt_polys") or []

        for i, txt in enumerate(texts):
            if not str(txt).strip():
                continue
            score = float(confs[i]) if i < len(confs) else 0.0
            poly = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            words.append({
                "text": str(txt),
                "conf": int(score * 100),
                "bbox": _poly_bbox(poly),
            })
            scores.append(score)

    words.sort(key=lambda w: (w["bbox"][1], w["bbox"][0]))
    full_text = "\n".join(w["text"] for w in words)
    mean_conf = (sum(scores) / len(scores) * 100) if scores else 0.0
    return full_text, mean_conf, words


def ocr_paddleocr(img_path: str, lang: str, caps: Caps,
                  verbose: bool = False) -> tuple[str, float, list[dict]]:
    """Run PaddleOCR 3.x. Returns (text, mean_conf, words)."""
    caps.require_paddleocr()
    with _engine_import(_paddle_import_requirement):
        from paddleocr import PaddleOCR

    paddle_lang = resolve_paddle_lang(lang)
    engine = _PADDLE_CACHE.get(paddle_lang)
    if engine is None:
        _log(f"loading PaddleOCR reader (lang={paddle_lang})...", verbose)
        engine = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
        _PADDLE_CACHE[paddle_lang] = engine

    result = engine.predict(img_path)
    text, mean_conf, words = _parse_paddle_result(result)
    _log(f"paddleocr: {len(words)} lines, mean_conf={mean_conf:.1f}", verbose)
    return text, mean_conf, words


def _extract_paddle_vl_markdown(result: Any) -> str:
    parts: list[str] = []
    for item in result:
        # PaddleX result objects subclass dict but expose rendered Markdown via
        # a property, not a top-level "markdown" mapping key.
        markdown_data = getattr(item, "markdown", None)
        if markdown_data is None and isinstance(item, dict):
            markdown_data = item.get("markdown")
        try:
            markdown_text = markdown_data["markdown_texts"]
        except (KeyError, TypeError):
            markdown_text = ""
        if isinstance(markdown_text, (list, tuple)):
            markdown_text = "\n".join(str(part) for part in markdown_text)
        normalized = str(markdown_text).strip()
        if normalized:
            parts.append(normalized)
    return "\n\n".join(parts)


def markdown_to_text(markdown: str) -> str:
    """Produce a conservative plain-text view without altering native Markdown."""
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"\|?[\s:|-]+\|?", line) and "-" in line:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            line = "\t".join(cells)
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[*_`~]", "", line)
        lines.append(line)
    return general_cleanup("\n".join(lines))


def ocr_paddleocr_vl_mlx(
    img_path: str,
    server_url: str,
    model: str,
    caps: Caps,
    verbose: bool = False,
) -> tuple[str, str]:
    """Run full PaddleOCR-VL parsing with only its VLM stage served by MLX."""
    caps.require_paddleocr_vl(server_url, model)
    with _engine_import(_paddle_vl_import_requirement):
        from paddleocr import PaddleOCRVL

    key = (server_url.rstrip("/") + "/", model)
    engine = _PADDLE_VL_CACHE.get(key)
    if engine is None:
        _log(f"loading PaddleOCR-VL pipeline (model={model})...", verbose)
        try:
            engine = PaddleOCRVL(
                vl_rec_backend="mlx-vlm-server",
                vl_rec_server_url=key[0],
                vl_rec_max_concurrency=1,
                vl_rec_api_model_name=model,
                # Medical transcription must not silently drop headers,
                # footers, footnotes, or page-number blocks.
                markdown_ignore_labels=[],
            )
        except Exception as exc:
            raise OcrError(f"paddleocr-vl-mlx initialization failed: {exc}") from exc
        _PADDLE_VL_CACHE[key] = engine

    markdown = ""
    for attempt in range(2):
        try:
            result = engine.predict(img_path, temperature=0)
            markdown = _extract_paddle_vl_markdown(result)
        except Exception as exc:
            raise OcrError(f"paddleocr-vl-mlx request failed: {exc}") from exc
        if markdown:
            break
        if attempt == 0:
            _log("paddleocr-vl-mlx returned empty Markdown; retrying once", verbose)
    if not markdown:
        raise OcrError("paddleocr-vl-mlx returned empty Markdown")
    _log(f"paddleocr-vl-mlx: {len(markdown)} Markdown chars", verbose)
    return markdown, markdown_to_text(markdown)


def resolve_vision_prompt(vision_prompt: str = "") -> str:
    """Return a custom vision prompt or the built-in faithful extraction prompt."""
    return vision_prompt if vision_prompt.strip() else DEFAULT_VISION_PROMPT


def vision_handoff(
    img_paths: list[tuple[int, str]],
    verbose: bool = False,
    *,
    vision_prompt: str = "",
) -> str:
    """
    Print a manifest of rendered PNG paths for agent-driven vision OCR.
    Returns a manifest string; the agent should read each PNG.
    """
    prompt = resolve_vision_prompt(vision_prompt)
    lines = [
        "== Vision OCR: agent read required ==",
        "",
        "The following pages were rendered as PNG for vision-based OCR.",
        "Read each image using the prompt below.",
        "",
        "Prompt to use:",
        prompt,
        "",
        "Pages to read:",
    ]
    for pnum, png_path in img_paths:
        lines.append(f"  Page {pnum}: {png_path}")
    manifest = "\n".join(lines)
    print(manifest, file=sys.stderr, flush=True)
    return manifest


def resolve_vision_config(
    vision_api_key: str = "",
    vision_model: str = "",
    vision_api_url: str = "",
) -> tuple[str, str, str | None]:
    """
    Validate and normalize vision credentials passed explicitly by the
    caller (CLI flags or a library's RecognizeOptions). Never reads
    OPENAI_API_KEY or any other environment variable. Raises OcrError if key
    or model is empty.
    """
    key = (vision_api_key or "").strip()
    model = (vision_model or "").strip()
    endpoint = (vision_api_url or "").strip() or None
    if not key:
        _fatal("vision_api_key is required for engine=vision (CLI: --vision-api-key).", EXIT_BAD_ARGS)
    if not model:
        _fatal("vision_model is required for engine=vision (CLI: --vision-model).", EXIT_BAD_ARGS)
    return key, model, endpoint


VISION_IMAGE_BYTE_LIMIT = 7_500_000   # raw bytes; safely under the API 10 MB cap
VISION_JPEG_QUALITY = 92


def _detect_media_type(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _encode_page_b64(img_path: str, verbose: bool = False) -> tuple[str, str]:
    """Return (base64_data, media_type), downscaling oversized images.

    Fast path: images already under the byte limit are sent untouched with their
    detected media type — no PIL, no re-encode, no quality loss.
    Oversized images are re-encoded to JPEG at high quality (full resolution
    first); if still over the limit, dimensions are reduced in a verify loop.
    """
    raw = Path(img_path).read_bytes()
    if len(raw) <= VISION_IMAGE_BYTE_LIMIT:
        return base64.b64encode(raw).decode(), _detect_media_type(raw)
    try:
        from PIL import Image
    except (ImportError, OSError):
        _fatal(
            "Pillow is required to downscale an oversized vision page image. "
            "Install package 'Pillow' alongside 'pro-ledin-ocr[vision]'. "
            + ENGINE_SETUP_GUIDANCE,
            EXIT_MISSING_BINARY,
        )
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    def encode_jpeg(im) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=VISION_JPEG_QUALITY, subsampling=0)
        return buf.getvalue()

    # Step 1: JPEG re-encode at full resolution (keeps text sharp).
    data = encode_jpeg(img)
    # Step 2: reduce dimensions only if still oversized.
    scale = 1.0
    while len(data) > VISION_IMAGE_BYTE_LIMIT:
        scale *= 0.85
        w, h = img.size
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        data = encode_jpeg(img.resize((nw, nh), Image.LANCZOS))
        if nw <= 1 or nh <= 1:
            break
    _log(f"downscaled oversized page image {len(raw)}B -> {len(data)}B JPEG", verbose)
    return base64.b64encode(data).decode(), "image/jpeg"


def vision_ocr(
    img_paths: list[tuple[int, str]],
    *,
    vision_api_url: str = "",
    vision_api_key: str = "",
    vision_model: str = "",
    vision_prompt: str = "",
    timeout: float | None = None,
    verbose: bool = False,
) -> str:
    """Call an OpenAI-compatible vision API for pages using explicit config.

    `timeout` (seconds) bounds each request via the openai SDK's own client
    timeout, so a stalled request is actually cancelled — unlike a generic
    in-process wall-clock timeout, which cannot forcibly stop a call already
    running. `None` keeps the SDK's own default.
    """
    key, model, endpoint = resolve_vision_config(vision_api_key, vision_model, vision_api_url)
    prompt = resolve_vision_prompt(vision_prompt)
    with _engine_import(_unavailable_requirement(
        "vision",
        vision_api_key=vision_api_key,
        vision_model=vision_model,
        has_openai=False,
    )):
        from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": key, "base_url": endpoint}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    try:
        client = OpenAI(**client_kwargs)
    except Exception as exc:
        raise OcrError("vision client initialization failed") from exc
    parts_by_page: list[str] = []

    for pnum, png_path in img_paths:
        _log(f"vision API: page {pnum} (model={model})", verbose)
        b64, media_type = _encode_page_b64(png_path, verbose)
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
            )
            page_text = resp.choices[0].message.content or ""
        except Exception as exc:
            raise OcrError(f"vision request failed for page {pnum}") from exc
        if not page_text.strip():
            raise OcrError(f"vision returned an empty result for page {pnum}")
        parts_by_page.append(f"## Page {pnum}\n\n{page_text}")

    return "\n\n".join(parts_by_page)


# ── post-processing ───────────────────────────────────────────────────────────

def general_cleanup(text: str) -> str:
    """
    Light general-purpose cleanup (no domain dictionaries).
    - Collapse runs of spaces/tabs to single space
    - Join hyphenated line-breaks (сло-\nво → слово)
    - Normalize common ligatures (ﬁ→fi, ﬂ→fl, ﬀ→ff, ﬃ→ffi, ﬄ→ffl)
    - Strip control characters (keep newlines and tabs)
    - Collapse 3+ blank lines to 2
    """
    # Ligatures
    for lig, rep in [("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl"),
                     ("ﬅ", "st"), ("ﬆ", "st")]:
        text = text.replace(lig, rep)
    # Strip control characters (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Join hyphenated line-breaks (word-\n  nextword → wordnextword)
    text = re.sub(r"(\w)-\n[ \t]*(\w)", r"\1\2", text)
    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── tabular heuristic ─────────────────────────────────────────────────────────

def looks_tabular(words: list[dict]) -> bool:
    """
    Detect table structure: multiple rows each having words at 3+ distinct
    x-column positions (not just 3 words on the same line — prose does that too).
    Requires the x-positions to span the page width with visible gaps.
    """
    if len(words) < 9:
        return False
    # Group words by y-bucket (10px)
    y_groups: dict[int, list[int]] = {}
    for w in words:
        y = w["bbox"][1] // 10 * 10
        y_groups.setdefault(y, []).append(w["bbox"][0])

    # A table row: ≥ 3 words at x-positions spanning ≥ 3 distinct column buckets
    # Column bucket = x // (page_width / 6)  — divides page into 6 zones
    x_all = [w["bbox"][0] for w in words]
    if not x_all:
        return False
    page_width = max(x_all) - min(x_all) + 1
    col_bucket_size = max(page_width // 6, 50)

    table_rows = 0
    for xs in y_groups.values():
        col_buckets = {x // col_bucket_size for x in xs}
        if len(col_buckets) >= 3 and len(xs) >= 4:
            table_rows += 1

    return table_rows >= 3


def quality_issues(confidence: float | None, words: list[dict], min_conf: float) -> list[str]:
    issues: list[str] = []
    if confidence is not None and confidence < min_conf:
        issues.append("low_confidence")
    if looks_tabular(words):
        issues.append("table_like_unstructured")
    return issues


# ── output formatters ─────────────────────────────────────────────────────────

def to_markdown(pages_data: list[dict], filename: str) -> str:
    parts = [f"# {filename}", ""]
    for page in pages_data:
        body = page.get("markdown") or page.get("text", "")
        if page.get("n") != 0:
            parts.append(f"## Page {page['n']}")
            parts.append("")
        parts.append(str(body).strip())
        parts.append("")
    return "\n".join(parts)


def to_text(pages_data: list[dict]) -> str:
    parts = []
    for page in pages_data:
        parts.append(f"----- Page {page['n']} -----")
        parts.append(page.get("text", "").strip())
        parts.append("")
    return "\n".join(parts)


def to_json(pages_data: list[dict], meta: dict) -> str:
    low_conf = [
        p["n"]
        for p in pages_data
        if p.get("mean_conf") is not None
        and p["mean_conf"] < meta.get("min_conf", 60)
    ]
    vision_recs = [p["n"] for p in pages_data if p.get("flag") == "review-vision"]
    out = {
        "file": meta.get("file", ""),
        "engine": meta.get("engine", "auto"),
        "lang": meta.get("lang", ""),
        "dpi": meta.get("dpi", 0),
        "preprocess": meta.get("preprocess", "auto"),
        "pages": pages_data,
        "report": {
            "total_pages": len(pages_data),
            "low_conf_pages": low_conf,
            "recommend_vision": vision_recs,
        },
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


# ── caching ───────────────────────────────────────────────────────────────────

class Cache:
    def __init__(self, path: str | None):
        self._path = path
        self._data: dict = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                pass

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        missing = object()
        previous = self._data.get(key, missing)
        self._data[key] = value
        if self._path:
            path = Path(self._path)
            temp_path = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temp_path = handle.name
                    json.dump(self._data, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            except (OSError, TypeError, ValueError) as exc:
                if previous is missing:
                    self._data.pop(key, None)
                else:
                    self._data[key] = previous
                _fatal(f"failed to write cache: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)


# ── main processing ───────────────────────────────────────────────────────────

@dataclass
class RecognizeOptions:
    """Recognition parameters for `process_file()`/`recognize()`.

    Mirrors the CLI flags that control *how* a single file is recognized.
    Output-formatting flags (--out, --format, --searchable-pdf)
    are a CLI/output concern handled by `write_outputs()`, not part of this
    library-facing options object.
    """
    engine: str = "tesseract"
    lang: str = "auto"
    dpi: int = 0
    preprocess: str = "auto"
    pages: str = ""
    max_pages: int = 0
    psm: int = DEFAULT_PSM
    min_conf: float = DEFAULT_MIN_CONF
    no_cleanup: bool = False
    force: bool = False
    vision_api_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""
    # Seconds for the vision HTTP request (openai SDK client timeout).
    # None keeps the SDK default. Not used by local engines (tesseract,
    # easyocr, paddleocr) or the PaddleOCR-VL MLX client: once ocr.py is called as a library rather than run
    # as a CLI subprocess, there is no external process to kill, and a
    # generic in-process wall-clock timeout cannot forcibly cancel a running
    # local OCR call — only a real network request can be cancelled cleanly.
    timeout: float | None = None
    verbose: bool = False
    vision_prompt: str = ""
    auto_escalate: tuple[str, ...] = ()
    skip_ocr: bool = False
    paddle_vl_server_url: str = ""
    paddle_vl_model: str = ""
    extract_tables: bool = True
    table_flavor: str = "auto"


def _process_file_once(
    path: str,
    options: RecognizeOptions,
    caps: Caps,
    cache: Cache,
    tmpdir: str,
) -> list[dict]:
    """
    Process one input file. Returns list of page dicts.
    """
    verbose = options.verbose
    input_type = classify_input(path)

    if input_type == "unsupported":
        _fatal(f"Unsupported file type: {path}", EXIT_UNSUPPORTED)

    # Resolve DPI
    dpi = options.dpi
    if dpi == 0:  # 0 = auto
        if input_type == "pdf":
            dpi = auto_dpi(path, caps)
        else:
            dpi = 150  # images already rasterized

    # Probe PDF
    probe: dict | None = None
    if input_type == "pdf":
        probe = probe_pdf(path, caps, verbose)
        if probe.get("status") == "blocked":
            _fatal(probe["reason"], EXIT_BAD_ARGS)

    # Preprocess level
    pp_level = resolve_preprocess(options.preprocess, probe, caps, input_type)

    page_range = None
    if input_type == "pdf":
        page_range = _resolve_page_range(
            options.pages, probe["pages"] if probe else 0, options.max_pages
        )

    if options.skip_ocr and (
        input_type == "image" or (probe and probe["needs_ocr"])
    ):
        selected_pages = page_range or (
            [1]
            if input_type == "image"
            else list(range(1, (probe["pages"] if probe else 0) + 1))
        )
        return [{
            "n": page_number,
            "source": "skipped",
            "mean_conf": None,
            "flag": None,
            "text": "",
            "issues": [],
            "words": [],
            "skipped": True,
            "skip_reason": "input requires OCR and --skip-ocr is enabled",
        } for page_number in selected_pages]

    # Cache key
    page_selection = (
        ",".join(str(page) for page in page_range)
        if page_range is not None
        else "all"
    )
    engine_context = ""
    if options.engine == "vision":
        engine_context = json.dumps({
            "model": options.vision_model.strip(),
            "prompt": resolve_vision_prompt(options.vision_prompt),
            "url": options.vision_api_url.strip(),
        }, ensure_ascii=False, sort_keys=True)
    elif options.engine == "paddleocr-vl-mlx":
        engine_context = json.dumps({
            "model": options.paddle_vl_model.strip(),
            "url": options.paddle_vl_server_url.strip(),
        }, ensure_ascii=False, sort_keys=True)
    table_context = json.dumps({
        "enabled": options.extract_tables,
        "flavor": options.table_flavor,
        "output_version": TABLE_OUTPUT_VERSION,
    }, sort_keys=True)
    engine_context = f"{engine_context}|tables={table_context}" if engine_context else f"tables={table_context}"
    cache_key = _sha1_key(
        path,
        options.engine,
        str(dpi),
        pp_level,
        options.lang,
        page_selection,
        engine_context,
    )
    if not options.force and cache.get(cache_key):
        _log(f"cache hit for {path}", verbose)
        cached = cache.get(cache_key)
        return cached.get("pages", [])

    pages_data: list[dict] = []

    # Fast path: real text layer
    if (input_type == "pdf"
            and probe
            and not probe["needs_ocr"]):
        texts = extract_text_layer(path, page_range, caps)
        selected_pages = page_range or list(range(1, probe["pages"] + 1))
        layouts: dict[int, dict[str, Any]] = {}
        tables_by_page: dict[int, list[dict[str, Any]]] = {}
        table_page_issues: dict[int, list[str]] = {}
        table_error = ""
        if options.extract_tables:
            try:
                layouts = extract_text_layer_layout(path, selected_pages, caps)
                raw_tables = extract_text_pdf_tables(
                    path,
                    selected_pages,
                    options.table_flavor,
                    verbose,
                )
                blocking_issues = {
                    "table_too_small",
                    "table_text_coverage_low",
                    "table_numeric_coverage_low",
                    "table_region_text_missing",
                    "table_parse_quality_low",
                }
                for page_number, tables in raw_tables.items():
                    layout = layouts.get(page_number)
                    if not layout:
                        table_page_issues.setdefault(page_number, []).append(
                            "table_layout_missing"
                        )
                        continue
                    normalized = []
                    for table in tables:
                        table["bbox"] = _normalize_camelot_bbox(
                            table.pop("raw_bbox"), layout["height"]
                        )
                        table["issues"] = _validate_table(table, layout["blocks"])
                        table["accepted"] = not any(
                            issue in blocking_issues for issue in table["issues"]
                        )
                        normalized.append(table)
                    tables_by_page[page_number] = _deduplicate_tables(normalized)
            except OcrError:
                raise
            except Exception as exc:
                table_error = str(exc)
                _log(f"camelot failed; preserving text layer: {exc}", verbose)
        for i, text in enumerate(texts):
            pnum = (page_range[i] if page_range else i + 1)
            cleaned = general_cleanup(text) if not options.no_cleanup else text
            page_tables = tables_by_page.get(pnum, [])
            issues = []
            issues.extend(table_page_issues.get(pnum, []))
            if table_error:
                issues.append("table_extraction_failed")
            if (page_tables
                    and not any(table.get("accepted") for table in page_tables)
                    and any(not table.get("accepted") for table in page_tables)):
                issues.append("table_extraction_rejected")
            page_data = {
                "n": pnum,
                "source": "text_layer",
                "mean_conf": 100.0,
                "flag": None,
                "text": cleaned,
                "issues": issues,
                "words": [],
                "tables": page_tables,
            }
            if page_tables and any(table.get("accepted") for table in page_tables):
                page_data["markdown"] = _compose_page_markdown(
                    layouts[pnum]["blocks"], page_tables
                )
            pages_data.append(page_data)
        return _cache_and_return(cache, cache_key, pages_data, probe)

    # Automated vision path
    if options.engine == "vision":
        _require_engine("vision", caps, options)
        caps.require_render()
        if input_type == "pdf":
            rendered = render_pages(path, dpi, page_range, tmpdir, caps, verbose)
        else:
            rendered = [(1, path)]
        combined_md = vision_ocr(
            rendered,
            vision_api_url=options.vision_api_url,
            vision_api_key=options.vision_api_key,
            vision_model=options.vision_model,
            vision_prompt=options.vision_prompt,
            timeout=options.timeout,
            verbose=verbose,
        )
        pages_data.append({"n": 0, "source": "vision", "mean_conf": None,
                            "flag": None, "text": combined_md,
                            "markdown": combined_md, "issues": [], "words": []})
        return _cache_and_return(cache, cache_key, pages_data, probe)

    # EasyOCR path
    if options.engine == "easyocr":
        _require_engine("easyocr", caps, options)
        if input_type == "pdf":
            rendered = render_pages(path, dpi, page_range, tmpdir, caps, verbose)
        else:
            rendered = [(1, path)]
        for pnum, img_path in rendered:
            pp_path = preprocess(img_path, pp_level, caps, tmpdir, verbose)
            text, conf, words = ocr_easyocr(pp_path, caps, verbose)
            cleaned = general_cleanup(text) if not options.no_cleanup else text
            issues = quality_issues(conf, words, options.min_conf)
            flag = "review-vision" if issues else None
            pages_data.append({
                "n": pnum, "source": "easyocr",
                "mean_conf": round(conf, 1), "flag": flag,
                "text": cleaned, "issues": issues, "words": words,
            })
        return _cache_and_return(cache, cache_key, pages_data, probe)

    # PaddleOCR path (opt-in)
    if options.engine == "paddleocr":
        _require_engine("paddleocr", caps, options)
        if input_type == "pdf":
            rendered = render_pages(path, dpi, page_range, tmpdir, caps, verbose)
        else:
            rendered = [(1, path)]

        # Resolve language: OSD auto-detect on page 1, else the user-provided code
        lang = options.lang
        if lang == "auto" and rendered:
            lang = detect_lang(rendered[0][1], caps, verbose)
            _log(f"language detected: {lang}", verbose)

        for pnum, img_path in rendered:
            pp_path = preprocess(img_path, pp_level, caps, tmpdir, verbose)
            text, conf, words = ocr_paddleocr(pp_path, lang, caps, verbose)
            cleaned = general_cleanup(text) if not options.no_cleanup else text
            issues = quality_issues(conf, words, options.min_conf)
            flag = "review-vision" if issues else None
            pages_data.append({
                "n": pnum, "source": "paddleocr",
                "mean_conf": round(conf, 1), "flag": flag,
                "text": cleaned, "issues": issues, "words": words,
            })
        return _cache_and_return(cache, cache_key, pages_data, probe)

    # Full document parsing with PaddleOCR-VL and an MLX-served VLM stage.
    if options.engine == "paddleocr-vl-mlx":
        _require_engine("paddleocr-vl-mlx", caps, options)
        if input_type == "pdf":
            rendered = render_pages(path, dpi, page_range, tmpdir, caps, verbose)
        else:
            rendered = [(1, path)]
        for pnum, img_path in rendered:
            markdown, text = ocr_paddleocr_vl_mlx(
                img_path,
                options.paddle_vl_server_url,
                options.paddle_vl_model,
                caps,
                verbose,
            )
            pages_data.append({
                "n": pnum,
                "source": "paddleocr-vl-mlx",
                "mean_conf": None,
                "flag": None,
                "text": text,
                "markdown": markdown,
                "issues": [],
                "words": [],
            })
        return _cache_and_return(cache, cache_key, pages_data, probe)

    if options.engine != "tesseract":
        _fatal(f"Unsupported OCR engine: {options.engine}", EXIT_BAD_ARGS)

    # Tesseract
    caps.require_ocr()
    if input_type == "pdf":
        rendered = render_pages(path, dpi, page_range, tmpdir, caps, verbose)
    else:
        rendered = [(1, path)]

    # Auto-detect language from first page
    lang = options.lang
    if lang == "auto" and rendered:
        lang = detect_lang(rendered[0][1], caps, verbose)
        _log(f"language detected: {lang}", verbose)

    for pnum, img_path in rendered:
        t0 = time.time()
        pp_path = preprocess(img_path, pp_level, caps, tmpdir, verbose)
        text, conf, words = ocr_tesseract(pp_path, lang, options.psm, caps, verbose)
        elapsed = time.time() - t0

        cleaned = general_cleanup(text) if not options.no_cleanup else text

        # Determine flag
        issues = quality_issues(conf, words, options.min_conf)
        flag = "review-vision" if issues else None

        pages_data.append({
            "n": pnum, "source": "tesseract",
            "mean_conf": round(conf, 1), "flag": flag,
            "text": cleaned, "issues": issues, "words": words,
            "elapsed_s": round(elapsed, 2),
        })
        _log(f"page {pnum}: {len(words)} words, conf={conf:.1f}, {elapsed:.1f}s"
             + (f" [FLAGGED: {flag}]" if flag else ""), verbose)

    return _cache_and_return(cache, cache_key, pages_data, probe)


def _cache_and_return(
    cache: "Cache",
    cache_key: str,
    pages_data: list[dict],
    probe: dict[str, Any] | None,
) -> list[dict]:
    """Stamp the text-layer-rejected marker, persist, and return pages."""
    if probe and probe.get("text_layer_rejected"):
        for page in pages_data:
            issues = page.setdefault("issues", [])
            if "text_layer_rejected" not in issues:
                issues.append("text_layer_rejected")
    cache.set(cache_key, {"pages": pages_data})
    return pages_data


def _require_engine(engine: str, caps: Caps, options: RecognizeOptions) -> None:
    _raise_requirement(probe_engine_requirements(
        engine,
        vision_api_key=options.vision_api_key,
        vision_model=options.vision_model,
        paddle_vl_server_url=options.paddle_vl_server_url,
        paddle_vl_model=options.paddle_vl_model,
        caps=caps,
    ))


def _workflow_cache_key(path: str, options: RecognizeOptions) -> str:
    stat = os.stat(path)
    config = {
        "path": os.path.abspath(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "engine": options.engine,
        "lang": options.lang,
        "dpi": options.dpi,
        "preprocess": options.preprocess,
        "pages": options.pages,
        "max_pages": options.max_pages,
        "psm": options.psm,
        "min_conf": options.min_conf,
        "cleanup": not options.no_cleanup,
        "auto_escalate": options.auto_escalate,
        "vision_url": options.vision_api_url,
        "vision_model": options.vision_model,
        "vision_prompt": resolve_vision_prompt(options.vision_prompt),
        "paddle_vl_server_url": options.paddle_vl_server_url,
        "paddle_vl_model": options.paddle_vl_model,
        "extract_tables": options.extract_tables,
        "table_flavor": options.table_flavor,
        "table_output_version": TABLE_OUTPUT_VERSION,
        "output_schema": OCR_OUTPUT_SCHEMA_VERSION,
    }
    return hashlib.sha1(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _flag_reasons(page: dict, min_conf: float) -> list[str]:
    reasons = list(page.get("issues", []))
    if not reasons:
        reasons = quality_issues(
            page.get("mean_conf"), page.get("words", []), min_conf
        )
    return reasons or ([str(page["flag"])] if page.get("flag") else [])


def process_file(
    path: str,
    options: RecognizeOptions,
    caps: Caps,
    cache: Cache,
    tmpdir: str,
) -> list[dict]:
    """Process one file, optionally escalating flagged pages strictly."""
    if not options.auto_escalate or options.skip_ocr:
        return _process_file_once(path, options, caps, cache, tmpdir)

    if classify_input(path) == "pdf" and not options.force:
        initial_probe = probe_pdf(path, caps, options.verbose)
        if initial_probe.get("status") == "blocked":
            _fatal(initial_probe["reason"], EXIT_BAD_ARGS)
        if not initial_probe["needs_ocr"]:
            return _process_file_once(path, options, caps, cache, tmpdir)

    for engine in (options.engine, *options.auto_escalate):
        _require_engine(engine, caps, options)

    workflow_key = _workflow_cache_key(path, options)
    if not options.force:
        cached = cache.get(workflow_key)
        if cached:
            return cached.get("pages", [])

    single_options = replace(options, auto_escalate=())
    pages = _process_file_once(path, single_options, caps, Cache(None), tmpdir)
    flagged = [page for page in pages if page.get("flag")]
    if not flagged:
        cache.set(workflow_key, {"pages": pages})
        return pages

    by_number = {page["n"]: page for page in pages}
    for baseline in flagged:
        page_number = baseline["n"]
        attempts = [{
            "engine": options.engine,
            "status": "completed",
            "mean_conf": baseline.get("mean_conf"),
        }]
        selected = baseline
        selected_score = (
            not bool(selected.get("flag")),
            selected.get("mean_conf") or 0.0,
            len(selected.get("words", [])),
        )

        for engine in options.auto_escalate:
            attempt_options = replace(
                options,
                engine=engine,
                pages=str(page_number),
                force=True,
                auto_escalate=(),
                skip_ocr=False,
            )
            candidate_pages = _process_file_once(
                path, attempt_options, caps, Cache(None), tmpdir
            )
            if not candidate_pages:
                _fatal(f"Escalation engine {engine} returned no result")
            candidate = candidate_pages[0]
            if (not candidate.get("text", "").strip()
                    and not candidate.get("markdown", "").strip()
                    and not candidate.get("words")):
                _fatal(f"Escalation engine {engine} returned an empty result")
            candidate["n"] = page_number
            attempts.append({
                "engine": engine,
                "status": "completed",
                "mean_conf": candidate.get("mean_conf"),
            })
            if engine in {"vision", "paddleocr-vl-mlx"}:
                selected = candidate
            else:
                score = (
                    not bool(candidate.get("flag")),
                    candidate.get("mean_conf") or 0.0,
                    len(candidate.get("words", [])),
                )
                if score > selected_score:
                    selected = candidate
                    selected_score = score

        selected["decision"] = {
            "baseline": options.engine,
            "flag_reason": _flag_reasons(baseline, options.min_conf),
            "attempts": attempts,
            "selected": selected.get("source", options.engine),
        }
        by_number[page_number] = selected

    final_pages = [by_number[page["n"]] for page in pages]
    cache.set(workflow_key, {"pages": final_pages})
    return final_pages


def recognize(
    path: str | Path,
    options: RecognizeOptions | None = None,
    *,
    caps: Caps | None = None,
    cache: Cache | None = None,
) -> list[dict]:
    """Recognize one PDF/image file and return its page data (library entry point).

    Manages a throwaway temp directory for rendered pages and cleans it up
    before returning. Each returned page dict has: n, source, mean_conf,
    flag, text, words, issues, and optional tables/markdown. Engine=vision
    instead returns a single combined page whose `text` holds the whole
    document's Markdown. Format the result with `to_markdown()`, `to_text()`,
    or `to_json()`.

    `options.verbose` enables progress logging only. The capability dump stays
    a CLI diagnostic; pass `caps=Caps(report=True)` to opt into it.

    Raises:
        OcrError: unsupported input type, missing required binaries/packages,
            or a vision configuration/request failure.
        OcrRequirementError: a selected engine or PDF backend is unavailable.
    """
    options = options or RecognizeOptions()
    caps = caps or Caps(verbose=options.verbose)
    cache = cache if cache is not None else Cache(None)
    with tempfile.TemporaryDirectory(prefix="ocr_lib_") as tmpdir:
        return process_file(str(path), options, caps, cache, tmpdir)
