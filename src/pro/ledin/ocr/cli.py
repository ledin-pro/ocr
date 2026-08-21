#!/usr/bin/env python3
"""Command-line interface for the `pro.ledin.ocr` package (console script: ocr).

The recognition logic lives in `core`; this module owns only argument parsing
and output writing (formats, files, searchable PDF, quality report).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from .core import (
    Cache,
    Caps,
    DEFAULT_MIN_CONF,
    DEFAULT_PSM,
    OcrError,
    RecognizeOptions,
    TABLE_FLAVORS,
    _log,
    classify_input,
    probe_input,
    process_file,
    to_json,
    to_markdown,
    to_text,
)

# ── output writing ────────────────────────────────────────────────────────────

caps_global: Caps  # set in main()
OUTPUT_FORMATS = ("md", "txt", "json")
OCR_ENGINES = ("tesseract", "easyocr", "paddleocr", "paddleocr-vl-mlx", "vision")


def parse_formats(value: str) -> list[str]:
    formats: list[str] = []
    for raw_format in value.split(","):
        fmt = raw_format.strip()
        if not fmt:
            raise argparse.ArgumentTypeError("format list contains an empty value")
        expanded = OUTPUT_FORMATS if fmt == "all" else (fmt,)
        for item in expanded:
            if item not in OUTPUT_FORMATS:
                choices = ", ".join((*OUTPUT_FORMATS, "all"))
                raise argparse.ArgumentTypeError(
                    f"unknown format '{item}'; choose from {choices}"
                )
            if item not in formats:
                formats.append(item)
    return formats


def parse_engines(value: str) -> tuple[str, ...]:
    engines: list[str] = []
    for raw_engine in value.split(","):
        engine = raw_engine.strip()
        if not engine:
            raise argparse.ArgumentTypeError("engine list contains an empty value")
        if engine not in OCR_ENGINES:
            raise argparse.ArgumentTypeError(
                f"unknown engine '{engine}'; choose from {', '.join(OCR_ENGINES)}"
            )
        if engine not in engines:
            engines.append(engine)
    return tuple(engines)


def _write_file_set(files: list[tuple[str, str]]) -> None:
    staged: list[tuple[str, Path]] = []
    backups: dict[Path, str] = {}
    installed: list[Path] = []
    try:
        for path, content in files:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((handle.name, output))

        for _, output in staged:
            if output.exists():
                descriptor, backup = tempfile.mkstemp(
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".bak",
                )
                os.close(descriptor)
                os.unlink(backup)
                os.replace(output, backup)
                backups[output] = backup

        for temp_path, output in staged:
            os.replace(temp_path, output)
            installed.append(output)
    except OSError as exc:
        for output in installed:
            if output.exists():
                output.unlink()
        for output, backup in backups.items():
            if os.path.exists(backup):
                os.replace(backup, output)
        raise OcrError(f"failed to write output files: {exc}") from exc
    finally:
        for temp_path, _ in staged:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        for backup in backups.values():
            if os.path.exists(backup):
                os.unlink(backup)


def write_outputs(
    pages_data: list[dict],
    input_path: str,
    args: argparse.Namespace,
    lang: str,
    dpi: int,
) -> None:
    filename = os.path.basename(input_path)
    meta = {
        "file": filename,
        "engine": args.engine,
        "lang": lang,
        "dpi": dpi,
        "preprocess": args.preprocess,
        "min_conf": args.min_conf,
    }

    formats = args.format

    rendered_outputs = []
    for fmt in formats:
        if fmt == "md":
            content = to_markdown(pages_data, filename)
        elif fmt == "txt":
            content = to_text(pages_data)
        elif fmt == "json":
            content = to_json(pages_data, meta)
        rendered_outputs.append((fmt, content))

    if args.out:
        files = []
        for fmt, content in rendered_outputs:
            out_path = args.out
            multiple_inputs = len(getattr(args, "inputs", [input_path])) > 1
            if os.path.isdir(out_path) or len(formats) > 1 or multiple_inputs:
                os.makedirs(out_path, exist_ok=True)
                stem = Path(input_path).stem
                out_file = os.path.join(out_path, f"{stem}.{fmt}")
            else:
                out_file = out_path
            files.append((fmt, out_file, content))
        _write_file_set([(out_file, content) for _, out_file, content in files])
        for fmt, out_file, _ in files:
            print(f"[ocr] wrote {fmt} → {out_file}", file=sys.stderr)
    else:
        for fmt, content in rendered_outputs:
            if len(formats) > 1:
                print(f"\n{'='*60}\n[{fmt.upper()}]\n{'='*60}", flush=True)
            print(content, flush=True)

    # Searchable PDF
    if args.searchable_pdf:
        if not caps_global.bin_ocrmypdf:
            print("[ocr] WARNING: ocrmypdf not found. Install: brew install ocrmypdf", file=sys.stderr)
        else:
            cmd = [caps_global.bin_ocrmypdf, "-l", lang,
                   "--rotate-pages", "--deskew", "--force-ocr",
                   input_path, args.searchable_pdf]
            try:
                import subprocess
                subprocess.run(cmd, check=True)
                print(f"[ocr] searchable PDF → {args.searchable_pdf}", file=sys.stderr)
            except Exception as e:
                print(f"[ocr] ocrmypdf failed: {e}", file=sys.stderr)

    # Print quality report summary
    flagged = [p for p in pages_data if p.get("flag")]
    if flagged:
        print(f"\n[ocr] Quality report: {len(flagged)} page(s) flagged for review", file=sys.stderr)
        for p in flagged:
            print(f"  Page {p['n']}: conf={p.get('mean_conf', '?')}, flag={p['flag']}", file=sys.stderr)
        print("[ocr] Suggestion: re-run with --engine vision --pages "
              + ",".join(str(p["n"]) for p in flagged), file=sys.stderr)


# ── argparse ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr",
        description="Extract text and tables from PDFs and images using layered OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ocr scan.pdf --format md,json
  ocr photo.jpg --format md
  ocr doc.pdf --lang rus+eng --preprocess full --format all
  ocr report.pdf --probe
  ocr report.pdf --table-flavor lattice
  ocr report.pdf --no-tables
  ocr table.png --engine vision --vision-prompt-file table-prompt.txt
  ocr *.pdf --cache cache.json --format txt
        """,
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="PDF or image file(s)")
    p.add_argument(
        "--engine",
        choices=OCR_ENGINES,
        help="OCR backend (default: OCR_ENGINE or tesseract)",
    )
    p.add_argument("--probe", action="store_true",
                   help="Print NDJSON input probe results and exit")
    p.add_argument("--lang", default="auto",
                   help="Tesseract language code(s), e.g. rus+eng (default: auto via OSD)")
    p.add_argument(
        "--format",
        type=parse_formats,
        default=["md"],
        metavar="FORMAT[,FORMAT...]",
        help="Output formats: md,txt,json,all; comma-separated (default: md)",
    )
    p.add_argument("--out", default="",
                   help="Output file or directory (default: stdout)")
    p.add_argument("--dpi", type=int, default=0,
                   help="Rendering DPI (default: auto — 300 A4, 150 wide canvas)")
    p.add_argument("--preprocess", default="auto",
                   choices=["none", "basic", "enhanced", "full", "auto"],
                   help="Image preprocessing level (default: auto)")
    p.add_argument("--pages", default="",
                   help="Page range to process, e.g. 1-3,5 (default: all)")
    p.add_argument("--max-pages", type=int, default=0,
                   help="Maximum pages per file (default: all)")
    p.add_argument("--psm", type=int, default=DEFAULT_PSM,
                   help=f"Tesseract PSM (default: {DEFAULT_PSM}; 6 for single-block)")
    p.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF,
                   help=f"Confidence threshold for flagging pages (default: {DEFAULT_MIN_CONF})")
    p.add_argument("--cache", default="",
                   help="Cache file path (JSON) for skipping already-processed files")
    p.add_argument("--force", action="store_true",
                   help="Ignore cache and re-process all files")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Only process files with a real text layer; skip OCR")
    p.add_argument(
        "--tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract tables from readable text PDFs with Camelot (default: enabled)",
    )
    p.add_argument(
        "--table-flavor",
        choices=TABLE_FLAVORS,
        default="auto",
        help="Camelot parser for text-PDF tables (default: auto)",
    )
    p.add_argument(
        "--auto-escalate",
        type=parse_engines,
        help="Retry flagged pages through comma-separated engines",
    )
    p.add_argument("--no-cleanup", action="store_true",
                   help="Skip whitespace / ligature cleanup of OCR output")
    p.add_argument("--vision-api-url", default=None,
                   help="OpenAI-compatible base URL for --engine vision "
                        "(or OCR_VISION_API_URL)")
    p.add_argument("--vision-api-key", default=None,
                   help="API key for --engine vision (or OCR_VISION_API_KEY; required)")
    p.add_argument("--vision-model", default=None,
                   help="Model name for --engine vision "
                        "(or OCR_VISION_MODEL; required)")
    p.add_argument("--paddle-vl-server-url", default="",
                   help="Loopback MLX-VLM service URL for --engine paddleocr-vl-mlx")
    p.add_argument("--paddle-vl-model", default="",
                   help="PaddleOCR-VL model ID for --engine paddleocr-vl-mlx")
    prompt_group = p.add_mutually_exclusive_group()
    prompt_group.add_argument("--vision-prompt", default="",
                               help="Custom prompt for --engine vision")
    prompt_group.add_argument("--vision-prompt-file", type=Path,
                              help="UTF-8 file containing a custom vision prompt")
    p.add_argument("--searchable-pdf", default="",
                   help="Path for searchable PDF output (requires ocrmypdf)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging to stderr")
    return p


def _vision_prompt_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if ((args.vision_prompt or args.vision_prompt_file is not None)
            and args.engine != "vision"
            and "vision" not in (args.auto_escalate or ())):
        parser.error(
            "--vision-prompt and --vision-prompt-file require vision as the "
            "baseline or escalation engine"
        )
    if args.vision_prompt_file is None:
        return args.vision_prompt
    try:
        return args.vision_prompt_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"cannot read vision prompt file {args.vision_prompt_file}: {exc}")


def _validate_output_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if len(args.inputs) < 2:
        return

    if args.searchable_pdf:
        parser.error("--searchable-pdf supports only one input file")
    if not args.out and "json" in args.format:
        parser.error("JSON output with multiple inputs requires --out DIR")
    if not args.out:
        return

    stems: dict[str, list[str]] = {}
    for input_path in args.inputs:
        stem = Path(input_path).stem
        normalized_stem = unicodedata.normalize("NFC", stem).casefold()
        stems.setdefault(normalized_stem, []).append(stem)
    duplicates = sorted({
        stem
        for values in stems.values()
        if len(values) > 1
        for stem in values
    })
    if duplicates:
        parser.error(
            "--out with multiple inputs requires unique file stems; duplicate "
            f"stem(s): {', '.join(duplicates)}. Rename inputs or use separate output directories."
        )


def _validate_probe_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if not args.probe:
        return
    incompatible = []
    if args.engine is not None:
        incompatible.append("--engine")
    if args.out:
        incompatible.append("--out")
    if args.searchable_pdf:
        incompatible.append("--searchable-pdf")
    if args.auto_escalate:
        incompatible.append("--auto-escalate")
    if any((args.vision_api_url, args.vision_api_key, args.vision_model,
            args.paddle_vl_server_url, args.paddle_vl_model,
            args.vision_prompt, args.vision_prompt_file)):
        incompatible.append("engine options")
    if incompatible:
        parser.error(f"--probe cannot be combined with {', '.join(incompatible)}")


def _resolve_engine(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    engine = args.engine or os.environ.get("OCR_ENGINE", "tesseract")
    valid = set(OCR_ENGINES)
    if engine not in valid:
        parser.error(
            f"OCR_ENGINE has invalid value '{engine}'; choose from "
            f"{', '.join(sorted(valid))}"
        )
    return engine


def _resolve_auto_escalate(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    engines = args.auto_escalate
    if engines is None:
        raw = os.environ.get("OCR_AUTO_ESCALATE", "").strip()
        if not raw:
            return ()
        try:
            engines = parse_engines(raw)
        except argparse.ArgumentTypeError as exc:
            parser.error(f"OCR_AUTO_ESCALATE: {exc}")
    return tuple(engine for engine in engines if engine != args.engine)


def _resolve_vision_args(args: argparse.Namespace) -> None:
    """Fill omitted Vision CLI values from the OCR_VISION_* environment."""
    env_names = {
        "vision_api_url": "OCR_VISION_API_URL",
        "vision_api_key": "OCR_VISION_API_KEY",
        "vision_model": "OCR_VISION_MODEL",
    }
    for argument, env_name in env_names.items():
        if getattr(args, argument) is None:
            setattr(args, argument, os.environ.get(env_name, ""))


# ── entry point ───────────────────────────────────────────────────────────────

def run(argv: list[str] | None = None) -> None:
    global caps_global
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_probe_args(args, parser)
    if args.probe:
        caps = Caps(verbose=args.verbose, report=args.verbose)
        for input_path in args.inputs:
            if not os.path.exists(input_path):
                parser.error(f"file not found: {input_path}")
            try:
                result = probe_input(input_path, caps, args.verbose)
            except OcrError as exc:
                input_type = classify_input(input_path)
                print(json.dumps({
                    "path": os.path.abspath(input_path),
                    "input_type": input_type,
                    "status": "error",
                    "needs_ocr": False,
                    "reason": str(exc),
                }, ensure_ascii=False))
                raise
            print(json.dumps(result, ensure_ascii=False))
        return

    args.engine = _resolve_engine(args, parser)
    args.auto_escalate = _resolve_auto_escalate(args, parser)
    _validate_output_args(args, parser)
    vision_prompt = _vision_prompt_from_args(args, parser)
    _resolve_vision_args(args)

    options = RecognizeOptions(
        engine=args.engine,
        lang=args.lang,
        dpi=args.dpi,
        preprocess=args.preprocess,
        pages=args.pages,
        max_pages=args.max_pages,
        psm=args.psm,
        min_conf=args.min_conf,
        no_cleanup=args.no_cleanup,
        force=args.force,
        vision_api_url=args.vision_api_url,
        vision_api_key=args.vision_api_key,
        vision_model=args.vision_model,
        paddle_vl_server_url=args.paddle_vl_server_url,
        paddle_vl_model=args.paddle_vl_model,
        vision_prompt=vision_prompt,
        verbose=args.verbose,
        auto_escalate=args.auto_escalate,
        skip_ocr=args.skip_ocr,
        extract_tables=args.tables,
        table_flavor=args.table_flavor,
    )

    caps = Caps(verbose=args.verbose, report=args.verbose)
    caps_global = caps

    cache = Cache(args.cache if args.cache else None)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="ocr_skill_")
    tmpdir = tmp_ctx.name

    try:
        for input_path in args.inputs:
            if not os.path.exists(input_path):
                print(f"[ocr] WARNING: file not found: {input_path}", file=sys.stderr)
                continue

            _log(f"processing: {input_path}", args.verbose)
            t_start = time.time()

            pages_data = process_file(input_path, options, caps, cache, tmpdir)

            # Resolve effective lang for output meta (might have been auto-detected)
            effective_lang = args.lang
            if effective_lang == "auto" and pages_data:
                # Best we can do without re-running OSD
                effective_lang = "auto-detected"

            effective_dpi = args.dpi or 0

            elapsed = time.time() - t_start
            total_chars = sum(len(p.get("text", "")) for p in pages_data)
            _log(f"done: {len(pages_data)} pages, {total_chars} chars, {elapsed:.1f}s total",
                 args.verbose)
            write_outputs(pages_data, input_path, args, effective_lang, effective_dpi)
    finally:
        tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    try:
        run(argv)
    except OcrError as exc:
        print(f"[ocr] ERROR: {exc}", file=sys.stderr)
        return exc.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
