#!/usr/bin/env python3
"""Command-line interface for the `pro.ledin.ocr` package (console script: ocr).

The recognition logic lives in `core`; this module owns only argument parsing
and output writing (formats, files, searchable PDF, quality report).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from .core import (
    Cache,
    Caps,
    DEFAULT_MIN_CONF,
    DEFAULT_PSM,
    OcrError,
    RecognizeOptions,
    _log,
    process_file,
    to_json,
    to_markdown,
    to_text,
)

# ── output writing ────────────────────────────────────────────────────────────

caps_global: Caps  # set in main()


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

    formats = args.format.split(",") if "," in args.format else [args.format]
    if "all" in formats:
        formats = ["md", "txt", "json"]

    for fmt in formats:
        if fmt == "md":
            content = to_markdown(pages_data, filename)
        elif fmt == "txt":
            content = to_text(pages_data)
        elif fmt == "json":
            content = to_json(pages_data, meta)
        else:
            continue

        if args.out:
            out_path = args.out
            if os.path.isdir(out_path) or "all" in [args.format] or len(formats) > 1:
                os.makedirs(out_path, exist_ok=True)
                stem = Path(input_path).stem
                out_file = os.path.join(out_path, f"{stem}.{fmt}")
            else:
                out_file = out_path
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[ocr] wrote {fmt} → {out_file}", file=sys.stderr)
        else:
            if len(formats) > 1:
                print(f"\n{'='*60}\n[{fmt.upper()}]\n{'='*60}", flush=True)
            print(content, flush=True)

    # Optional JSON report
    if args.json_report:
        report_content = to_json(pages_data, meta)
        with open(args.json_report, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[ocr] quality report → {args.json_report}", file=sys.stderr)

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
        description="Extract text from scanned PDFs and images using layered OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ocr scan.pdf --format all
  ocr photo.jpg --format md
  ocr doc.pdf --lang rus+eng --preprocess full --format all
  ocr slides.pdf --engine vision --pages 9,12
  ocr *.pdf --cache cache.json --format txt
        """,
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="PDF or image file(s)")
    p.add_argument("--engine", default="auto",
                   choices=["auto", "tesseract", "easyocr", "paddleocr", "vision", "vision-api"],
                   help="OCR engine (default: auto)")
    p.add_argument("--lang", default="auto",
                   help="Tesseract language code(s), e.g. rus+eng (default: auto via OSD)")
    p.add_argument("--format", default="md",
                   help="Output format: md|txt|json|all (default: md)")
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
    p.add_argument("--no-cleanup", action="store_true",
                   help="Skip whitespace / ligature cleanup of OCR output")
    p.add_argument("--vision-api-url", default="",
                   help="OpenAI-compatible base URL for --engine vision-api")
    p.add_argument("--vision-api-key", default="",
                   help="API key for --engine vision-api (required; env vars are not read)")
    p.add_argument("--vision-model", default="",
                   help="Model name for --engine vision-api (required; no default)")
    p.add_argument("--searchable-pdf", default="",
                   help="Path for searchable PDF output (requires ocrmypdf)")
    p.add_argument("--json-report", default="",
                   help="Path to write JSON quality report")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging to stderr")
    return p


# ── entry point ───────────────────────────────────────────────────────────────

def run(argv: list[str] | None = None) -> None:
    global caps_global
    parser = build_parser()
    args = parser.parse_args(argv)

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
        verbose=args.verbose,
    )

    caps = Caps(verbose=args.verbose)
    caps_global = caps

    # Validate binary requirements for the chosen engine
    if args.engine in ("auto", "tesseract"):
        caps.require_ocr()

    cache = Cache(args.cache if args.cache else None)

    # Vision handoff must persist rendered PNGs after this process exits so the
    # calling agent (Claude, GPT, or any multimodal model) can read them.
    # Other engines use a throwaway temp dir.
    if args.engine == "vision":
        tmpdir = tempfile.mkdtemp(prefix="ocr_skill_vision_")
        print(f"[ocr] vision PNGs will persist in: {tmpdir}", file=sys.stderr)
        cleanup_tmp = False
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="ocr_skill_")
        tmpdir = tmp_ctx.name
        cleanup_tmp = True

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

            write_outputs(pages_data, input_path, args, effective_lang, effective_dpi)

            elapsed = time.time() - t_start
            total_chars = sum(len(p.get("text", "")) for p in pages_data)
            _log(f"done: {len(pages_data)} pages, {total_chars} chars, {elapsed:.1f}s total",
                 args.verbose)
    finally:
        if cleanup_tmp:
            tmp_ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    try:
        run(argv)
    except OcrError as exc:
        print(f"[ocr] ERROR: {exc}", file=sys.stderr)
        return exc.code
    return 0


def probe_main(argv: list[str] | None = None) -> int:
    """Console-script entry point (`ocr-probe`): run the bundled probe.sh helper.

    Triages a single file for OCR need and prints one-line JSON to stdout.
    """
    import subprocess

    from .core import probe_script_path

    args = list(sys.argv[1:] if argv is None else argv)
    return subprocess.run(["bash", probe_script_path(), *args]).returncode


if __name__ == "__main__":
    sys.exit(main())
