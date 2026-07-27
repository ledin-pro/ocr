"""Render pages and print a manifest for interactive multimodal OCR."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from .core import (
    Caps,
    OcrError,
    _parse_page_range,
    auto_dpi,
    classify_input,
    probe_pdf,
    render_pages,
    vision_handoff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pro.ledin.ocr.vision_handoff",
        description="Render OCR pages and print paths plus a multimodal prompt.",
    )
    parser.add_argument("inputs", nargs="+", metavar="INPUT")
    parser.add_argument("--pages", default="", help="Page range, e.g. 1-3,5")
    parser.add_argument("--dpi", type=int, default=0, help="Rendering DPI (default: auto)")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--vision-prompt", default="")
    prompt_group.add_argument("--vision-prompt-file", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.vision_prompt_file is None:
        return args.vision_prompt
    try:
        return args.vision_prompt_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"cannot read vision prompt file {args.vision_prompt_file}: {exc}")


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = _prompt(args, parser)
    caps = Caps(verbose=args.verbose)

    for input_path in args.inputs:
        if not os.path.exists(input_path):
            parser.error(f"file not found: {input_path}")
        input_type = classify_input(input_path)
        if input_type == "unsupported":
            parser.error(f"unsupported file type: {input_path}")
        if input_type == "image":
            if args.pages:
                parser.error("--pages applies only to PDF inputs")
            vision_handoff([(1, os.path.abspath(input_path))], vision_prompt=prompt)
            continue

        probe = probe_pdf(input_path, caps, args.verbose)
        if probe["status"] == "blocked":
            raise OcrError(probe["reason"])
        try:
            pages = _parse_page_range(args.pages, probe["pages"]) if args.pages else None
        except (TypeError, ValueError):
            parser.error(f"invalid page range: {args.pages}")
        if pages == []:
            parser.error(f"page selection matched no pages for {input_path}")
        dpi = args.dpi or auto_dpi(input_path, caps)
        output_dir = tempfile.mkdtemp(prefix="ocr_vision_handoff_")
        print(f"[ocr] vision handoff images persist in: {output_dir}", file=sys.stderr)
        rendered = render_pages(input_path, dpi, pages, output_dir, caps, args.verbose)
        vision_handoff(rendered, vision_prompt=prompt)


def main(argv: list[str] | None = None) -> int:
    try:
        run(argv)
    except OcrError as exc:
        print(f"[ocr-vision-handoff] ERROR: {exc}", file=sys.stderr)
        return exc.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
