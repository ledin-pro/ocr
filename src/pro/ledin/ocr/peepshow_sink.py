"""Peepshow sink entry point for OCRing extracted video frames."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .core import (
    DEFAULT_MIN_CONF,
    DEFAULT_PSM,
    EXIT_BAD_ARGS,
    OcrError,
    RecognizeOptions,
    __version__,
    recognize,
    to_markdown,
    to_text,
)

ENGINE_CHOICES = ("auto", "tesseract", "easyocr", "paddleocr", "vision", "vision-api")
PREPROCESS_CHOICES = ("none", "basic", "enhanced", "full", "auto")
EXIT_RUNTIME = 5


@dataclass(frozen=True)
class SinkConfig:
    options: RecognizeOptions
    output: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peepshow-sink-ocr",
        description="OCR peepshow frame paths from a JSON payload on stdin.",
    )
    parser.add_argument("--engine", choices=ENGINE_CHOICES)
    parser.add_argument("--lang")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--preprocess", choices=PREPROCESS_CHOICES)
    parser.add_argument("--psm", type=int)
    parser.add_argument("--min-conf", type=float)
    parser.add_argument("--vision-api-url")
    parser.add_argument("--vision-api-key")
    parser.add_argument("--vision-model")
    parser.add_argument("--timeout", type=float)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--vision-prompt")
    prompt_group.add_argument("--vision-prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _configured(
    args: argparse.Namespace,
    attr: str,
    env: Mapping[str, str],
    env_name: str,
    default: Any,
    cast: Callable[[str], Any] | None = None,
) -> Any:
    cli_value = getattr(args, attr)
    if cli_value is not None:
        return cli_value
    raw = env.get(env_name, "")
    if not raw:
        return default
    if cast is None:
        return raw
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise OcrError(f"{env_name} has invalid value", EXIT_BAD_ARGS) from exc


def _read_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OcrError(f"cannot read vision prompt file {path}: {exc}", EXIT_BAD_ARGS) from exc


def resolve_config(
    args: argparse.Namespace,
    env: Mapping[str, str] | None = None,
) -> SinkConfig:
    env = os.environ if env is None else env
    engine = _configured(args, "engine", env, "PEEPSHOW_SINK_OCR_ENGINE", "auto")
    if engine not in ENGINE_CHOICES:
        raise OcrError("PEEPSHOW_SINK_OCR_ENGINE has invalid value", EXIT_BAD_ARGS)
    if engine == "vision":
        raise OcrError(
            "engine='vision' requires an interactive agent; use vision-api or a local engine",
            EXIT_BAD_ARGS,
        )

    preprocess = _configured(
        args, "preprocess", env, "PEEPSHOW_SINK_OCR_PREPROCESS", "auto"
    )
    if preprocess not in PREPROCESS_CHOICES:
        raise OcrError("PEEPSHOW_SINK_OCR_PREPROCESS has invalid value", EXIT_BAD_ARGS)

    if args.vision_prompt is not None:
        vision_prompt = args.vision_prompt
    elif args.vision_prompt_file is not None:
        vision_prompt = _read_prompt(args.vision_prompt_file)
    else:
        env_prompt = env.get("PEEPSHOW_SINK_OCR_VISION_PROMPT", "")
        env_prompt_file = env.get("PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE", "")
        if env_prompt and env_prompt_file:
            raise OcrError(
                "set only one of PEEPSHOW_SINK_OCR_VISION_PROMPT and "
                "PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE",
                EXIT_BAD_ARGS,
            )
        vision_prompt = _read_prompt(Path(env_prompt_file)) if env_prompt_file else env_prompt

    output = args.output
    if output is None and env.get("PEEPSHOW_SINK_OCR_OUTPUT"):
        output = Path(env["PEEPSHOW_SINK_OCR_OUTPUT"])

    timeout = _configured(
        args, "timeout", env, "PEEPSHOW_SINK_OCR_TIMEOUT", None, float
    )
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise OcrError("PEEPSHOW_SINK_OCR_TIMEOUT must be a positive number", EXIT_BAD_ARGS)

    options = RecognizeOptions(
        engine=engine,
        lang=_configured(args, "lang", env, "PEEPSHOW_SINK_OCR_LANG", "auto"),
        dpi=_configured(args, "dpi", env, "PEEPSHOW_SINK_OCR_DPI", 0, int),
        preprocess=preprocess,
        psm=_configured(args, "psm", env, "PEEPSHOW_SINK_OCR_PSM", DEFAULT_PSM, int),
        min_conf=_configured(
            args, "min_conf", env, "PEEPSHOW_SINK_OCR_MIN_CONF", DEFAULT_MIN_CONF, float
        ),
        vision_api_url=_configured(
            args, "vision_api_url", env, "PEEPSHOW_SINK_OCR_VISION_API_URL", ""
        ),
        vision_api_key=_configured(
            args, "vision_api_key", env, "PEEPSHOW_SINK_OCR_VISION_API_KEY", ""
        ),
        vision_model=_configured(
            args, "vision_model", env, "PEEPSHOW_SINK_OCR_VISION_MODEL", ""
        ),
        vision_prompt=vision_prompt,
        timeout=timeout,
    )
    return SinkConfig(options=options, output=output)


def read_payload(stream: TextIO) -> dict[str, Any]:
    raw = stream.read()
    if not raw.strip():
        raise OcrError("stdin was empty; expected peepshow JSON payload", EXIT_BAD_ARGS)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OcrError(f"malformed JSON on stdin: {exc.msg}", EXIT_BAD_ARGS) from exc
    if not isinstance(payload, dict):
        raise OcrError("peepshow payload must be a JSON object", EXIT_BAD_ARGS)
    return payload


def validate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = payload.get("outputDir")
    if not isinstance(output_dir, str) or not output_dir or not os.path.isabs(output_dir):
        raise OcrError("payload.outputDir must be an absolute path", EXIT_BAD_ARGS)
    if not os.path.isdir(output_dir):
        raise OcrError("payload.outputDir does not exist", EXIT_BAD_ARGS)
    if payload.get("strategy") not in ("scene", "fps"):
        raise OcrError("payload.strategy must be 'scene' or 'fps'", EXIT_BAD_ARGS)
    if not isinstance(payload.get("video"), dict):
        raise OcrError("payload.video must be an object", EXIT_BAD_ARGS)
    if not isinstance(payload.get("extraction"), dict):
        raise OcrError("payload.extraction must be an object", EXIT_BAD_ARGS)

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise OcrError("payload.frames must contain at least one frame", EXIT_BAD_ARGS)
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise OcrError(f"payload.frames[{index}] must be an object", EXIT_BAD_ARGS)
        path = frame.get("path")
        if not isinstance(path, str) or not path:
            raise OcrError(f"payload.frames[{index}].path must be a string", EXIT_BAD_ARGS)
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            raise OcrError(f"frame is not readable: {path}", EXIT_BAD_ARGS)
    return frames


def process_payload(
    payload: dict[str, Any],
    config: SinkConfig,
    recognize_func: Callable[[str, RecognizeOptions], list[dict]] | None = None,
) -> tuple[dict[str, Any], Path]:
    frames = validate_payload(payload)
    recognize_func = recognize if recognize_func is None else recognize_func
    frame_results: list[dict[str, Any]] = []

    for index, frame in enumerate(frames):
        path = frame["path"]
        pages = recognize_func(path, config.options)
        result: dict[str, Any] = {
            "index": index,
            "path": path,
            "bytes": frame.get("bytes", os.path.getsize(path)),
            "pages": pages,
            "text": to_text(pages),
            "markdown": to_markdown(pages, os.path.basename(path)),
        }
        for key in ("timestamp", "timestampSeconds", "timeSeconds", "ptsSeconds"):
            if key in frame:
                result[key] = frame[key]
        frame_results.append(result)

    combined_text = "\n\n".join(result["text"] for result in frame_results)
    combined_markdown = "\n\n".join(
        f"## Frame {result['index'] + 1}\n\n{result['markdown']}"
        for result in frame_results
    )
    output = config.output or Path(payload["outputDir"]) / "ocr.json"
    document = {
        "schemaVersion": 1,
        "packageVersion": __version__,
        "source": "peepshow",
        "peepshowOutputDir": payload["outputDir"],
        "strategy": payload["strategy"],
        "engine": config.options.engine,
        "lang": config.options.lang,
        "frames": frame_results,
        "text": combined_text,
        "markdown": combined_markdown,
    }
    return document, output


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        config = resolve_config(args)
        payload = read_payload(sys.stdin)
        document, output = process_payload(payload, config)
        write_json_atomic(output, document)
    except OcrError as exc:
        print(f"[peepshow-sink-ocr] ERROR: {exc}", file=sys.stderr)
        return exc.code
    except OSError:
        print("[peepshow-sink-ocr] ERROR: filesystem operation failed", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception:
        print("[peepshow-sink-ocr] ERROR: recognition failed unexpectedly", file=sys.stderr)
        return EXIT_RUNTIME
    return 0


if __name__ == "__main__":
    sys.exit(main())
