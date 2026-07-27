# Peepshow OCR Sink

`peepshow-sink-ocr` is peepshow-compatible console script included in
`pro-ledin-ocr`. It reads one `--emit json` payload from stdin, recognizes each
primary frame, and atomically writes one JSON sidecar.

Current console scripts are `ocr` and `peepshow-sink-ocr`. Interactive handoff
is Python module `python -m pro.ledin.ocr.vision_handoff`, not sink engine.

## Dependencies

| Engine | Requirement |
|---|---|
| `tesseract` | Tesseract binary and requested language data |
| `easyocr` | `pro-ledin-ocr[easyocr]` and downloaded models |
| `paddleocr` | `pro-ledin-ocr[paddle]`, matching PaddlePaddle runtime, models |
| `vision` | `pro-ledin-ocr[vision]`, endpoint key and model |

Default sink engine is `tesseract`. There is no `auto` engine. Poppler is not
needed because peepshow supplies image frames. Missing baseline or escalation
dependency stops sink before recognition; follow approval workflow in
`engine-setup.md`.

## Usage

```bash
peepshow video.mp4 --sink ocr

peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'

peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --auto-escalate easyocr'
```

Named sink with automated vision escalation:

```bash
export PEEPSHOW_SINK_OCR_ENGINE=tesseract
export PEEPSHOW_SINK_OCR_AUTO_ESCALATE=easyocr,vision
export PEEPSHOW_SINK_OCR_VISION_API_URL=https://api.example.com/v1
export PEEPSHOW_SINK_OCR_VISION_API_KEY="$KEY"
export PEEPSHOW_SINK_OCR_VISION_MODEL=my-vision-model
export PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE=/absolute/path/to/prompt.txt
peepshow video.mp4 --sink ocr
```

## Configuration

Command flags override environment values.

| Flag | Environment variable | Default |
|---|---|---|
| `--engine` | `PEEPSHOW_SINK_OCR_ENGINE` | `tesseract` |
| `--auto-escalate` | `PEEPSHOW_SINK_OCR_AUTO_ESCALATE` | none |
| `--lang` | `PEEPSHOW_SINK_OCR_LANG` | `auto` |
| `--dpi` | `PEEPSHOW_SINK_OCR_DPI` | `0` (automatic) |
| `--preprocess` | `PEEPSHOW_SINK_OCR_PREPROCESS` | `auto` |
| `--psm` | `PEEPSHOW_SINK_OCR_PSM` | `3` |
| `--min-conf` | `PEEPSHOW_SINK_OCR_MIN_CONF` | `60` |
| `--vision-api-url` | `PEEPSHOW_SINK_OCR_VISION_API_URL` | empty |
| `--vision-api-key` | `PEEPSHOW_SINK_OCR_VISION_API_KEY` | empty |
| `--vision-model` | `PEEPSHOW_SINK_OCR_VISION_MODEL` | empty |
| `--timeout` | `PEEPSHOW_SINK_OCR_TIMEOUT` | SDK default |
| `--vision-prompt` | `PEEPSHOW_SINK_OCR_VISION_PROMPT` | built-in prompt |
| `--vision-prompt-file` | `PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE` | empty |
| `--output` | `PEEPSHOW_SINK_OCR_OUTPUT` | `<outputDir>/ocr.json` |

Escalation value is ordered comma-separated engine chain. Duplicate entries and
baseline engine are omitted. All dependencies are checked before baseline.
Flagged frames run through every engine in chain. Any attempt failure aborts
sink and prevents partial sidecar write.

Prompt text and prompt file are mutually exclusive. Keep API keys in environment
rather than shell history or process arguments.

## Input contract

```json
{
  "outputDir": "/tmp/peepshow-run",
  "strategy": "scene",
  "frames": [
    {"path": "/tmp/peepshow-run/frame_0001.jpg", "bytes": 42321}
  ],
  "video": {},
  "extraction": {}
}
```

`outputDir` must be existing absolute directory. At least one frame path must
name readable file. Thumbnails are ignored; only `frames[].path` is recognized.

## Output contract

Default output is atomically written to `<outputDir>/ocr.json`:

```json
{
  "schemaVersion": 1,
  "packageVersion": "0.5.0",
  "source": "peepshow",
  "peepshowOutputDir": "/tmp/peepshow-run",
  "strategy": "scene",
  "engine": "tesseract",
  "lang": "auto",
  "frames": [
    {
      "index": 0,
      "path": "/tmp/peepshow-run/frame_0001.jpg",
      "bytes": 42321,
      "pages": [],
      "text": "recognized frame text",
      "markdown": "# frame_0001.jpg"
    }
  ],
  "text": "combined frame text",
  "markdown": "combined frame markdown"
}
```

Frame order and available timestamps are preserved. Sink does not modify
`manifest.json`, frames, or stdout.

## Failure semantics

- Empty/malformed stdin, invalid payload, missing frames, invalid engine,
  missing dependency, or OCR attempt failure writes stderr and exits non-zero.
- Peepshow keeps successful extraction and treats sink failure as warning.
- Result writes only after every frame succeeds; atomic replacement avoids
  partial `ocr.json`.

## Privacy

- Frame paths and recognized text may contain sensitive information.
- Automated `vision` sends selected frames to configured external endpoint.
- API keys and prompt text are not written to `ocr.json`.
- Protect output directory and review retention/backup behavior.
