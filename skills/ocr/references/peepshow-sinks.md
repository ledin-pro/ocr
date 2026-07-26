# Peepshow OCR Sink

`peepshow-sink-ocr` is a peepshow-compatible sink executable included in the
`pro-ledin-ocr` package. It reads one peepshow `--emit json` payload from stdin,
recognizes each primary frame path, and writes one JSON sidecar.

## Dependencies

The interface adds no mandatory Python dependency and does not import the npm
`peepshow` package. Peepshow discovers `peepshow-sink-ocr` on `PATH`.

Engine requirements remain unchanged:

| Engine | Requirement |
|---|---|
| `auto`, `tesseract` | Tesseract binary and requested language packs |
| `easyocr` | `pro-ledin-ocr[easyocr]` |
| `paddleocr` | `pro-ledin-ocr[paddle]` |
| `vision-api` | `pro-ledin-ocr[vision]` plus endpoint key/model |
| `vision` | Unsupported because it requires interactive agent handoff |

Peepshow supplies static image frames, so Poppler is not needed by this sink
path.

## Usage

Named sink with defaults:

```bash
peepshow video.mp4 --sink ocr
```

Explicit command and flags:

```bash
peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'
```

Named sink with `vision-api` configuration:

```bash
export PEEPSHOW_SINK_OCR_ENGINE=vision-api
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
| `--engine` | `PEEPSHOW_SINK_OCR_ENGINE` | `auto` |
| `--lang` | `PEEPSHOW_SINK_OCR_LANG` | `auto` |
| `--dpi` | `PEEPSHOW_SINK_OCR_DPI` | `0` |
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

Prompt text and prompt file are mutually exclusive. Keep API keys in the
environment rather than shell history or process arguments.

## Input Contract

The sink requires peepshow's core payload fields:

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

`outputDir` must be an existing absolute directory. At least one frame path
must name a readable file. Optional thumbnails are ignored; only primary
`frames[].path` files are recognized.

## Output Contract

Default output is written atomically to `<outputDir>/ocr.json`:

```json
{
  "schemaVersion": 1,
  "packageVersion": "0.3.0",
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

Original frame order is preserved. Timestamp metadata is copied when present.
The sink does not modify `manifest.json`, frame files, or stdout.

## Failure Semantics

- Empty/malformed stdin, invalid payload, missing frames, unsupported
  interactive `vision`, or OCR failure produces stderr error and non-zero exit.
- Peepshow treats sink failure as a warning and keeps its successful extraction.
- Result is written only after every frame succeeds. Atomic replacement avoids
  partially written `ocr.json`.

## Privacy

- Frame paths and recognized text may contain sensitive information.
- `vision-api` sends every selected frame to configured external endpoint.
- API keys and custom prompt text are not written to `ocr.json`.
- Protect peepshow output directory and review retention/backup behavior.
