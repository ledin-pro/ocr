# pro-ledin-ocr

[![skills.sh](https://skills.sh/b/ledin-pro/ocr)](https://skills.sh/ledin-pro/ocr)

Layered OCR workhorse for scanned PDFs and images (PNG/JPG/TIFF/HEIC/WEBP).
Baseline OCR uses Poppler and Tesseract; EasyOCR, PaddleOCR, and automated vision
are opt-in engines.

- Import name: `pro.ledin.ocr`
- Console scripts: `ocr`, `peepshow-sink-ocr`
- Interactive vision handoff: `python -m pro.ledin.ocr.vision_handoff`
- PyPI: `pro-ledin-ocr`

## Install

```bash
pip install pro-ledin-ocr             # baseline
pip install "pro-ledin-ocr[easyocr]" # EasyOCR
pip install "pro-ledin-ocr[paddle]"  # PaddleOCR
pip install "pro-ledin-ocr[vision]"  # automated OpenAI-compatible vision
pip install "pro-ledin-ocr[all]"     # all optional Python engines/tools
```

Baseline PDF OCR also needs Poppler (`pdftoppm`, `pdftotext`, `pdfinfo`) and
Tesseract with required language data:

```bash
brew install poppler tesseract tesseract-lang       # macOS
sudo apt install poppler-utils tesseract-ocr-all    # Debian/Ubuntu
```

Windows and engine-specific setup: [`skills/ocr/references/engine-setup.md`](skills/ocr/references/engine-setup.md).

## CLI

```bash
ocr myfile.pdf --probe                         # triage only; NDJSON result
ocr myfile.pdf --format md,json                # multiple formats to stdout
ocr myfile.pdf --format md,txt,json --out results/
ocr scan.png --format md
ocr russian_doc.pdf --lang rus+eng --format md
ocr scan.pdf --preprocess full
ocr slides.pdf --engine vision --pages 9,12 \
  --vision-api-key "$KEY" --vision-model my-vision-model \
  --vision-prompt "Preserve tables and empty cells"
ocr scan.pdf --auto-escalate easyocr,vision \
  --vision-api-key "$KEY" --vision-model my-vision-model
```

Engine resolution is explicit: `--engine` overrides `OCR_ENGINE`, otherwise
`tesseract` is used. There is no `auto` engine. Valid engines are `tesseract`,
`easyocr`, `paddleocr`, and `vision`. `vision` is automated, headless extraction
through an OpenAI-compatible endpoint; former automated-engine alias is gone.

`--auto-escalate` overrides `OCR_AUTO_ESCALATE`. Both accept an ordered,
comma-separated chain such as `easyocr,vision`. OCR validates every dependency
before baseline work. Flagged pages run through every engine in chain, in order.
Any attempt failure stops run and prevents workflow-cache write. Without a
configured chain, quality report recommends targeted page rerun.

Interactive multimodal-agent handoff is separate and does not call vision API:

```bash
python -m pro.ledin.ocr.vision_handoff slides.pdf \
  --pages 9,12 --dpi 200 \
  --vision-prompt "Read visible text and preserve table structure"
```

Handoff accepts `--pages`, `--dpi`, `--vision-prompt`, and
`--vision-prompt-file`; it intentionally has no `--max-pages` option.

See `ocr --help` for complete flag reference.

## Output formats

`--format` accepts `md`, `txt`, `json`, comma-separated combinations, or `all`
(`md,txt,json`). Comma-format behavior introduced in `0.4.0` remains unchanged:
with multiple formats, `--out` is a directory and files use input stem. Multiple
inputs also use directory mode; duplicate stems are rejected. JSON output for
multiple inputs requires `--out`; `--searchable-pdf` accepts one input.

`--json-report` was removed in `0.4.0`. Use
`--format json --out report.json` for JSON-only output or
`--format md,json --out results/` for multiple files. One invocation cannot mix
stdout format with independently named JSON sidecar.

## Peepshow sink

`peepshow-sink-ocr` reads peepshow JSON from stdin, recognizes primary frames,
and atomically writes `<outputDir>/ocr.json`:

```bash
peepshow video.mp4 --sink ocr
peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'
```

Named sink configuration uses `PEEPSHOW_SINK_OCR_*` variables, including
`PEEPSHOW_SINK_OCR_AUTO_ESCALATE`. Automated `vision` works in sink mode with
`pro-ledin-ocr[vision]`, key, and model. Interactive handoff does not run inside
sink processes. See [`skills/ocr/references/peepshow-sinks.md`](skills/ocr/references/peepshow-sinks.md).

## Library

```python
from pro.ledin import ocr

pages = ocr.recognize(
    "scan.pdf",
    ocr.RecognizeOptions(
        engine="vision",
        vision_api_key="key",
        vision_model="model",
        vision_prompt="Preserve checkbox states and labels.",
    ),
)
markdown = ocr.to_markdown(pages, "scan.pdf")
```

`recognize()` never calls `sys.exit()`; catch `ocr.OcrError` for recoverable
failures such as unsupported input, missing dependencies, or vision config.

## Engine tiers

| Tier | Engine | Best for | Cost |
|---|---|---|---|
| 0 | pdftotext / PyMuPDF | Real text layers | Free, instant |
| 1 | tesseract (default) | Clean scans and typed text | Free |
| 2 | easyocr | Handwriting and degraded scans | Free, heavy |
| 2.5 | paddleocr | CJK, multilingual, angled text | Free, heavy |
| 3 | vision | Automated tables, charts, complex layouts | API cost |
| Handoff | `python -m pro.ledin.ocr.vision_handoff` | Interactive agent reading | Agent/model tokens |

Full docs: [`skills/ocr/SKILL.md`](skills/ocr/SKILL.md),
[`skills/ocr/references/engines.md`](skills/ocr/references/engines.md),
[`skills/ocr/references/engine-setup.md`](skills/ocr/references/engine-setup.md),
[`skills/ocr/references/peepshow-sinks.md`](skills/ocr/references/peepshow-sinks.md), and
[`skills/ocr/references/troubleshooting.md`](skills/ocr/references/troubleshooting.md).

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv build
```

## License

MIT
