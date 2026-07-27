# pro-ledin-ocr

[![skills.sh](https://skills.sh/b/ledin-pro/ocr)](https://skills.sh/ledin-pro/ocr)

Layered OCR workhorse: extract text from scanned PDFs and images (PNG/JPG/TIFF/
HEIC/WEBP) using a tiered engine stack. The baseline path (poppler + tesseract)
needs zero extra Python installs; heavier engines are opt-in extras.

- Import name: `pro.ledin.ocr`
- Console scripts: `ocr`, `ocr-probe`, `peepshow-sink-ocr`
- PyPI: `pro-ledin-ocr`

## Install

```bash
pip install pro-ledin-ocr            # baseline
pip install "pro-ledin-ocr[vision]"  # + OpenAI-compatible vision-api engine
pip install "pro-ledin-ocr[all]"     # + pymupdf, opencv, easyocr, paddleocr
```

System binaries required for the local path: `poppler` (pdftoppm, pdftotext,
pdfinfo) and `tesseract` (with language packs).

```bash
brew install poppler tesseract tesseract-lang      # macOS
sudo apt install poppler-utils tesseract-ocr-all   # Debian/Ubuntu
```

## CLI

```bash
ocr-probe myfile.pdf                          # triage: does it need OCR?
ocr myfile.pdf --format md,json               # multiple formats to stdout
ocr myfile.pdf --format md,txt,json --out results/  # one file per format
ocr scan.png --format md
ocr russian_doc.pdf --lang rus+eng --format md
ocr scan.pdf --preprocess full                # deskew + denoise
ocr slides.pdf --engine vision --pages 9,12   # hand pages to a multimodal agent
ocr table.png --engine vision \
  --vision-prompt "Extract only table rows and preserve empty cells"
ocr slides.pdf --engine vision-api \
  --vision-api-url https://api.example.com/v1 \
  --vision-api-key "$KEY" --vision-model my-vision-model \
  --vision-prompt-file prompts/faithful-ocr.txt
```

See `ocr --help` for the full flag reference.

`--format` accepts `md`, `txt`, `json`, or comma-separated combinations. `all`
remains shorthand for `md,txt,json`. With multiple formats, `--out` is treated
as a directory and files are named from the input stem. Multiple inputs also
use directory mode; duplicate input stems are rejected to prevent overwrites.
JSON output for multiple inputs requires `--out`, and `--searchable-pdf`
accepts only one input.

`--json-report` was removed in `0.4.0`. Use `--format json --out report.json`
for JSON-only output or `--format md,json --out results/` for multiple files.
One invocation no longer mixes a stdout format with an independently named JSON
sidecar.

## Peepshow sink

`peepshow-sink-ocr` reads peepshow's JSON payload from stdin, recognizes each
primary frame, and atomically writes `<outputDir>/ocr.json`:

```bash
peepshow video.mp4 --sink ocr
peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'
```

No extra Python dependency is required for the sink interface. Peepshow only
needs the installed `peepshow-sink-ocr` executable on `PATH`. The default local
engine still requires Tesseract; `vision-api` requires
`pip install "pro-ledin-ocr[vision]"`.

Configure named sink runs through `PEEPSHOW_SINK_OCR_*` variables:

```bash
export PEEPSHOW_SINK_OCR_ENGINE=vision-api
export PEEPSHOW_SINK_OCR_VISION_API_URL=https://api.example.com/v1
export PEEPSHOW_SINK_OCR_VISION_API_KEY="$KEY"
export PEEPSHOW_SINK_OCR_VISION_MODEL=my-vision-model
export PEEPSHOW_SINK_OCR_TIMEOUT=120
peepshow video.mp4 --sink ocr
```

The sink never changes peepshow's manifest or frames and prints no OCR content
to stdout. `vision-api` sends frame images to the configured external endpoint.
Protect both API credentials and output directories containing recognized text.

## Library

```python
from pro.ledin import ocr

pages = ocr.recognize(
    "scan.pdf",
    ocr.RecognizeOptions(
        engine="vision-api",
        vision_api_key="key",
        vision_model="model",
        vision_prompt="Preserve checkbox states and labels.",
    ),
)
markdown = ocr.to_markdown(pages, "scan.pdf")
```

`recognize()` never calls `sys.exit()`; catch `ocr.OcrError` for recoverable
failures (unsupported input, missing binaries/packages, vision-api config).

## Engine tiers

| Tier | Engine | Best for | Cost |
|------|--------|----------|------|
| 0 | pdftotext / PyMuPDF | Real text layers | Free, instant |
| 1 | tesseract (default) | Clean scans, typed text, 160+ languages | Free |
| 2 | easyocr | Handwriting, degraded scans | Free, heavy |
| 2.5 | paddleocr | CJK, multilingual, angled text | Free |
| 3 | vision (agent reads PNGs) | Tables, charts, complex layouts | Agent tokens |
| 3.5 | vision-api (OpenAI-compatible) | Headless batch, complex layouts | API cost |

Full docs: `skills/ocr/SKILL.md`, `skills/ocr/references/engines.md`,
`skills/ocr/references/peepshow-sinks.md`, and
`skills/ocr/references/troubleshooting.md`.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv build
```

## License

MIT
