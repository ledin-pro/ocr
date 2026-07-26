# pro-ledin-ocr

[![skills.sh](https://skills.sh/b/ledin-pro/ocr)](https://skills.sh/ledin-pro/ocr)

Layered OCR workhorse: extract text from scanned PDFs and images (PNG/JPG/TIFF/
HEIC/WEBP) using a tiered engine stack. The baseline path (poppler + tesseract)
needs zero extra Python installs; heavier engines are opt-in extras.

- Import name: `pro.ledin.ocr`
- Console scripts: `ocr`, `ocr-probe`
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
ocr myfile.pdf --format all                   # md + txt + json
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
`skills/ocr/references/troubleshooting.md`.

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest
uv build
```

## License

MIT
