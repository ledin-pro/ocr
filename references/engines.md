# OCR Engines Reference

## Engine selection table

| Tier | Engine | Trigger condition | Languages | Speed | Cost | Requires |
|------|--------|-------------------|-----------|-------|------|----------|
| 0 | Text layer (pdftotext / PyMuPDF) | probe.sh → needs_ocr=false | Whatever is embedded | Instant | Free | poppler (present) |
| 1 | tesseract (default) | Clean rasterized text, typed docs | 160+ incl. rus, eng, chi_sim | ~3–4 s/page @ 300 DPI | Free | tesseract binary (present) |
| 2 | pytesseract wrapper | Same as tesseract; adds TSV confidence | Same | Same | Free | `uv run --with pytesseract` |
| 3 | easyocr | Handwriting, degraded/blurry scans | ru, en (and many others) | ~8–20 s/page (CPU) | Free | `uv run --with easyocr` + 2 GB models |
| 3.5 | paddleocr (opt-in) | CJK, multilingual, angled/rotated text | 100+ incl. ch, japan, korean, ru, en | ~2–10 s/page (CPU) | Free | `uv run --with paddleocr,paddlepaddle` + models on first run |
| 4 | Vision (agent reads PNGs) | Tables, charts, forms, complex layouts | Any | Agent latency | Agent tokens | None — agent-native |
| 5 | Vision API (OpenAI-compatible) | Headless batch, tables, complex layouts | Any | Fast | API cost | Key + model + `uv run --with openai` |
| 6 | Cloud APIs | High-volume, regulated, max accuracy | Any | Fast | Paid | Key + SDK |

Measured baseline on clean rasterized Russian financial doc:
- tesseract `rus+eng`, 300 DPI: **mean_conf ≈ 80, 464 words, ~3.8 s/page**

---

## Escalation thresholds

Default `--min-conf 60`. The system flags a page for vision escalation when:
- Page mean confidence < `--min-conf` (likely noisy, rotated, or handwritten)
- Detected grid of words with close horizontal alignment (table heuristic)
- Tesseract returns very few words relative to image complexity (sparse layout)

You can force escalation for specific pages: `--engine vision --pages 5,9`.

---

## Language auto-detection (OSD)

Tesseract's orientation and script detection (`--psm 0`) runs on the first page
before full OCR. It identifies the script family and drives language selection.

### Script → tesseract language map

| OSD script | Languages used | Notes |
|------------|---------------|-------|
| Cyrillic | `rus+eng` | Covers Russian + embedded Latin terms |
| Latin | `eng` | Falls back if no other Latin lang specified |
| Han | `chi_sim+eng` | Simplified Chinese; use `chi_tra` for Traditional |
| Arabic | `ara+eng` | |
| Devanagari | `hin+eng` | Hindi; add `san` for Sanskrit |
| Bengali | `ben+eng` | |
| Korean | `kor+eng` | |
| Japanese | `jpn+eng` | |
| Greek | `ell+eng` | |
| Hebrew | `heb+eng` | |
| Unknown / low confidence | `eng` | Safe fallback; warn user |

`+eng` is always appended so embedded Latin abbreviations, numbers, and
branded terms (EBITDA, BBE, AI) are read correctly alongside non-Latin scripts.

Override auto-detection with `--lang rus+eng` (or any tesseract lang codes).

### Installing language packs

```bash
# macOS — all 160+ languages at once
brew install tesseract-lang

# macOS — individual pack
brew install tesseract
# then download .traineddata manually to $(brew --prefix)/share/tessdata/

# Ubuntu / Debian
sudo apt install tesseract-ocr-rus   # Russian
sudo apt install tesseract-ocr-all   # everything

# List installed languages
tesseract --list-langs
```

---

## DPI guidance

Higher DPI = better accuracy, larger images, more memory, slower.

| Page type | Recommended DPI | Reasoning |
|-----------|----------------|-----------|
| A4 / Letter (595×842 pt, 1240×841 pt) | **300** | 300 DPI gives ~2480×3508 px — tesseract sweet spot |
| Large canvas slides (1920×1080 pt) | **150** | 1920 pt @ 150 ≈ 4000 px wide — sufficient; 300 would be 8000+ px |
| Small / thumbnail scans (< 800 px wide) | **auto-upscale ×2** | Minimum ~1400 px width for tesseract; Pillow LANCZOS |
| Phone photos of documents | **as-is** (already high-res) | Usually 3000+ px — no upscaling needed |
| Fax / low-res (< 150 DPI equivalent) | **full preprocess** | Denoise + adaptive threshold help most |

`--dpi auto` selects based on the page rectangle from `pdfinfo` / PyMuPDF.

---

## Preprocessing levels

Preprocessing converts the rendered page image into a form tesseract handles
better. More preprocessing = slower but better for messy scans; can *hurt* on
clean digital renders (over-thresholding removes fine detail).

### none
Raw rendered image, no modification. Use when the source is a clean digital
render you trust (e.g. a good Word/PowerPoint export).

### basic (Pillow only, no OpenCV required)
- Convert to grayscale
- Boost contrast (factor 1.3)
- Sharpen
- Upscale if width < 1400 px (LANCZOS ×2)

Good default for clean-ish scans and digital rasters.

### enhanced (requires `opencv-python numpy`)
Everything in basic, plus:
- `fastNlMeansDenoising` — removes scanner noise, JPEG artifacts, grain
- `adaptiveThreshold (GAUSSIAN_C, blockSize=31, C=10)` — binarizes with local
  contrast adjustment, handles uneven lighting across the page

Best for: scanned documents, faxes, photocopied pages, documents photographed
under uneven lighting.

**Warning:** adaptive threshold on a clean digital render can introduce false
binarization artifacts. Use `basic` or `none` for clean PDFs.

### full (requires `opencv-python numpy`)
Everything in enhanced, plus:
- **Deskew** — detects page rotation via `minAreaRect` on binarized foreground
  pixels; corrects up to ±45°; skips correction if angle < 0.5°
  (algorithm: `cv2.minAreaRect → warpAffine(INTER_CUBIC, BORDER_REPLICATE)`)

Best for: phone photos of documents, flatbed scans with misalignment, old
archives with accumulated skew.

### auto (default)
- Clean digital render (probe reports low image coverage, no full-page masks) → `basic`
- Scan-like (full-page image masks detected) → `enhanced`
- Detected significant skew (tesseract OSD reports rotation ≠ 0) → `full`

### Installing OpenCV for enhanced/full

```bash
# Temporary (used just for this run)
uv run --with opencv-python,numpy python3 scripts/ocr.py FILE --preprocess enhanced

# Permanent in a project venv
uv pip install opencv-python numpy
```

---

## PaddleOCR tier (opt-in)

PaddleOCR is a strong multilingual engine (100+ languages) that excels at CJK
scripts (Chinese/Japanese/Korean), dense layouts, and angled/rotated text via
its built-in angle classifier. It is **opt-in only** — `--engine auto` never
selects it. Reach for it when tesseract struggles on CJK or mixed-script pages.

```bash
uv run --with paddleocr,paddlepaddle python3 scripts/ocr.py doc.png --engine paddleocr
# First run downloads detection/recognition/angle models (cached afterward).
```

### API note (PaddleOCR 3.x only)

The engine targets the PaddleOCR **3.x** API: `PaddleOCR(...).predict(path)`,
which returns result items exposing `rec_texts`, `rec_scores`, and `rec_polys`.
It does **not** support the 2.x `.ocr()` list-of-lists format. The reader is
constructed once per language and reused across pages.

### Language codes

`--lang` accepts tesseract codes (`auto` via OSD, or e.g. `rus+eng`). The
primary code is mapped to a PaddleOCR code internally:

| Tesseract | PaddleOCR | | Tesseract | PaddleOCR |
|-----------|-----------|-|-----------|-----------|
| eng | en | | jpn | japan |
| rus | ru | | kor | korean |
| chi_sim | ch | | ara | arabic |
| chi_tra | chinese_cht | | hin | hi |

Composite specs like `rus+eng` use the primary code (`ru`); unknown codes fall
back to `en`.

### When to prefer PaddleOCR

- CJK documents (Chinese/Japanese/Korean) — usually beats tesseract
- Mixed-script or non-Latin pages where OSD is unreliable
- Pages with rotated/angled text lines (angle classifier handles them)

Prefer tesseract for clean Latin/Cyrillic typed docs (faster, zero install) and
easyocr for handwriting.

---

## Vision tier — agent-driven

When `--engine vision` is used (or when flagged pages trigger escalation),
`ocr.py` renders those pages to persistent PNG files and exits with a manifest
pointing to them.

The agent then reads each PNG and produces text. This is the default "vision
tier" — no API key required. It uses the multimodal model already running the
conversation. If the current agent is GPT, GPT reads the PNGs. If the current
agent is Claude, Claude reads them. If the current agent cannot read images,
use `--engine vision-api` or another OCR backend instead.

### Recommended prompt for the agent

```
Read the attached page image faithfully. Reproduce all visible text in the same
order as it appears. For tables, use Markdown table syntax (| col | col |).
For charts, describe the key values (axis labels, bar heights, trend lines).
Do not add commentary or interpretation — only the content visible in the image.
```

### Vision API path (headless / batch use)

`--engine vision-api` sends rendered pages to any **OpenAI-compatible** vision
endpoint with `detail: high`. All configuration is passed via flags — the key is
**never** read from `OPENAI_API_KEY` (or any environment variable), and there is
**no default model**. Both `--vision-api-key` and `--vision-model` are required;
`--vision-api-url` is optional (defaults to the OpenAI base URL).

```bash
# OpenAI
uv run --with openai python3 scripts/ocr.py FILE --engine vision-api \
  --vision-api-key "$OPENAI_KEY" --vision-model gpt-4o

# Any OpenAI-compatible gateway / self-hosted model
uv run --with openai python3 scripts/ocr.py FILE --engine vision-api \
  --vision-api-url https://gateway.example.com/v1 \
  --vision-api-key "$GATEWAY_KEY" --vision-model qwen2-vl
```

---

## Making PDFs searchable (ocrmypdf)

`ocrmypdf` adds an invisible text layer to the original PDF, keeping its visual
appearance while making text selectable/searchable.

```bash
# Install
brew install ocrmypdf   # macOS
sudo apt install ocrmypdf  # Ubuntu

# Usage via ocr.py
python3 scripts/ocr.py input.pdf --searchable-pdf output.pdf --lang rus+eng

# Direct usage
ocrmypdf -l rus+eng input.pdf output.pdf
ocrmypdf -l rus+eng --rotate-pages --deskew input.pdf output.pdf  # auto-fix
```

Note: `ocrmypdf` also does deskew and rotation correction internally when
`--rotate-pages --deskew` flags are passed.

---

## Cloud APIs (optional, not the default)

These are worth considering for high-volume production pipelines or when
maximum accuracy on complex layouts is needed.

### AWS Textract
Strengths: forms (key-value pairs), tables, handwriting, multi-column.
```bash
pip install boto3
aws textract detect-document-text --document '{"S3Object":{"Bucket":"…","Name":"…"}}'
```

### Google Document AI
Strengths: best-in-class table extraction, invoices, receipts, ID cards.
```bash
pip install google-cloud-documentai
```

### Azure Document Intelligence (formerly Form Recognizer)
Strengths: pre-built models for invoices, receipts, business cards.
```bash
pip install azure-ai-documentintelligence
```

### Mistral OCR
Strengths: multilingual, good on Cyrillic, fast API, competitive cost.
```bash
pip install mistralai
```

For any cloud API: you need an account, a key, and network access. Files are
sent to external servers — check your data-handling requirements before use.

---

## Install commands summary

```bash
# Required (already present on this machine)
# poppler: pdftoppm, pdftotext, pdffonts, pdfimages, pdfinfo
# tesseract 5.5.2 with 163 languages including rus + eng

# Optional tiers — install only when needed

# Tier 1 upgrade: pytesseract (TSV confidence output)
uv run --with pytesseract python3 scripts/ocr.py FILE

# Tier 2 preprocessing: OpenCV
uv run --with opencv-python,numpy python3 scripts/ocr.py FILE --preprocess enhanced

# Tier 2 engine: easyocr (handwriting)
uv run --with easyocr python3 scripts/ocr.py FILE --engine easyocr
# Note: first run downloads ~2 GB of models

# PaddleOCR engine (CJK / multilingual, opt-in)
uv run --with paddleocr,paddlepaddle python3 scripts/ocr.py FILE --engine paddleocr
# Note: first run downloads OCR models; PaddleOCR 3.x API only

# Tier 1 renderer upgrade: PyMuPDF
uv run --with pymupdf python3 scripts/ocr.py FILE
# Note: PyMuPDF is AGPL licensed

# Searchable PDF output
brew install ocrmypdf

# Vision API (OpenAI-compatible; key + model required, endpoint optional)
uv run --with openai python3 scripts/ocr.py FILE --engine vision-api \
  --vision-api-key KEY --vision-model MODEL [--vision-api-url URL]

# All optional Python tiers at once
uv run --with pytesseract,pymupdf,opencv-python,numpy python3 scripts/ocr.py FILE
```
