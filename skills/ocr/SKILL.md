---
name: ocr
description: >
  Extract text from scanned PDFs and images (PNG/JPG/TIFF/HEIC) using OCR. Use
  this skill whenever a PDF's text cannot be selected or copied, the document is a
  scan or photo, text is rendered as images rather than a selectable layer, or the
  file is a receipt, screenshot, fax, ID card, form, or presentation slide image.
  Also use for non-English and Cyrillic-language scans, when pdftotext or pypdf
  return empty or garbled output, or when a user says "this PDF has no text" or
  "I can't copy from this file". Handles language auto-detection, deskew and
  denoise for messy scans, tables and charts via vision escalation, and produces
  Markdown plus plain-text output. Always reach for this skill before giving up on
  a document that appears to have no readable text.
---

# OCR Skill

Extracts text from scanned PDFs and images using a layered engine stack.
The baseline path runs with zero additional installs (poppler + tesseract are
assumed present). Heavier tools are added only when a page needs them.

## Installation

```bash
pip install pro-ledin-ocr            # baseline (tesseract + poppler on PATH)
pip install "pro-ledin-ocr[vision]"  # + OpenAI-compatible vision-api engine
pip install "pro-ledin-ocr[all]"     # + pymupdf, opencv, easyocr, paddleocr
```

System binaries still required for the local path: `poppler` (pdftoppm,
pdftotext, pdfinfo) and `tesseract`. Installs the `ocr`, `ocr-probe`, and
`peepshow-sink-ocr` console scripts.

## When to use this skill

- PDF where text cannot be selected/copied in a viewer
- PDF produced by scanning a paper document or photographing a page
- Office-generated PDF where text is rasterized into image masks (common with
  some Word/PowerPoint exports — `pdftotext` returns near-empty output)
- Any standalone image file containing text (PNG, JPG, TIFF, HEIC, WEBP)
- Non-English documents, especially Cyrillic/Russian
- Receipts, invoices, forms, ID cards, screenshots of documents
- When a previous attempt with `pdftotext`, `pypdf`, or `pdfplumber` failed

## Decision tree

Work through this in order. Stop at the first successful step.

```
1. Is input an image (png/jpg/tiff/heic/webp)?
   └─ Yes → OCR directly (skip probe). Go to step 3.

2. PDF input: run ocr-probe FILE
   ├─ needs_ocr = false  → real text layer exists.
   │   Run: pdftotext -layout FILE -  (or PyMuPDF get_text())
   │   Done — fast, free, no OCR needed.
   └─ needs_ocr = true   → continue to step 3.

3. Baseline OCR:
   ocr FILE --format all --out .
   (auto-detects language via OSD, DPI from page size, preprocessing level)
   Emits: <stem>.md  <stem>.txt  <stem>.json
   quality report on stderr: per-page confidence, flagged pages.

4. Review quality report. Escalate flagged pages only:
   ├─ low confidence OR tables/charts/forms
   │   → ocr FILE --engine vision --pages <flagged>
   │     Renders persistent PNGs and hands them to the current multimodal agent
   │     to read (Claude, GPT, or another model with image/file reading).
   │     Agent produces Markdown (use Markdown table syntax for tables).
   ├─ CJK / multilingual / angled dense text
   │   → ocr FILE --engine paddleocr
   │     (opt-in; installs paddleocr+paddlepaddle, downloads models on first run)
   ├─ handwriting detected (very low conf, cursive)
   │   → ocr FILE --engine easyocr
   └─ skewed/noisy scan (scanned paper, phone photo)
       → ocr FILE --preprocess full

5. Need a selectable/searchable PDF?
   → ocr FILE --searchable-pdf OUT.pdf
     (requires: brew install ocrmypdf)

6. Processing a folder or re-running repeatedly?
   → ocr FILE1 FILE2 … --cache ocr_cache.json
     Add --skip-ocr to triage-only mode (skip OCR, text-layer files only).
     Add --force to ignore cache and re-process.
```

## Quick start (90% of cases)

```bash
# Probe first to confirm OCR is needed
ocr-probe myfile.pdf

# Extract everything (md + txt + json quality report)
ocr myfile.pdf --format all

# Image input
ocr scan.png --format all

# Russian/Cyrillic doc — language auto-detected, but can be forced
ocr russian_doc.pdf --lang rus+eng --format md

# Messy scan (skewed, noisy)
ocr scan.pdf --preprocess full --format all

# Table-heavy slide / complex layout → vision tier
ocr slides.pdf --engine vision \
  --vision-prompt "Extract tables and preserve empty cells"

# CJK / multilingual doc → PaddleOCR (opt-in)
ocr doc.png --engine paddleocr

# Headless vision via an OpenAI-compatible endpoint (all config via flags)
ocr slides.pdf --engine vision-api \
  --vision-api-url https://api.example.com/v1 \
  --vision-api-key "$MY_KEY" --vision-model my-vision-model \
  --vision-prompt-file prompts/faithful-ocr.txt
```

For `vision-api`, page images over ~7.5 MB are auto-re-encoded to JPEG (full
resolution first, dimensions reduced only if still too large) to stay under
the API's 10 MB per-image cap; this re-encode path requires Pillow. Images
already under the limit are sent untouched with their detected media type.

## Peepshow sink

Use the installed sink to recognize frames already extracted by peepshow:

```bash
peepshow video.mp4 --sink ocr
peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'
```

The sink reads peepshow JSON from stdin and writes only
`<outputDir>/ocr.json`. It does not modify frame files or peepshow's manifest.
No additional Python package is required for the interface itself. Local
recognition still requires Tesseract; `vision-api` requires the existing
`pro-ledin-ocr[vision]` extra.

For named `--sink ocr` use, configure `PEEPSHOW_SINK_OCR_*` environment
variables. `engine=vision` is unavailable because a sink cannot perform an
interactive agent handoff; use `vision-api` or a local engine. See
`references/peepshow-sinks.md` for flags, environment variables, JSON schema,
and privacy rules.

## Engine tiers (summary)

| Tier | Engine | Best for | Cost |
|------|--------|----------|------|
| 0 | pdftotext / PyMuPDF | Real text layers | Free, instant |
| 1 | tesseract (default) | Clean scans, typed text, 160+ languages | Free, ~3–4s/page |
| 2 | easyocr | Handwriting, degraded scans | Free, heavy (~2 GB) |
| 2.5 | paddleocr (opt-in) | CJK, multilingual (100+), angled text | Free, models on first run |
| 3 | vision (agent reads PNGs) | Tables, charts, complex layouts | Agent/model tokens |
| 4 | cloud APIs | High-volume, max accuracy | Paid + key |

See `references/engines.md` for full details, escalation thresholds, language
maps, DPI guidance, preprocessing levels, and install commands.

## Output formats

| Flag | Output |
|------|--------|
| `--format md` | `# filename` + `## Page N` headers, prose text |
| `--format txt` | pages separated by `----- Page N -----` |
| `--format json` | per-page text + word confidence + bboxes + quality report |
| `--format md,json` | comma-separated formats in requested order |
| `--format all` | shorthand for `md,txt,json` |
| `--searchable-pdf OUT` | invisible text layer overlaid on original PDF |

Without `--out`, selected formats print to stdout. With one format, `--out`
is an exact file path. With multiple formats or multiple inputs, `--out` is a
directory containing `<input-stem>.<format>` files. Inputs sharing a stem are
rejected to prevent overwrites. JSON output for multiple inputs requires
`--out`; `--searchable-pdf` accepts one input only.

## All CLI flags

```
ocr INPUT [INPUT ...]
  --engine   auto|tesseract|easyocr|paddleocr|vision|vision-api   default: auto
  --lang     auto|<tesseract codes>           default: auto (OSD detection)
  --format   FORMAT[,FORMAT...]  md|txt|json|all   default: md
  --out      PATH                            default: stdout
  --dpi      N|auto                          default: auto (300 A4, 150 wide)
  --preprocess  none|basic|enhanced|full|auto  default: auto
  --pages    RANGE  (e.g. 1-3,5)
  --max-pages N
  --psm      N      (default 3; use 6 for dense single-block pages)
  --min-conf F      (default 60.0 — flag pages below this for review)
  --cache    PATH   --force   --skip-ocr
  --no-cleanup      (skip whitespace / ligature cleanup)
  --vision-api-url  URL   (OpenAI-compatible base URL for vision-api)
  --vision-api-key  KEY   (required for vision-api; env vars are NOT read)
  --vision-model    NAME  (required for vision-api; no default)
  --vision-prompt   TEXT  (custom prompt for vision or vision-api)
  --vision-prompt-file PATH  (UTF-8 prompt file; mutually exclusive with above)
  --searchable-pdf OUT.pdf
  --verbose
```

## Library usage

The package also works as an importable library for other skills/scripts that
want structured OCR results in-process instead of shelling out to the CLI:

```python
from pro.ledin import ocr

pages = ocr.recognize("scan.pdf", ocr.RecognizeOptions(engine="tesseract", lang="rus+eng"))
markdown = ocr.to_markdown(pages, "scan.pdf")
```

- `recognize(path, options=None, *, caps=None, cache=None)` is the entry
  point: it manages a throwaway render directory and returns the same
  per-page dicts the CLI builds internally. Format the result with
  `to_markdown()`, `to_text()`, or `to_json()`.
- `RecognizeOptions` mirrors the CLI's recognition flags (`engine`, `lang`,
  `dpi`, `preprocess`, `pages`, `max_pages`, `psm`, `min_conf`, `no_cleanup`,
  `force`, `vision_api_url`, `vision_api_key`, `vision_model`, `vision_prompt`, `timeout`,
  `verbose`). Output-only flags (`--out`, `--format`,
  `--searchable-pdf`) are CLI-only and have no library equivalent.
- `--engine vision` is an interactive agent handoff (renders pages and prints
  a manifest for a multimodal agent to read) and is not usable via
  `recognize()`; it raises `OcrError` if requested. Use another engine or the
  CLI directly.
- Catch `OcrError` for recoverable failures (unsupported input, missing
  binaries/packages, vision-api config/request errors) — the library never
  calls `sys.exit()`, unlike the CLI.
- For `engine="vision-api"`, `RecognizeOptions.timeout` (seconds) bounds the
  HTTP request via the openai SDK's own client timeout. Local engines
  (tesseract/easyocr/paddleocr) run in-process with no external kill switch,
  since there is no longer a subprocess to terminate on timeout.
- The `healthos` skill uses this API (via `import`) to recognize family medical
  documents without spawning a subprocess per file.

## Troubleshooting

See `references/troubleshooting.md` for: rasterized-text PDFs, garbled Cyrillic,
rotated pages, table/chart handling, handwriting, multi-column layouts, and
large-folder batch jobs.
