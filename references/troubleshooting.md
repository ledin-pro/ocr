# OCR Troubleshooting

Each section: **Symptom → Cause → Fix**.

---

## "PDF returns no text / pdftotext gives nothing"

**Cause:** The PDF stores text as rasterized image bitmaps rather than a text
layer. Common with some Word/PowerPoint exports, scanned-then-exported documents,
and PDFs produced by certain printers or copiers. `pdffonts` shows fonts with
`uni: no` / not embedded; `pdfimages -list` shows full-page `smask` masks.

**Fix:** This is exactly what this skill handles. Run:
```bash
bash scripts/probe.sh myfile.pdf   # confirms needs_ocr=true
python3 scripts/ocr.py myfile.pdf --format all
```
The probe returns `needs_ocr=true` with `reason: "rasterized text — low char yield"`.
The OCR script renders pages and runs tesseract.

---

## Garbled text / transliterated Cyrillic (e.g. "рт/л" instead of "мг/л")

**Cause A:** Wrong tesseract language. If `--lang eng` is used on a Russian
document, tesseract maps Cyrillic glyphs to the closest Latin equivalents → garbage.

**Fix:** Let OSD auto-detect, or force the language:
```bash
python3 scripts/ocr.py FILE --lang rus+eng
```

**Cause B:** OSD script detection failed (very short page, all-caps header,
logo-heavy first page → low OSD confidence → fallback to `eng`).

**Fix:** Force language explicitly with `--lang rus+eng`.

**Cause C:** Mixed-script document where auto-detection picks the wrong dominant script.

**Fix:** Specify all scripts: `--lang rus+eng+fra` etc.

---

## Rotated or upside-down pages

**Cause:** The page was scanned upside-down or at 90°/270°. Tesseract OSD
detects rotation but the rendering step doesn't always auto-correct.

**Fix — automatic rotation (recommended):**
```bash
# ocrmypdf handles this automatically
python3 scripts/ocr.py FILE --searchable-pdf OUT.pdf --lang rus+eng
# ocrmypdf --rotate-pages is applied automatically

# Or force full preprocessing (deskew handles small angles; rotation needs explicit fix)
python3 scripts/ocr.py FILE --preprocess full
```

**Fix — manual:** Check `pdfinfo FILE` for `Page rot:`. Use `pdftk` or
`qpdf --rotate=+90:1` to fix the PDF before OCR.

---

## Tiny or low-resolution text (OCR reads partial words or misses characters)

**Cause:** The rendered image resolution is too low. Tesseract needs roughly
300 DPI (≈ 1200 px width for A4) to handle typical font sizes reliably.
At 150 DPI, small fonts (< 8pt) become unreliable.

**Fix:**
```bash
# Increase DPI
python3 scripts/ocr.py FILE --dpi 400

# For small scans already at low pixel count, auto-upscale kicks in at < 1400 px
# — enable basic/enhanced preprocess to get upscaling
python3 scripts/ocr.py FILE --preprocess basic
```

For phone photos (already high-res, just blurry): try `--preprocess enhanced`
for denoising.

---

## Tables come out as jumbled text (rows merged, columns lost)

**Cause:** Tesseract reads left-to-right in reading order; it doesn't understand
table structure. Multi-column grids confuse its segmentation.

**Fix — recommended:** Use the vision tier. The agent (or GPT-4o) reads the
rendered page and produces Markdown table syntax:
```bash
python3 scripts/ocr.py FILE --engine vision --pages <table-page-numbers>
# e.g. --pages 2,5,7
```

**Fix — tesseract PSM tweak:** For simple two-column tables, try:
```bash
python3 scripts/ocr.py FILE --psm 6   # treat page as single uniform block
```
or
```bash
python3 scripts/ocr.py FILE --psm 4   # single column — processes each column separately
```

---

## Charts, graphs, bar charts — can't extract meaningful numbers

**Cause:** Charts are graphical elements. Tesseract reads embedded text labels
but cannot interpret visual data (bar heights, trend lines, axes).

**Fix:** Always use the vision tier for charts:
```bash
python3 scripts/ocr.py FILE --engine vision --pages <slide-with-chart>
```
The agent reads the rendered PNG and describes: axis labels, key values,
trend direction, data series. For very small rotated axis labels (common in
financial slides), the vision model usually handles them better than tesseract.

---

## Rotated axis labels / small text at angles

**Cause:** Tesseract `--psm 3` (auto) misreads rotated text because it assumes
horizontal reading direction.

**Fix A:** Vision tier (best):
```bash
python3 scripts/ocr.py FILE --engine vision --pages <page>
```

**Fix B:** Try `--psm 12` (sparse text with OSD) or `--psm 11` (sparse text):
```bash
python3 scripts/ocr.py FILE --psm 12
```

---

## Handwriting not recognized / very low confidence

**Cause:** Tesseract is trained primarily on printed fonts. Handwritten text,
especially cursive, is at the edge of its capability.

**Fix — easyocr** (better for handwriting):
```bash
uv run --with easyocr python3 scripts/ocr.py FILE --engine easyocr
# Note: downloads ~2 GB models on first run
```

**Fix — vision tier** (best for complex handwriting):
```bash
python3 scripts/ocr.py FILE --engine vision
```

---

## Multi-column document (newspaper, magazine, two-column academic paper)

**Cause:** Tesseract default `--psm 3` (auto) sometimes merges columns into
mixed-up lines across the page.

**Fix:**
```bash
# PSM 3 with OSD — usually works for 2-column
python3 scripts/ocr.py FILE --psm 3

# If still wrong: treat as single column (reads left column then right)
python3 scripts/ocr.py FILE --psm 4

# For best results on newspaper-style: vision tier
python3 scripts/ocr.py FILE --engine vision
```

---

## Noisy / speckled scan (scanner noise, grain, JPEG artifacts)

**Cause:** Scanner noise, old photocopies, JPEG compression artifacts confuse
tesseract's character segmentation.

**Fix:**
```bash
# enhanced: adds fastNlMeansDenoising + adaptive threshold
uv run --with opencv-python,numpy python3 scripts/ocr.py FILE --preprocess enhanced
```

The denoising step removes high-frequency noise while preserving character edges.
The adaptive threshold handles uneven lighting and watermarks.

---

## Skewed scan (page tilted in scanner, phone photo at an angle)

**Cause:** The document is physically rotated a few degrees, which degrades
tesseract's line-finding and character segmentation significantly.

**Fix:**
```bash
# full: enhanced + deskew (detects and corrects up to ~45° tilt)
uv run --with opencv-python,numpy python3 scripts/ocr.py FILE --preprocess full
```

Deskew uses `cv2.minAreaRect` on the binarized foreground to measure the text
block angle, then `warpAffine` with cubic interpolation to correct it.
Corrections < 0.5° are skipped as measurement noise.

---

## Preprocessing makes output worse (over-binarization)

**Cause:** `enhanced` or `full` preprocessing applies adaptive threshold, which
can introduce artifacts on clean digital renders (high-contrast PDF renders that
are already effectively binary).

**Fix:**
```bash
# Use basic or none for clean digital PDFs
python3 scripts/ocr.py FILE --preprocess basic
python3 scripts/ocr.py FILE --preprocess none
```

Rule of thumb: `none`/`basic` for clean Office/LaTeX PDFs; `enhanced`/`full`
for scanned paper.

---

## Slow processing / large folder of files

**Causes:** High DPI, large images, many pages, no caching.

**Fixes:**

```bash
# Use caching — skip already-processed files on re-run
python3 scripts/ocr.py *.pdf --cache ocr_cache.json

# Triage first — only OCR files that need it
python3 scripts/ocr.py *.pdf --skip-ocr --cache ocr_cache.json   # text-layer only pass
python3 scripts/ocr.py *.pdf --cache ocr_cache.json               # OCR remaining

# Limit pages per file (useful for quick scan / first pass)
python3 scripts/ocr.py FILE --max-pages 3

# Lower DPI for wide slides (150 is usually enough for 1920-pt pages)
python3 scripts/ocr.py slides.pdf --dpi 150

# Force re-process despite cache
python3 scripts/ocr.py FILE --cache cache.json --force
```

---

## Choosing output format

| Use case | Recommended format |
|----------|--------------------|
| Read by agent / LLM downstream | `--format md` (structured, page headers) |
| Full-text search / grep | `--format txt` |
| Programmatic processing | `--format json` (confidence, bboxes, source) |
| Deliver all three | `--format all` |
| Human-readable selectable PDF | `--searchable-pdf OUT.pdf` (needs ocrmypdf) |

The JSON report (`--format json` or `--json-report PATH`) always includes
`recommend_vision: [page numbers]` listing pages the agent should read manually.

---

## Encrypted PDF

**Symptom:** `probe.sh` returns `encrypted: true`; `pdftotext` exits with error.

**Fix:** Decrypt first with the password:
```bash
qpdf --password=YOURPASS --decrypt encrypted.pdf decrypted.pdf
python3 scripts/ocr.py decrypted.pdf --format all
```
Or: `pdftk encrypted.pdf input_pw YOURPASS output decrypted.pdf`

---

## "Required binary missing" error (exit code 4)

**Symptom:** `ocr.py` exits with code 4 and an install hint.

**Fix:** Install the missing binary:
```bash
# poppler (pdftotext, pdftoppm, pdfinfo, pdffonts, pdfimages)
brew install poppler         # macOS
sudo apt install poppler-utils  # Ubuntu

# tesseract
brew install tesseract tesseract-lang   # macOS (includes all langs)
sudo apt install tesseract-ocr tesseract-ocr-rus   # Ubuntu
```
