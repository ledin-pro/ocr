# OCR Troubleshooting

Each section follows symptom, cause, fix. For missing dependencies, detect
platform and request install approval using `engine-setup.md` before changing
environment.

## PDF returns no text

**Cause:** PDF contains page images or rasterized glyph masks instead of usable
text layer.

**Fix:** Run normal OCR once:

```bash
ocr FILE --format md,txt,json --out results/
```

Use `ocr FILE --probe` only for triage-only NDJSON. No separate probe command.

## Garbled or non-Unicode text layer accepted

**Symptom:** A PDF has a dense embedded text layer, but the extracted characters
are punctuation/symbol soup (e.g. `D ! 9 *!1E (&%($*'++`). Common for scans saved
with fonts that lack a `ToUnicode` map.

**Cause:** The text-layer probe used to trust raw character volume alone. A broken
CMap yields many characters but no readable words.

**Behavior:** The probe now scores readability of the extracted text
(`letter_digit_ratio`, `word_score`, `replacement_ratio`). When density is high
but readability is low, the text layer is rejected: `needs_ocr=True`,
`text_layer_rejected=True`, and pages carry a `text_layer_rejected` issue.
Recognition then renders pages and runs the selected engine. Absence of a
`ToUnicode` map on every font is reported (`all fonts non-Unicode`) but is not the
deciding signal — legitimate base-14 fonts also report no `ToUnicode` yet extract
cleanly, so readability, not font metadata, gates the decision.

**Fix:** No action needed; the correct engine runs automatically. To inspect the
decision, run `ocr FILE --probe` and read `reason`, `text_layer_rejected`, and the
`readability` scores in the NDJSON.

## Wrong or garbled Cyrillic

**Cause:** Tesseract selected English after weak OSD result, or requested
language data is absent.

```bash
tesseract --list-langs
ocr FILE --lang rus+eng --format md
```

If `rus` is missing, follow approved platform install in `engine-setup.md`.

## Missing engine dependency

**Symptom:** Requested baseline/escalation engine fails before OCR.

**Behavior:** Strict chain validates all dependencies before baseline. It does
not silently remove requested engine or fall back to another engine.

**Fix:** Detect OS/version, architecture, Python version/bitness, and GPU driver
when relevant. Propose matching commands from `engine-setup.md`, request user
approval, install, verify, then rerun exact command. If verification fails, stop.

For callers embedding package, inspect without importing engine:

```python
result = ocr.probe_engine_requirements(
    selected_engine,
    vision_api_key=key,
    vision_model=model,
)
```

Branch on stable `result.code`; render every item in `missing_components`, then
use `components_relation`, `ocr_extra`, `component_type`, and `first_run_note`
for remediation. `components_relation="any"` means one listed component suffices;
`all` means every listed component is required. `missing_component` remains first
tuple item for older integrations. Recognition raises `OcrRequirementError`
carrying same fields. Numeric error `.code` remains CLI exit status; stable code
is `.requirement_code`/`.stable_code`.

Use `ocr.probe_pdf_requirements()` for PDF render/text-layer backends.

## Engine installed but import fails

**Symptom:** Package is present yet OCR stops with requirement error naming same
package.

**Cause:** Broken build, namespace-package shell, or failed native library load.
Spec-based detection reports package present; actual import fails.

**Fix:** Reinstall reported component for detected platform following
`engine-setup.md`. Error keeps stable code, so remediation is unchanged. Raw
`ImportError` never escapes recognition.

EasyOCR transitive failures use `missing_easyocr_dependency` and identify the
component where possible. pytesseract, Pillow preprocessing, and OpenCV/NumPy
preprocessing are optional layers: import failures log fallback instead of
aborting OCR.

## Escalation stops midway

**Cause:** Engine attempt failed or returned no page result.

**Behavior:** Later engines do not run and workflow cache is not written. Chain
is strict, not best-effort.

**Fix:** Report failing engine and error. Repair dependency/config only with
approval, then rerun full original command so baseline and ordered decisions are
reproducible.

## No escalation happened

Check resolution:

```bash
# CLI chain wins
ocr FILE --auto-escalate easyocr,vision ...

# Otherwise environment chain is used
export OCR_AUTO_ESCALATE=easyocr,vision
ocr FILE ...
```

No chain is default. Only flagged pages escalate. Duplicate engines and baseline
engine are omitted. If no chain was configured, obtain approval before targeted
rerun of reported pages.

## Engine value rejected

Valid engine values: `tesseract`, `easyocr`, `paddleocr`, `vision`.

Resolution is `--engine` > `OCR_ENGINE` > `tesseract`. `auto` and old automated
vision engine name are invalid.

## Automated vision configuration error

Automated engine is `vision` and requires the base package's OpenAI client, a
key, and a model. Configure the key and model with CLI flags or `OCR_VISION_*`
environment variables:

```bash
ocr FILE --engine vision \
  --vision-api-key "$KEY" \
  --vision-model MODEL \
  --vision-api-url https://api.example.com/v1
```

`--vision-api-url` is optional. Key is not read from `OPENAI_API_KEY`. Confirm
external-data approval before sending sensitive pages.

## Need interactive agent reading instead of API

Use module handoff:

```bash
python -m pro.ledin.ocr.vision_handoff FILE \
  --pages 2,5 --dpi 200 \
  --vision-prompt "Preserve table rows and empty cells"
```

Handoff has no page-count cap option. It prints persistent image paths for
current multimodal agent and does not act as automated escalation engine.

## Tables are jumbled

First determine whether the PDF has a readable text layer.

For a text PDF, Camelot runs automatically. Try one targeted parser override:

```bash
# Explicit rules
ocr FILE --table-flavor lattice --pages 2,5 --format md,json --out results/

# Borderless alignment
ocr FILE --table-flavor stream --pages 2,5 --format md,json --out results/
```

Inspect JSON table fields: `accepted`, `flavor`, `accuracy`, `confidence`,
`text_coverage`, `numeric_coverage`, and `issues`. A rejected candidate leaves
the original linear text intact. `table_parse_quality_low`,
`table_text_coverage_low`, or `table_numeric_coverage_low` explains rejection.

Use `--no-tables` to compare against legacy linear text:

```bash
ocr FILE --no-tables --pages 2,5 --format md
```

Do not use vision for a selectable text PDF solely because columns are wrong.

For a scanned table, Tesseract preserves words but not table structure. Use
local PaddleOCR-VL or approved vision:

```bash
ocr FILE --engine paddleocr-vl-mlx --pages 2,5 --format md \
  --paddle-vl-server-url http://127.0.0.1:8111/ \
  --paddle-vl-model PaddlePaddle/PaddleOCR-VL-1.6

ocr FILE --engine vision --pages 2,5 \
  --vision-api-key "$KEY" --vision-model MODEL \
  --vision-prompt "Return Markdown tables; preserve empty cells"
```

For simple dense blocks, try `--psm 6`. For interactive inspection, use handoff
module.

## Camelot is missing or fails to import

Camelot is a baseline dependency. A missing import indicates an incomplete or
broken `pro-ledin-ocr` installation rather than an optional engine.

Reinstall the package in its owning Python environment after approval, then
verify `python -c 'import camelot; print(camelot.__version__)'`. Runtime parse
failures degrade to ordinary text and add `table_extraction_failed`; they do not
send the document to vision automatically.

## Charts lack meaningful values

Tesseract reads labels but cannot infer graphical relationships. Use automated
vision with approved external processing or interactive handoff for selected
pages.

## Handwriting has low confidence

Use EasyOCR or vision after approval:

```bash
ocr FILE --engine easyocr --pages 1-3
ocr FILE --engine vision --pages 1-3 --vision-api-key "$KEY" --vision-model MODEL
```

First EasyOCR run downloads models. See `engine-setup.md`.

## CJK, multilingual, or angled text fails

Use PaddleOCR:

```bash
ocr FILE --engine paddleocr --pages 1-3
```

PaddlePaddle runtime must match platform and GPU driver. First run downloads
models. Do not guess CUDA package; follow `engine-setup.md` approval workflow.

## Rotated or skewed pages

```bash
ocr FILE --preprocess full --format md
ocr FILE --searchable-pdf OUT.pdf --lang rus+eng
```

`full` handles deskew; quarter-turn rotation may need source correction or
OCRmyPDF rotation. Inspect `pdfinfo FILE` before changing source.

## Tiny text

```bash
ocr FILE --dpi 400 --format md
ocr FILE --preprocess basic --format md
```

High DPI costs memory/time. Wide slides often need only 150 DPI.

## Noisy scan

```bash
ocr FILE --preprocess enhanced --format md
ocr FILE --preprocess full --format md
```

These modes require OpenCV/NumPy. Missing dependency follows same detection,
proposal, approval, verification, rerun workflow.

## Preprocessing worsens output

Adaptive threshold can damage clean digital renders:

```bash
ocr FILE --preprocess basic --format md
ocr FILE --preprocess none --format md
```

## Slow batch

```bash
ocr *.pdf --cache ocr-cache.json --format txt --out results/
ocr slides.pdf --dpi 150 --format md
ocr FILE --pages 1-3 --format md
ocr FILE --cache ocr-cache.json --force --format md
```

Use `--pages` for bounded selection. Interactive handoff also uses `--pages` and
does not support page-count cap option.

## Output path surprises

Comma-format behavior from `0.4.0` remains:

- One format plus `--out`: exact file path.
- Multiple formats plus `--out`: output directory.
- Multiple inputs plus `--out`: output directory.
- Duplicate input stems: rejected.
- Multiple-input JSON without `--out`: rejected.
- Searchable PDF: one input only.

```bash
ocr FILE --format json --out report.json
ocr FILE --format md,json --out results/
```

## Encrypted PDF

Probe reports blocked input. Decrypt only with user-provided authorization and
password, then OCR decrypted copy:

```bash
qpdf --password=PASSWORD --decrypt encrypted.pdf decrypted.pdf
ocr decrypted.pdf --format md,txt,json --out results/
```

Do not place password in documentation, logs, or vault files.
