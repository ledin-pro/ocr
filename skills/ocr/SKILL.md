---
name: ocr
description: >
  Extract text and tables from text-based PDFs, scanned PDFs, and images
  (PNG/JPG/TIFF/HEIC). Use this skill for PDF-to-Markdown extraction, selectable
  PDF tables, scans or photos, receipts, screenshots, faxes, IDs, forms, and
  presentation slides. Also use for non-English and Cyrillic documents, empty or
  garbled PDF text, and image-only PDFs. Handles native Camelot table extraction,
  language detection, preprocessing, strict OCR engine escalation, charts and
  scanned tables through layout OCR or approved vision, and Markdown, plain-text,
  and JSON output.
---

# OCR Skill

Use one initial `ocr` command. Readable text PDFs automatically use native text
and Camelot table extraction; scans use the selected OCR engine. Let configured
escalation chain handle flagged OCR pages. Do not probe first unless user asks
for a triage-only result.

## Initial command

```bash
ocr FILE --format md,txt,json --out OUTPUT_DIR
```

For explicit triage without extraction:

```bash
ocr FILE --probe
```

`--probe` prints one NDJSON object per input and exits. It is part of `ocr`; no
separate probe executable exists. Each object reports `needs_ocr`, the
`text_layer_rejected` flag, `readability` scores, and a human-readable `reason`.
A dense but unreadable embedded text layer (broken/`ToUnicode`-less fonts) is
rejected and routed to OCR rather than emitting glyph soup.

## Native text-PDF tables

Camelot is a local text-PDF stage, not an OCR engine. It runs automatically only
after the PDF probe accepts the embedded text layer. It never receives rendered
page images and does not call a model or external service.

- Default: `--tables --table-flavor auto`.
- `auto` compares classic auto-detection with a Camelot `stream` candidate.
- Use `lattice` for ruled tables.
- Use `stream` or `network` for borderless aligned tables.
- Use `hybrid` when line and alignment cues are mixed.
- `ml` is unsupported because native table extraction must remain model-free.
- Use `--no-tables` only for legacy linear-text output or diagnosis.

Accepted table regions replace their linear text in Markdown, preventing
duplication. TXT keeps the original text layer. JSON includes normalized table
rows, bbox, parser metrics, validation coverage, rendered output, and issues.
Simple grids use pipe Markdown; complex grids use HTML inside Markdown. A
rejected candidate preserves source text and records why it was rejected.

Do not reroute a selectable text PDF to vision merely because it contains a
table. Vision or PaddleOCR-VL is appropriate only when the page is a scan, the
text layer is rejected, or native extraction cannot access the visual structure.

## Engine and escalation resolution

- Baseline engine: `--engine` > `OCR_ENGINE` > `tesseract`.
- Valid engines: `tesseract`, `easyocr`, `paddleocr`, `paddleocr-vl-mlx`, `vision`.
- No `auto` engine exists.
- Automated OpenAI-compatible vision engine is `vision`; former alias is
  removed.
- Escalation chain: `--auto-escalate` > `OCR_AUTO_ESCALATE` > no escalation.
- Chains are comma-separated and ordered, for example `easyocr,vision`.
- Duplicate engines and baseline engine are omitted.

With escalation configured, all requested engine dependencies are validated
before baseline OCR. Flagged pages then run through every chain engine in strict
order. Local candidates compete by unflagged status, confidence, then word
count; `vision` becomes final candidate when present. Each selected page records
baseline, flag reasons, attempts, and selected source under `decision`.

Any missing dependency or engine attempt failure stops whole workflow. No later
engine runs and no workflow result is cached. Never silently skip, reorder, or
replace user-requested engines.

For programmatic preflight, use public side-effect-free probe instead of
importing engine packages:

```python
from pro.ledin import ocr

requirement = ocr.probe_engine_requirements(
    "paddleocr",
    vision_api_key="",  # used only for engine="vision"
    vision_model="",
)
if not requirement.available:
    handle(requirement.to_dict())
```

`ocr.probe_pdf_requirements()` covers PDF render and text-layer backends.

Result fields: `engine`, `available`, stable `code`, backward-compatible
`missing_component`, complete `missing_components` tuple, `components_relation`,
`ocr_extra`, `component_type`, optional `first_run_note`. Singular value is first
tuple item. `components_relation` is `all` for required sets and `any` when one
listed component suffices, as for PDF backends. Paddle probe reports `paddleocr`
and `paddle` together when both modules are missing; other engine failures use
one-item tuples. Recognition uses same probe and raises
`ocr.OcrRequirementError` on failure. Error retains numeric `.code` for CLI
compatibility and exposes stable code as `.requirement_code` or `.stable_code`.
Probe performs no engine import, model download, network call, credential
environment lookup, or mutation.

Probe uses module specs, so broken installs can still fail at import time. Every
engine import site converts that failure into same structured
`ocr.OcrRequirementError` with same stable code; raw `ImportError` never escapes.
EasyOCR transitive import failures use `missing_easyocr_dependency` and name
actual component when determinable (`numpy`, `pillow`, `torch`, etc.).

Optional helper failures are fallback conditions, not fatal requirements:

- Broken/missing pytesseract falls back to Tesseract CLI.
- Broken/missing Pillow during basic preprocessing returns original image.
- Broken/missing cv2, NumPy, or Pillow during enhanced/full preprocessing falls
  back to basic, then original image if Pillow is also unavailable.

## Workflow

1. Run one initial command with requested formats, language, pages, or existing
   escalation configuration.
2. For a readable text PDF, use the automatic Camelot result. If a user reports
   wrong columns, obtain approval for one targeted rerun with `--table-flavor`.
3. If command succeeds, use output and report escalation decisions. Do not run
   speculative second pass.
4. If no escalation is configured and report flags OCR pages, inspect reasons. Ask
   approval for one targeted rerun, then rerun only flagged pages with selected
   engine or preprocessing change.
5. If user-selected engine or escalation dependency is missing, detect platform
   before proposing install: OS/version, architecture, Python version/bitness,
   and GPU/vendor/driver/CUDA where relevant.
6. Propose exact platform-specific commands from `references/engine-setup.md`.
   State package downloads, model downloads, disk/network impact, and GPU/CPU
   choice. Request approval before installing anything.
7. After approval, install only requested dependency, verify executable/import,
   language/model availability, and GPU visibility where applicable. Rerun same
   failed OCR command. If verification fails, stop and report exact failure.

Do not downgrade to Tesseract when user explicitly requested EasyOCR,
PaddleOCR, PaddleOCR-VL, or vision. Do not remove an escalation engine to make run pass.

## Targeted approved reruns

Use only when no auto-escalation chain was configured:

```bash
# Flagged handwriting/degraded pages
ocr FILE --engine easyocr --pages 2,5 --format md

# Flagged CJK, multilingual, or angled pages
ocr FILE --engine paddleocr --pages 2,5 --format md

# Structured local tables on Apple Silicon; MLX service is VLM-only
ocr FILE --engine paddleocr-vl-mlx --pages 2,5 --format md \
  --paddle-vl-server-url http://127.0.0.1:8111/ \
  --paddle-vl-model PaddlePaddle/PaddleOCR-VL-1.6

# Automated vision; endpoint URL optional, key/model required
ocr FILE --engine vision --pages 2,5 --format md \
  --vision-api-key "$KEY" --vision-model MODEL \
  --vision-prompt "Preserve tables and empty cells"

# Noisy or skewed pages while retaining selected engine
ocr FILE --pages 2,5 --preprocess full --format md
```

For automated strict escalation:

```bash
OCR_AUTO_ESCALATE=easyocr,paddleocr ocr FILE --format md,json --out results/

ocr FILE --auto-escalate easyocr,vision --format md,json --out results/ \
  --vision-api-key "$KEY" --vision-model MODEL
```

## Interactive vision handoff

Interactive handoff is not OCR engine. Use module when current multimodal agent
must inspect rendered pages:

```bash
python -m pro.ledin.ocr.vision_handoff FILE \
  --pages 2,5 --dpi 200 \
  --vision-prompt "Read all visible text in reading order"
```

Supported handoff controls: `--pages`, `--dpi`, `--vision-prompt`,
`--vision-prompt-file`, `--verbose`. No `--max-pages` option. Module prints
persistent image paths and prompt; agent reads images and returns faithful text.

## Common commands

```bash
ocr scan.png --format md,txt,json --out results/
ocr russian.pdf --lang rus+eng --format md
ocr report.pdf --table-flavor lattice --format md,json --out results/
ocr report.pdf --no-tables --format md
ocr scan.pdf --preprocess full --format md
ocr report.pdf --probe
ocr FILE1 FILE2 --cache ocr-cache.json --format txt --out results/
ocr FILE --searchable-pdf searchable.pdf
```

## OCR Profile Benchmark

Use `scripts/ocr-profile-benchmark.py` to compare temporary oMLX profiles with
OpenAI compatible vision models. The benchmark calls `pro.ledin.ocr` directly and
stores resumable reports under the configured `run_root`:

```bash
export OMLX_API_KEY=your-omlx-key
export OPENAI_API_KEY=your-openai-key
python scripts/ocr-profile-benchmark.py --config benchmark.json
python scripts/ocr-profile-benchmark.py --resume path/to/run
```

See `references/benchmark.md` and
`scripts/ocr-profile-benchmark.example.json` for configuration details.

## Output behavior

| Flag | Output |
|---|---|
| `--format md` | Markdown with page headings |
| `--format txt` | Plain text with page separators |
| `--format json` | Page text, tables, confidence, boxes, flags, decisions |
| `--format md,json` | Requested comma-separated formats |
| `--format all` | `md,txt,json` shorthand |
| `--searchable-pdf OUT` | Original PDF with invisible text layer |

Preserve `0.4.0` comma-format behavior: without `--out`, selected formats print
to stdout. With one format, `--out` is exact file path. With multiple formats or
inputs, `--out` is directory containing `<stem>.<format>`. Duplicate input stems
are rejected. Multiple-input JSON requires `--out`; searchable PDF accepts one
input. `--json-report` remains removed.

## Peepshow sink

```bash
peepshow video.mp4 --sink ocr
peepshow video.mp4 \
  --sink-cmd 'peepshow-sink-ocr --engine tesseract --lang rus+eng'
```

Sink supports same non-interactive engines and strict escalation through
`--auto-escalate` or `PEEPSHOW_SINK_OCR_AUTO_ESCALATE`. Automated `vision`
uses the base package's OpenAI client and requires a key and model. Interactive handoff is outside
sink process. See `references/peepshow-sinks.md`.

## Library use

```python
from pro.ledin import ocr

pages = ocr.recognize(
    "scan.pdf",
    ocr.RecognizeOptions(
        engine="tesseract",
        lang="rus+eng",
        auto_escalate=("easyocr", "vision"),
        vision_api_key="key",
        vision_model="model",
        extract_tables=True,
        table_flavor="auto",
    ),
)
markdown = ocr.to_markdown(pages, "scan.pdf")
```

Catch `ocr.OcrError` for recoverable failures. Catch
`ocr.OcrRequirementError` when structured engine requirement fields matter.
Library never calls `sys.exit()`. `verbose=True` enables progress logging only;
capability dump stays CLI diagnostic and library callers opt in with
`caps=ocr.Caps(report=True)`.
`RecognizeOptions` mirrors recognition flags, including `max_pages`; this does
not add `--max-pages` to interactive handoff module.

## References

- `references/engines.md`: selection, strict escalation, quality, languages.
- `references/engine-setup.md`: approved platform-specific dependency setup.
- `references/troubleshooting.md`: symptom-based diagnosis.
- `references/peepshow-sinks.md`: sink configuration and schema.
