# pro-ledin-ocr

[![skills.sh](https://skills.sh/b/ledin-pro/ocr)](https://skills.sh/ledin-pro/ocr)

Layered extraction for text PDFs, scanned PDFs, and images
(PNG/JPG/TIFF/HEIC/WEBP). Readable PDF text layers are extracted directly and
tables are reconstructed with Camelot. Scans use Poppler and Tesseract;
EasyOCR, PaddleOCR, structured PaddleOCR-VL over MLX, and automated vision are
opt-in engines.

- Import name: `pro.ledin.ocr`
- Console scripts: `ocr`, `peepshow-sink-ocr`
- Interactive vision handoff: `python -m pro.ledin.ocr.vision_handoff`
- PyPI: `pro-ledin-ocr`

## Install

```bash
pip install pro-ledin-ocr             # baseline, including Camelot tables
pip install "pro-ledin-ocr[easyocr]" # EasyOCR
pip install "pro-ledin-ocr[paddle]"  # PaddleOCR
pip install "pro-ledin-ocr[paddle-vl]" # PaddleOCR-VL document parser client
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
ocr report.pdf --table-flavor lattice            # ruled tables
ocr report.pdf --table-flavor stream             # borderless tables
ocr report.pdf --no-tables                        # legacy linear text
ocr scan.png --format md
ocr russian_doc.pdf --lang rus+eng --format md
ocr scan.pdf --preprocess full
ocr slides.pdf --engine vision --pages 9,12 \
  --vision-api-key "$KEY" --vision-model my-vision-model \
  --vision-prompt "Preserve tables and empty cells"
ocr scan.pdf --auto-escalate easyocr,vision \
  --vision-api-key "$KEY" --vision-model my-vision-model
export OCR_VISION_API_KEY="$KEY"
export OCR_VISION_MODEL="my-vision-model"
export OCR_VISION_API_URL="http://localhost:8000/v1"
ocr scan.pdf --engine vision
ocr table.pdf --engine paddleocr-vl-mlx \
  --paddle-vl-server-url http://127.0.0.1:8111/ \
  --paddle-vl-model PaddlePaddle/PaddleOCR-VL-1.6
```

Engine resolution is explicit: `--engine` overrides `OCR_ENGINE`, otherwise
`tesseract` is used. There is no `auto` engine. Valid engines are `tesseract`,
`easyocr`, `paddleocr`, `paddleocr-vl-mlx`, and `vision`. The MLX engine runs
full layout parsing inside `PaddleOCRVL`; its loopback service handles only the
VLM stage and must not receive document images directly.

For readable text PDFs, table extraction is a native pre-OCR stage rather than
an OCR engine. It is enabled by default. Camelot `auto` is attempted first and
non-stream page results are compared with a `stream` candidate. Override with
`--table-flavor lattice|stream|network|hybrid`, or disable with `--no-tables`.
The `ml` flavor is intentionally unsupported because this path must not require
a visual model.

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

Markdown replaces accepted table regions with pipe tables. Complex or spanning
grids use HTML tables inside Markdown; uncertain spans are flattened and marked
with `table_spans_flattened`. Plain-text output remains the original linear text.
JSON pages add a `tables` array containing normalized rows, bbox, Camelot flavor,
parsing metrics, validation coverage, rendered representation, and issues.
Rejected candidates never replace source text.

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
        extract_tables=True,
        table_flavor="auto",
    ),
)
markdown = ocr.to_markdown(pages, "scan.pdf")
```

`recognize()` never calls `sys.exit()`; catch `ocr.OcrError` for recoverable
failures such as unsupported input, missing dependencies, or vision config.

Call `probe_engine_requirements()` before recognition when an orchestrator needs
structured, side-effect-free dependency diagnostics. It does not import OCR
engines, download models, access network, or read generic credential variables:

```python
requirement = ocr.probe_engine_requirements(
    "vision",
    vision_api_key="key",
    vision_model="model",
)
if not requirement.available:
    print(requirement.to_dict())
```

`probe_pdf_requirements()` covers PDF rendering and text-layer extraction:

```python
requirement = ocr.probe_pdf_requirements()
```

`RequirementResult` carries `engine`, stable `code`, backward-compatible
`missing_component`, complete `missing_components` tuple, `components_relation`,
`ocr_extra`, `component_type`, and optional `first_run_note`.
`missing_component` is always first tuple item. `components_relation` is `all`
when every listed component is required and `any` when installing one listed
component is enough — PDF backends use `any` because Poppler or PyMuPDF
satisfies them. Paddle preflight reports both `paddleocr` and `paddle` when both
import modules are absent. EasyOCR, OpenAI, binary, and configuration failures
use one-item tuples. Recognition raises `OcrRequirementError`, an `OcrError`
subclass, with same data under `.result` and direct attributes. Existing numeric
`.code` remains CLI exit status;
`.requirement_code`/`.stable_code` is stable requirement code. Current codes:
`ok`, `missing_tesseract_binary`, `missing_easyocr_package`,
`missing_easyocr_dependency`, `missing_paddleocr_package`,
`missing_paddle_runtime`, `missing_openai_package`,
`missing_vision_api_key`, `missing_vision_model`, `unsupported_engine`,
`missing_pdf_render_backend`, and `missing_pdf_text_backend`.

Probes stay cheap by checking module specs rather than importing. A module can
still exist by spec and fail to import (broken build, namespace shell, failed
native library load), so every engine import site converts that failure into the
same `OcrRequirementError` with the same stable code. A raw `ImportError` never
escapes recognition.

EasyOCR transitive failures use `missing_easyocr_dependency` and name the
failing component when determinable, for example `numpy`, `pillow`, or `torch`.
Optional local helpers degrade safely: broken pytesseract falls back to the
Tesseract CLI; broken Pillow skips basic preprocessing; broken OpenCV, NumPy, or
Pillow in enhanced/full preprocessing falls back to basic, then the original
image when Pillow is also unusable.

Python package failures identify distribution extra such as
`pro-ledin-ocr[easyocr]`; platform-specific binary/runtime commands remain in
the engine setup guide rather than assuming a source-tree `ocr.py` invocation.

`RecognizeOptions(verbose=True)` enables progress logging only. The 17-line
capability dump is a CLI diagnostic (`ocr -v`); library callers opt in with
`ocr.recognize(path, options, caps=ocr.Caps(report=True))`.

## Engine tiers

| Tier | Engine | Best for | Cost |
|---|---|---|---|
| 0 | Text layer + Camelot | Real text layers and native tables | Free, local |
| 1 | tesseract (default) | Clean scans and typed text | Free |
| 2 | easyocr | Handwriting and degraded scans | Free, heavy |
| 2.5 | paddleocr | CJK, multilingual, angled text | Free, heavy |
| 2.75 | paddleocr-vl-mlx | Local tables and complex document layout on Apple Silicon | Free, heavy |
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
