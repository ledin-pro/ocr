# OCR Engines Reference

## Selection

| Tier | Engine | Best use | Languages | Requirement |
|---|---|---|---|---|
| 0 | Text layer + Camelot | Readable text PDF and native tables | Embedded text | Poppler/PyMuPDF plus bundled Camelot |
| Local opt-in | `apple-vision` | Photos with Russian captions on macOS; validate per document | OS/request-dependent | `pro-ledin-ocr[apple-vision]`, system Vision |
| 1 | `tesseract` | Clean scans, typed text | 160+ with language data | Tesseract binary |
| 2 | `easyocr` | Handwriting, degraded scans | 80+ | `pro-ledin-ocr[easyocr]`, models |
| 2.5 | `paddleocr` | CJK, multilingual, angled text | 100+ | `pro-ledin-ocr[paddle]`, PaddlePaddle, models |
| 2.75 | `paddleocr-vl-mlx` | Tables and document layout on Apple Silicon | Model-supported | `pro-ledin-ocr[paddle-vl]`, loopback MLX-VLM service |
| 3 | `vision` | Tables, charts, forms, complex layout | Any model-supported language | `pro-ledin-ocr`, key, model |
| Handoff | `python -m pro.ledin.ocr.vision_handoff` | Current multimodal agent reads selected pages | Model-dependent | Image-capable agent |

Engine resolution: `--engine` > `OCR_ENGINE` > `tesseract`. Valid values are
`tesseract`, `easyocr`, `paddleocr`, `paddleocr-vl-mlx`, `apple-vision`, and
`vision`. There is no `auto` value.
Automated OpenAI-compatible extraction uses `vision`; former alias is removed.

## Local Apple Vision

`apple-vision` explicitly selects macOS Vision through PyObjC, with recognition
level `.accurate` (no fast mode). It is a local option for photos with Russian
captions, not a replacement for the default `tesseract` or the OpenAI-compatible
API engine `vision`.

```bash
ocr photo.jpg --engine apple-vision --lang rus+eng --format md,json --out results/
ocr scan.pdf --engine apple-vision --lang rus+eng --format md,json --out results/
ocr scan.pdf --lang rus+eng --auto-escalate apple-vision --format md,json --out results/
```

`rus+eng` maps to `ru-RU` and `en-US`. Explicit language identifiers are checked
against the current OS/request revision; unsupported languages fail explicitly.
`--lang auto` uses Vision language detection where available (macOS 13+), not
Tesseract OSD. Custom `--psm` settings are unsupported.

Scanned PDFs use existing page rendering and preprocessing; readable text PDFs
keep native extraction and Camelot. JSON `words` contains whole-line observations
with `granularity="line"`, not inferred word boxes. Confidence is scaled to
0–100; `bbox=[left, top, width, height]` is in pixels with a top-left origin on
the decoded image passed to recognition (after rendering/preprocessing), not
necessarily the original photo or EXIF display coordinates.

There is no orientation search or automatic best-rotation selection, and no
Apple Vision table/layout reconstruction. Inspect source orientation and line
content rather than choosing a rotation by character count.

Apple Vision can be baseline or a strict escalation candidate. All requested
requirements are preflighted; platform, dependency, or attempt failures stop the
workflow without substitution or hidden API upload. Local candidates follow the
same selection rules described below.

### Verified local results

Verified on macOS 27.2 arm64, Python 3.14.3, and PyObjC 13.0 using the installed
CLI from the reviewed local repository with extra `apple-vision`; system Python
was unchanged. CLI checks used `--lang rus+eng --preprocess none` and JSON output:

| Input | Mean confidence | Content check | Elapsed |
|---|---|---|---|
| Local photo A | 75.8 | 3/4 expected labels | 7.600 s |
| Local photo B | 75.7 | 4/4 expected labels | 0.521 s |
| Synthetic two-page PDF | 100.0 on each page | All expected lines/numbers | 0.798 s |

Pixel bounds were valid. Photo sources and recognized content remain outside
the repository/CI and were not sent to external APIs. The missed label is a
real limitation, not a successful full transcription. The local suite passed
232 tests without skips, including real image, automatic-language, and PDF
smoke checks. macOS 14 CI has not yet run.

These measurements cover this host and these inputs only, not a compatibility
or performance matrix. Record versions, material lines, confidence, box bounds,
and timings for other hosts/documents; do not infer cold-start causes or promise
better accuracy/speed on untested documents.

## Native Camelot table stage

Camelot runs only after the text-layer readability gate accepts a PDF. It is not
an OCR engine and does not participate in `--auto-escalate`.

| Flavor | Best use |
|---|---|
| `auto` | Default; compare classic auto-detection with a `stream` candidate |
| `lattice` | Tables with explicit horizontal and vertical rules |
| `stream` | Borderless tables separated by whitespace and alignment |
| `network` | Repeated coordinate alignments and form-like grids |
| `hybrid` | Mixed line and alignment cues |

`ml` is intentionally not exposed. Automatic native extraction must not load a
visual model. Disable the stage with `--no-tables`.

Camelot candidates are normalized and checked against the source table region.
Low parse quality, weak text coverage, or lost numeric tokens prevents
replacement. Accepted regions become pipe Markdown or HTML; the original linear
text remains available in TXT and JSON `text`.

Limitations: image-only tables require OCR; malformed text layers are rejected
before Camelot; cross-page tables remain page-local; merged cells may be
flattened and marked `table_spans_flattened` when geometry is ambiguous.

## Requirement probe

`ocr.probe_engine_requirements(engine, *, vision_api_key="",
vision_model="")` checks one selected engine without importing OCR packages or
triggering model/network work. It returns `RequirementResult`; recognition uses
same probe and raises `OcrRequirementError` for unavailable requirements.

| Stable code | Missing component | Extra | Component type |
|---|---|---|---|
| `unsupported_apple_vision_platform` | `macOS` | `apple-vision` | `platform` |
| `missing_apple_vision_packages` | absent `Vision`, `Foundation`, `Quartz`, `objc` modules | `apple-vision` | `python-package` |
| `apple_vision_import_failed` | `PyObjC` | `apple-vision` | `python-runtime` |
| `missing_tesseract_binary` | `tesseract` | none | `binary` |
| `missing_easyocr_package` | `easyocr` | `easyocr` | `python-package` |
| `missing_easyocr_dependency` | detected transitive module | `easyocr` | `python-package` |
| `missing_paddleocr_package` | `paddleocr` | `paddle` | `python-package` |
| `missing_paddle_runtime` | `paddle` | `paddle` | `python-runtime` |
| `missing_paddleocr_doc_parser` | `paddleocr` and/or `paddle` | `paddle-vl` | `python-package` |
| `missing_paddle_vl_server_url` | `paddle_vl_server_url` | `paddle-vl` | `configuration` |
| `unsafe_paddle_vl_server_url` | `paddle_vl_server_url` | `paddle-vl` | `configuration` |
| `missing_paddle_vl_model` | `paddle_vl_model` | `paddle-vl` | `configuration` |
| `missing_openai_package` | `openai` | `vision` | `python-package` |
| `missing_vision_api_key` | `vision_api_key` | `vision` | `configuration` |
| `missing_vision_model` | `vision_model` | `vision` | `configuration` |
| `unsupported_engine` | supplied engine | none | `engine` |
| `missing_pdf_render_backend` | `pdftoppm`, `pymupdf` | `pdf` | `binary-or-python-package` |
| `missing_pdf_text_backend` | `pdftotext`, `pymupdf` | `pdf` | `binary-or-python-package` |

Successful result uses `code="ok"`. EasyOCR and PaddleOCR results include
first-run model-download note. Probe checks OpenAI package before explicit vision
key/model, never reads `OPENAI_API_KEY`, and returns first unmet requirement.
`OcrRequirementError.code` remains numeric process exit code;
`requirement_code`/`stable_code` carries table value.

`ocr.probe_pdf_requirements(caps=None)` checks PDF render and text-layer needs
with same result type. Render needs `pdftoppm` or PyMuPDF; text layer needs
`pdftotext` or PyMuPDF. Those results set `components_relation="any"`, meaning
one listed component satisfies requirement. Engine results use
`components_relation="all"`. `Caps.require_render()` and
`Caps.require_pdftotext()` raise `OcrRequirementError` built from that probe.

## Text-layer readability gate

Tier 0 is chosen only when an embedded text layer is both dense enough
(`median >= MIN_TEXT_CHARS` non-space bytes/page) and readable. The probe scores
extracted sample text with `ocr.text_readability(text)`, returning
`letter_digit_ratio`, `word_score` (share of tokens that are real words or bare
numeric/measurement values), and `replacement_ratio` (U+FFFD and control chars).
A dense-but-unreadable layer — typically a scan whose fonts have no `ToUnicode`
map, producing symbol soup — is rejected: `needs_ocr=True`,
`text_layer_rejected=True`, and every produced page carries a
`text_layer_rejected` issue. Recognition then renders pages and runs the selected
engine. Font metadata (`all fonts non-Unicode`) is reported but not decisive:
legitimate base-14 fonts also report no `ToUnicode` yet score readable. When no
text can be extracted, the probe falls back to the density-only decision so real
text is never rejected on missing signals. Bumping the readability policy bumps
`OCR_OUTPUT_SCHEMA_VERSION`, invalidating stale cache entries.

Probe reads module specs and executable paths only. Broken installs can still
raise at import time, so engine import sites convert `ImportError` and native
load failures into `OcrRequirementError` carrying same stable code. A failing
`from paddleocr import PaddleOCR` is attributed to `missing_paddle_runtime` when
underlying `paddle` module fails, otherwise `missing_paddleocr_package`. When
PyMuPDF is installed but broken, PDF paths fall back to Poppler where available
and raise structured PDF codes only when no backend remains.

EasyOCR imports its own package plus NumPy and Pillow. Failure of EasyOCR package
uses `missing_easyocr_package`; a determinable transitive failure uses
`missing_easyocr_dependency` with component such as `numpy`, `pillow`, or
`torch`. Both retain `engine="easyocr"`, `ocr_extra="easyocr"`, and
`component_type="python-package"`.

`missing_components` is tuple containing every missing module found during one
engine preflight. `missing_component` remains first tuple item for compatibility.
If both Paddle modules are absent, result keeps code
`missing_paddleocr_package`, component type `python-package`, singular
`paddleocr`, and reports `("paddleocr", "paddle")` in plural field. If only
runtime is absent, result reports one-item tuple `("paddle",)`; PaddlePaddle is
the distribution providing that import module.
Apple Vision similarly reports every absent PyObjC module in one result.
Single-component failures expose one-item tuples; successful results expose an
empty tuple. `apple_vision_import_failed` is a runtime import error, not a
side-effect-free probe result.

## Strict auto-escalation

Configure ordered engines with CLI or environment:

```bash
ocr FILE --auto-escalate easyocr,paddleocr --format md,json --out results/
OCR_AUTO_ESCALATE=easyocr,vision ocr FILE \
  --vision-api-key "$KEY" --vision-model MODEL
```

`--auto-escalate` overrides `OCR_AUTO_ESCALATE`. Duplicate entries are removed;
baseline engine is omitted from chain.

Strict semantics:

1. Baseline engine and every escalation engine are dependency-checked before
   OCR starts.
2. Baseline runs once. Only flagged pages enter escalation.
3. Every flagged page runs through every chain engine in listed order.
4. Local candidates are ranked by unflagged status, mean confidence, then word
   count. `vision` is final selected candidate when chain reaches it.
5. Output page receives `decision.baseline`, `decision.flag_reason`, ordered
   `decision.attempts`, and `decision.selected`.
6. Missing dependency, empty result, or engine failure aborts workflow. Later
   attempts do not run and workflow cache is not written.

No chain means no automatic retry. CLI reports flagged pages and suggests a
targeted rerun; skill must obtain user approval before that rerun.

## Flagging

Default `--min-conf` is `60`. Pages are flagged for low mean confidence,
table-like word geometry, or engine-specific review reason. Typical targeted
commands:

```bash
ocr FILE --engine easyocr --pages 5,9
ocr FILE --engine paddleocr --pages 5,9
ocr FILE --engine vision --pages 5,9 \
  --vision-api-key "$KEY" --vision-model MODEL
```

## Language detection

Tesseract OSD inspects script and orientation before OCR when `--lang auto`.
Common mapping:

| Script | Tesseract languages |
|---|---|
| Cyrillic | `rus+eng` |
| Latin | `eng` |
| Han | `chi_sim+eng` |
| Arabic | `ara+eng` |
| Devanagari | `hin+eng` |
| Korean | `kor+eng` |
| Japanese | `jpn+eng` |
| Greek | `ell+eng` |
| Hebrew | `heb+eng` |
| Unknown | `eng` with warning |

Override with `--lang rus+eng`. Verify installed data with
`tesseract --list-langs`; setup commands live in `engine-setup.md`.

PaddleOCR maps primary Tesseract code internally: `eng` to `en`, `rus` to `ru`,
`jpn` to `japan`, `kor` to `korean`, `chi_sim` to `ch`, `chi_tra` to
`chinese_cht`, `ara` to `arabic`, and `hin` to `hi`. Composite specs use first
code. Engine targets PaddleOCR 3.x `predict()` API.

## DPI

| Page type | DPI | Notes |
|---|---:|---|
| A4 / Letter | 300 | Tesseract baseline sweet spot |
| Wide slides | 150 | Avoid oversized renders |
| Small image | upscale through preprocessing | Aim for roughly 1400 px width |
| Phone photo | source resolution | Usually already high resolution |
| Fax / low resolution | 300-400 plus full preprocessing | Denoise and threshold |

`--dpi 0`/default selects automatically. Interactive handoff uses same automatic
choice unless `--dpi` is supplied.

## Preprocessing

| Level | Behavior | Best use |
|---|---|---|
| `none` | Raw image | Clean digital render |
| `basic` | Grayscale, contrast, sharpen, small-image upscale | Clean scan |
| `enhanced` | Basic plus denoise and adaptive threshold | Noise, uneven light |
| `full` | Enhanced plus deskew | Tilted scans and phone photos |
| `auto` | Chooses from source characteristics | Default |

`enhanced` and `full` require OpenCV and NumPy. Adaptive threshold can damage
clean digital text; use `basic` or `none` when output worsens.
Import/load failures degrade safely: enhanced/full falls back to basic; basic
returns original image if Pillow cannot load. pytesseract is optional even when
detected by spec; failed import falls back to Tesseract CLI.

## Automated vision

`ocr --engine vision` sends rendered pages to an OpenAI-compatible vision endpoint
with high-detail image input. The base `pro-ledin-ocr` package includes the
OpenAI client. Model and API key are required; endpoint URL is optional:

```bash
ocr FILE --engine vision \
  --vision-api-key "$KEY" \
  --vision-model MODEL \
  --vision-api-url https://api.example.com/v1 \
  --vision-prompt-file prompts/faithful-ocr.txt
```

Key is never read from `OPENAI_API_KEY`. `--vision-prompt` and
`--vision-prompt-file` are mutually exclusive. Effective prompt participates in
cache key. Review privacy requirements before sending documents externally.

## Interactive handoff

Interactive handoff only renders selected pages and prints persistent image
paths plus prompt for current agent:

```bash
python -m pro.ledin.ocr.vision_handoff FILE \
  --pages 1-3,5 --dpi 200 \
  --vision-prompt-file prompts/faithful-ocr.txt
```

It accepts page selection, DPI, prompt text/file, and verbose output. It has no
page-count cap option. It is not valid escalation engine and cannot run inside
peepshow sink.

## Searchable PDFs

```bash
brew install ocrmypdf
sudo apt install ocrmypdf
ocr input.pdf --searchable-pdf output.pdf --lang rus+eng
```

See `engine-setup.md` before installing requested engines. Detect platform and
ask approval first.
