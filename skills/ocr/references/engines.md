# OCR Engines Reference

## Selection

| Tier | Engine | Best use | Languages | Requirement |
|---|---|---|---|---|
| 0 | Text layer | PDF already contains a readable text layer | Embedded text | Poppler or PyMuPDF |
| 1 | `tesseract` | Clean scans, typed text | 160+ with language data | Tesseract binary |
| 2 | `easyocr` | Handwriting, degraded scans | 80+ | `pro-ledin-ocr[easyocr]`, models |
| 2.5 | `paddleocr` | CJK, multilingual, angled text | 100+ | `pro-ledin-ocr[paddle]`, PaddlePaddle, models |
| 2.75 | `paddleocr-vl-mlx` | Tables and document layout on Apple Silicon | Model-supported | `pro-ledin-ocr[paddle-vl]`, loopback MLX-VLM service |
| 3 | `vision` | Tables, charts, forms, complex layout | Any model-supported language | `pro-ledin-ocr[vision]`, key, model |
| Handoff | `python -m pro.ledin.ocr.vision_handoff` | Current multimodal agent reads selected pages | Model-dependent | Image-capable agent |

Engine resolution: `--engine` > `OCR_ENGINE` > `tesseract`. Valid values are
`tesseract`, `easyocr`, `paddleocr`, `paddleocr-vl-mlx`, and `vision`. There is no `auto` value.
Automated OpenAI-compatible extraction uses `vision`; former alias is removed.

## Requirement probe

`ocr.probe_engine_requirements(engine, *, vision_api_key="",
vision_model="")` checks one selected engine without importing OCR packages or
triggering model/network work. It returns `RequirementResult`; recognition uses
same probe and raises `OcrRequirementError` for unavailable requirements.

| Stable code | Missing component | Extra | Component type |
|---|---|---|---|
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
Other current failures expose one-item tuples; successful result exposes empty
tuple.

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

`ocr --engine vision` sends rendered pages to OpenAI-compatible vision endpoint
with high-detail image input. Install `pro-ledin-ocr[vision]`. Model and API key
are required; endpoint URL is optional:

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
