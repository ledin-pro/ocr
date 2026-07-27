# OCR Engines Reference

## Selection

| Tier | Engine | Best use | Languages | Requirement |
|---|---|---|---|---|
| 0 | Text layer | PDF already contains usable text | Embedded text | Poppler or PyMuPDF |
| 1 | `tesseract` | Clean scans, typed text | 160+ with language data | Tesseract binary |
| 2 | `easyocr` | Handwriting, degraded scans | 80+ | `pro-ledin-ocr[easyocr]`, models |
| 2.5 | `paddleocr` | CJK, multilingual, angled text | 100+ | `pro-ledin-ocr[paddle]`, PaddlePaddle, models |
| 3 | `vision` | Tables, charts, forms, complex layout | Any model-supported language | `pro-ledin-ocr[vision]`, key, model |
| Handoff | `python -m pro.ledin.ocr.vision_handoff` | Current multimodal agent reads selected pages | Model-dependent | Image-capable agent |

Engine resolution: `--engine` > `OCR_ENGINE` > `tesseract`. Valid values are
`tesseract`, `easyocr`, `paddleocr`, and `vision`. There is no `auto` value.
Automated OpenAI-compatible extraction uses `vision`; former alias is removed.

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
