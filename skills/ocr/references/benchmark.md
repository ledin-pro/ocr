# OCR Profile Benchmark

`scripts/ocr-profile-benchmark.py` compares OCR profiles served by oMLX with
OpenAI compatible models. It calls `pro.ledin.ocr` directly;
it does not invoke HealthOS or another OCR wrapper.

## Run

Copy `scripts/ocr-profile-benchmark.example.json` and set:

- `source_dir` — directory containing input PDFs.
- `run_root` — directory for reports and resume state.
- `targets` — objects with `kind` (`omlx` or `gpt`), `model`, and `slug`.
- `judge.model` — model used by the OpenAI compatible judge.
- `OMLX_API_KEY` and `OPENAI_API_KEY` — service credentials named by the config.

```bash
export OMLX_API_KEY=your-omlx-key
export OPENAI_API_KEY=your-openai-key
python scripts/ocr-profile-benchmark.py --config benchmark.json
```

Relative paths are resolved relative to the configuration file. Use
`--only-model MODEL` to run one target, `--page DOCUMENT.pdf:PAGE` to create a
single-page benchmark input, and `--skip-judge` to collect OCR outputs without
the judge phase.

## Configuration

Important settings include:

| Key | Default | Purpose |
|---|---:|---|
| `ocr_timeout_seconds` | `900` | Timeout for each vision HTTP request |
| `http_timeout_seconds` | `60` | oMLX/OpenAI compatible control and judge requests |
| `profile_exposure_poll_attempts` | `30` | Attempts to wait for an exposed oMLX profile |
| `judge_attempts` | `5` | Retries for a judge request |
| `judge_retry_delay_seconds` | `5` | Delay multiplier between judge retries |
| `pdftoppm_bin` | `pdftoppm` | PDF page renderer for judging |
| `pdfseparate_bin` | `pdfseparate` | Tool for `--page` selection |

The benchmark always uses the OCR `vision` engine. For each PDF it constructs
`RecognizeOptions` with the target endpoint, API key, exposed model, and
`ocr_timeout_seconds`, then writes `to_markdown()` output to:

```text
<run_root>/<target>/output/<document-stem>.md
```

Per-target logs are stored as `ocr.log`. API keys are never written to the
benchmark event log, OCR log, summary, or run configuration.

## Resume

```bash
python scripts/ocr-profile-benchmark.py --resume path/to/run
```

New runs save their normalized settings in `run-config.json`. Completed targets
are skipped when all expected Markdown documents exist. Run configurations must
use the current format version; old runs must be started again with a new
configuration.
