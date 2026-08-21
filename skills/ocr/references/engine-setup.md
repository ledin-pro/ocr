# OCR Engine Setup

Use this reference only after selected engine or escalation dependency fails.
Never install automatically.

Library orchestrators should call `ocr.probe_engine_requirements()` first and
branch on stable result `code`, `component_type`, and `ocr_extra`. Probe is
diagnostic only: no package imports, installation, model download, network call,
or environment credential lookup. `missing_components` names every absent
module; `missing_component` remains first item for compatibility. Paddle failures
therefore distinguish and, when needed, jointly report `paddleocr` and `paddle`
import modules. `paddle` is provided by PaddlePaddle runtime distribution.
EasyOCR/Paddle results include optional first-run note. `ocr.probe_pdf_requirements()`
covers PDF render and text-layer backends and marks alternatives with
`components_relation="any"`.

Probe is spec-based, so an installed-but-broken package still fails at import.
Those failures surface as same structured `OcrRequirementError` with same stable
code; treat them as reinstall of reported component for detected platform.
EasyOCR transitive failures use `missing_easyocr_dependency` and name component
when determinable. pytesseract and preprocessing dependencies are optional and
fall back without requiring installation.

## Approval workflow

1. Detect OS and version, CPU architecture, Python version and bitness.
2. For GPU engines, detect GPU vendor/model, driver, and supported CUDA runtime.
3. Select commands below matching detected platform. Explain downloads, model
   cache, disk use, and whether CPU or GPU build will be installed.
4. Ask user approval for exact commands.
5. Install only approved packages.
6. Verify dependency, then rerun original OCR command unchanged.

Useful detection commands:

```bash
# macOS
sw_vers
uname -m
python -c "import platform,sys; print(sys.version); print(platform.architecture())"
system_profiler SPDisplaysDataType

# Linux
uname -a
cat /etc/os-release
python -c "import platform,sys; print(sys.version); print(platform.architecture())"
lspci | grep -Ei 'vga|3d|display'
nvidia-smi

# Windows PowerShell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsArchitecture
python -c "import platform,sys; print(sys.version); print(platform.architecture())"
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
nvidia-smi
```

Commands may be absent; report that and use available platform facts. Python
3.10-3.12 is safest shared range for this package, PaddleOCR 3.x, and
PaddlePaddle 3.2.

## Baseline: Poppler and Tesseract

### macOS

```bash
brew install poppler tesseract tesseract-lang
pdftoppm -v
pdfinfo -v
tesseract --version
tesseract --list-langs
```

Homebrew `tesseract` contains `eng`, `osd`, and `snum`; `tesseract-lang` adds
other language data. Official sources:

- [Homebrew Poppler formula](https://formulae.brew.sh/formula/poppler)
- [Homebrew Tesseract formula](https://formulae.brew.sh/formula/tesseract)
- [Homebrew tesseract-lang formula](https://formulae.brew.sh/formula/tesseract-lang)

### Debian / Ubuntu

Install only needed language package where possible:

```bash
sudo apt update
sudo apt install poppler-utils tesseract-ocr tesseract-ocr-rus

# Or all packaged Tesseract languages:
sudo apt install poppler-utils tesseract-ocr-all

pdftoppm -v
pdfinfo -v
tesseract --version
tesseract --list-langs
```

Language package naming follows `tesseract-ocr-<langcode>`, for example
`tesseract-ocr-deu`. Official sources:

- [Tesseract installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- [Ubuntu poppler-utils package](https://packages.ubuntu.com/search?keywords=poppler-utils)
- [Debian tesseract-ocr packages](https://packages.debian.org/search?keywords=tesseract-ocr)

### Windows

Tesseract: use Windows installer maintained by UB Mannheim. During install,
select needed language data. Add install directory, commonly
`C:\Program Files\Tesseract-OCR`, to system or user `PATH`. Additional
`.traineddata` files belong in `C:\Program Files\Tesseract-OCR\tessdata` unless
installation uses another data directory. Download trained data only from
official Tesseract repositories and verify `TESSDATA_PREFIX` if language lookup
fails.

Poppler through WinGet:

```powershell
winget install --exact --id oschwartz10612.Poppler
```

Restart terminal after PATH changes, then verify:

```powershell
where.exe tesseract
where.exe pdftoppm
tesseract --version
tesseract --list-langs
pdftoppm -v
```

Official/maintainer sources:

- [Tesseract Windows installation guidance](https://tesseract-ocr.github.io/tessdoc/Installation.html#windows)
- [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki)
- [Tesseract tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
- [Tesseract tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
- [oschwartz10612 Poppler Windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
- [WinGet package manifests](https://github.com/microsoft/winget-pkgs)

## EasyOCR

Install package extra after platform checks:

```bash
python -m pip install "pro-ledin-ocr[easyocr]"
python -c "import easyocr,torch; print(easyocr.__version__); print(torch.__version__); print(torch.cuda.is_available())"
```

On Windows, and whenever GPU acceleration matters, install `torch` and
`torchvision` first using command generated by official PyTorch selector for
detected OS, package manager, language, compute platform, and CUDA version. Do
not guess CUDA wheel. Select CPU when no compatible NVIDIA GPU/driver exists.
Then install OCR extra and verify again.

- [Official PyTorch install selector](https://pytorch.org/get-started/locally/)
- [EasyOCR official repository and install notes](https://github.com/JaidedAI/EasyOCR)
- [EasyOCR model hub](https://www.jaided.ai/easyocr/modelhub/)

First use downloads language-specific detection/recognition weights. Default
cache is `~/.EasyOCR/model`; offline installs can place official model files
there manually. Warn user before download and ensure writable cache/disk space.

## PaddleOCR and PaddlePaddle 3.2

Use Python 3.10-3.12 for supported overlap. Install PaddlePaddle runtime matching
platform before `pro-ledin-ocr[paddle]`. Never install CPU and GPU runtimes
together in same environment.

### Linux or Windows CPU

```bash
python -m pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "pro-ledin-ocr[paddle]"
```

### Linux or Windows NVIDIA GPU

Choose index matching driver-supported CUDA family. CUDA 11.8 generally needs
NVIDIA driver at least 450.80.02 on Linux or 452.39 on Windows; CUDA 12.6 needs
driver at least 550.54.14. Confirm current official matrix before approval.

```bash
# CUDA 11.8
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.6
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

python -m pip install pro-ledin-ocr paddleocr
```

Do not install `pro-ledin-ocr[paddle]` in GPU environment because that extra
declares CPU `paddlepaddle` runtime. Install package plus `paddleocr` as above
after GPU runtime.

### macOS CPU

PaddlePaddle supports CPU only on macOS. Confirm current architecture support in
official installer before approval; current releases target Apple Silicon.

```bash
python -m pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "pro-ledin-ocr[paddle]"
```

### Verify and models

```bash
python -c "import paddle; paddle.utils.run_check(); print(paddle.__version__); print(paddle.device.get_device())"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
```

First OCR run downloads detection, recognition, and orientation models into
PaddleOCR/PaddleX cache. Warn user about network and disk use. For offline use,
follow official model download/cache documentation rather than copying
unverified weights.

- [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
- [PaddlePaddle Linux pip guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html)
- [PaddlePaddle macOS pip guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html)
- [PaddleOCR installation](https://www.paddleocr.ai/latest/en/version3.x/installation.html)
- [PaddleOCR model list and downloads](https://www.paddleocr.ai/latest/en/version3.x/module_usage/ocr_modules/text_detection.html)

### PaddleOCR-VL with MLX on Apple Silicon

Follow the official sequence in an isolated environment. First validate full
direct `PaddleOCRVL` inference on CPU using PaddlePaddle 3.2.1 or later and
`paddleocr[doc-parser]`. Only then install `mlx-vlm>=0.3.11` and start its
VLM-only service:

```bash
python -m pip install "pro-ledin-ocr[paddle-vl]"
python -m pip install "mlx-vlm>=0.3.11"
mlx_vlm.server --host 127.0.0.1 --port 8111
```

The MLX service is not an end-to-end parsing API. Never send document images to
it directly; run the complete `PaddleOCRVL` client with backend
`mlx-vlm-server`. Apple M1 is listed as supported Apple Silicon, but PaddleOCR
currently reports accuracy and speed verification only on M4.

- [Official PaddleOCR-VL Apple Silicon guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html#31-starting-the-vlm-inference-service)

## Automated vision

```bash
python -m pip install pro-ledin-ocr
python -c "import openai; print(openai.__version__)"
```

`vision` needs `--vision-api-key` and `--vision-model`, or the corresponding
`OCR_VISION_API_KEY` and `OCR_VISION_MODEL` environment variables. Optional
`--vision-api-url` / `OCR_VISION_API_URL` selects the OpenAI-compatible endpoint;
CLI flags take precedence. Credentials are not read from generic OpenAI
environment variables. Verification should avoid uploading user document until
user approves external processing. Package source:

- [OpenAI Python library](https://github.com/openai/openai-python)

## Rerun rule

After successful verification, rerun exact failed OCR command. Preserve input,
engine, escalation chain, page selection, formats, output path, language,
preprocessing, and vision options. Report any first-run model download separately.
