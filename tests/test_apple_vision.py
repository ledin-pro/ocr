"""Apple Vision contracts at the public probe/recognize and CLI seams.

Framework doubles are an external boundary, not adapter-internal mocks. The
separate smoke test exercises the real bridge when explicitly requested.
"""
import builtins
import contextlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest

from pro.ledin.ocr import core, cli


def test_probe_rejects_non_macos_without_importing_frameworks():
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"Vision", "Foundation", "Quartz", "objc"}
        return original_import(name, *args, **kwargs)

    with mock.patch.object(core.sys, "platform", "linux"), mock.patch(
        "builtins.__import__", side_effect=guarded_import
    ):
        result = core.probe_engine_requirements("apple-vision")
    assert not result.available
    assert result.code == "unsupported_apple_vision_platform"
    assert result.component_type == "platform"
    assert result.missing_component == "macOS"


def test_probe_lists_missing_bridge_components_without_imports():
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"Vision", "Foundation", "Quartz", "objc"}
        return original_import(name, *args, **kwargs)

    caps = {"has_vision": False, "has_foundation": True,
            "has_quartz": False, "has_objc": True}
    with mock.patch.object(core.sys, "platform", "darwin"), mock.patch(
        "builtins.__import__", side_effect=guarded_import
    ):
        result = core.probe_engine_requirements("apple-vision", caps=caps)
        available = core.probe_engine_requirements(
            "apple-vision", caps={key: True for key in caps}
        )
    assert result.code == "missing_apple_vision_packages"
    assert result.missing_components == ("Vision", "Quartz")
    assert result.ocr_extra == "apple-vision"
    assert result.components_relation == "all"
    assert available.available


def test_caps_report_includes_spec_based_apple_dependencies(capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {"Vision", "Foundation", "Quartz", "objc"}
        return original_import(name, *args, **kwargs)

    def module_spec(name):
        return object() if name in {"Vision", "objc"} else None

    with mock.patch.object(core.sys, "platform", "darwin"), mock.patch(
        "builtins.__import__", side_effect=guarded_import
    ), mock.patch("importlib.util.find_spec", side_effect=module_spec):
        caps = core.Caps(report=True)
        result = core.probe_engine_requirements("apple-vision", caps=caps)
    report = capsys.readouterr().err
    rows = {row.split()[0]: row.split()[1] for row in report.splitlines()[1:]}
    assert rows["has_vision"] == "OK"
    assert rows["has_foundation"] == "MISSING"
    assert rows["has_quartz"] == "MISSING"
    assert rows["has_objc"] == "OK"
    assert result.missing_components == ("Foundation", "Quartz")


@pytest.fixture
def apple_frameworks(monkeypatch):
    """A documented PyObjC boundary; geometry is normalized bottom-left."""
    state = types.SimpleNamespace(
        observations=[("Hello world", 0.9, (0.1, 0.7, 0.5, 0.1)),
                      ("Привет мир", 0.8, (0.2, 0.2, 0.4, 0.2))],
        calls=[], error=None,
    )

    class Request:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setRecognitionLevel_(self, level):
            assert level == 0  # Apple's accurate enum, never fast.

        def supportedRecognitionLanguagesAndReturnError_(self, error):
            return ["en-US", "ru-RU", "fr-FR"], None

        def setRecognitionLanguages_(self, languages):
            state.languages = languages

        def setAutomaticallyDetectsLanguage_(self, value):
            state.auto = value

        def results(self):
            results = []
            for text, conf, rect in state.observations:
                x, y, w, h = rect
                candidate = types.SimpleNamespace(
                    string=lambda text=text: text,
                    confidence=lambda conf=conf: conf,
                )
                results.append(types.SimpleNamespace(
                    topCandidates_=lambda n, candidate=candidate: [candidate],
                    boundingBox=lambda x=x, y=y, w=w, h=h: types.SimpleNamespace(
                        origin=types.SimpleNamespace(x=x, y=y),
                        size=types.SimpleNamespace(width=w, height=h),
                    ),
                ))
            return results

    class Handler:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithCGImage_options_(self, image, options):
            return self

        def performRequests_error_(self, requests, error):
            state.calls.append("apple-vision")
            return (False, state.error) if state.error else (True, None)

    modules = {
        "Vision": types.SimpleNamespace(VNRecognizeTextRequest=Request,
                    VNImageRequestHandler=Handler, VNRequestTextRecognitionLevelAccurate=0),
        "Foundation": types.SimpleNamespace(
            NSDictionary=types.SimpleNamespace(dictionary=lambda: {}),
            NSURL=types.SimpleNamespace(fileURLWithPath_=lambda path: path),
        ),
        "Quartz": types.SimpleNamespace(
            CGImageSourceCreateWithURL=lambda url, options: url,
            CGImageSourceCreateImageAtIndex=lambda source, index, options: object(),
            CGImageGetWidth=lambda image: 1000,
            CGImageGetHeight=lambda image: 500,
        ),
        "objc": types.SimpleNamespace(autorelease_pool=contextlib.nullcontext),
    }
    monkeypatch.setattr(core.sys, "platform", "darwin")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    state.caps = core.Caps()
    for name in modules:
        setattr(state.caps, f"has_{name.lower()}", True)
    return state


def recognize_image(state, **kwargs):
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png") as image:
        Image.new("RGB", (1000, 500), "white").save(image.name)
        options = {"engine": "apple-vision", "preprocess": "none", "no_cleanup": True}
        options.update(kwargs)
        return core.recognize(image.name, core.RecognizeOptions(**options), caps=state.caps)


def test_recognize_image_returns_line_geometry_and_scaled_confidence(apple_frameworks):
    page, = recognize_image(apple_frameworks, lang="rus+eng")
    assert page["source"] == "apple-vision"
    assert page["n"] == 1
    assert page["text"] == "Hello world\nПривет мир"
    assert page["mean_conf"] == 85.0
    assert page["words"] == [
        {"text": "Hello world", "conf": 90.0, "bbox": [100, 100, 500, 50],
         "granularity": "line"},
        {"text": "Привет мир", "conf": 80.0, "bbox": [200, 300, 400, 100],
         "granularity": "line"},
    ]
    assert apple_frameworks.languages == ["ru-RU", "en-US"]
    assert apple_frameworks.calls == ["apple-vision"]


@pytest.mark.parametrize("failure", [ImportError, OSError, RuntimeError])
def test_recognize_reports_broken_bridge_without_fallback(apple_frameworks, failure):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "Vision":
            raise failure("broken native library")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=broken_import):
        with pytest.raises(core.OcrRequirementError) as caught:
            recognize_image(apple_frameworks, lang="eng")
    assert caught.value.requirement_code == "apple_vision_import_failed"
    assert caught.value.ocr_extra == "apple-vision"
    assert caught.value.code == core.EXIT_MISSING_BINARY
    assert apple_frameworks.calls == []


@pytest.mark.parametrize("options, message", [
    ({"lang": "xyz"}, "Unsupported apple-vision language"),
    ({"lang": "eng", "psm": 6}, "psm"),
])
def test_recognize_rejects_unsupported_settings(apple_frameworks, options, message):
    with pytest.raises(core.OcrError, match=message):
        recognize_image(apple_frameworks, **options)
    assert apple_frameworks.calls == []


def test_cli_exposes_apple_vision_without_changing_existing_engines(apple_frameworks, capsys):
    with tempfile.NamedTemporaryFile(suffix=".png") as image, mock.patch.object(
        cli, "Caps", return_value=apple_frameworks.caps
    ):
        code = cli.main([image.name, "--engine", "apple-vision", "--lang", "eng",
                         "--preprocess", "none", "--format", "txt"])
    assert code == 0
    assert "Hello world" in capsys.readouterr().out
    assert core.RecognizeOptions().engine == "tesseract"
    assert core.probe_engine_requirements("vision", vision_api_key="key",
        vision_model="model", caps={"has_openai": True}).available


@pytest.mark.parametrize("preprocess, dimensions, box", [
    ("none", (1000, 500), [100, 100, 500, 50]),
    ("basic", (2000, 1000), [200, 200, 1000, 100]),
])
def test_scanned_pdf_renders_selected_pages_before_apple_vision(
    apple_frameworks, preprocess, dimensions, box
):
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "scan.pdf"
        canvas = Canvas(str(pdf), pagesize=(400, 200))
        for _ in range(3):
            canvas.drawImage(ImageReader(Image.new("RGB", (400, 200), "white")),
                             0, 0, width=400, height=200)
            canvas.showPage()
        canvas.save()
        decoded = []
        quartz = sys.modules["Quartz"]

        def decode(source, index, options):
            assert source.endswith(".png")
            assert source != str(pdf)
            with Image.open(source) as image:
                decoded.append(image.size)
            return decoded[-1]

        with mock.patch.object(quartz, "CGImageSourceCreateImageAtIndex", side_effect=decode), mock.patch.object(
            quartz, "CGImageGetWidth", side_effect=lambda image: image[0]
        ), mock.patch.object(quartz, "CGImageGetHeight", side_effect=lambda image: image[1]):
            pages = core.recognize(pdf, core.RecognizeOptions(
                engine="apple-vision", lang="eng", pages="2-3", dpi=180,
                preprocess=preprocess, no_cleanup=True,
            ), caps=apple_frameworks.caps)
    assert [page["n"] for page in pages] == [2, 3]
    assert [page["source"] for page in pages] == ["apple-vision", "apple-vision"]
    assert decoded == [dimensions, dimensions]
    assert pages[0]["words"][0]["bbox"] == box


@pytest.fixture
def escalation_engines(apple_frameworks, monkeypatch):
    state = apple_frameworks

    def tesseract_data(*args, **kwargs):
        state.calls.append("tesseract")
        return {"text": ["baseline"], "conf": [30], "left": [0], "top": [0],
                "width": [100], "height": [50]}

    monkeypatch.setitem(sys.modules, "pytesseract", types.SimpleNamespace(
        Output=types.SimpleNamespace(DICT="dict"), image_to_data=tesseract_data,
        image_to_string=lambda *args, **kwargs: "baseline",
    ))

    def easy_result(*args, **kwargs):
        state.calls.append("easyocr")
        return [([[0, 0], [100, 0], [100, 50], [0, 50]], "alternative", 0.7)]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(
        Reader=lambda *args, **kwargs: types.SimpleNamespace(readtext=easy_result)
    ))
    state.caps.has_pytesseract = True
    state.caps.bin_tesseract = "synthetic-tesseract"
    state.caps.has_easyocr = True
    return state


def test_strict_chain_preserves_order_and_selects_best_confidence(escalation_engines):
    state = escalation_engines
    page, = recognize_image(state, engine="tesseract", lang="rus+eng",
                            auto_escalate=("apple-vision", "easyocr"))
    assert state.calls == ["tesseract", "apple-vision", "easyocr"]
    assert page["source"] == "apple-vision"
    assert page["decision"] == {
        "baseline": "tesseract", "flag_reason": ["low_confidence"],
        "attempts": [
            {"engine": "tesseract", "status": "completed", "mean_conf": 30.0},
            {"engine": "apple-vision", "status": "completed", "mean_conf": 85.0},
            {"engine": "easyocr", "status": "completed", "mean_conf": 70.0},
        ],
        "selected": "apple-vision",
    }


def test_strict_chain_preflights_apple_before_baseline(escalation_engines):
    state = escalation_engines
    state.caps.has_vision = False
    with pytest.raises(core.OcrRequirementError) as caught:
        recognize_image(state, engine="tesseract", lang="rus+eng",
                        auto_escalate=("apple-vision", "easyocr"))
    assert caught.value.requirement_code == "missing_apple_vision_packages"
    assert state.calls == []


def test_strict_chain_stops_on_apple_failure(escalation_engines):
    state = escalation_engines
    state.error = "synthetic framework failure"
    with pytest.raises(core.OcrError, match="synthetic framework failure"):
        recognize_image(state, engine="tesseract", lang="rus+eng",
                        auto_escalate=("apple-vision", "easyocr"))
    assert state.calls == ["tesseract", "apple-vision"]


@pytest.mark.parametrize("format", ["md", "txt", "json"])
def test_cli_output_formats_and_escalation(apple_frameworks, capsys, format):
    with tempfile.NamedTemporaryFile(suffix=".png") as image, mock.patch.object(
        cli, "Caps", return_value=apple_frameworks.caps
    ):
        code = cli.main([image.name, "--engine", "apple-vision", "--lang", "eng",
            "--preprocess", "none", "--format", format,
            "--auto-escalate", "apple-vision"])
    assert code == 0
    output = capsys.readouterr().out
    assert "Hello world" in output
    if format == "json":
        report = json.loads(output)
        assert report["pages"][0]["source"] == "apple-vision"
        assert report["pages"][0]["words"][0]["granularity"] == "line"
    elif format == "md":
        assert "# " in output
    else:
        assert "Hello world\nПривет мир" in output


def test_cli_runs_strict_escalation_and_reports_decision(escalation_engines, capsys):
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png") as image, mock.patch.object(
        cli, "Caps", return_value=escalation_engines.caps
    ):
        Image.new("RGB", (1000, 500), "white").save(image.name)
        code = cli.main([image.name, "--engine", "tesseract", "--lang", "rus+eng",
            "--preprocess", "none", "--format", "json",
            "--auto-escalate", "apple-vision,easyocr"])
    assert code == 0
    page, = json.loads(capsys.readouterr().out)["pages"]
    assert page["source"] == "apple-vision"
    assert [attempt["engine"] for attempt in page["decision"]["attempts"]] == [
        "tesseract", "apple-vision", "easyocr"
    ]


@pytest.mark.parametrize("platform, missing, exit_code, message", [
    ("linux", False, core.EXIT_UNSUPPORTED, "requires macOS"),
    ("darwin", True, core.EXIT_MISSING_BINARY, "apple-vision"),
])
def test_cli_unavailable_apple_exits_without_ocr(
    apple_frameworks, capsys, monkeypatch, platform, missing, exit_code, message
):
    monkeypatch.setattr(core.sys, "platform", platform)
    apple_frameworks.caps.has_vision = not missing
    with tempfile.NamedTemporaryFile(suffix=".png") as image, mock.patch.object(
        cli, "Caps", return_value=apple_frameworks.caps
    ):
        code = cli.main([image.name, "--engine", "apple-vision", "--preprocess", "none"])
    assert code == exit_code
    streams = capsys.readouterr()
    assert streams.out == ""
    assert message in streams.err
    assert apple_frameworks.calls == []


def test_cached_result_does_not_hide_unsupported_settings(apple_frameworks):
    cache = core.Cache(None)
    with tempfile.NamedTemporaryFile(suffix=".png") as image:
        options = core.RecognizeOptions(engine="apple-vision", lang="eng", preprocess="none")
        core.recognize(image.name, options, caps=apple_frameworks.caps, cache=cache)
        options.psm = 6
        with pytest.raises(core.OcrError, match="psm"):
            core.recognize(image.name, options, caps=apple_frameworks.caps, cache=cache)


def test_existing_language_codes_map_to_runtime_supported_languages(apple_frameworks):
    recognize_image(apple_frameworks, lang="fra+eng")
    assert apple_frameworks.languages == ["fr-FR", "en-US"]


def test_auto_language_and_empty_result_are_explicit(apple_frameworks):
    apple_frameworks.observations = []
    page, = recognize_image(apple_frameworks)
    assert apple_frameworks.auto is True
    assert page["text"] == ""
    assert page["words"] == []
    assert page["mean_conf"] == 0.0
    assert page["flag"] == "review-vision"
    assert page["issues"] == ["low_confidence"]
