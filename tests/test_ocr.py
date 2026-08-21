#!/usr/bin/env python3
"""Unit tests for pro.ledin.ocr pure helpers. Run: python3 -m pytest tests"""

import base64
import builtins
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from pro.ledin.ocr import core as ocr  # noqa: E402
from pro.ledin.ocr import cli as ocr_cli  # noqa: E402


class ResolvePaddleLang(unittest.TestCase):
    def test_composite_takes_primary(self):
        self.assertEqual(ocr.resolve_paddle_lang("rus+eng"), "ru")

    def test_eng(self):
        self.assertEqual(ocr.resolve_paddle_lang("eng"), "en")

    def test_chi_sim(self):
        self.assertEqual(ocr.resolve_paddle_lang("chi_sim"), "ch")

    def test_auto_defaults_en(self):
        self.assertEqual(ocr.resolve_paddle_lang("auto"), "en")

    def test_unknown_defaults_en(self):
        self.assertEqual(ocr.resolve_paddle_lang("xyz"), "en")


class ParsePaddleResult(unittest.TestCase):
    def _stub(self, texts, scores, polys):
        # PaddleOCR 3.x result item exposes attributes; emulate via SimpleNamespace
        return types.SimpleNamespace(rec_texts=texts, rec_scores=scores, rec_polys=polys)

    def test_basic_parse(self):
        # two lines, second one higher on the page (smaller y) than first
        poly_top = [[10, 5], [110, 5], [110, 25], [10, 25]]      # y=5
        poly_bottom = [[10, 60], [90, 60], [90, 80], [10, 80]]    # y=60
        item = self._stub(["world", "hello"], [0.90, 0.80],
                          [poly_bottom, poly_top])
        text, mean_conf, words = ocr._parse_paddle_result([item])
        # position-sorted top-to-bottom → hello then world
        self.assertEqual(text.splitlines(), ["hello", "world"])
        self.assertAlmostEqual(mean_conf, 85.0, places=1)
        self.assertEqual(len(words), 2)

    def test_bbox_from_poly(self):
        poly = [[10, 5], [110, 5], [110, 25], [10, 25]]
        item = self._stub(["x"], [0.5], [poly])
        _, _, words = ocr._parse_paddle_result([item])
        # bbox = [min_x, min_y, w, h]
        self.assertEqual(words[0]["bbox"], [10, 5, 100, 20])
        self.assertEqual(words[0]["conf"], 50)

    def test_empty(self):
        item = self._stub([], [], [])
        text, mean_conf, words = ocr._parse_paddle_result([item])
        self.assertEqual(text, "")
        self.assertEqual(mean_conf, 0.0)
        self.assertEqual(words, [])


class PaddleOCRVLMarkdown(unittest.TestCase):
    def test_extracts_native_markdown(self):
        item = types.SimpleNamespace(markdown={
            "markdown_texts": "| Eye | SPH |\n| --- | --- |\n| OD | +2.50 |",
            "markdown_images": {},
        })
        self.assertEqual(
            ocr._extract_paddle_vl_markdown([item]),
            "| Eye | SPH |\n| --- | --- |\n| OD | +2.50 |",
        )

    def test_extracts_from_paddlex_dict_subclass_property(self):
        class FakeResult(dict):
            @property
            def markdown(self):
                return {"markdown_texts": "<table><tr><td>A</td></tr></table>"}

        self.assertEqual(
            ocr._extract_paddle_vl_markdown([FakeResult(res={})]),
            "<table><tr><td>A</td></tr></table>",
        )

    def test_to_markdown_prefers_native_structure(self):
        pages = [{
            "n": 1,
            "source": "paddleocr-vl-mlx",
            "text": "Eye\tSPH\nOD\t+2.50",
            "markdown": "| Eye | SPH |\n| --- | --- |\n| OD | +2.50 |",
            "mean_conf": None,
            "flag": None,
            "issues": [],
            "words": [],
        }]
        rendered = ocr.to_markdown(pages, "table.png")
        self.assertIn("| --- | --- |", rendered)
        self.assertNotIn("Eye\tSPH", rendered)

    def test_markdown_to_text_keeps_cells_without_separator_row(self):
        text = ocr.markdown_to_text(
            "## Prescription\n\n| Eye | SPH |\n| --- | --- |\n| OD | +2.50 |"
        )
        self.assertEqual(text, "Prescription\n\nEye\tSPH\nOD\t+2.50")

    def test_full_pipeline_uses_mlx_only_as_vlm_backend(self):
        captured = {}

        class FakePipeline:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def predict(self, path, **kwargs):
                captured["path"] = path
                captured["predict"] = kwargs
                return [types.SimpleNamespace(markdown={
                    "markdown_texts": "| A | B |\n| --- | --- |\n| 1 | 2 |"
                })]

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCRVL = FakePipeline
        ocr._PADDLE_VL_CACHE.clear()
        with mock.patch.dict(sys.modules, {"paddleocr": fake_module}):
            markdown, text = ocr.ocr_paddleocr_vl_mlx(
                "page.png",
                "http://127.0.0.1:8111/",
                "PaddlePaddle/PaddleOCR-VL-1.6",
                fake_caps(),
            )
        self.assertEqual(captured["vl_rec_backend"], "mlx-vlm-server")
        self.assertEqual(captured["vl_rec_max_concurrency"], 1)
        self.assertEqual(captured["markdown_ignore_labels"], [])
        self.assertEqual(captured["predict"]["temperature"], 0)
        self.assertIn("| --- | --- |", markdown)
        self.assertIn("A\tB", text)

    def test_empty_lazy_load_response_is_retried_once(self):
        calls = 0

        class FakePipeline:
            def __init__(self, **kwargs):
                pass

            def predict(self, path, **kwargs):
                nonlocal calls
                calls += 1
                text = "" if calls == 1 else "| A |\n| --- |\n| 1 |"
                return [types.SimpleNamespace(markdown={"markdown_texts": text})]

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCRVL = FakePipeline
        ocr._PADDLE_VL_CACHE.clear()
        with mock.patch.dict(sys.modules, {"paddleocr": fake_module}):
            markdown, _ = ocr.ocr_paddleocr_vl_mlx(
                "page.png", "http://127.0.0.1:8111/", "model", fake_caps()
            )
        self.assertEqual(calls, 2)
        self.assertIn("| 1 |", markdown)


class TextPdfTables(unittest.TestCase):
    def _write_ruled_pdf(self, path):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        document = canvas.Canvas(path, pagesize=letter)
        for x in (50, 200, 300, 400):
            document.line(x, 650, x, 750)
        for y in (650, 680, 715, 750):
            document.line(50, y, 400, y)
        for x, y, text in (
            (60, 730, "Test"),
            (210, 730, "Result"),
            (310, 730, "Unit"),
            (60, 695, "ALT"),
            (210, 695, "35.5"),
            (310, 695, "U/L"),
            (60, 660, "AST"),
            (210, 660, "29.3"),
            (310, 660, "U/L"),
        ):
            document.drawString(x, y, text)
        document.save()

    def test_pipe_table_escapes_cells(self):
        rendered = ocr._render_pipe_table([["Name", "Value"], ["A|B", "one\ntwo"]])
        self.assertIn("A\\|B", rendered)
        self.assertIn("one<br>two", rendered)

    def test_complex_table_uses_html(self):
        rendered = ocr._render_html_table([["Name", "Value"], ["ALT", "35.5"]])
        self.assertIn("<table>", rendered)
        self.assertIn("<th>Name</th>", rendered)
        self.assertIn("<td>35.5</td>", rendered)

    def test_camelot_bbox_is_converted_to_top_left_coordinates(self):
        self.assertEqual(
            ocr._normalize_camelot_bbox([10, 20, 30, 80], 100),
            [10, 20, 30, 80],
        )

    def test_markdown_composition_preserves_text_table_text_order(self):
        blocks = [
            {"bbox": [0, 0, 100, 10], "text": "Before"},
            {"bbox": [0, 20, 100, 30], "text": "duplicate table text"},
            {"bbox": [0, 40, 100, 50], "text": "After"},
        ]
        tables = [{
            "bbox": [0, 15, 100, 35],
            "rendered": "| A | B |\n| --- | --- |\n| 1 | 2 |",
            "accepted": True,
        }]
        rendered = ocr._compose_page_markdown(blocks, tables)
        self.assertLess(rendered.index("Before"), rendered.index("| A | B |"))
        self.assertLess(rendered.index("| A | B |"), rendered.index("After"))
        self.assertNotIn("duplicate table text", rendered)

    def test_numeric_coverage_rejects_lost_value(self):
        table = {
            "rows": [["Test", "Result"], ["ALT", "35.5"]],
            "bbox": [0, 0, 200, 100],
            "issues": [],
        }
        blocks = [{"bbox": [0, 0, 200, 100], "text": "ALT 35.5 AST 29.3"}]
        issues = ocr._validate_table(table, blocks)
        self.assertIn("table_numeric_coverage_low", issues)

    def test_recognize_formats_text_pdf_table_without_changing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.pdf")
            self._write_ruled_pdf(path)
            pages = ocr.recognize(path, ocr.RecognizeOptions())
        self.assertEqual(len(pages), 1)
        self.assertIn("ALT", pages[0]["text"])
        self.assertEqual(len(pages[0]["tables"]), 1)
        self.assertTrue(pages[0]["tables"][0]["accepted"])
        self.assertIn("| Test | Result | Unit |", pages[0]["markdown"])
        self.assertEqual(pages[0]["markdown"].count("ALT"), 1)

    def test_no_tables_preserves_original_fast_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.pdf")
            self._write_ruled_pdf(path)
            with mock.patch.object(ocr, "extract_text_pdf_tables") as extract:
                pages = ocr.recognize(
                    path,
                    ocr.RecognizeOptions(extract_tables=False),
                )
        extract.assert_not_called()
        self.assertNotIn("markdown", pages[0])
        self.assertEqual(pages[0]["tables"], [])

    def test_camelot_failure_preserves_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.pdf")
            self._write_ruled_pdf(path)
            with mock.patch.object(
                ocr,
                "extract_text_pdf_tables",
                side_effect=RuntimeError("broken parser"),
            ):
                pages = ocr.recognize(path, ocr.RecognizeOptions())
        self.assertIn("ALT", pages[0]["text"])
        self.assertIn("table_extraction_failed", pages[0]["issues"])

    def test_workflow_cache_key_includes_table_options(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            enabled = ocr._workflow_cache_key(
                source.name,
                ocr.RecognizeOptions(extract_tables=True),
            )
            disabled = ocr._workflow_cache_key(
                source.name,
                ocr.RecognizeOptions(extract_tables=False),
            )
            lattice = ocr._workflow_cache_key(
                source.name,
                ocr.RecognizeOptions(table_flavor="lattice"),
            )
        self.assertNotEqual(enabled, disabled)
        self.assertNotEqual(enabled, lattice)

    def test_auto_falls_back_to_stream_for_page_without_usable_grid(self):
        calls = []

        class Values:
            def __init__(self, rows):
                self._rows = rows

            def tolist(self):
                return self._rows

        class Frame:
            def __init__(self, rows):
                self.values = Values(rows)

        def table(rows, flavor):
            return types.SimpleNamespace(
                df=Frame(rows),
                page=1,
                flavor=flavor,
                _bbox=(0, 0, 100, 100),
                parsing_report={"page": 1, "accuracy": 100, "confidence": 1},
                cells=[],
            )

        fake_camelot = types.ModuleType("camelot")

        def read_pdf(path, pages, flavor):
            calls.append(flavor)
            if flavor == "auto":
                return [table([["one"], ["two"]], "network")]
            return [table([["A", "B"], ["1", "2"]], "stream")]

        fake_camelot.read_pdf = read_pdf
        with mock.patch.dict(sys.modules, {"camelot": fake_camelot}):
            result = ocr.extract_text_pdf_tables("report.pdf", [1], "auto")
        self.assertEqual(calls, ["auto", "stream"])
        self.assertEqual(len(result[1]), 2)

    def test_malformed_camelot_metrics_are_treated_as_missing(self):
        class Values:
            def tolist(self):
                return [["A", "B"], ["1", "2"]]

        table = types.SimpleNamespace(
            df=types.SimpleNamespace(values=Values()),
            page=1,
            flavor="lattice",
            _bbox=(0, 0, 100, 100),
            parsing_report={
                "page": 1,
                "accuracy": "unknown",
                "confidence": {},
                "whitespace": [],
            },
            cells=[],
        )
        fake_camelot = types.ModuleType("camelot")
        fake_camelot.read_pdf = lambda path, pages, flavor: [table]
        with mock.patch.dict(sys.modules, {"camelot": fake_camelot}):
            result = ocr.extract_text_pdf_tables("report.pdf", [1], "lattice")
        self.assertIsNone(result[1][0]["accuracy"])
        self.assertIsNone(result[1][0]["confidence"])


class ResolveVisionConfig(unittest.TestCase):
    def test_ok(self):
        key, model, endpoint = ocr.resolve_vision_config(
            vision_api_key="k", vision_model="m", vision_api_url="http://x"
        )
        self.assertEqual((key, model, endpoint), ("k", "m", "http://x"))

    def test_empty_model_errors(self):
        with self.assertRaises(ocr.OcrError):
            ocr.resolve_vision_config(vision_api_key="k", vision_model="")

    def test_empty_key_errors(self):
        with self.assertRaises(ocr.OcrError):
            ocr.resolve_vision_config(vision_api_key="", vision_model="m")

    def test_does_not_read_env(self):
        os.environ["OPENAI_API_KEY"] = "should-not-be-used"
        try:
            with self.assertRaises(ocr.OcrError):
                ocr.resolve_vision_config(vision_api_key="", vision_model="m")
        finally:
            del os.environ["OPENAI_API_KEY"]


class ResolveVisionPrompt(unittest.TestCase):
    def test_empty_uses_default(self):
        self.assertEqual(ocr.resolve_vision_prompt(""), ocr.DEFAULT_VISION_PROMPT)

    def test_whitespace_uses_default(self):
        self.assertEqual(ocr.resolve_vision_prompt(" \n\t"), ocr.DEFAULT_VISION_PROMPT)

    def test_custom_prompt_is_preserved(self):
        prompt = "Extract checkboxes.\nPreserve labels exactly."
        self.assertEqual(ocr.resolve_vision_prompt(prompt), prompt)


class VisionHandoffPrompt(unittest.TestCase):
    def test_default_prompt_in_manifest(self):
        manifest = ocr.vision_handoff([(1, "/tmp/page.png")])
        self.assertIn(ocr.DEFAULT_VISION_PROMPT, manifest)

    def test_custom_prompt_in_manifest(self):
        prompt = "Extract only the table.\nKeep empty cells."
        manifest = ocr.vision_handoff(
            [(1, "/tmp/page.png")], vision_prompt=prompt
        )
        self.assertIn(prompt, manifest)
        self.assertNotIn(ocr.DEFAULT_VISION_PROMPT, manifest)


class Regression(unittest.TestCase):
    def test_parse_page_range(self):
        self.assertEqual(ocr._parse_page_range("1-3,5", 10), [1, 2, 3, 5])
        self.assertEqual(ocr._parse_page_range("1-3,5", 4), [1, 2, 3])

    def test_resolve_preprocess_image_none(self):
        caps = types.SimpleNamespace(has_cv2=True, has_numpy=True)
        self.assertEqual(ocr.resolve_preprocess("auto", None, caps, "image"), "none")

    def test_resolve_preprocess_explicit(self):
        caps = types.SimpleNamespace(has_cv2=False, has_numpy=False)
        self.assertEqual(ocr.resolve_preprocess("full", None, caps, "pdf"), "full")

    def test_max_pages_caps_explicit_range(self):
        self.assertEqual(ocr._resolve_page_range("2-5", 10, 2), [2, 3])

    def test_max_pages_caps_default_range(self):
        self.assertEqual(ocr._resolve_page_range("", 10, 2), [1, 2])

    def test_page_range_matching_no_pages_errors(self):
        with self.assertRaises(ocr.OcrError):
            ocr._resolve_page_range("999", 10, 0)

    def test_negative_max_pages_errors(self):
        with self.assertRaises(ocr.OcrError):
            ocr._resolve_page_range("", 10, -1)


class FatalRaisesOcrError(unittest.TestCase):
    """`_fatal` used to call sys.exit() directly, which killed the whole host
    process when ocr.py was imported as a library. It must raise instead."""

    def test_raises_ocr_error_with_code(self):
        with self.assertRaises(ocr.OcrError) as ctx:
            ocr._fatal("boom", ocr.EXIT_MISSING_BINARY)
        self.assertEqual(str(ctx.exception), "boom")
        self.assertEqual(ctx.exception.code, ocr.EXIT_MISSING_BINARY)

    def test_default_code_is_bad_args(self):
        with self.assertRaises(ocr.OcrError) as ctx:
            ocr._fatal("boom")
        self.assertEqual(ctx.exception.code, ocr.EXIT_BAD_ARGS)


class EngineRequirementProbe(unittest.TestCase):
    def caps(self, **overrides):
        values = {
            "bin_tesseract": "tesseract",
            "has_easyocr": True,
            "has_paddleocr": True,
            "has_paddle": True,
            "has_paddlex_ocr": True,
            "has_openai": True,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_tesseract_missing_binary_is_structured(self):
        result = ocr.probe_engine_requirements(
            "tesseract", caps=self.caps(bin_tesseract=None)
        )
        self.assertEqual(result.code, "missing_tesseract_binary")
        self.assertEqual(result.missing_component, "tesseract")
        self.assertEqual(result.missing_components, ("tesseract",))
        self.assertEqual(result.component_type, "binary")
        self.assertIsNone(result.ocr_extra)

    def test_easyocr_result_includes_extra_and_first_run_note(self):
        result = ocr.probe_engine_requirements(
            "easyocr", caps=self.caps(has_easyocr=False)
        )
        self.assertEqual(result.code, "missing_easyocr_package")
        self.assertEqual(result.missing_components, ("easyocr",))
        self.assertEqual(result.ocr_extra, "easyocr")
        self.assertEqual(result.component_type, "python-package")
        self.assertIn("model", result.first_run_note.lower())

    def test_paddle_package_and_runtime_are_distinct(self):
        package = ocr.probe_engine_requirements(
            "paddleocr", caps=self.caps(has_paddleocr=False)
        )
        runtime = ocr.probe_engine_requirements(
            "paddleocr", caps=self.caps(has_paddle=False)
        )
        self.assertEqual(package.code, "missing_paddleocr_package")
        self.assertEqual(package.missing_component, "paddleocr")
        self.assertEqual(package.missing_components, ("paddleocr",))
        self.assertEqual(runtime.code, "missing_paddle_runtime")
        self.assertEqual(runtime.missing_component, "paddle")
        self.assertEqual(runtime.missing_components, ("paddle",))
        self.assertEqual(runtime.component_type, "python-runtime")

    def test_paddle_reports_package_and_runtime_together(self):
        result = ocr.probe_engine_requirements(
            "paddleocr",
            caps=self.caps(has_paddleocr=False, has_paddle=False),
        )
        self.assertEqual(result.code, "missing_paddleocr_package")
        self.assertEqual(result.missing_component, "paddleocr")
        self.assertEqual(
            result.missing_components,
            ("paddleocr", "paddle"),
        )
        self.assertEqual(
            result.to_dict()["missing_components"],
            ("paddleocr", "paddle"),
        )

    def test_paddle_vl_requires_loopback_server_and_model(self):
        missing_url = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_model="PaddlePaddle/PaddleOCR-VL-1.6",
            caps=self.caps(),
        )
        unsafe_url = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url="https://example.com/v1",
            paddle_vl_model="PaddlePaddle/PaddleOCR-VL-1.6",
            caps=self.caps(),
        )
        missing_model = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url="http://localhost:8111/",
            caps=self.caps(),
        )
        ready = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url="http://127.0.0.1:8111/",
            paddle_vl_model="PaddlePaddle/PaddleOCR-VL-1.6",
            caps=self.caps(),
        )
        self.assertEqual(missing_url.code, "missing_paddle_vl_server_url")
        self.assertEqual(unsafe_url.code, "unsafe_paddle_vl_server_url")
        self.assertEqual(missing_model.code, "missing_paddle_vl_model")
        self.assertTrue(ready.available)

    def test_paddle_vl_requires_doc_parser_package_and_runtime(self):
        result = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url="http://localhost:8111/",
            paddle_vl_model="model",
            caps=self.caps(has_paddleocr=False, has_paddle=False),
        )
        self.assertEqual(result.code, "missing_paddleocr_doc_parser")
        self.assertEqual(result.ocr_extra, "paddle-vl")
        self.assertEqual(result.missing_components, ("paddleocr", "paddle"))

        missing_extra = ocr.probe_engine_requirements(
            "paddleocr-vl-mlx",
            paddle_vl_server_url="http://localhost:8111/",
            paddle_vl_model="model",
            caps=self.caps(has_paddlex_ocr=False),
        )
        self.assertEqual(missing_extra.code, "missing_paddleocr_doc_parser")
        self.assertEqual(missing_extra.missing_component, "paddleocr[doc-parser]")

    def test_vision_checks_package_then_explicit_config(self):
        package = ocr.probe_engine_requirements(
            "vision", vision_api_key="k", vision_model="m",
            caps=self.caps(has_openai=False),
        )
        key = ocr.probe_engine_requirements(
            "vision", vision_model="m", caps=self.caps()
        )
        model = ocr.probe_engine_requirements(
            "vision", vision_api_key="k", caps=self.caps()
        )
        ready = ocr.probe_engine_requirements(
            "vision", vision_api_key="k", vision_model="m", caps=self.caps()
        )
        self.assertEqual(package.code, "missing_openai_package")
        self.assertEqual(package.missing_components, ("openai",))
        self.assertEqual(key.code, "missing_vision_api_key")
        self.assertEqual(key.missing_components, ("vision_api_key",))
        self.assertEqual(model.code, "missing_vision_model")
        self.assertEqual(model.missing_components, ("vision_model",))
        self.assertEqual(ready.code, "ok")
        self.assertEqual(ready.missing_components, ())
        self.assertTrue(ready.available)

    def test_probe_does_not_import_engine_packages(self):
        with (mock.patch.object(ocr.importlib.util, "find_spec", return_value=None),
              mock.patch.object(ocr.shutil, "which", return_value=None),
              mock.patch("builtins.__import__", side_effect=AssertionError("imported"))):
            result = ocr.probe_engine_requirements("easyocr")
        self.assertEqual(result.code, "missing_easyocr_package")

    def test_requirement_error_exposes_result_fields(self):
        options = ocr.RecognizeOptions(engine="vision", vision_model="m")
        with self.assertRaises(ocr.OcrRequirementError) as context:
            ocr._require_engine("vision", self.caps(), options)
        error = context.exception
        self.assertEqual(error.code, ocr.EXIT_BAD_ARGS)
        self.assertEqual(error.requirement_code, "missing_vision_api_key")
        self.assertEqual(error.engine, "vision")
        self.assertEqual(error.missing_component, "vision_api_key")
        self.assertEqual(error.missing_components, ("vision_api_key",))
        self.assertEqual(error.ocr_extra, "vision")
        self.assertEqual(error.component_type, "configuration")
        self.assertEqual(error.result.to_dict()["code"], error.requirement_code)

    def test_missing_package_message_uses_distribution_extra_not_ocr_py(self):
        options = ocr.RecognizeOptions(engine="easyocr")
        with self.assertRaises(ocr.OcrRequirementError) as context:
            ocr._require_engine(
                "easyocr", self.caps(has_easyocr=False), options
            )
        message = str(context.exception)
        self.assertIn("pro-ledin-ocr[easyocr]", message)
        self.assertNotIn("ocr.py", message)

    def test_multi_component_error_names_every_missing_paddle_module(self):
        options = ocr.RecognizeOptions(engine="paddleocr")
        with self.assertRaises(ocr.OcrRequirementError) as context:
            ocr._require_engine(
                "paddleocr",
                self.caps(has_paddleocr=False, has_paddle=False),
                options,
            )
        error = context.exception
        self.assertEqual(
            error.missing_components,
            ("paddleocr", "paddle"),
        )
        self.assertIn("'paddleocr'", str(error))
        self.assertIn("'paddle'", str(error))

    def test_unsupported_engine_is_structured_bad_args(self):
        result = ocr.probe_engine_requirements("unknown", caps=self.caps())
        self.assertEqual(result.code, "unsupported_engine")
        with self.assertRaises(ocr.OcrRequirementError) as context:
            ocr._raise_requirement(result)
        self.assertEqual(context.exception.code, ocr.EXIT_BAD_ARGS)

    def test_public_package_exports_probe_types(self):
        from pro.ledin import ocr as public_ocr

        self.assertIs(public_ocr.RequirementResult, ocr.RequirementResult)
        self.assertIs(public_ocr.OcrRequirementError, ocr.OcrRequirementError)
        self.assertIs(public_ocr.probe_engine_requirements, ocr.probe_engine_requirements)

    def _stderr_visible_to_reader(self, emit) -> str:
        """Return what a concurrent reader sees on a buffered stderr."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stderr.log")
            with open(path, "w", encoding="utf-8") as stream:
                with mock.patch.object(ocr.sys, "stderr", stream):
                    emit()
                with open(path, encoding="utf-8") as reader:
                    return reader.read()

    def test_verbose_progress_reaches_stderr_before_process_exit(self):
        observed = self._stderr_visible_to_reader(
            lambda: ocr._log("checking", True)
        )
        self.assertIn("checking", observed)

    def test_vision_handoff_manifest_reaches_stderr_before_process_exit(self):
        observed = self._stderr_visible_to_reader(
            lambda: ocr.vision_handoff([(1, "/tmp/page_0001.png")])
        )
        self.assertIn("/tmp/page_0001.png", observed)


def fake_caps(**overrides):
    """Build a real `Caps` instance (with its require_* methods) without
    running hardware/dependency detection."""
    caps = object.__new__(ocr.Caps)
    defaults = {
        "verbose": False,
        "bin_tesseract": "tesseract",
        "bin_pdftoppm": "/usr/bin/pdftoppm",
        "bin_pdftotext": "/usr/bin/pdftotext",
        "bin_pdfinfo": None,
        "bin_pdffonts": None,
        "bin_pdfimages": None,
        "bin_ocrmypdf": None,
        "has_pytesseract": False,
        "has_fitz": False,
        "has_cv2": False,
        "has_numpy": False,
        "has_pil": False,
        "has_easyocr": True,
        "has_openai": True,
        "has_paddleocr": True,
        "has_paddle": True,
        "has_paddlex_ocr": True,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(caps, name, value)
    return caps


def broken_import(name: str, error: BaseException):
    """Patch imports so `name` resolves by spec but fails on import."""
    real_import = builtins.__import__

    def fake_import(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] == name:
            raise error
        return real_import(module_name, *args, **kwargs)

    return mock.patch.object(builtins, "__import__", side_effect=fake_import)


def broken_imports(errors: dict[str, BaseException]):
    """Patch several import roots with independent failures."""
    real_import = builtins.__import__

    def fake_import(module_name, *args, **kwargs):
        root = module_name.split(".", 1)[0]
        if root in errors:
            raise errors[root]
        return real_import(module_name, *args, **kwargs)

    return mock.patch.object(builtins, "__import__", side_effect=fake_import)


class BrokenEngineInstall(unittest.TestCase):
    """A module can exist by spec yet still fail to import (broken build,
    namespace shell, failed native load). Those must stay inside the documented
    OcrError contract instead of escaping as a raw ImportError."""

    def caps(self, **overrides):
        return fake_caps(**overrides)

    def _broken_module(self, name: str, error: BaseException):
        return broken_import(name, error), None

    def test_broken_easyocr_import_raises_structured_requirement_error(self):
        patcher, _ = self._broken_module("easyocr", ImportError("bad build", name="easyocr"))
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.ocr_easyocr("page.png", self.caps(), False)
        error = context.exception
        self.assertIsInstance(error, ocr.OcrError)
        self.assertEqual(error.requirement_code, "missing_easyocr_package")
        self.assertEqual(error.missing_components, ("easyocr",))
        self.assertEqual(error.component_type, "python-package")
        self.assertEqual(error.ocr_extra, "easyocr")

    def test_broken_paddleocr_import_is_attributed_to_package(self):
        patcher, _ = self._broken_module(
            "paddleocr", ImportError("bad build", name="paddleocr")
        )
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.ocr_paddleocr("page.png", "eng", self.caps(), False)
        error = context.exception
        self.assertEqual(error.requirement_code, "missing_paddleocr_package")
        self.assertEqual(error.missing_components, ("paddleocr",))
        self.assertEqual(error.ocr_extra, "paddle")

    def test_failed_paddle_runtime_load_is_attributed_to_runtime(self):
        patcher, _ = self._broken_module(
            "paddleocr", ImportError("libpaddle.so missing", name="paddle")
        )
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.ocr_paddleocr("page.png", "eng", self.caps(), False)
        error = context.exception
        self.assertEqual(error.requirement_code, "missing_paddle_runtime")
        self.assertEqual(error.missing_components, ("paddle",))
        self.assertEqual(error.component_type, "python-runtime")

    def test_native_load_failure_is_converted_not_leaked(self):
        patcher, _ = self._broken_module(
            "easyocr", OSError("cannot load shared object")
        )
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.ocr_easyocr("page.png", self.caps(), False)
        self.assertEqual(
            context.exception.requirement_code, "missing_easyocr_package"
        )

    def test_broken_openai_import_raises_structured_requirement_error(self):
        patcher, _ = self._broken_module("openai", ImportError("bad build", name="openai"))
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.vision_ocr(
                    [(1, "page.png")], vision_api_key="k", vision_model="m"
                )
        error = context.exception
        self.assertEqual(error.requirement_code, "missing_openai_package")
        self.assertEqual(error.missing_components, ("openai",))
        self.assertEqual(error.ocr_extra, "vision")

    def test_broken_pymupdf_falls_back_to_poppler(self):
        caps = self.caps(has_fitz=True, bin_pdftoppm="/usr/bin/pdftoppm")
        patcher, _ = self._broken_module("fitz", ImportError("bad build", name="fitz"))
        with patcher:
            with mock.patch.object(
                ocr, "_render_pdftoppm", return_value=[(1, "page-0001.png")]
            ) as fallback:
                rendered = ocr.render_pages("scan.pdf", 300, None, "/tmp", caps, False)
        self.assertEqual(rendered, [(1, "page-0001.png")])
        fallback.assert_called_once()

    def test_broken_pymupdf_without_poppler_raises_structured_error(self):
        caps = self.caps(has_fitz=True, bin_pdftoppm=None, bin_pdftotext=None)
        patcher, _ = self._broken_module("fitz", ImportError("bad build", name="fitz"))
        with patcher:
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.render_pages("scan.pdf", 300, None, "/tmp", caps, False)
        error = context.exception
        self.assertEqual(error.requirement_code, "missing_pdf_render_backend")
        self.assertEqual(error.missing_components, ("pdftoppm", "pymupdf"))

    def test_easyocr_transitive_import_failure_names_dependency(self):
        fake_easyocr = types.ModuleType("easyocr")
        for failure in (
            ImportError("numpy missing", name="numpy"),
            OSError("numpy native extension failed"),
        ):
            with self.subTest(failure=type(failure).__name__):
                with mock.patch.dict(sys.modules, {"easyocr": fake_easyocr}):
                    with broken_import("numpy", failure):
                        with self.assertRaises(ocr.OcrRequirementError) as context:
                            ocr.ocr_easyocr("page.png", self.caps(), False)
                error = context.exception
                self.assertEqual(
                    error.requirement_code, "missing_easyocr_dependency"
                )
                self.assertEqual(error.engine, "easyocr")
                self.assertEqual(error.missing_components, ("numpy",))
                self.assertEqual(error.component_type, "python-package")
                self.assertEqual(error.ocr_extra, "easyocr")

    def test_easyocr_pillow_failure_names_dependency(self):
        fake_easyocr = types.ModuleType("easyocr")
        fake_numpy = types.ModuleType("numpy")
        with mock.patch.dict(
            sys.modules, {"easyocr": fake_easyocr, "numpy": fake_numpy}
        ):
            with broken_import("PIL", OSError("Pillow native load failed")):
                with self.assertRaises(ocr.OcrRequirementError) as context:
                    ocr.ocr_easyocr("page.png", self.caps(), False)
        error = context.exception
        self.assertEqual(error.requirement_code, "missing_easyocr_dependency")
        self.assertEqual(error.missing_components, ("pillow",))
        self.assertEqual(error.ocr_extra, "easyocr")

    def test_easyocr_import_attributes_known_transitive_module(self):
        with broken_import(
            "easyocr", ImportError("torch missing", name="torch")
        ):
            with self.assertRaises(ocr.OcrRequirementError) as context:
                ocr.ocr_easyocr("page.png", self.caps(), False)
        self.assertEqual(
            context.exception.requirement_code, "missing_easyocr_dependency"
        )
        self.assertEqual(context.exception.missing_components, ("torch",))


class OptionalImportFallbacks(unittest.TestCase):
    def _stderr_from(self, emit) -> str:
        stream = io.StringIO()
        with mock.patch.object(sys, "stderr", stream):
            emit()
        return stream.getvalue()

    def test_broken_pytesseract_falls_back_to_cli(self):
        sentinel = ("cli text", 88.0, [])
        for failure in (
            ImportError("bad install", name="pytesseract"),
            OSError("native dependency failed"),
        ):
            with self.subTest(failure=type(failure).__name__):
                caps = fake_caps(has_pytesseract=True)
                with broken_import("pytesseract", failure):
                    with mock.patch.object(
                        ocr, "_ocr_tesseract_cli", return_value=sentinel
                    ):
                        stream = io.StringIO()
                        with mock.patch.object(sys, "stderr", stream):
                            result = ocr.ocr_tesseract(
                                "page.png", "eng", 3, caps, True
                            )
                self.assertEqual(result, sentinel)
                self.assertIn("falling back to tesseract CLI", stream.getvalue())

    def test_broken_pillow_basic_returns_original_path(self):
        for failure in (
            ImportError("Pillow missing", name="PIL"),
            OSError("Pillow native load failed"),
        ):
            with self.subTest(failure=type(failure).__name__):
                stream = io.StringIO()
                with broken_import("PIL", failure):
                    with mock.patch.object(sys, "stderr", stream):
                        result = ocr._preprocess_basic(
                            "page.png", "processed.png", True
                        )
                self.assertEqual(result, "page.png")
                self.assertIn("skipping preprocessing", stream.getvalue())

    def test_enhanced_falls_through_basic_to_original(self):
        failures = {
            "cv2": ImportError("OpenCV missing", name="cv2"),
            "PIL": OSError("Pillow native load failed"),
        }
        stream = io.StringIO()
        with broken_imports(failures):
            with mock.patch.object(sys, "stderr", stream):
                result = ocr._preprocess_opencv(
                    "page.png", "processed.png", "enhanced", True
                )
        self.assertEqual(result, "page.png")
        output = stream.getvalue()
        self.assertIn("falling back to basic preprocessing", output)
        self.assertIn("skipping preprocessing", output)

    def test_numpy_native_failure_falls_back_to_basic(self):
        fake_cv2 = types.ModuleType("cv2")
        stream = io.StringIO()
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            with broken_import("numpy", OSError("NumPy native load failed")):
                with mock.patch.object(
                    ocr, "_preprocess_basic", return_value="basic.png"
                ):
                    with mock.patch.object(sys, "stderr", stream):
                        result = ocr._preprocess_opencv(
                            "page.png", "processed.png", "full", True
                        )
        self.assertEqual(result, "basic.png")
        self.assertIn("falling back to basic preprocessing", stream.getvalue())

    def test_cli_contains_broken_easyocr_import_without_traceback(self):
        real_module_available = ocr._module_available

        def module_available(name):
            return True if name == "easyocr" else real_module_available(name)

        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            stderr = io.StringIO()
            with mock.patch.object(ocr, "_module_available", side_effect=module_available):
                with broken_import(
                    "easyocr", ImportError("broken package", name="easyocr")
                ):
                    with mock.patch.object(sys, "stderr", stderr):
                        code = ocr_cli.main([
                            source.name,
                            "--engine", "easyocr",
                            "--preprocess", "none",
                            "--lang", "eng",
                            "--format", "txt",
                        ])
        output = stderr.getvalue()
        self.assertEqual(code, ocr.EXIT_MISSING_BINARY)
        self.assertIn("[ocr] ERROR:", output)
        self.assertNotIn("Traceback", output)


class PdfRequirementProbe(unittest.TestCase):
    def test_satisfied_by_poppler_only(self):
        result = ocr.probe_pdf_requirements(caps={
            "has_fitz": False,
            "bin_pdftoppm": "/usr/bin/pdftoppm",
            "bin_pdftotext": "/usr/bin/pdftotext",
        })
        self.assertTrue(result.available)
        self.assertEqual(result.code, "ok")
        self.assertEqual(result.missing_components, ())

    def test_satisfied_by_pymupdf_only(self):
        result = ocr.probe_pdf_requirements(caps={
            "has_fitz": True,
            "bin_pdftoppm": None,
            "bin_pdftotext": None,
        })
        self.assertTrue(result.available)
        self.assertEqual(result.code, "ok")

    def test_missing_render_backend(self):
        result = ocr.probe_pdf_requirements(caps={
            "has_fitz": False,
            "bin_pdftoppm": None,
            "bin_pdftotext": "/usr/bin/pdftotext",
        })
        self.assertFalse(result.available)
        self.assertEqual(result.code, "missing_pdf_render_backend")
        self.assertEqual(result.missing_components, ("pdftoppm", "pymupdf"))
        self.assertEqual(result.missing_component, "pdftoppm")
        self.assertEqual(result.components_relation, "any")
        self.assertEqual(result.ocr_extra, "pdf")
        self.assertEqual(result.component_type, "binary-or-python-package")

    def test_missing_text_backend(self):
        result = ocr.probe_pdf_requirements(caps={
            "has_fitz": False,
            "bin_pdftoppm": "/usr/bin/pdftoppm",
            "bin_pdftotext": None,
        })
        self.assertEqual(result.code, "missing_pdf_text_backend")
        self.assertEqual(result.missing_components, ("pdftotext", "pymupdf"))

    def test_caps_require_helpers_raise_structured_error(self):
        caps = fake_caps(has_fitz=False, bin_pdftoppm=None, bin_pdftotext=None)
        with self.assertRaises(ocr.OcrRequirementError) as render:
            caps.require_render()
        with self.assertRaises(ocr.OcrRequirementError) as text:
            caps.require_pdftotext()
        self.assertEqual(render.exception.requirement_code, "missing_pdf_render_backend")
        self.assertEqual(text.exception.requirement_code, "missing_pdf_text_backend")
        self.assertEqual(render.exception.code, ocr.EXIT_MISSING_BINARY)
        self.assertIn("pymupdf", str(render.exception))

    def test_probe_is_exported_publicly(self):
        from pro.ledin import ocr as public_ocr

        self.assertIs(public_ocr.probe_pdf_requirements, ocr.probe_pdf_requirements)


class CapabilityReporting(unittest.TestCase):
    def _stderr_from(self, emit) -> str:
        stream = io.StringIO()
        with mock.patch.object(sys, "stderr", stream):
            emit()
        return stream.getvalue()

    def test_recognize_with_verbose_does_not_dump_capabilities(self):
        options = ocr.RecognizeOptions(skip_ocr=True, verbose=True)
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            output = self._stderr_from(
                lambda: ocr.recognize(source.name, options)
            )
        self.assertNotIn("[caps]", output)

    def test_recognize_still_emits_verbose_progress(self):
        options = ocr.RecognizeOptions(
            engine="vision",
            dpi=150,
            preprocess="none",
            verbose=True,
            vision_api_key="k",
            vision_model="m",
        )
        caps = fake_caps(has_openai=True)
        cache = ocr.Cache(None)
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(ocr, "vision_ocr", return_value="text"):
                ocr.recognize(source.name, options, caps=caps, cache=cache)
                # Second run hits the cache and logs progress from recognize()
                # itself rather than from a mocked engine call.
                output = self._stderr_from(
                    lambda: ocr.recognize(
                        source.name, options, caps=caps, cache=cache
                    )
                )
        self.assertIn("[ocr]", output)
        self.assertNotIn("[caps]", output)

    def test_caps_report_is_opt_in(self):
        self.assertNotIn("[caps]", self._stderr_from(lambda: ocr.Caps(verbose=True)))
        self.assertIn("[caps]", self._stderr_from(lambda: ocr.Caps(report=True)))


class TesseractCliFailures(unittest.TestCase):
    def test_text_command_failure_raises_sanitized_error(self):
        failed = types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="sensitive tesseract diagnostics",
        )
        caps = types.SimpleNamespace(bin_tesseract="tesseract")
        with mock.patch.object(ocr.subprocess, "run", return_value=failed):
            with self.assertRaises(ocr.OcrError) as context:
                ocr._ocr_tesseract_cli("frame.png", "missing-lang", 3, caps, False)
        self.assertNotIn("sensitive", str(context.exception))
        self.assertIn("exit 1", str(context.exception))


class RecognizeOptionsDefaults(unittest.TestCase):
    """Guard against silent drift between library defaults and prior CLI defaults."""

    def test_defaults_match_former_argparse_defaults(self):
        options = ocr.RecognizeOptions()
        self.assertEqual(options.engine, "tesseract")
        self.assertEqual(options.lang, "auto")
        self.assertEqual(options.dpi, 0)
        self.assertEqual(options.preprocess, "auto")
        self.assertEqual(options.psm, ocr.DEFAULT_PSM)
        self.assertEqual(options.min_conf, ocr.DEFAULT_MIN_CONF)
        self.assertFalse(options.no_cleanup)
        self.assertFalse(options.force)
        self.assertEqual(options.vision_prompt, "")
        self.assertIsNone(options.timeout)

    def test_new_prompt_field_preserves_old_positional_order(self):
        options = ocr.RecognizeOptions(
            "auto", "auto", 0, "auto", "", 0,
            ocr.DEFAULT_PSM, ocr.DEFAULT_MIN_CONF,
            False, False, "", "", "", 12.5, True,
        )
        self.assertEqual(options.timeout, 12.5)
        self.assertTrue(options.verbose)
        self.assertEqual(options.vision_prompt, "")


class ProcessFileGuards(unittest.TestCase):
    def test_unsupported_extension_raises_ocr_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = os.path.join(tmp, "file.docx")
            with open(bogus, "w", encoding="utf-8") as f:
                f.write("x")
            caps = ocr.Caps()
            cache = ocr.Cache(None)
            with self.assertRaises(ocr.OcrError) as ctx:
                ocr.process_file(bogus, ocr.RecognizeOptions(), caps, cache, tmp)
            self.assertEqual(ctx.exception.code, ocr.EXIT_UNSUPPORTED)

    def test_recognize_supports_automated_vision_engine(self):
        caps = types.SimpleNamespace(require_render=lambda: None, has_openai=True)
        options = ocr.RecognizeOptions(
            engine="vision",
            vision_api_key="k",
            vision_model="m",
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(ocr, "vision_ocr", return_value="recognized"):
                pages = ocr.recognize(source.name, options, caps=caps)
        self.assertEqual(pages[0]["source"], "vision")
        self.assertEqual(pages[0]["text"], "recognized")

    def test_vision_engine_respects_max_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scan.pdf")
            with open(path, "wb") as f:
                f.write(b"pdf")
            caps = types.SimpleNamespace(require_render=lambda: None, has_openai=True)
            probe = {"pages": 5, "needs_ocr": True, "reason": "scan"}
            options = ocr.RecognizeOptions(
                engine="vision",
                dpi=150,
                preprocess="none",
                max_pages=2,
                vision_api_key="k",
                vision_model="m",
            )
            with (mock.patch.object(ocr, "probe_pdf", return_value=probe),
                  mock.patch.object(ocr, "render_pages", return_value=[(1, "page.png")]) as render,
                  mock.patch.object(ocr, "vision_ocr", return_value="text")):
                ocr.process_file(path, options, caps, ocr.Cache(None), tmp)
            self.assertEqual(render.call_args.args[2], [1, 2])


class VisionApiRequest(unittest.TestCase):
    """`timeout=None` must mean "SDK default", not an explicit None passed to
    the OpenAI client (which would disable the client's own default timeout).
    """

    def _install_fake_openai(self, captured: dict) -> None:
        class FakeMessage:
            content = "recognized text"

        class FakeChoice:
            message = FakeMessage()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured["request"] = kwargs
                return FakeCompletion()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.chat = FakeChat()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI
        sys.modules["openai"] = fake_module

    def _call_vision_api(self, **kwargs) -> dict:
        captured: dict = {}
        self._install_fake_openai(captured)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                png = os.path.join(tmp, "p.png")
                with open(png, "wb") as f:
                    f.write(b"\x89PNG\r\n\x1a\n")
                ocr.vision_ocr(
                    [(1, png)],
                    vision_api_url="http://x",
                    vision_api_key="k",
                    vision_model="m",
                    **kwargs,
                )
        finally:
            del sys.modules["openai"]
        return captured

    def test_omits_timeout_kwarg_when_none(self):
        captured = self._call_vision_api(timeout=None)
        self.assertNotIn("timeout", captured)

    def test_passes_timeout_kwarg_when_set(self):
        captured = self._call_vision_api(timeout=12.5)
        self.assertEqual(captured.get("timeout"), 12.5)

    def test_default_prompt_in_request(self):
        captured = self._call_vision_api()
        content = captured["request"]["messages"][0]["content"]
        self.assertEqual(content[1]["text"], ocr.DEFAULT_VISION_PROMPT)

    def test_custom_prompt_in_request(self):
        prompt = "Return CSV rows only."
        captured = self._call_vision_api(vision_prompt=prompt)
        content = captured["request"]["messages"][0]["content"]
        self.assertEqual(content[1]["text"], prompt)

    def test_provider_error_is_sanitized(self):
        class FailingCompletions:
            def create(self, **kwargs):
                raise RuntimeError("provider secret diagnostics")

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=FailingCompletions())

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI
        sys.modules["openai"] = fake_module
        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as source:
                with self.assertRaises(ocr.OcrError) as context:
                    ocr.vision_ocr(
                        [(1, source.name)],
                        vision_api_key="k",
                        vision_model="m",
                    )
            self.assertNotIn("provider secret", str(context.exception))
            self.assertIn("vision request failed", str(context.exception))
        finally:
            del sys.modules["openai"]

    def test_client_constructor_error_is_sanitized(self):
        class FakeOpenAI:
            def __init__(self, **kwargs):
                raise RuntimeError("constructor secret")

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI
        sys.modules["openai"] = fake_module
        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as source:
                with self.assertRaises(ocr.OcrError) as context:
                    ocr.vision_ocr(
                        [(1, source.name)],
                        vision_api_key="k",
                        vision_model="m",
                    )
            self.assertNotIn("constructor secret", str(context.exception))
            self.assertIn("initialization failed", str(context.exception))
        finally:
            del sys.modules["openai"]

    def test_blank_response_errors(self):
        class BlankMessage:
            content = "  "

        class BlankChoice:
            message = BlankMessage()

        class BlankCompletions:
            def create(self, **kwargs):
                return types.SimpleNamespace(choices=[BlankChoice()])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=BlankCompletions())

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeOpenAI
        sys.modules["openai"] = fake_module
        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as source:
                with self.assertRaises(ocr.OcrError) as context:
                    ocr.vision_ocr(
                        [(1, source.name)],
                        vision_api_key="k",
                        vision_model="m",
                    )
            self.assertIn("empty result", str(context.exception))
        finally:
            del sys.modules["openai"]


class VisionPromptCli(unittest.TestCase):
    def test_inline_and_file_are_mutually_exclusive(self):
        parser = ocr_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "scan.png",
                "--vision-prompt", "inline",
                "--vision-prompt-file", "prompt.txt",
            ])

    def test_reads_utf8_prompt_file(self):
        parser = ocr_cli.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prompt.txt")
            prompt = "Preserve the table structure."
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompt)
            args = parser.parse_args([
                "scan.png", "--engine", "vision", "--vision-prompt-file", path
            ])
            self.assertEqual(ocr_cli._vision_prompt_from_args(args, parser), prompt)

    def test_missing_prompt_file_errors(self):
        parser = ocr_cli.build_parser()
        args = parser.parse_args([
            "scan.png", "--engine", "vision",
            "--vision-prompt-file", "/missing/prompt.txt"
        ])
        with self.assertRaises(SystemExit):
            ocr_cli._vision_prompt_from_args(args, parser)

    def test_prompt_rejected_for_non_vision_engine(self):
        parser = ocr_cli.build_parser()
        args = parser.parse_args([
            "scan.png", "--engine", "tesseract", "--vision-prompt", "text"
        ])
        with self.assertRaises(SystemExit):
            ocr_cli._vision_prompt_from_args(args, parser)


class VisionPromptCache(unittest.TestCase):
    def _process(
        self,
        path: str,
        cache: ocr.Cache,
        prompt: str,
        *,
        model: str = "m",
        url: str = "",
    ) -> None:
        options = ocr.RecognizeOptions(
            engine="vision",
            dpi=150,
            preprocess="none",
            vision_api_key="k",
            vision_model=model,
            vision_api_url=url,
            vision_prompt=prompt,
        )
        caps = types.SimpleNamespace(require_render=lambda: None, has_openai=True)
        with tempfile.TemporaryDirectory() as tmp:
            ocr.process_file(path, options, caps, cache, tmp)

    def test_custom_prompt_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_ocr", return_value="text") as call:
                self._process(path, cache, "Prompt A")
                self._process(path, cache, "Prompt B")
            self.assertEqual(call.call_count, 2)

    def test_blank_prompts_share_default_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_ocr", return_value="text") as call:
                self._process(path, cache, "")
                self._process(path, cache, " \n")
            self.assertEqual(call.call_count, 1)

    def test_model_and_provider_change_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_ocr", return_value="text") as call:
                self._process(path, cache, "Prompt", model="model-a", url="http://a")
                self._process(path, cache, "Prompt", model="model-b", url="http://a")
                self._process(path, cache, "Prompt", model="model-b", url="http://b")
            self.assertEqual(call.call_count, 3)

    def test_page_selection_changes_cache_key(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            cache = ocr.Cache(None)
            caps = types.SimpleNamespace(require_render=lambda: None, has_openai=True)
            probe = {"pages": 5, "needs_ocr": True, "reason": "scan"}
            with (mock.patch.object(ocr, "probe_pdf", return_value=probe),
                  mock.patch.object(ocr, "render_pages", return_value=[(1, "page.png")]),
                  mock.patch.object(ocr, "vision_ocr", return_value="text") as call):
                for pages in ("1", "2"):
                    options = ocr.RecognizeOptions(
                        engine="vision",
                        dpi=150,
                        preprocess="none",
                        pages=pages,
                        vision_api_key="k",
                        vision_model="m",
                    )
                    ocr.process_file(source.name, options, caps, cache, tempfile.gettempdir())
            self.assertEqual(call.call_count, 2)


try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class EncodePageB64(unittest.TestCase):
    """_encode_page_b64 keeps small images untouched (correct media type) and
    re-encodes oversized images to JPEG under the byte limit.
    """

    def _write(self, data: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        tmp.write(data)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_small_png_passthrough(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        path = self._write(raw)
        b64, media_type = ocr._encode_page_b64(path)
        self.assertEqual(media_type, "image/png")
        self.assertEqual(base64.b64decode(b64), raw)

    def test_small_jpeg_passthrough(self):
        raw = b"\xff\xd8\xff\xe0" + b"\x00" * 32
        path = self._write(raw)
        b64, media_type = ocr._encode_page_b64(path)
        self.assertEqual(media_type, "image/jpeg")
        self.assertEqual(base64.b64decode(b64), raw)

    @unittest.skipUnless(_HAS_PIL, "Pillow required")
    def test_oversized_reencoded_under_limit(self):
        # Noisy large image so PNG stays incompressible and exceeds the limit.
        w = h = 4000
        img = _PILImage.frombytes("RGB", (w, h), os.urandom(w * h * 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        self.assertGreater(len(raw), ocr.VISION_IMAGE_BYTE_LIMIT)
        path = self._write(raw)
        b64, media_type = ocr._encode_page_b64(path)
        self.assertEqual(media_type, "image/jpeg")
        self.assertLessEqual(len(base64.b64decode(b64)), ocr.VISION_IMAGE_BYTE_LIMIT)

    def test_oversized_without_pil_fatals(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * (ocr.VISION_IMAGE_BYTE_LIMIT + 1)
        path = self._write(raw)
        saved = sys.modules.get("PIL")
        sys.modules["PIL"] = None  # force ImportError on `from PIL import Image`
        try:
            with self.assertRaises(ocr.OcrError) as ctx:
                ocr._encode_page_b64(path)
            self.assertEqual(ctx.exception.code, ocr.EXIT_MISSING_BINARY)
        finally:
            if saved is None:
                del sys.modules["PIL"]
            else:
                sys.modules["PIL"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
