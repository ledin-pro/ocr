import argparse
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from pro.ledin.ocr import cli
from pro.ledin.ocr import core
from pro.ledin.ocr import vision_handoff


class ProbeDecision(unittest.TestCase):
    def decide(self, **overrides):
        signals = {
            "encrypted": False,
            "median_chars": 0,
            "total_fonts": 0,
            "non_unicode_fonts": 0,
            "smask_count": 0,
        }
        signals.update(overrides)
        return core.decide_probe(signals)

    def test_encrypted_is_blocked(self):
        result = self.decide(encrypted=True, median_chars=500)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["needs_ocr"])

    def test_threshold_boundary(self):
        self.assertTrue(self.decide(median_chars=29)["needs_ocr"])
        self.assertFalse(
            self.decide(
                median_chars=30,
                total_fonts=1,
                non_unicode_fonts=0,
            )["needs_ocr"]
        )

    def test_raster_and_masks_override_text_yield(self):
        result = self.decide(
            median_chars=30,
            total_fonts=1,
            non_unicode_fonts=1,
            smask_count=3,
        )
        self.assertTrue(result["needs_ocr"])

    def test_one_secondary_signal_does_not_override_text_yield(self):
        self.assertFalse(
            self.decide(
                median_chars=30,
                total_fonts=1,
                non_unicode_fonts=1,
                smask_count=2,
            )["needs_ocr"]
        )

    def test_image_probe(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            result = core.probe_input(source.name, types.SimpleNamespace())
        self.assertEqual(result["input_type"], "image")
        self.assertTrue(result["needs_ocr"])

    def test_unicode_counts_match_shell_byte_semantics(self):
        self.assertEqual(core._nonspace_byte_count("я" * 20), 40)


class EngineConfiguration(unittest.TestCase):
    def parse(self, *args):
        return cli.build_parser().parse_args(["scan.png", *args])

    def test_default_engine_is_tesseract(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                cli._resolve_engine(self.parse(), cli.build_parser()),
                "tesseract",
            )

    def test_environment_engine(self):
        with mock.patch.dict(os.environ, {"OCR_ENGINE": "paddleocr"}, clear=True):
            self.assertEqual(
                cli._resolve_engine(self.parse(), cli.build_parser()),
                "paddleocr",
            )

    def test_cli_overrides_environment(self):
        with mock.patch.dict(os.environ, {"OCR_ENGINE": "paddleocr"}, clear=True):
            self.assertEqual(
                cli._resolve_engine(
                    self.parse("--engine", "easyocr"), cli.build_parser()
                ),
                "easyocr",
            )

    def test_invalid_environment_engine_errors(self):
        with mock.patch.dict(os.environ, {"OCR_ENGINE": "auto"}, clear=True):
            with self.assertRaises(SystemExit):
                cli._resolve_engine(self.parse(), cli.build_parser())

    def test_auto_escalate_environment(self):
        args = self.parse()
        args.engine = "tesseract"
        with mock.patch.dict(
            os.environ,
            {"OCR_AUTO_ESCALATE": "easyocr, vision,easyocr"},
            clear=True,
        ):
            self.assertEqual(
                cli._resolve_auto_escalate(args, cli.build_parser()),
                ("easyocr", "vision"),
            )

    def test_prompt_allowed_for_escalation_only_vision(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "scan.png",
            "--auto-escalate",
            "vision",
            "--vision-prompt",
            "Read table",
        ])
        args.engine = cli._resolve_engine(args, parser)
        args.auto_escalate = cli._resolve_auto_escalate(args, parser)
        self.assertEqual(cli._vision_prompt_from_args(args, parser), "Read table")

    def test_probe_mode_does_not_resolve_engine_or_ocr(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            stdout = io.StringIO()
            with (mock.patch.dict(os.environ, {"OCR_ENGINE": "invalid"}, clear=True),
                  mock.patch.object(sys, "stdout", stdout),
                  mock.patch.object(cli, "process_file") as process):
                cli.run([source.name, "--probe"])
        process.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["input_type"], "image")


class StrictEscalation(unittest.TestCase):
    def caps(self):
        return types.SimpleNamespace(
            require_ocr=lambda: None,
            bin_tesseract="tesseract",
            has_easyocr=True,
            require_paddleocr=lambda: None,
            has_paddleocr=True,
            has_paddle=True,
            has_openai=True,
        )

    def test_chain_runs_and_vision_becomes_final(self):
        options = core.RecognizeOptions(
            engine="tesseract",
            auto_escalate=("easyocr", "vision"),
            vision_api_key="k",
            vision_model="m",
        )

        def recognize_once(path, attempt, caps, cache, tmpdir):
            if attempt.engine == "tesseract":
                return [{
                    "n": 1, "source": "tesseract", "mean_conf": 40.0,
                    "flag": "review-vision", "text": "base", "words": [],
                }]
            if attempt.engine == "easyocr":
                return [{
                    "n": 1, "source": "easyocr", "mean_conf": 80.0,
                    "flag": None, "text": "easy", "words": [{"text": "easy"}],
                }]
            return [{
                "n": 0, "source": "vision", "mean_conf": None,
                "flag": None, "text": "vision", "words": [],
            }]

        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(core, "_process_file_once", side_effect=recognize_once):
                pages = core.process_file(
                    source.name, options, self.caps(), core.Cache(None), tempfile.gettempdir()
                )
        self.assertEqual(pages[0]["source"], "vision")
        self.assertEqual(pages[0]["n"], 1)
        self.assertEqual(
            [item["engine"] for item in pages[0]["decision"]["attempts"]],
            ["tesseract", "easyocr", "vision"],
        )

    def test_missing_requested_engine_stops_before_baseline(self):
        options = core.RecognizeOptions(
            auto_escalate=("easyocr",),
        )
        caps = self.caps()
        caps.has_easyocr = False
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(core, "_process_file_once") as process:
                with self.assertRaises(core.OcrError):
                    core.process_file(
                        source.name, options, caps, core.Cache(None), tempfile.gettempdir()
                    )
        process.assert_not_called()

    def test_attempt_failure_stops_without_cache(self):
        options = core.RecognizeOptions(auto_escalate=("easyocr",))
        cache = core.Cache(None)

        def recognize_once(path, attempt, caps, ignored_cache, tmpdir):
            if attempt.engine == "tesseract":
                return [{
                    "n": 1, "source": "tesseract", "mean_conf": 40.0,
                    "flag": "review-vision", "text": "base", "words": [],
                }]
            raise core.OcrError("engine failed")

        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(core, "_process_file_once", side_effect=recognize_once):
                with self.assertRaises(core.OcrError):
                    core.process_file(
                        source.name, options, self.caps(), cache, tempfile.gettempdir()
                    )
        self.assertEqual(cache._data, {})

    def test_empty_attempt_result_stops_run(self):
        options = core.RecognizeOptions(auto_escalate=("easyocr",))

        def recognize_once(path, attempt, caps, ignored_cache, tmpdir):
            if attempt.engine == "tesseract":
                return [{
                    "n": 1, "source": "tesseract", "mean_conf": 10.0,
                    "flag": "review-vision", "text": "base", "words": [],
                }]
            return [{
                "n": 1, "source": "easyocr", "mean_conf": 0.0,
                "flag": "review-vision", "text": "", "words": [],
            }]

        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(core, "_process_file_once", side_effect=recognize_once):
                with self.assertRaises(core.OcrError):
                    core.process_file(
                        source.name,
                        options,
                        self.caps(),
                        core.Cache(None),
                        tempfile.gettempdir(),
                    )

    def test_skip_ocr_returns_all_pdf_pages(self):
        options = core.RecognizeOptions(skip_ocr=True, dpi=150, preprocess="none")
        caps = types.SimpleNamespace()
        probe = {
            "status": "ready", "pages": 2, "needs_ocr": True,
            "reason": "scan",
        }
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            with mock.patch.object(core, "probe_pdf", return_value=probe):
                pages = core.process_file(
                    source.name, options, caps, core.Cache(None), tempfile.gettempdir()
                )
        self.assertEqual([page["n"] for page in pages], [1, 2])
        self.assertTrue(all(page["skipped"] for page in pages))

    def test_easyocr_low_confidence_is_flagged(self):
        options = core.RecognizeOptions(
            engine="easyocr", dpi=150, preprocess="none", min_conf=60
        )
        caps = types.SimpleNamespace(has_easyocr=True)
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with mock.patch.object(
                core, "ocr_easyocr", return_value=("weak", 20.0, [{"text": "weak"}])
            ):
                pages = core.process_file(
                    source.name,
                    options,
                    caps,
                    core.Cache(None),
                    tempfile.gettempdir(),
                )
        self.assertEqual(pages[0]["flag"], "review-vision")


class AtomicCache(unittest.TestCase):
    def test_cache_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.json")
            cache = core.Cache(path)
            cache.set("key", {"pages": []})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["key"], {"pages": []})
            self.assertEqual(
                [name for name in os.listdir(tmp) if name.endswith(".tmp")], []
            )

    def test_failed_cache_write_restores_memory_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.json")
            cache = core.Cache(path)
            cache.set("key", {"value": "old"})
            with mock.patch.object(core.os, "replace", side_effect=OSError("disk")):
                with self.assertRaises(core.OcrError):
                    cache.set("key", {"value": "new"})
            self.assertEqual(cache.get("key"), {"value": "old"})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["key"], {"value": "old"})

    def test_cache_parent_failure_restores_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "not-a-directory")
            with open(parent, "w", encoding="utf-8") as handle:
                handle.write("file")
            cache = core.Cache(os.path.join(parent, "cache.json"))
            with self.assertRaises(core.OcrError):
                cache.set("key", {"value": "new"})
            self.assertIsNone(cache.get("key"))

    def test_cache_serialization_failure_restores_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = core.Cache(os.path.join(tmp, "cache.json"))
            with self.assertRaises(core.OcrError):
                cache.set("key", {"value": {"not-json"}})
            self.assertIsNone(cache.get("key"))


class PublicLibraryBehavior(unittest.TestCase):
    def test_skip_ocr_does_not_require_tesseract(self):
        options = core.RecognizeOptions(skip_ocr=True)
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            pages = core.recognize(source.name, options, caps=types.SimpleNamespace())
        self.assertTrue(pages[0]["skipped"])

    def test_text_layer_does_not_require_selected_backend(self):
        options = core.RecognizeOptions(engine="vision")
        probe = {"status": "ready", "pages": 1, "needs_ocr": False}
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            with (mock.patch.object(core, "probe_pdf", return_value=probe),
                  mock.patch.object(core, "auto_dpi", return_value=150),
                  mock.patch.object(core, "resolve_preprocess", return_value="none"),
                  mock.patch.object(core, "extract_text_layer", return_value=["text"])):
                pages = core.recognize(
                    source.name,
                    options,
                    caps=types.SimpleNamespace(),
                )
        self.assertEqual(pages[0]["source"], "text_layer")


class VisionHandoffModule(unittest.TestCase):
    def test_image_manifest_uses_source_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            stderr = io.StringIO()
            with mock.patch.object(sys, "stderr", stderr):
                vision_handoff.run([source.name, "--vision-prompt", "Read it"])
        self.assertIn(os.path.abspath(source.name), stderr.getvalue())
        self.assertIn("Read it", stderr.getvalue())

    def test_pdf_renders_selected_pages_without_ocr(self):
        probe = {"status": "ready", "pages": 5}
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            with (mock.patch.object(vision_handoff, "probe_pdf", return_value=probe),
                  mock.patch.object(vision_handoff, "auto_dpi", return_value=200),
                  mock.patch.object(
                      vision_handoff,
                      "render_pages",
                      return_value=[(2, "/tmp/page_0002.png")],
                  ) as render,
                  mock.patch.object(vision_handoff, "vision_handoff") as manifest):
                vision_handoff.run([source.name, "--pages", "2"])
        self.assertEqual(render.call_args.args[2], [2])
        manifest.assert_called_once()

    def test_max_pages_flag_is_absent(self):
        with self.assertRaises(SystemExit):
            vision_handoff.build_parser().parse_args(["scan.pdf", "--max-pages", "1"])

    def test_invalid_page_range_is_parser_error(self):
        probe = {"status": "ready", "pages": 5}
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            with mock.patch.object(vision_handoff, "probe_pdf", return_value=probe):
                with self.assertRaises(SystemExit):
                    vision_handoff.run([source.name, "--pages", "bad"])

    def test_pages_rejected_for_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as source:
            with self.assertRaises(SystemExit):
                vision_handoff.run([source.name, "--pages", "1"])


if __name__ == "__main__":
    unittest.main()
