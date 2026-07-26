#!/usr/bin/env python3
"""Unit tests for pro.ledin.ocr pure helpers. Run: python3 -m pytest tests"""

import base64
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


class RecognizeOptionsDefaults(unittest.TestCase):
    """Guard against silent drift between library defaults and prior CLI defaults."""

    def test_defaults_match_former_argparse_defaults(self):
        options = ocr.RecognizeOptions()
        self.assertEqual(options.engine, "auto")
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

    def test_recognize_rejects_vision_engine(self):
        with self.assertRaises(ocr.OcrError):
            ocr.recognize("whatever.png", ocr.RecognizeOptions(engine="vision"))

    def test_vision_engines_respect_max_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scan.pdf")
            with open(path, "wb") as f:
                f.write(b"pdf")
            caps = types.SimpleNamespace(require_render=lambda: None)
            probe = {"pages": 5, "needs_ocr": True, "reason": "scan"}
            for engine in ("vision", "vision-api"):
                with self.subTest(engine=engine):
                    options = ocr.RecognizeOptions(
                        engine=engine,
                        dpi=150,
                        preprocess="none",
                        max_pages=2,
                        vision_api_key="k",
                        vision_model="m",
                    )
                    with (mock.patch.object(ocr, "probe_pdf", return_value=probe),
                          mock.patch.object(ocr, "render_pages", return_value=[(1, "page.png")]) as render,
                          mock.patch.object(ocr, "vision_handoff"),
                          mock.patch.object(ocr, "vision_api", return_value="text")):
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
                ocr.vision_api(
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
            engine="vision-api",
            dpi=150,
            preprocess="none",
            vision_api_key="k",
            vision_model=model,
            vision_api_url=url,
            vision_prompt=prompt,
        )
        caps = types.SimpleNamespace(require_render=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            ocr.process_file(path, options, caps, cache, tmp)

    def test_custom_prompt_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_api", return_value="text") as call:
                self._process(path, cache, "Prompt A")
                self._process(path, cache, "Prompt B")
            self.assertEqual(call.call_count, 2)

    def test_blank_prompts_share_default_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_api", return_value="text") as call:
                self._process(path, cache, "")
                self._process(path, cache, " \n")
            self.assertEqual(call.call_count, 1)

    def test_model_and_provider_change_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as f:
                f.write(b"png")
            cache = ocr.Cache(None)
            with mock.patch.object(ocr, "vision_api", return_value="text") as call:
                self._process(path, cache, "Prompt", model="model-a", url="http://a")
                self._process(path, cache, "Prompt", model="model-b", url="http://a")
                self._process(path, cache, "Prompt", model="model-b", url="http://b")
            self.assertEqual(call.call_count, 3)

    def test_page_selection_changes_cache_key(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            cache = ocr.Cache(None)
            caps = types.SimpleNamespace(require_render=lambda: None)
            probe = {"pages": 5, "needs_ocr": True, "reason": "scan"}
            with (mock.patch.object(ocr, "probe_pdf", return_value=probe),
                  mock.patch.object(ocr, "render_pages", return_value=[(1, "page.png")]),
                  mock.patch.object(ocr, "vision_api", return_value="text") as call):
                for pages in ("1", "2"):
                    options = ocr.RecognizeOptions(
                        engine="vision-api",
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
