import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pro.ledin.ocr import peepshow_sink as sink


def page_result(text: str = "recognized text") -> list[dict]:
    return [{
        "n": 1,
        "source": "tesseract",
        "mean_conf": 95.0,
        "flag": None,
        "text": text,
        "words": [],
    }]


class PeepshowSinkTestCase(unittest.TestCase):
    def make_payload(self, directory: str, frame_count: int = 1) -> dict:
        frames = []
        for index in range(frame_count):
            path = os.path.join(directory, f"frame_{index + 1:04d}.jpg")
            with open(path, "wb") as handle:
                handle.write(f"frame-{index}".encode())
            frames.append({
                "path": path,
                "bytes": os.path.getsize(path),
                "timestampSeconds": index * 1.5,
            })
        return {
            "outputDir": directory,
            "strategy": "scene",
            "frames": frames,
            "video": {"durationSeconds": 3.0},
            "extraction": {"framesEmitted": frame_count},
        }


class PayloadValidation(PeepshowSinkTestCase):
    def test_empty_stdin_errors(self):
        with self.assertRaises(sink.OcrError):
            sink.read_payload(io.StringIO(""))

    def test_malformed_json_errors(self):
        with self.assertRaises(sink.OcrError):
            sink.read_payload(io.StringIO("{"))

    def test_non_object_json_errors(self):
        with self.assertRaises(sink.OcrError):
            sink.read_payload(io.StringIO("[]"))

    def test_valid_payload_returns_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp, 2)
            frames = sink.validate_payload(payload)
            self.assertEqual(len(frames), 2)

    def test_relative_output_dir_errors(self):
        payload = {
            "outputDir": "relative",
            "strategy": "scene",
            "frames": [],
            "video": {},
            "extraction": {},
        }
        with self.assertRaises(sink.OcrError):
            sink.validate_payload(payload)

    def test_missing_frame_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp)
            payload["frames"][0]["path"] = os.path.join(tmp, "missing.jpg")
            with self.assertRaises(sink.OcrError):
                sink.validate_payload(payload)


class Configuration(unittest.TestCase):
    def parse(self, *args: str):
        return sink.build_parser().parse_args(list(args))

    def test_environment_defaults(self):
        config = sink.resolve_config(self.parse(), {
            "PEEPSHOW_SINK_OCR_ENGINE": "tesseract",
            "PEEPSHOW_SINK_OCR_LANG": "rus+eng",
            "PEEPSHOW_SINK_OCR_DPI": "200",
            "PEEPSHOW_SINK_OCR_TIMEOUT": "12.5",
        })
        self.assertEqual(config.options.engine, "tesseract")
        self.assertEqual(config.options.lang, "rus+eng")
        self.assertEqual(config.options.dpi, 200)
        self.assertEqual(config.options.timeout, 12.5)

    def test_flags_override_environment(self):
        config = sink.resolve_config(
            self.parse(
                "--engine", "easyocr", "--lang", "eng", "--timeout", "3"
            ),
            {
                "PEEPSHOW_SINK_OCR_ENGINE": "tesseract",
                "PEEPSHOW_SINK_OCR_LANG": "rus",
                "PEEPSHOW_SINK_OCR_TIMEOUT": "20",
            },
        )
        self.assertEqual(config.options.engine, "easyocr")
        self.assertEqual(config.options.lang, "eng")
        self.assertEqual(config.options.timeout, 3.0)

    def test_prompt_file_is_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text("Preserve table cells.", encoding="utf-8")
            config = sink.resolve_config(
                self.parse("--vision-prompt-file", str(path)), {}
            )
            self.assertEqual(config.options.vision_prompt, "Preserve table cells.")

    def test_prompt_flag_overrides_environment_file(self):
        config = sink.resolve_config(
            self.parse("--vision-prompt", "Inline"),
            {"PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE": "/missing/prompt.txt"},
        )
        self.assertEqual(config.options.vision_prompt, "Inline")

    def test_conflicting_environment_prompts_error(self):
        with self.assertRaises(sink.OcrError):
            sink.resolve_config(self.parse(), {
                "PEEPSHOW_SINK_OCR_VISION_PROMPT": "Inline",
                "PEEPSHOW_SINK_OCR_VISION_PROMPT_FILE": "/tmp/prompt.txt",
            })

    def test_interactive_vision_engine_errors(self):
        with self.assertRaises(sink.OcrError):
            sink.resolve_config(self.parse("--engine", "vision"), {})

    def test_invalid_numeric_environment_errors_without_leaking_value(self):
        with self.assertRaises(sink.OcrError) as context:
            sink.resolve_config(
                self.parse(), {"PEEPSHOW_SINK_OCR_DPI": "secret-value"}
            )
        self.assertNotIn("secret-value", str(context.exception))

    def test_timeout_must_be_finite_and_positive(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with self.assertRaises(sink.OcrError):
                    sink.resolve_config(
                        self.parse(), {"PEEPSHOW_SINK_OCR_TIMEOUT": value}
                    )


class Processing(PeepshowSinkTestCase):
    def test_processes_frames_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp, 2)
            calls = []

            def recognize_frame(path, options):
                calls.append((path, options.engine))
                return page_result(os.path.basename(path))

            config = sink.SinkConfig(
                options=sink.RecognizeOptions(engine="tesseract"),
                output=None,
            )
            document, output = sink.process_payload(payload, config, recognize_frame)

            self.assertEqual([call[0] for call in calls], [
                payload["frames"][0]["path"],
                payload["frames"][1]["path"],
            ])
            self.assertEqual(output, Path(tmp) / "ocr.json")
            self.assertEqual(document["schemaVersion"], 1)
            self.assertEqual(document["source"], "peepshow")
            self.assertEqual(document["frames"][1]["timestampSeconds"], 1.5)
            self.assertIn("frame_0001.jpg", document["text"])
            self.assertIn("## Frame 2", document["markdown"])

    def test_output_excludes_credentials_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp)
            config = sink.SinkConfig(
                options=sink.RecognizeOptions(
                    engine="vision-api",
                    vision_api_key="top-secret",
                    vision_prompt="private instructions",
                ),
                output=None,
            )
            document, _ = sink.process_payload(
                payload, config, lambda path, options: page_result()
            )
            serialized = json.dumps(document)
            self.assertNotIn("top-secret", serialized)
            self.assertNotIn("private instructions", serialized)

    def test_atomic_write_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocr.json"
            sink.write_json_atomic(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_recognition_failure_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp)
            output = Path(tmp) / "ocr.json"
            config = sink.SinkConfig(sink.RecognizeOptions(), output)

            def fail(path, options):
                raise sink.OcrError("recognition failed")

            with self.assertRaises(sink.OcrError):
                document, path = sink.process_payload(payload, config, fail)
                sink.write_json_atomic(path, document)
            self.assertFalse(output.exists())


class EntryPoint(PeepshowSinkTestCase):
    def test_main_writes_json_without_stdout_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (mock.patch.dict(os.environ, {}, clear=True),
                  mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))),
                  mock.patch.object(sys, "stdout", stdout),
                  mock.patch.object(sys, "stderr", stderr),
                  mock.patch.object(sink, "recognize", return_value=page_result())):
                code = sink.main(["--engine", "tesseract"])

            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            output = Path(tmp) / "ocr.json"
            self.assertTrue(output.exists())

    def test_main_reports_bad_payload(self):
        stderr = io.StringIO()
        with (mock.patch.dict(os.environ, {}, clear=True),
              mock.patch.object(sys, "stdin", io.StringIO("{}")),
              mock.patch.object(sys, "stderr", stderr)):
            code = sink.main([])
        self.assertEqual(code, sink.EXIT_BAD_ARGS)
        self.assertIn("ERROR", stderr.getvalue())

    def test_main_redacts_unexpected_engine_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.make_payload(tmp)
            stderr = io.StringIO()
            with (mock.patch.dict(os.environ, {}, clear=True),
                  mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))),
                  mock.patch.object(sys, "stderr", stderr),
                  mock.patch.object(
                      sink, "recognize", side_effect=RuntimeError("provider secret")
                  )):
                code = sink.main([])
            self.assertEqual(code, sink.EXIT_RUNTIME)
            self.assertNotIn("provider secret", stderr.getvalue())
            self.assertIn("failed unexpectedly", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
