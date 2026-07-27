import argparse
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from pro.ledin.ocr import cli
from pro.ledin.ocr import core


def page_result() -> list[dict]:
    return [{
        "n": 1,
        "source": "tesseract",
        "mean_conf": 92.0,
        "flag": None,
        "text": "Hello world",
        "words": [],
    }]


class FormatParsing(unittest.TestCase):
    def test_default(self):
        args = cli.build_parser().parse_args(["scan.png"])
        self.assertEqual(args.format, ["md"])

    def test_single_format(self):
        self.assertEqual(cli.parse_formats("json"), ["json"])

    def test_comma_separated_formats_preserve_order(self):
        self.assertEqual(cli.parse_formats("json,md,txt"), ["json", "md", "txt"])

    def test_whitespace_is_trimmed(self):
        self.assertEqual(cli.parse_formats("md, txt"), ["md", "txt"])

    def test_duplicates_are_removed(self):
        self.assertEqual(cli.parse_formats("md,md,json"), ["md", "json"])

    def test_all_expands(self):
        self.assertEqual(cli.parse_formats("all"), ["md", "txt", "json"])

    def test_all_mixed_with_formats_deduplicates_in_order(self):
        self.assertEqual(
            cli.parse_formats("json,all,md"),
            ["json", "md", "txt"],
        )

    def test_unknown_format_errors(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.parse_formats("md,pdf")

    def test_empty_format_errors(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.parse_formats("md,,json")

    def test_json_report_flag_is_removed(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args([
                "scan.png", "--json-report", "report.json"
            ])


class WriteOutputs(unittest.TestCase):
    def args(
        self,
        formats: list[str],
        out: str = "",
        inputs: list[str] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            engine="tesseract",
            preprocess="none",
            min_conf=60.0,
            format=formats,
            out=out,
            inputs=inputs or ["scan.png"],
            searchable_pdf="",
        )

    def test_single_format_writes_exact_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.md"
            cli.write_outputs(
                page_result(), "scan.png", self.args(["md"], str(output)), "eng", 300
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                core.to_markdown(page_result(), "scan.png"),
            )

    def test_multiple_formats_create_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            cli.write_outputs(
                page_result(),
                "scan.png",
                self.args(["md", "json"], str(output)),
                "eng",
                300,
            )
            self.assertTrue((output / "scan.md").is_file())
            self.assertTrue((output / "scan.json").is_file())

    def test_multiple_inputs_create_output_directory_for_single_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            args = self.args(
                ["md"], str(output), ["one.png", "two.png"]
            )
            cli.write_outputs(page_result(), "one.png", args, "eng", 300)
            cli.write_outputs(page_result(), "two.png", args, "eng", 300)
            self.assertTrue((output / "one.md").is_file())
            self.assertTrue((output / "two.md").is_file())

    def test_multiple_inputs_and_formats_write_complete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            args = self.args(
                ["md", "json"], str(output), ["one.png", "two.png"]
            )
            cli.write_outputs(page_result(), "one.png", args, "eng", 300)
            cli.write_outputs(page_result(), "two.png", args, "eng", 300)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["one.json", "one.md", "two.json", "two.md"],
            )

    def test_all_formats_write_once_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            stderr = io.StringIO()
            with mock.patch.object(sys, "stderr", stderr):
                cli.write_outputs(
                    page_result(),
                    "scan.png",
                    self.args(["md", "txt", "json"], str(output)),
                    "eng",
                    300,
                )
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["scan.json", "scan.md", "scan.txt"],
            )
            self.assertEqual(stderr.getvalue().count("[ocr] wrote"), 3)

    def test_single_stdout_contains_only_content(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            cli.write_outputs(
                page_result(), "scan.png", self.args(["txt"]), "eng", 300
            )
        self.assertEqual(stdout.getvalue().strip(), core.to_text(page_result()).strip())
        self.assertNotIn("[TXT]", stdout.getvalue())

    def test_multiple_stdout_preserves_requested_order(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            cli.write_outputs(
                page_result(),
                "scan.png",
                self.args(["json", "md"]),
                "eng",
                300,
            )
        content = stdout.getvalue()
        self.assertLess(content.index("[JSON]"), content.index("[MD]"))

    def test_json_matches_standalone_formatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            cli.write_outputs(
                page_result(),
                "scan.png",
                self.args(["md", "json"], str(output)),
                "eng",
                300,
            )
            actual = json.loads((output / "scan.json").read_text(encoding="utf-8"))
            expected = json.loads(core.to_json(page_result(), {
                "file": "scan.png",
                "engine": "tesseract",
                "lang": "eng",
                "dpi": 300,
                "preprocess": "none",
                "min_conf": 60.0,
            }))
            self.assertEqual(actual, expected)


class OutputArgumentValidation(unittest.TestCase):
    def test_duplicate_stems_are_rejected_before_processing(self):
        with mock.patch.object(cli, "process_file") as process:
            with self.assertRaises(SystemExit):
                cli.run([
                    "/one/scan.png",
                    "/two/scan.jpg",
                    "--out",
                    "results",
                ])
        process.assert_not_called()

    def test_case_only_duplicate_stems_are_rejected(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "/one/Scan.png",
            "/two/scan.jpg",
            "--out",
            "results",
        ])
        with self.assertRaises(SystemExit):
            cli._validate_output_args(args, parser)

    def test_unicode_normalization_duplicate_stems_are_rejected(self):
        parser = cli.build_parser()
        args = parser.parse_args([
            "/one/caf\u00e9.png",
            "/two/cafe\u0301.jpg",
            "--out",
            "results",
        ])
        with self.assertRaises(SystemExit):
            cli._validate_output_args(args, parser)

    def test_multiple_inputs_without_output_path_are_valid(self):
        parser = cli.build_parser()
        args = parser.parse_args(["one.png", "two.png", "--format", "md"])
        cli._validate_output_args(args, parser)

    def test_multiple_inputs_with_json_require_output_directory(self):
        with mock.patch.object(cli, "process_file") as process:
            with self.assertRaises(SystemExit):
                cli.run(["one.png", "two.png", "--format", "json"])
        process.assert_not_called()

    def test_multiple_inputs_with_searchable_pdf_are_rejected(self):
        with mock.patch.object(cli, "process_file") as process:
            with self.assertRaises(SystemExit):
                cli.run([
                    "one.png",
                    "two.png",
                    "--searchable-pdf",
                    "output.pdf",
                ])
        process.assert_not_called()


class OutputTransactions(unittest.TestCase):
    def test_file_set_failure_restores_all_previous_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.md"
            second = Path(tmp) / "second.json"
            first.write_text("old first", encoding="utf-8")
            second.write_text("old second", encoding="utf-8")
            real_replace = os.replace

            def replace_with_failure(source, destination):
                if str(source).endswith(".tmp") and Path(destination) == second:
                    raise OSError("disk full")
                return real_replace(source, destination)

            with mock.patch.object(cli.os, "replace", side_effect=replace_with_failure):
                with self.assertRaises(cli.OcrError):
                    cli._write_file_set([
                        (str(first), "new first"),
                        (str(second), "new second"),
                    ])
            self.assertEqual(first.read_text(encoding="utf-8"), "old first")
            self.assertEqual(second.read_text(encoding="utf-8"), "old second")
            self.assertFalse(list(Path(tmp).glob("*.tmp")))
            self.assertFalse(list(Path(tmp).glob("*.bak")))

    def test_later_input_failure_preserves_first_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one.png"
            second = Path(tmp) / "two.png"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            output = Path(tmp) / "results"
            pages = page_result()
            with (mock.patch.object(cli, "Caps", return_value=types.SimpleNamespace()),
                  mock.patch.object(
                      cli,
                      "process_file",
                      side_effect=[pages, cli.OcrError("second failed")],
                  )):
                with self.assertRaises(cli.OcrError):
                    cli.run([
                        str(first),
                        str(second),
                        "--format",
                        "md",
                        "--out",
                        str(output),
                    ])
            self.assertTrue((output / "one.md").is_file())
            self.assertFalse((output / "two.md").exists())


class ProbeErrorOutput(unittest.TestCase):
    def test_probe_runtime_error_preserves_pdf_type(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as source:
            stdout = io.StringIO()
            with (mock.patch.object(sys, "stdout", stdout),
                  mock.patch.object(cli, "probe_input", side_effect=cli.OcrError("probe failed"))):
                with self.assertRaises(cli.OcrError):
                    cli.run([source.name, "--probe"])
        self.assertEqual(json.loads(stdout.getvalue())["input_type"], "pdf")

    def test_unsupported_probe_emits_error_json(self):
        with tempfile.NamedTemporaryFile(suffix=".docx") as source:
            stdout = io.StringIO()
            with mock.patch.object(sys, "stdout", stdout):
                with self.assertRaises(cli.OcrError):
                    cli.run([source.name, "--probe"])
        self.assertEqual(json.loads(stdout.getvalue())["input_type"], "unsupported")


if __name__ == "__main__":
    unittest.main()
