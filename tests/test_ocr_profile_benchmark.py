import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "ocr-profile-benchmark.py"
    spec = importlib.util.spec_from_file_location("ocr_profile_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_resolves_relative_paths_and_external_targets(tmp_path: Path):
    module = load_benchmark_module()
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(
        json.dumps(
            {
                "source_dir": "input",
                "run_root": "reports",
                "targets": [
                    {"kind": "gpt", "model": "vision-model", "slug": "vision-model"}
                ],
                "judge": {"model": "judge-model"},
            }
        )
    )

    config = module.load_config(config_path)

    assert config["source_dir"] == str((tmp_path / "input").resolve())
    assert config["run_root"] == str((tmp_path / "reports").resolve())
    assert config["targets"] == [["gpt", "vision-model", "vision-model"]]
    assert config["judge"]["model"] == "judge-model"
    assert config["version"] == 5
    assert "healthos_bin" not in config
    assert config["ocr_timeout_seconds"] == 900
    assert config["openai_api_key_env"] == "OPENAI_API_KEY"


def test_legacy_config_fields_are_rejected(tmp_path: Path):
    module = load_benchmark_module()
    with pytest.raises(ValueError, match="Unsupported legacy benchmark fields"):
        module.normalize_config(
            {
                "source_dir": "input",
                "targets": [{"kind": "gpt", "model": "model", "slug": "model"}],
                "judge": {"model": "judge-model"},
                "healthos_ocr_timeout_seconds": 900,
            },
            tmp_path,
        )


def test_old_run_config_version_is_rejected(tmp_path: Path):
    module = load_benchmark_module()
    with pytest.raises(ValueError, match="Unsupported benchmark config version"):
        module.normalize_config(
            {
                "version": 2,
                "source_dir": "input",
                "targets": [{"kind": "gpt", "model": "model", "slug": "model"}],
                "judge": {"model": "judge-model"},
            },
            tmp_path,
        )


def test_new_run_passes_skip_judge_to_benchmark(tmp_path: Path, monkeypatch):
    module = load_benchmark_module()
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(
        json.dumps(
            {
                "source_dir": "input",
                "targets": [
                    {"kind": "gpt", "model": "vision-model", "slug": "vision-model"}
                ],
                "judge": {"model": "judge-model"},
            }
        )
    )
    captured = {}

    class FakeBenchmark:
        def __init__(self, run_root, skip_judge, *, resume, source_dir, config):
            captured["skip_judge"] = skip_judge
            captured["run_root_existed"] = run_root.exists()
            run_root.mkdir(parents=True, exist_ok=True)
            self.run_root = run_root
            self.summary_path = run_root / "summary.tsv"
            self.judge_settings = config["judge"]

        def log(self, *_args, **_kwargs):
            pass

        def completed_profiles(self):
            return set()

        def run_openai(self, *_args):
            pass

        def print_results(self):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(module, "Benchmark", FakeBenchmark)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ocr-profile-benchmark.py",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "run"),
            "--skip-judge",
        ],
    )

    assert module.main() == 0
    assert captured["skip_judge"] is True
    assert captured["run_root_existed"] is False


def test_login_admin_session_exchanges_api_key_for_cookie(monkeypatch):
    module = load_benchmark_module()

    def fake_http_json_with_headers(*_args, **_kwargs):
        return {"success": True}, {"set-cookie": "omlx_admin_session=signed-token; Path=/"}

    monkeypatch.setattr(module, "http_json_with_headers", fake_http_json_with_headers)

    assert module.login_admin_session("http://omlx", "api-key", 10) == "signed-token"


def test_judge_filters_by_document_count_not_page_count(tmp_path: Path):
    module = load_benchmark_module()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "profile\tmodel\tkind\tresult\telapsed_seconds\toutput_documents\tprofile_name\texposed_model\tcleanup\n"
        "profile\tmodel\tomlx\tok\t10\t4\tname\texposed\tdeleted\n"
    )

    assert len(module.successful_judge_rows(summary_path, expected_documents=4)) == 1
    assert module.successful_judge_rows(summary_path, expected_documents=17) == []


def test_run_target_uses_ocr_library_and_writes_markdown(tmp_path: Path, monkeypatch):
    module = load_benchmark_module()
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    document = source_dir / "scan.pdf"
    document.write_bytes(b"pdf")
    config = module.normalize_config(
        {
            "source_dir": str(source_dir),
            "run_root": "run",
            "targets": [{"kind": "gpt", "model": "model", "slug": "model"}],
            "judge": {"model": "judge-model"},
        },
        tmp_path,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "judge-key")

    captured = {}

    class FakeOcr:
        RecognizeOptions = module.ocr.RecognizeOptions

        @staticmethod
        def recognize(path, options):
            captured["path"] = path
            captured["options"] = options
            return [{"n": 1, "text": "recognized"}]

        @staticmethod
        def to_markdown(pages, filename):
            return f"# {filename}\n\n{pages[0]['text']}\n"

    monkeypatch.setattr(module, "ocr", FakeOcr)
    benchmark = module.Benchmark(
        tmp_path / "run",
        skip_judge=True,
        source_dir=source_dir,
        config=config,
    )

    benchmark.run_target(
        profile="vision-openai-model",
        model="model",
        kind="gpt",
        endpoint="http://openai",
        api_key="vision-key",
        exposed_model="model",
        profile_name="-",
    )

    output = tmp_path / "run" / "vision-openai-model" / "output" / "scan.md"
    assert output.read_text() == "# scan.pdf\n\nrecognized\n"
    assert captured["path"] == document
    assert captured["options"].engine == "vision"
    assert captured["options"].vision_api_url == "http://openai/v1"
    assert captured["options"].vision_api_key == "vision-key"
    assert captured["options"].vision_model == "model"
    assert captured["options"].timeout == 900.0


def test_run_target_continues_after_one_ocr_error(tmp_path: Path, monkeypatch):
    module = load_benchmark_module()
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "failed.pdf").write_bytes(b"pdf")
    (source_dir / "ok.pdf").write_bytes(b"pdf")
    config = module.normalize_config(
        {
            "source_dir": str(source_dir),
            "run_root": "run",
            "targets": [{"kind": "gpt", "model": "model", "slug": "model"}],
            "judge": {"model": "judge-model"},
        },
        tmp_path,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "judge-key")

    class FakeOcr:
        RecognizeOptions = module.ocr.RecognizeOptions

        @staticmethod
        def recognize(path, _options):
            if path.stem == "failed":
                raise module.ocr.OcrError("vision request failed")
            return [{"n": 1, "text": "recognized"}]

        @staticmethod
        def to_markdown(_pages, filename):
            return f"# {filename}\n"

    monkeypatch.setattr(module, "ocr", FakeOcr)
    benchmark = module.Benchmark(
        tmp_path / "run",
        skip_judge=True,
        source_dir=source_dir,
        config=config,
    )

    benchmark.run_target(
        profile="vision-openai-model",
        model="model",
        kind="gpt",
        endpoint="http://openai",
        api_key="secret-key",
        exposed_model="model",
        profile_name="-",
    )

    assert not (tmp_path / "run" / "vision-openai-model" / "output" / "failed.md").exists()
    assert (tmp_path / "run" / "vision-openai-model" / "output" / "ok.md").exists()
    log = (tmp_path / "run" / "vision-openai-model" / "ocr.log").read_text()
    assert "OCR_FAILED\tfailed.pdf" in log
    assert "secret-key" not in log
    assert "healthos" not in log.lower()
