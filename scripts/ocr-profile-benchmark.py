#!/usr/bin/env python3
"""Run the OCR benchmark with temporary oMLX profiles and a GPT judge."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pro.ledin import ocr


DEFAULT_PROFILE_SETTINGS = {
    "enable_thinking": False,
    "thinking_budget_enabled": False,
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "min_p": 0,
    "repetition_penalty": 1,
    "presence_penalty": 0,
    "max_tokens": 4096,
}

DEFAULT_JUDGE_SETTINGS = {
    "temperature": 1,
    "reasoning_effort": "high",
    "max_tokens": 4096,
}

RUN_CONFIG_VERSION = 5

Target = tuple[str, str, str]


class ApiError(RuntimeError):
    pass


def read_api_key(environment_name: str) -> str:
    value = os.environ.get(environment_name)
    if not value:
        raise RuntimeError(f"API key environment variable is not set: {environment_name}")
    return value


def http_json_with_headers(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    ) -> tuple[dict[str, Any], dict[str, str]]:
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return (
                json.loads(body) if body else {},
                {key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise ApiError(f"HTTP {error.code} {url}: {detail}") from error
    except urllib.error.URLError as error:
        raise ApiError(f"Request failed {url}: {error.reason}") from error


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response, _ = http_json_with_headers(
        url,
        method=method,
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    return response


def login_admin_session(api_url: str, api_key: str, timeout: int) -> str:
    response, headers = http_json_with_headers(
        f"{api_url}/admin/api/login",
        method="POST",
        payload={"api_key": api_key, "remember": True},
        timeout=timeout,
    )
    if response.get("success") is not True:
        raise RuntimeError(f"oMLX admin login failed: {response}")
    set_cookie = headers.get("set-cookie", "")
    match = re.search(r"omlx_admin_session=([^;]+)", set_cookie)
    if not match:
        raise RuntimeError("oMLX admin login did not return an admin session cookie")
    return match.group(1)


def count_outputs(path: Path) -> int:
    return len(list(path.glob("*.md")))


def safe_error(error: BaseException, *secrets: str) -> str:
    message = str(error).replace("\t", " ").replace("\n", " ")
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def successful_judge_rows(summary_path: Path, expected_documents: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    columns = ("profile", "model", "kind", "result", "elapsed", "documents", "profile_name", "exposed", "cleanup")
    for line in summary_path.read_text().splitlines()[1:]:
        values = line.split("\t")
        if len(values) < len(columns):
            continue
        row = dict(zip(columns, values))
        if row["result"] == "ok" and row["documents"] == str(expected_documents):
            rows.append(row)
    return rows


def page_map(markdown: str) -> dict[int, str]:
    body = re.sub(r"^---\n[\s\S]*?\n---\n", "", markdown, count=1)
    matches = list(re.finditer(r"^## Page (\d+)\s*$", body, re.MULTILINE))
    if not matches:
        return {1: body.strip()}
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        pages[int(match.group(1))] = body[match.end() : end].strip()
    return pages


def parse_judge_json(content: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.IGNORECASE)
    return json.loads((fenced.group(1) if fenced else content).strip())


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_command(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or "/" in value or value.startswith("."):
        return str(resolve_path(path, base_dir))
    return value


def normalize_target(value: Any) -> Target:
    if isinstance(value, dict):
        kind, model, slug = value.get("kind"), value.get("model"), value.get("slug")
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        kind, model, slug = value
    else:
        raise ValueError("Each target must contain kind, model, and slug")
    if kind not in {"omlx", "gpt"} or not all(isinstance(item, str) and item for item in (model, slug)):
        raise ValueError(f"Invalid benchmark target: {value!r}")
    return kind, model, slug


def normalize_config(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if "version" in raw and raw["version"] != RUN_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported benchmark config version: {raw['version']}; "
            f"expected {RUN_CONFIG_VERSION}"
        )
    legacy_fields = {"healthos_bin", "healthos_ocr_timeout_seconds"} & raw.keys()
    if legacy_fields:
        names = ", ".join(sorted(legacy_fields))
        raise ValueError(f"Unsupported legacy benchmark fields: {names}")
    source_dir = raw.get("source_dir")
    if not source_dir:
        raise ValueError("Config must define source_dir")
    raw_targets = raw.get("targets", [])
    if not raw_targets:
        raise ValueError("Config must define at least one target")
    targets = [normalize_target(value) for value in raw_targets]
    judge = dict(DEFAULT_JUDGE_SETTINGS)
    judge.update(raw.get("judge", {}))
    if not judge.get("model"):
        raise ValueError("Config judge must define model")
    profile_settings = dict(DEFAULT_PROFILE_SETTINGS)
    profile_settings.update(raw.get("profile_settings", {}))
    return {
        "version": RUN_CONFIG_VERSION,
        "source_dir": str(resolve_path(source_dir, base_dir)),
        "run_root": str(resolve_path(raw.get("run_root", "ocr-bench-report"), base_dir)),
        "omlx_api": str(raw.get("omlx_api", "http://127.0.0.1:8000")).rstrip("/"),
        "omlx_api_key_env": str(raw.get("omlx_api_key_env", "OMLX_API_KEY")),
        "openai_api": str(
            raw.get("openai_api", "http://127.0.0.1:4000")
        ).rstrip("/"),
        "openai_api_key_env": str(
            raw.get("openai_api_key_env", "OPENAI_API_KEY")
        ),
        "pdftoppm_bin": resolve_command(str(raw.get("pdftoppm_bin", "pdftoppm")), base_dir),
        "pdfseparate_bin": resolve_command(str(raw.get("pdfseparate_bin", "pdfseparate")), base_dir),
        "ocr_timeout_seconds": int(raw.get("ocr_timeout_seconds", 900)),
        "http_timeout_seconds": int(raw.get("http_timeout_seconds", 60)),
        "profile_exposure_poll_attempts": int(raw.get("profile_exposure_poll_attempts", 30)),
        "judge_attempts": int(raw.get("judge_attempts", 5)),
        "judge_retry_delay_seconds": int(raw.get("judge_retry_delay_seconds", 5)),
        "targets": [list(target) for target in targets],
        "profile_settings": profile_settings,
        "judge": judge,
        "mtp_enabled_by_model": dict(raw.get("mtp_enabled_by_model", {})),
        "only_model": raw.get("only_model"),
        "page_selectors": raw.get("page_selectors"),
        "skip_judge": bool(raw.get("skip_judge", False)),
    }


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON config {path}: {error}") from error
    return normalize_config(raw, path.resolve().parent)


def write_run_config(run_root: Path, config: dict[str, Any]) -> None:
    (run_root / "run-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )


@dataclass
class Benchmark:
    run_root: Path
    skip_judge: bool
    source_dir: Path
    resume: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_root.exists() and any(self.run_root.iterdir()) and not self.resume:
            raise RuntimeError(f"Run root is not empty: {self.run_root}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_root / "events.log"
        self.summary_path = self.run_root / "summary.tsv"
        self.profile_log_path = self.run_root / "profile-lifecycle.tsv"
        self.current_profile: tuple[str, str] | None = None
        self.pending_cleanup: list[tuple[str, str]] = []
        self.profile_settings = dict(self.config["profile_settings"])
        self.judge_settings = dict(self.config["judge"])
        self.omlx_api = self.config["omlx_api"]
        self.omlx_api_key = ""
        self.openai_api = self.config["openai_api"]
        self.pdftoppm_bin = self.config["pdftoppm_bin"]
        self.pdfseparate_bin = self.config["pdfseparate_bin"]
        self.http_timeout = self.config["http_timeout_seconds"]
        self.ocr_timeout = self.config["ocr_timeout_seconds"]
        self.profile_poll_attempts = self.config["profile_exposure_poll_attempts"]
        self.judge_attempts = self.config["judge_attempts"]
        self.judge_retry_delay = self.config["judge_retry_delay_seconds"]
        has_omlx = any(target[0] == "omlx" for target in self.config["targets"])
        needs_openai = not self.skip_judge or any(
            target[0] == "gpt" for target in self.config["targets"]
        )
        if has_omlx:
            self.omlx_api_key = read_api_key(self.config["omlx_api_key_env"])
            self.admin_session = login_admin_session(self.omlx_api, self.omlx_api_key, self.http_timeout)
        else:
            self.admin_session = ""
        self.openai_key = (
            read_api_key(self.config["openai_api_key_env"])
            if needs_openai
            else ""
        )
        if not self.summary_path.exists() or self.summary_path.stat().st_size == 0:
            self.summary_path.write_text(
                "profile\tmodel\tkind\tresult\telapsed_seconds\toutput_documents\tprofile_name\texposed_model\tcleanup\n"
            )
        if not self.profile_log_path.exists() or self.profile_log_path.stat().st_size == 0:
            self.profile_log_path.write_text(
                "model\tprofile_name\texposed_model\tcreated\tdeleted\tdelete_result\n"
            )

    def completed_profiles(self) -> set[str]:
        completed: set[str] = set()
        if not self.summary_path.exists():
            return completed
        expected_documents = len(list(self.source_dir.glob("*.pdf")))
        for line in self.summary_path.read_text().splitlines()[1:]:
            values = line.split("\t")
            if len(values) >= 6 and values[3] == "ok" and values[5] == str(expected_documents):
                completed.add(values[0])
        return completed

    def log(self, event: str, **fields: Any) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"{datetime.now(timezone.utc).isoformat()}\t{event}\t{details}\n"
        with self.events_path.open("a") as stream:
            stream.write(line)
        print(line, end="", flush=True)

    def append_summary(self, values: list[Any]) -> None:
        with self.summary_path.open("a") as stream:
            stream.write("\t".join(str(value) for value in values) + "\n")

    def append_profile_log(self, values: list[Any]) -> None:
        with self.profile_log_path.open("a") as stream:
            stream.write("\t".join(str(value) for value in values) + "\n")

    def profile_url(self, model: str, profile_name: str) -> str:
        encoded_model = urllib.parse.quote(model, safe="")
        return f"{self.omlx_api}/admin/api/models/{encoded_model}/profiles/{profile_name}"

    def model_mtp_compatibility(self, model: str) -> tuple[bool, str]:
        models = http_json(
            f"{self.omlx_api}/admin/api/models",
            headers={"Cookie": f"omlx_admin_session={self.admin_session}"},
            timeout=self.http_timeout,
        ).get("models", [])
        entry = next((item for item in models if item.get("id") == model), None)
        if entry is None:
            return False, "model not found in admin inventory"
        return bool(entry.get("mtp_compatible")), entry.get("mtp_compatibility_reason", "")

    def create_profile(self, model: str, profile_name: str, *, mtp_enabled: bool) -> str:
        encoded_model = urllib.parse.quote(model, safe="")
        settings = dict(self.profile_settings)
        if mtp_enabled:
            settings["mtp_enabled"] = True
        payload = {
            "name": profile_name,
            "display_name": f"OCR {profile_name}",
            "api_name": profile_name,
            "description": "Temporary OCR benchmark profile",
            "settings": settings,
            "expose_as_model": True,
        }
        http_json(
            f"{self.omlx_api}/admin/api/models/{encoded_model}/profiles",
            method="POST",
            payload=payload,
            headers={"Cookie": f"omlx_admin_session={self.admin_session}"},
            timeout=self.http_timeout,
        )
        return f"{model}:{profile_name}"

    def delete_profile(self, model: str, profile_name: str) -> bool:
        try:
            http_json(
                self.profile_url(model, profile_name),
                method="DELETE",
                headers={"Cookie": f"omlx_admin_session={self.admin_session}"},
                timeout=self.http_timeout,
            )
            return True
        except ApiError as error:
            self.log("PROFILE_DELETE_ERROR", model=model, profile=profile_name, error=str(error).replace("\t", " "))
            return False

    def profile_is_exposed(self, exposed_model: str) -> bool:
        models = http_json(
            f"{self.omlx_api}/v1/models",
            headers={"Authorization": f"Bearer {self.omlx_api_key}"},
            timeout=self.http_timeout,
        )
        return any(model.get("id") == exposed_model for model in models.get("data", []))

    def cleanup(self) -> None:
        pending = list(self.pending_cleanup)
        if self.current_profile is not None:
            pending.append(self.current_profile)
            self.current_profile = None
        for model, profile_name in pending:
            if self.delete_profile(model, profile_name):
                self.log("PROFILE_CLEANUP_DONE", model=model, profile=profile_name)
            else:
                self.log("PROFILE_CLEANUP_FAILED", model=model, profile=profile_name)

    def run_target(
        self,
        *,
        profile: str,
        model: str,
        kind: str,
        endpoint: str,
        api_key: str,
        exposed_model: str,
        profile_name: str,
    ) -> None:
        target = self.run_root / profile / "output"
        log_path = self.run_root / profile / "ocr.log"
        target.mkdir(parents=True, exist_ok=True)
        for output_path in target.glob("*.md"):
            output_path.unlink()
        self.log("EXTRACTION_START", profile=profile, model=model, kind=kind, exposed=exposed_model)
        started = time.monotonic()
        source_documents = sorted(self.source_dir.glob("*.pdf"))
        failed_documents = 0
        with log_path.open("w") as log_stream:
            with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                for document in source_documents:
                    try:
                        options = ocr.RecognizeOptions(
                            engine="vision",
                            vision_api_url=f"{endpoint.rstrip('/')}/v1",
                            vision_api_key=api_key,
                            vision_model=exposed_model,
                            timeout=float(self.ocr_timeout),
                            verbose=True,
                        )
                        pages = ocr.recognize(document, options)
                        output_path = target / f"{document.stem}.md"
                        output_path.write_text(
                            ocr.to_markdown(pages, document.name),
                            encoding="utf-8",
                        )
                    except Exception as error:
                        failed_documents += 1
                        log_stream.write(
                            f"OCR_FAILED\t{document.name}\t{safe_error(error, api_key)}\n"
                        )
                    log_stream.flush()
        elapsed = int(time.monotonic() - started)
        documents = count_outputs(target)
        result_name = "ok" if failed_documents == 0 and documents == len(source_documents) else "failed"
        self.log("EXTRACTION_DONE", profile=profile, model=model, result=result_name, elapsed=elapsed, documents=documents)
        cleanup_result = "n/a"
        if kind == "omlx":
            self.log("PROFILE_DELETE_START", model=model, profile=profile_name)
            if self.delete_profile(model, profile_name):
                cleanup_result = "deleted"
                self.log("PROFILE_DELETE_DONE", model=model, profile=profile_name)
            else:
                cleanup_result = "delete_failed"
                self.pending_cleanup.append((model, profile_name))
                self.log("PROFILE_DELETE_FAILED", model=model, profile=profile_name)
            self.append_profile_log([model, profile_name, exposed_model, "yes", cleanup_result == "deleted", cleanup_result])
        self.append_summary([profile, model, kind, result_name, elapsed, documents, profile_name, exposed_model, cleanup_result])

    def run_omlx(self, model: str, slug: str, profile_name: str) -> None:
        profile = f"vision-omlx-{slug}"
        exposed_model = f"{model}:{profile_name}"
        self.current_profile = (model, profile_name)
        try:
            mtp_enabled, mtp_reason = self.model_mtp_compatibility(model)
        except Exception as error:
            mtp_enabled = False
            mtp_reason = f"compatibility lookup failed: {error}"
        saved_mtp = self.config.setdefault("mtp_enabled_by_model", {})
        if model in saved_mtp:
            mtp_enabled = bool(saved_mtp[model])
            mtp_reason = "saved run configuration"
        else:
            saved_mtp[model] = mtp_enabled
            write_run_config(self.run_root, self.config)
        self.log("MTP_COMPATIBILITY", model=model, compatible=mtp_enabled, reason=mtp_reason or "none")
        self.log("PROFILE_CREATE_START", model=model, profile=profile_name, exposed=exposed_model, lightning_mtp=mtp_enabled)
        try:
            exposed_model = self.create_profile(model, profile_name, mtp_enabled=mtp_enabled)
            self.log("PROFILE_CREATE_DONE", model=model, profile=profile_name, exposed=exposed_model, lightning_mtp=mtp_enabled)
        except Exception as error:
            self.log("PROFILE_CREATE_FAILED", model=model, profile=profile_name, error=str(error).replace("\t", " "))
            self.append_summary([profile, model, "omlx", "profile_create_failed", 0, 0, profile_name, exposed_model, "create_failed"])
            self.current_profile = None
            return
        for _ in range(self.profile_poll_attempts):
            try:
                if self.profile_is_exposed(exposed_model):
                    break
            except ApiError:
                pass
            time.sleep(1)
        else:
            self.log("PROFILE_EXPOSE_FAILED", model=model, profile=profile_name, exposed=exposed_model)
            self.delete_profile(model, profile_name)
            self.append_summary([profile, model, "omlx", "profile_not_exposed", 0, 0, profile_name, exposed_model, "deleted_after_failure"])
            self.current_profile = None
            return
        self.run_target(
            profile=profile,
            model=model,
            kind="omlx",
            endpoint=self.omlx_api,
            api_key=self.omlx_api_key,
            exposed_model=exposed_model,
            profile_name=profile_name,
        )
        self.current_profile = None

    def run_openai(self, model: str, slug: str) -> None:
        self.run_target(
            profile=f"vision-openai-{slug}",
            model=model,
            kind="gpt",
            endpoint=self.openai_api,
            api_key=self.openai_key,
            exposed_model=model,
            profile_name="-",
        )

    def render_pages(self) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        for document in sorted(self.source_dir.glob("*.pdf")):
            stem = document.stem
            directory = self.run_root / "source-pages" / stem
            directory.mkdir(parents=True, exist_ok=True)
            if not any(directory.glob("*.png")):
                subprocess.run([self.pdftoppm_bin, "-r", "160", "-png", str(document), str(directory / "page")], check=True)
            images = sorted(directory.glob("page-*.png"), key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)))
            for image in images:
                pages.append({"document": document.name, "stem": stem, "page": int(re.search(r"(\d+)$", image.stem).group(1)), "image": image})
        return pages

    def judge(self) -> None:
        judge_log = self.run_root / "judge.log"
        errors_path = self.run_root / "judge-errors.tsv"
        judge_summary_tsv = self.run_root / "judge-summary.tsv"
        errors_path.write_text("profile\tdocument\tpage\terror\n")
        pages = self.render_pages()
        expected_documents = len({item["document"] for item in pages})
        rows = successful_judge_rows(self.summary_path, expected_documents)
        aggregate: dict[str, Any] = {}
        with judge_log.open("a") as log_stream:
            for row in rows:
                aggregate[row["profile"]] = {"model": row["model"], "kind": row["kind"], "profile_name": row["profile_name"], "exposed_model": row["exposed"], "pages_expected": len(pages), "pages_judged": 0, "totals": {field: 0 for field in ("transcription", "numbers_dates", "layout", "hallucination", "overall")}}
                for item in pages:
                    result_path = self.run_root / "judge" / row["profile"] / item["stem"] / f"page-{item['page']}.json"
                    try:
                        if result_path.exists():
                            judged = json.loads(result_path.read_text())["judged"]
                        else:
                            candidate_path = self.run_root / row["profile"] / "output" / f"{item['stem']}.md"
                            candidate = page_map(candidate_path.read_text()).get(item["page"], "[missing]")
                            prompt = "Evaluate OCR fidelity only. Compare source image and candidate transcription. Return JSON only: {\"scores\":{\"transcription\":0,\"numbers_dates\":0,\"layout\":0,\"hallucination\":0,\"overall\":0},\"evidence\":[\"...\"]}. Score 0-100; overall conservative. Do not reproduce personal medical data.\n\nCandidate:\n" + candidate
                            image = base64.b64encode(item["image"].read_bytes()).decode()
                            body = {**self.judge_settings, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}]}
                            last_error: Exception | None = None
                            for attempt in range(1, self.judge_attempts + 1):
                                try:
                                    response = http_json(
                                        f"{self.openai_api}/v1/chat/completions",
                                        method="POST",
                                        payload=body,
                                        headers={"Authorization": f"Bearer {self.openai_key}"},
                                        timeout=self.http_timeout,
                                    )
                                    judged = parse_judge_json(response["choices"][0]["message"]["content"])
                                    break
                                except Exception as error:
                                    last_error = error
                                    time.sleep(attempt * self.judge_retry_delay)
                            else:
                                raise last_error or RuntimeError("judge failed")
                            result_path.parent.mkdir(parents=True, exist_ok=True)
                            result_path.write_text(json.dumps({"judged": judged}, ensure_ascii=False, indent=2))
                        aggregate[row["profile"]]["pages_judged"] += 1
                        for field in aggregate[row["profile"]]["totals"]:
                            aggregate[row["profile"]]["totals"][field] += float(judged.get("scores", {}).get(field, 0))
                        log_stream.write(f"{datetime.now(timezone.utc).isoformat()}\tPAGE_DONE\t{row['profile']}\t{item['document']}\t{item['page']}\t{judged.get('scores', {}).get('overall', '')}\n")
                        log_stream.flush()
                    except Exception as error:
                        message = str(error).replace("\t", " ").replace("\n", " ")
                        with errors_path.open("a") as error_stream:
                            error_stream.write(f"{row['profile']}\t{item['document']}\t{item['page']}\t{message}\n")
                        log_stream.write(f"{datetime.now(timezone.utc).isoformat()}\tPAGE_FAILED\t{row['profile']}\t{item['document']}\t{item['page']}\t{message}\n")
                        log_stream.flush()
                pages_judged = max(aggregate[row["profile"]]["pages_judged"], 1)
                aggregate[row["profile"]]["averages"] = {field: round(total / pages_judged, 2) for field, total in aggregate[row["profile"]]["totals"].items()}
        ranked = sorted(aggregate.items(), key=lambda item: item[1]["averages"]["overall"], reverse=True)
        (self.run_root / "judge-summary.json").write_text(
            json.dumps(
                {
                    "judge_model": self.judge_settings["model"],
                    "judge_reasoning_effort": self.judge_settings.get("reasoning_effort"),
                    "aggregate": aggregate,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines = ["rank\tprofile\tmodel\tkind\tpages_expected\tpages_judged\ttranscription\tnumbers_dates\tlayout\thallucination\toverall"]
        for rank, (profile, value) in enumerate(ranked, 1):
            averages = value["averages"]
            lines.append("\t".join(map(str, [rank, profile, value["model"], value["kind"], value["pages_expected"], value["pages_judged"], averages["transcription"], averages["numbers_dates"], averages["layout"], averages["hallucination"], averages["overall"]])))
        judge_summary_tsv.write_text("\n".join(lines) + "\n")

    def judge_is_complete(self) -> bool:
        judge_summary = self.run_root / "judge-summary.tsv"
        if not judge_summary.exists():
            return False
        rows = judge_summary.read_text().splitlines()[1:]
        if not rows:
            return False
        expected_pages = len(self.render_pages())
        return all(
            len(values := line.split("\t")) >= 6
            and values[4] == str(expected_pages)
            and values[5] == str(expected_pages)
            for line in rows
        )

    def print_results(self) -> None:
        latest: dict[str, list[str]] = {}
        for line in self.summary_path.read_text().splitlines()[1:]:
            values = line.split("\t")
            if len(values) >= 9:
                latest[values[0]] = values

        print("\n=== Extraction results ===")
        print("model\tkind\tresult\telapsed_seconds\toutput_documents\tcleanup")
        for values in latest.values():
            print("\t".join((values[1], values[2], values[3], values[4], values[5], values[8])))

        judge_summary = self.run_root / "judge-summary.tsv"
        if not judge_summary.exists():
            return
        print("\n=== Judge results ===")
        print(judge_summary.read_text().rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON benchmark configuration for a new run")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume an existing run root")
    parser.add_argument("--only-model", help="Run only this exact model ID")
    parser.add_argument("--page", action="append", help="Page selector DOCUMENT.pdf:PAGE; repeatable")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    if args.run_root and args.resume:
        parser.error("use either --run-root or --resume, not both")
    resume = args.resume is not None
    if resume and args.config:
        parser.error("--config cannot be used with --resume")
    if resume and (args.page or args.only_model or args.skip_judge):
        parser.error("--resume uses the saved run configuration; do not pass --page, --only-model, or --skip-judge")

    if resume:
        run_root = args.resume.resolve()
        config_path = run_root / "run-config.json"
        if config_path.exists():
            try:
                config = normalize_config(json.loads(config_path.read_text()), run_root)
            except (ValueError, json.JSONDecodeError) as error:
                parser.error(f"invalid run configuration: {error}")
        else:
            parser.error(
                f"run configuration not found: {config_path}; "
                "legacy runs are not supported"
            )
        source_dir = Path(config["source_dir"])
        targets = [tuple(target) for target in config["targets"]]
        skip_judge = bool(config["skip_judge"])
    else:
        if not args.config:
            parser.error("--config is required for a new run")
        try:
            config = load_config(args.config)
        except ValueError as error:
            parser.error(str(error))
        run_root = (args.run_root or Path(config["run_root"])).resolve()
        input_source_dir = Path(config["source_dir"])
        targets = [tuple(target) for target in config["targets"]]
        if args.only_model:
            targets = [target for target in targets if target[1] == args.only_model]
        if not targets:
            raise SystemExit(f"Model not found in configured targets: {args.only_model}")
        source_dir = run_root / "target-pages" if args.page else input_source_dir
        config["source_dir"] = str(source_dir)
        config["input_source_dir"] = str(input_source_dir)
        config["run_root"] = str(run_root)
        config["targets"] = [list(target) for target in targets]
        config["only_model"] = args.only_model
        config["page_selectors"] = args.page
        config["skip_judge"] = args.skip_judge
        skip_judge = args.skip_judge

    benchmark = Benchmark(run_root, skip_judge, resume=resume, source_dir=source_dir, config=config)

    if not resume:
        write_run_config(run_root, config)

    if not resume and args.page:
        source_dir.mkdir(parents=True, exist_ok=True)
        for selector in args.page:
            try:
                document_name, page_text = selector.rsplit(":", 1)
                page_number = int(page_text)
            except ValueError as error:
                raise SystemExit(f"Invalid --page selector: {selector!r}; use DOCUMENT.pdf:PAGE") from error
            source = input_source_dir / document_name
            if not source.is_file() or page_number < 1:
                raise SystemExit(f"Invalid source page: {selector!r}")
            stem = f"{source.stem}-page-{page_number}"
            destination = source_dir / f"{stem}.pdf"
            generated = source_dir / f"{stem}.%d.pdf"
            subprocess.run([benchmark.pdfseparate_bin, "-f", str(page_number), "-l", str(page_number), str(source), str(generated)], check=True)
            generated_page = source_dir / f"{stem}.{page_number}.pdf"
            generated_page.replace(destination)

    cleanup_done = False

    def handle_signal(signum: int, _frame: Any) -> None:
        benchmark.log("INTERRUPTED", signal=signum)
        benchmark.cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        benchmark.log("BENCHMARK_START", run_root=run_root)
        profile_name = f"ocr-bench-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        completed = benchmark.completed_profiles()
        for kind, model, slug in targets:
            profile = f"vision-{'omlx' if kind == 'omlx' else 'openai'}-{slug}"
            if profile in completed:
                benchmark.log("TARGET_SKIPPED", profile=profile, reason="already_completed")
                continue
            if kind == "omlx":
                benchmark.run_omlx(model, slug, profile_name)
            else:
                benchmark.run_openai(model, slug)
        benchmark.log("EXTRACTION_PHASE_DONE", summary=benchmark.summary_path)
        if skip_judge:
            benchmark.log("JUDGE_SKIPPED")
        elif benchmark.judge_is_complete():
            benchmark.log("JUDGE_SKIPPED", reason="already_completed")
        else:
            benchmark.log("JUDGE_START", model=benchmark.judge_settings["model"], reasoning_effort=benchmark.judge_settings["reasoning_effort"])
            benchmark.judge()
            benchmark.log("JUDGE_DONE", summary=run_root / "judge-summary.tsv")
        benchmark.print_results()
        benchmark.log("BENCHMARK_DONE", run_root=run_root)
    finally:
        if not cleanup_done:
            benchmark.cleanup()
            cleanup_done = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
