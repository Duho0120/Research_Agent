from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .. import paths
from ..paths import trial_dir
from ..store import read_text, write_text
from .coding_result_validator import render_coding_result, validate_coding_result
from .memory import log_decision
from .policy_gate import should_call_llm


class CodeWriterClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class OpenAIResponsesClient:
    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.openai.com/v1/responses"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {error.code}: {detail}") from error


class FileResponseClient:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))


def run_code_writer(
    competition: str,
    trial_id: str,
    *,
    client: CodeWriterClient | None = None,
    model: str = "gpt-5",
    allow_api: bool = False,
    trial_llm_calls: int = 0,
    strategy_calls_today: int = 0,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff_path = out_dir / "coding_handoff.json"
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing coding handoff: {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("status") != "ready":
        return _write_blocked_result(competition, trial_id, handoff, ["coding_handoff_not_ready"])

    token_decision = should_call_llm(
        "code_writing",
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
    )
    if token_decision["decision"] != "call_llm":
        return _write_blocked_result(competition, trial_id, handoff, ["token_policy_blocked"], token_decision)

    if client is None:
        if not allow_api:
            return _write_blocked_result(competition, trial_id, handoff, ["api_call_not_enabled"], token_decision)
        client = OpenAIResponsesClient()

    payload = build_code_writer_payload(handoff, model=model)
    write_text(out_dir / "coding_api_request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    try:
        raw_response = client.create_response(payload)
    except RuntimeError as error:
        return _write_blocked_result(competition, trial_id, handoff, [f"api_error:{error}"], token_decision)
    write_text(out_dir / "coding_api_response.json", json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n")

    coding_result = _extract_coding_result(raw_response)
    coding_result = _normalize_coding_result(coding_result, competition, trial_id, handoff)
    blocking_issues = list(coding_result.get("blocking_issues", []))
    blocking_issues.extend(_apply_file_updates(coding_result, handoff))
    if blocking_issues:
        coding_result["status"] = "blocked"
        coding_result["blocking_issues"] = blocking_issues
    _write_coding_result(out_dir, coding_result)
    validation = validate_coding_result(competition, trial_id)
    validation["blocking_issues"] = coding_result.get("blocking_issues", [])
    log_decision(
        competition,
        trial_id,
        decision_type="code_writer_api",
        decision=validation["status"],
        reason="Code writer adapter produced a coding_result and ran the result validator.",
        evidence={
            "model": model,
            "token_decision": token_decision,
            "changed_files": coding_result.get("changed_files", []),
            "validation_issues": validation.get("issues", []),
        },
        next_action=validation["next_action"],
    )
    return validation


def build_code_writer_payload(handoff: dict[str, Any], *, model: str) -> dict[str, Any]:
    request_text = _build_prompt(handoff)
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You are a careful code writer. Return only JSON that follows the requested schema.",
            },
            {"role": "user", "content": request_text},
        ],
    }


def _build_prompt(handoff: dict[str, Any]) -> str:
    context = _read_context(handoff.get("context_files", []))
    return "\n".join(
        [
            "Implement the validated Kaggle research code change.",
            "",
            "Return one JSON object with:",
            "- status: completed | blocked | failed",
            "- summary",
            "- changed_files",
            "- file_updates: list of {path, content}; paths must be allowed",
            "- validation_results",
            "- blocking_issues",
            "",
            f"Objective: {handoff.get('objective')}",
            f"Allowed write files: {handoff.get('allowed_write_files', [])}",
            f"Declared new files: {handoff.get('create_files', [])}",
            f"Forbidden paths: {handoff.get('forbidden_paths', [])}",
            f"Implementation steps: {handoff.get('implementation_steps', [])}",
            f"Validation commands: {handoff.get('validation_commands', [])}",
            "",
            "Context files:",
            context,
        ]
    )


def _read_context(files: list[str]) -> str:
    root = paths.project_root()
    chunks = []
    for item in files:
        path = root / item
        if path.exists() and path.is_file():
            text = read_text(path)
            chunks.append(f"## {item}\n{text[:4000]}")
    return "\n\n".join(chunks) if chunks else "No context file contents available."


def _extract_coding_result(raw_response: dict[str, Any]) -> dict[str, Any]:
    text = raw_response.get("output_text")
    if not text:
        text = _extract_output_text(raw_response)
    if not text:
        return {
            "status": "blocked",
            "summary": "The code writer response did not include output text.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["missing_output_text"],
        }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "summary": "The code writer response was not valid JSON.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["invalid_json_output"],
        }
    return parsed


def _extract_output_text(raw_response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in raw_response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part)


def _apply_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    allowed = set(handoff.get("allowed_write_files", [])) | set(handoff.get("create_files", []))
    forbidden = handoff.get("forbidden_paths", [])
    issues: list[str] = []
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        content = update.get("content") if isinstance(update, dict) else None
        if not path or not isinstance(content, str):
            issues.append("invalid_file_update")
            continue
        if path not in allowed or _matches_forbidden_path(path, forbidden):
            issues.append(f"file_update_not_allowed:{path}")
            continue
        target = _resolve_allowed_path(path)
        write_text(target, content)
    return issues


def _write_blocked_result(
    competition: str,
    trial_id: str,
    handoff: dict[str, Any],
    issues: list[str],
    token_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    result = {
        "schema_version": "1.0",
        "request_id": handoff.get("request_id"),
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked",
        "summary": "Code writer adapter did not execute a code change.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": issues,
        "token_decision": token_decision or {},
        "next_action": "revise-code-writer-request",
    }
    _write_coding_result(out_dir, result)
    log_decision(
        competition,
        trial_id,
        decision_type="code_writer_api",
        decision="blocked",
        reason="Code writer adapter blocked before applying file updates.",
        evidence={"blocking_issues": issues, "token_decision": token_decision or {}},
        next_action="revise-code-writer-request",
    )
    return result


def _write_coding_result(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "coding_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "coding_result.md", render_coding_result(result))


def _normalize_coding_result(
    result: dict[str, Any],
    competition: str,
    trial_id: str,
    handoff: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
    normalized.setdefault("schema_version", "1.0")
    normalized.setdefault("request_id", handoff.get("request_id"))
    normalized.setdefault("competition", competition)
    normalized.setdefault("trial_id", trial_id)
    normalized.setdefault("status", "blocked")
    normalized.setdefault("summary", "")
    normalized.setdefault("changed_files", [])
    normalized.setdefault("validation_results", [])
    normalized.setdefault("blocking_issues", [])
    normalized.setdefault("next_action", "validate-code-change")
    return normalized


def _resolve_allowed_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = paths.project_root() / target
    return target


def _matches_forbidden_path(path: str, forbidden_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for forbidden in forbidden_paths:
        marker = str(forbidden).replace("\\", "/")
        if marker.endswith("/") and normalized.startswith(marker):
            return True
        if normalized == marker:
            return True
    return False
