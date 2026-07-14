from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from . import paths
from .agents.code_writer_adapter import CodeWriterClient, create_llm_client, provider_log_name
from .agents.memory import log_decision, log_token_usage
from .agents.policy_gate import should_call_llm
from .paths import trial_dir
from .store import read_text, write_text
from .trial_artifacts import trial_artifact_path


def run_workspace_code_writer(
    competition: str,
    trial_id: str,
    *,
    client: CodeWriterClient | None = None,
    model: str = "gpt-5",
    provider: str = "openai",
    allow_api: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff = _load_handoff(out_dir)
    if handoff.get("status") != "ready":
        return _write_blocked_result(competition, trial_id, handoff, ["workspace_coding_handoff_not_ready"])

    token_decision = should_call_llm(
        "code_writing",
        competition=competition,
        trial_id=trial_id,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
    )
    if token_decision["decision"] != "call_llm":
        return _write_blocked_result(competition, trial_id, handoff, ["token_policy_blocked"], token_decision)

    if client is None:
        if not allow_api:
            return _write_blocked_result(competition, trial_id, handoff, ["api_call_not_enabled"], token_decision)
        client = create_llm_client(provider)

    payload = build_workspace_code_writer_payload(handoff, model=model)
    write_text(out_dir / "workspace_coding_api_request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    try:
        raw_response = client.create_response(payload)
    except RuntimeError as error:
        return _write_blocked_result(competition, trial_id, handoff, [f"api_error:{error}"], token_decision)
    write_text(out_dir / "workspace_coding_api_response.json", json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n")
    token_usage = _record_response_token_usage(
        competition,
        trial_id,
        raw_response,
        provider=provider_log_name(provider),
        model=model,
    )

    coding_result = _normalize_coding_result(_extract_coding_result(raw_response), competition, trial_id, handoff)
    blocking_issues = list(coding_result.get("blocking_issues", []))
    blocking_issues.extend(_apply_file_updates(coding_result, handoff))
    if blocking_issues:
        coding_result["status"] = "blocked"
        coding_result["blocking_issues"] = _unique(blocking_issues)
    _write_coding_result(out_dir, coding_result)
    validation = validate_workspace_coding_result(competition, trial_id)
    validation["blocking_issues"] = coding_result.get("blocking_issues", [])
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_code_writer_api",
        decision=validation["status"],
        reason="Workspace code writer produced a coding result and ran scope validation.",
        evidence={
            "model": model,
            "token_decision": token_decision,
            "token_usage": token_usage,
            "changed_files": coding_result.get("changed_files", []),
            "validation_issues": validation.get("issues", []),
        },
        next_action=validation["next_action"],
    )
    return validation


def validate_workspace_coding_result(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff = _load_handoff(out_dir)
    result_path = trial_artifact_path(out_dir, "workspace_coding_result.json")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing workspace coding result: {result_path}")
    coding_result = json.loads(result_path.read_text(encoding="utf-8"))
    if "validation_results" in coding_result:
        coding_result["validation_results"] = _normalize_validation_results(coding_result.get("validation_results"))
    issues = _validate_schema(coding_result, handoff)
    issues.extend(_validate_changed_files(coding_result, handoff))
    issues.extend(_validate_file_updates(coding_result, handoff))
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "accepted" if not issues and coding_result.get("status") == "completed" else "blocked",
        "issues": _unique(issues),
        "coding_result_status": coding_result.get("status"),
        "changed_files": coding_result.get("changed_files", []),
        "allowed_write_paths": handoff.get("allowed_write_paths", []),
        "forbidden_paths": handoff.get("forbidden_paths", []),
        "next_action": "run-workspace-validation-commands"
        if not issues and coding_result.get("status") == "completed"
        else "revise-workspace-code-result",
    }
    write_text(out_dir / "workspace_coding_result_validation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_coding_result_validation.md", render_workspace_coding_result_validation(result))
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_coding_result_validation",
        decision=result["status"],
        reason="Workspace coding result was checked against Execution Profile-derived scope.",
        evidence={
            "issues": result["issues"],
            "coding_result_status": result["coding_result_status"],
            "changed_files": result["changed_files"],
        },
        next_action=result["next_action"],
    )
    return result


def build_workspace_code_writer_payload(handoff: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You are a careful code writer. Return only JSON that follows the requested schema.",
            },
            {"role": "user", "content": _build_prompt(handoff)},
        ],
    }


def render_workspace_coding_result(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Workspace Coding Result",
        "",
        f"- status: {result['status']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Summary",
        "",
        result.get("summary", ""),
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("changed_files", []) or ["None"])
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(f"- {item}" for item in result.get("blocking_issues", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def render_workspace_coding_result_validation(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Workspace Coding Result Validation",
        "",
        f"- status: {result['status']}",
        f"- coding_result_status: {result.get('coding_result_status')}",
        f"- next_action: {result['next_action']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- {item}" for item in result.get("changed_files", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _build_prompt(handoff: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Implement the next workspace experiment within the declared project write scope.",
            "",
            "Return one JSON object with:",
            "- status: completed | blocked | failed",
            "- summary",
            "- changed_files: project-root-relative paths only",
            "- file_updates: list of {path, content}; paths must be project-root-relative and allowed",
            "- validation_results",
            "- blocking_issues",
            "",
            "Use the RAG context pack and retrieval manifest when present. Prefer concrete retrieved evidence over generic changes.",
            "Your summary should state the actual pipeline changes you implemented, not only a generic performance goal.",
            "",
            f"Objective: {handoff.get('objective')}",
            f"Project root: {handoff.get('project_root')}",
            f"Allowed external write paths: {handoff.get('allowed_write_paths', [])}",
            f"Forbidden external paths: {handoff.get('forbidden_paths', [])}",
            f"Validation commands: {handoff.get('validation_commands', [])}",
            "Artifact policy:",
            json.dumps(handoff.get("artifact_policy", {}), ensure_ascii=False, indent=2),
            "Do not persist trained model/checkpoint artifacts by default. If you persist one, record the allowed policy reason.",
            "",
            "Retrieval context metadata:",
            json.dumps(handoff.get("retrieval_context", {}), ensure_ascii=False, indent=2),
            "",
            "Context files:",
            _read_context(handoff.get("context_files", [])),
        ]
    )


def _read_context(files: list[str]) -> str:
    root = paths.project_root()
    chunks = []
    for item in files:
        path = root / item
        if path.exists() and path.is_file():
            chunks.append(f"## {item}\n{read_text(path)[:4000]}")
    return "\n\n".join(chunks) if chunks else "No context file contents available."


def _load_handoff(out_dir: Path) -> dict[str, Any]:
    handoff_path = trial_artifact_path(out_dir, "workspace_coding_handoff.json")
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing workspace coding handoff: {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(handoff, dict):
        raise ValueError(f"Workspace coding handoff must be a JSON object: {handoff_path}")
    return handoff


def _extract_coding_result(raw_response: dict[str, Any]) -> dict[str, Any]:
    text = raw_response.get("output_text") or _extract_output_text(raw_response)
    if not text:
        return {
            "status": "blocked",
            "summary": "The workspace code writer response did not include output text.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["missing_output_text"],
        }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "summary": "The workspace code writer response was not valid JSON.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["invalid_json_output"],
        }
    return parsed if isinstance(parsed, dict) else {
        "status": "blocked",
        "summary": "The workspace code writer response JSON was not an object.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": ["invalid_json_output"],
    }


def _extract_output_text(raw_response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in raw_response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part)


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
    normalized.setdefault("next_action", "validate-workspace-code-change")
    normalized["validation_results"] = _normalize_validation_results(normalized.get("validation_results"))
    return normalized


def _normalize_validation_results(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        commands = value.get("commands")
        if isinstance(commands, list):
            return commands
    return value


def _apply_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    issues: list[str] = []
    project_root = Path(str(handoff.get("project_root", "")))
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        content = update.get("content") if isinstance(update, dict) else None
        if not isinstance(path, str) or not isinstance(content, str):
            issues.append("invalid_file_update")
            continue
        path_issues = _path_scope_issues(path, handoff, update_label="file_update")
        if path_issues:
            issues.extend(path_issues)
            continue
        target = project_root / path.replace("\\", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    return issues


def _validate_schema(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    required = handoff.get("required_output", {}).get("required_fields", [])
    status_values = set(handoff.get("required_output", {}).get("status_values", ["completed", "blocked", "failed"]))
    issues = [f"missing_required_field:{field}" for field in required if field not in coding_result]
    status = coding_result.get("status")
    if status not in status_values:
        issues.append(f"invalid_status:{status}")
    if "changed_files" in coding_result and not isinstance(coding_result["changed_files"], list):
        issues.append("invalid_type:changed_files")
    if "validation_results" in coding_result and not isinstance(coding_result["validation_results"], list):
        issues.append("invalid_type:validation_results")
    if "blocking_issues" in coding_result and not isinstance(coding_result["blocking_issues"], list):
        issues.append("invalid_type:blocking_issues")
    if status in {"blocked", "failed"} and not coding_result.get("blocking_issues"):
        issues.append("missing_blocking_issue_for_incomplete_result")
    return issues


def _validate_changed_files(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    changed_files = coding_result.get("changed_files", [])
    if not isinstance(changed_files, list):
        return []
    issues: list[str] = []
    for item in changed_files:
        if not isinstance(item, str):
            issues.append("invalid_changed_file")
            continue
        issues.extend(_path_scope_issues(item, handoff, update_label="changed_file"))
    return issues


def _validate_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not updates:
        return []
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    issues: list[str] = []
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        if not isinstance(path, str):
            issues.append("invalid_file_update")
            continue
        issues.extend(_path_scope_issues(path, handoff, update_label="file_update"))
    return issues


def _path_scope_issues(path: str, handoff: dict[str, Any], *, update_label: str) -> list[str]:
    issues: list[str] = []
    if not _is_safe_relative_path(path):
        issues.append(f"{update_label}_path_not_safe:{path}")
        return issues
    if not _is_allowed_path(path, handoff.get("allowed_write_paths", [])):
        issues.append(f"{update_label}_not_allowed:{path}")
    if _matches_forbidden_path(path, handoff.get("forbidden_paths", [])):
        issues.append(f"forbidden_path_touched:{path}")
    return issues


def _is_safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized.rstrip("/"))
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts and not _looks_like_windows_absolute(normalized)


def _looks_like_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] == "/"


def _is_allowed_path(path: str, allowed_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for allowed in allowed_paths:
        marker = str(allowed).replace("\\", "/").rstrip("/")
        if str(allowed).replace("\\", "/").endswith("/"):
            if normalized.startswith(marker + "/"):
                return True
        elif normalized == marker:
            return True
    return False


def _matches_forbidden_path(path: str, forbidden_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for forbidden in forbidden_paths:
        marker_raw = str(forbidden).replace("\\", "/")
        marker = marker_raw.rstrip("/")
        if marker_raw.endswith("/") and normalized.startswith(marker + "/"):
            return True
        if normalized == marker:
            return True
    return False


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
        "summary": "Workspace code writer did not execute a code change.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": issues,
        "token_decision": token_decision or {},
        "next_action": "revise-workspace-code-writer-request",
    }
    _write_coding_result(out_dir, result)
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_code_writer_api",
        decision="blocked",
        reason="Workspace code writer blocked before applying file updates.",
        evidence={"blocking_issues": issues, "token_decision": token_decision or {}},
        next_action="revise-workspace-code-writer-request",
    )
    return result


def _write_coding_result(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "workspace_coding_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_coding_result.md", render_workspace_coding_result(result))


def _record_response_token_usage(
    competition: str,
    trial_id: str,
    raw_response: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return None
    return log_token_usage(
        competition,
        trial_id,
        provider=provider,
        model=model,
        call_type="workspace_code_writing",
        usage=usage,
        request_id=raw_response.get("id"),
    )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
