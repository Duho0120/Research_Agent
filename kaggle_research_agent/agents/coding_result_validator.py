from __future__ import annotations

import json
from typing import Any

from ..paths import trial_dir
from ..store import write_text
from .memory import log_decision


def validate_coding_result(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff_path = out_dir / "coding_handoff.json"
    result_path = out_dir / "coding_result.json"
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing coding handoff: {handoff_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing coding result: {result_path}")

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    coding_result = json.loads(result_path.read_text(encoding="utf-8"))
    issues = _validate_schema(coding_result, handoff)
    issues.extend(_validate_changed_files(coding_result, handoff))
    issues.extend(_validate_file_updates(coding_result, handoff))

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "accepted" if not issues and coding_result.get("status") == "completed" else "blocked",
        "issues": issues,
        "coding_result_status": coding_result.get("status"),
        "changed_files": coding_result.get("changed_files", []),
        "allowed_write_files": handoff.get("allowed_write_files", []),
        "create_files": handoff.get("create_files", []),
        "forbidden_paths": handoff.get("forbidden_paths", []),
        "next_action": "run-validation-commands" if not issues and coding_result.get("status") == "completed" else "revise-code-result",
    }
    write_text(out_dir / "coding_result_validation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "coding_result_validation.md", render_coding_result_validation(result))
    log_decision(
        competition,
        trial_id,
        decision_type="coding_result_validation",
        decision=result["status"],
        reason="Coding result was checked against the handoff contract before downstream execution.",
        evidence={
            "issues": result["issues"],
            "coding_result_status": result["coding_result_status"],
            "changed_files": result["changed_files"],
        },
        next_action=result["next_action"],
    )
    return result


def create_dry_run_coding_result(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff_path = out_dir / "coding_handoff.json"
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing coding handoff: {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "1.0",
        "request_id": handoff.get("request_id"),
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked",
        "summary": "Dry run only. No coding agent or API call was executed.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": ["dry_run_no_code_writer"],
        "next_action": "connect-code-writer-api",
    }
    write_text(out_dir / "coding_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "coding_result.md", render_coding_result(result))
    log_decision(
        competition,
        trial_id,
        decision_type="code_writer_dry_run",
        decision="blocked",
        reason="Dry-run coding adapter produced a placeholder result without editing files.",
        evidence={"request_id": result["request_id"], "blocking_issues": result["blocking_issues"]},
        next_action="connect-code-writer-api",
    )
    return result


def render_coding_result_validation(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Coding Result Validation",
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


def render_coding_result(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Coding Result",
        "",
        f"- status: {result['status']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Summary",
        "",
        result["summary"],
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- {item}" for item in result["changed_files"] or ["None"])
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(f"- {item}" for item in result["blocking_issues"] or ["None"])
    lines.append("")
    return "\n".join(lines)


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
    allowed = set(handoff.get("allowed_write_files", [])) | set(handoff.get("create_files", []))
    forbidden = handoff.get("forbidden_paths", [])
    issues: list[str] = []
    for item in changed_files:
        if item not in allowed:
            issues.append(f"changed_file_not_allowed:{item}")
        if _matches_forbidden_path(item, forbidden):
            issues.append(f"forbidden_path_touched:{item}")
    return issues


def _validate_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not updates:
        return []
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    allowed = set(handoff.get("allowed_write_files", [])) | set(handoff.get("create_files", []))
    forbidden = handoff.get("forbidden_paths", [])
    issues: list[str] = []
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        if not path:
            issues.append("invalid_file_update")
            continue
        if path not in allowed:
            issues.append(f"file_update_not_allowed:{path}")
        if _matches_forbidden_path(path, forbidden):
            issues.append(f"forbidden_path_touched:{path}")
    return issues


def _matches_forbidden_path(path: str, forbidden_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for forbidden in forbidden_paths:
        marker = str(forbidden).replace("\\", "/")
        if marker.endswith("/") and normalized.startswith(marker):
            return True
        if normalized == marker:
            return True
    return False
