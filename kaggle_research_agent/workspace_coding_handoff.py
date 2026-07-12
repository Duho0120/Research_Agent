from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .execution_profile import load_execution_profile, validate_execution_profile
from .agents.memory import log_decision
from .paths import trial_dir
from .store import read_text, write_text


def prepare_workspace_coding_handoff(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    next_experiment_path = out_dir / "next_experiment.md"
    continuation_path = out_dir / "continuation_context.json"
    blocking_issues: list[str] = []

    next_experiment = read_text(next_experiment_path, default="").strip()
    if not next_experiment:
        blocking_issues.append("missing_next_experiment")
    continuation = _load_json_object(continuation_path)
    if continuation is None:
        blocking_issues.append("missing_or_invalid_continuation_context")
        continuation = {}
    if continuation.get("continuation_mode") == "must_wait":
        blocking_issues.append("continuation_requires_user_feedback")

    validation = validate_execution_profile(competition)
    profile: dict[str, Any] = {}
    if validation["status"] != "ready":
        blocking_issues.append("execution_profile_not_ready")
    else:
        profile = load_execution_profile(competition)

    status = "ready" if not blocking_issues else "blocked"
    handoff = _build_handoff(
        competition,
        trial_id,
        status=status,
        blocking_issues=blocking_issues,
        profile=profile,
        profile_validation=validation,
        continuation=continuation,
    )
    write_text(out_dir / "workspace_coding_handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    if status == "ready":
        write_text(out_dir / "workspace_coding_agent_request.md", render_workspace_coding_request(handoff, next_experiment))
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_coding_handoff",
        decision=status,
        reason="Workspace coding handoff prepared from next-experiment and Execution Profile scope.",
        evidence={
            "continuation_mode": handoff.get("continuation_mode"),
            "pending_human_review": handoff.get("pending_human_review"),
            "allowed_write_paths": handoff.get("allowed_write_paths", []),
            "blocking_issues": blocking_issues,
        },
        next_action=handoff["next_action"],
    )
    return handoff


def _build_handoff(
    competition: str,
    trial_id: str,
    *,
    status: str,
    blocking_issues: list[str],
    profile: dict[str, Any],
    profile_validation: dict[str, Any],
    continuation: dict[str, Any],
) -> dict[str, Any]:
    artifacts = profile.get("artifacts", {}) if profile else {}
    scope = profile.get("write_scope", {}) if profile else {}
    allowed = list(scope.get("allowed", [])) if isinstance(scope, dict) else []
    forbidden = list(scope.get("forbidden", [])) if isinstance(scope, dict) else []
    forbidden = _unique([*forbidden, *artifacts.get("metrics", []), *artifacts.get("submission", [])])
    validation_commands = []
    commands = profile.get("commands", {}) if profile else {}
    if isinstance(commands, dict):
        validation_commands = list(commands.get("test", []))
    next_action = "send-to-workspace-coding-agent" if status == "ready" else "resolve-workspace-handoff-blockers"
    return {
        "schema_version": "1.0",
        "request_id": f"{competition}:{trial_id}:workspace-coding",
        "competition": competition,
        "trial_id": trial_id,
        "handoff_type": "workspace_coding_agent_request",
        "status": status,
        "objective": "Implement the next workspace experiment within the Execution Profile write scope.",
        "project_root": profile.get("project_root") if profile else None,
        "platform": profile.get("platform") if profile else None,
        "continuation_mode": continuation.get("continuation_mode"),
        "pending_human_review": bool(continuation.get("pending_human_review")),
        "review_source_trial": continuation.get("review_source_trial"),
        "context_files": _context_files(competition, trial_id),
        "allowed_write_paths": allowed,
        "forbidden_paths": forbidden,
        "validation_commands": validation_commands,
        "execution_constraints": {
            "do_not_run_training": True,
            "do_not_submit": True,
            "do_not_edit_data_or_outputs": True,
            "do_not_write_outside_allowed_paths": True,
            "use_project_root_as_cwd": True,
        },
        "required_output": {
            "json_file": f"experiments/{competition}/{trial_id}/workspace_coding_result.json",
            "markdown_file": f"experiments/{competition}/{trial_id}/workspace_coding_result.md",
            "required_fields": ["status", "summary", "changed_files", "validation_results", "blocking_issues"],
            "status_values": ["completed", "blocked", "failed"],
            "next_action": "validate-workspace-code-change",
        },
        "profile_validation_status": profile_validation["status"],
        "profile_validation_issues": profile_validation.get("issues", []),
        "blocking_issues": _unique(blocking_issues),
        "next_action": next_action,
    }


def render_workspace_coding_request(handoff: dict[str, Any], next_experiment: str) -> str:
    lines = [
        f"# {handoff['trial_id']} Workspace Coding Agent Request",
        "",
        "## Objective",
        "",
        handoff["objective"],
        "",
        f"- competition: {handoff['competition']}",
        f"- trial_id: {handoff['trial_id']}",
        f"- request_id: {handoff['request_id']}",
        f"- project_root: {handoff['project_root']}",
        f"- continuation_mode: {handoff['continuation_mode']}",
        f"- pending_human_review: {handoff['pending_human_review']}",
        "",
        "## Input Context Files",
        "",
    ]
    lines.extend(f"- {item}" for item in handoff["context_files"] or ["None"])
    lines.extend(["", "## Allowed External Write Paths", ""])
    lines.extend(f"- {item}" for item in handoff["allowed_write_paths"] or ["None"])
    lines.extend(["", "## Forbidden External Paths", ""])
    lines.extend(f"- {item}" for item in handoff["forbidden_paths"] or ["None"])
    lines.extend(
        [
            "",
            "## Execution Constraints",
            "",
            "- Do not run training.",
            "- Do not submit to any competition platform.",
            "- Do not edit data, metrics, submission, or output artifacts.",
            "- Do not write outside the allowed external write paths.",
            "",
            "## Validation Commands",
            "",
        ]
    )
    lines.extend(f"```powershell\n{command}\n```" for command in handoff["validation_commands"] or ["No validation command declared."])
    required = handoff["required_output"]
    lines.extend(
        [
            "",
            "## Required Result Contract",
            "",
            f"- json_file: {required['json_file']}",
            f"- markdown_file: {required['markdown_file']}",
            f"- status_values: {', '.join(required['status_values'])}",
            f"- next_action: {required['next_action']}",
            "- required_fields:",
        ]
    )
    lines.extend(f"  - {field}" for field in required["required_fields"])
    if next_experiment:
        lines.extend(["", "## Next Experiment", "", next_experiment])
    lines.append("")
    return "\n".join(lines)


def _context_files(competition: str, trial_id: str) -> list[str]:
    out_dir = trial_dir(competition, trial_id)
    candidates = [
        "next_experiment.md",
        "continuation_context.json",
        "continuation_context.md",
    ]
    return [f"experiments/{competition}/{trial_id}/{name}" for name in candidates if (out_dir / name).exists()]


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
