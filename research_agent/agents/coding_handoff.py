from __future__ import annotations

import json
from typing import Any

from ..paths import trial_dir
from ..store import read_text, write_text
from .memory import log_decision
from .patch_validator import validate_patch_plan


def prepare_coding_handoff(
    competition: str,
    trial_id: str,
    *,
    user_approved: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    plan_path = out_dir / "code_patch_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing code patch plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validation = _load_or_validate_patch_plan(competition, trial_id, user_approved=user_approved)
    status = "ready" if validation["status"] == "ready" else "blocked"
    next_action = "send-to-coding-agent" if status == "ready" else "revise-patch-plan"
    target_files = plan.get("target_files", [])
    handoff = {
        "schema_version": "1.0",
        "request_id": f"{competition}:{trial_id}:coding",
        "competition": competition,
        "trial_id": trial_id,
        "handoff_type": "coding_agent_request",
        "status": status,
        "objective": "Implement the validated code patch plan without expanding its approved scope.",
        "strategy": plan.get("strategy"),
        "pipeline_axis": plan.get("pipeline_axis"),
        "context_files": _context_files(competition, trial_id),
        "target_files": target_files,
        "allowed_write_files": target_files,
        "create_files": plan.get("create_files", []),
        "forbidden_paths": [
            "data/",
            "submissions/",
            f"experiments/{competition}/{trial_id}/submission.csv",
            f"experiments/{competition}/{trial_id}/metrics.json",
        ],
        "config_changes": plan.get("config_changes", {}),
        "implementation_steps": plan.get("implementation_steps", []),
        "validation_commands": plan.get("validation_commands", []),
        "execution_constraints": {
            "do_not_run_training": True,
            "do_not_submit": True,
            "do_not_change_protected_axes": True,
            "do_not_write_outside_allowed_files": True,
        },
        "required_output": {
            "json_file": f"experiments/{competition}/{trial_id}/coding_result.json",
            "markdown_file": f"experiments/{competition}/{trial_id}/coding_result.md",
            "required_fields": ["status", "summary", "changed_files", "validation_results", "blocking_issues"],
            "status_values": ["completed", "blocked", "failed"],
            "next_action": "validate-code-change",
        },
        "requires_user_approval": bool(plan.get("requires_user_approval")),
        "user_approved": user_approved,
        "protected_axes": plan.get("protected_axes", []),
        "blocking_issues": validation["issues"],
        "patch_validation_status": validation["status"],
        "next_action": next_action,
    }
    write_text(out_dir / "coding_handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    if status == "ready":
        write_text(out_dir / "coding_agent_request.md", render_coding_agent_request(handoff, out_dir))
    log_decision(
        competition,
        trial_id,
        decision_type="coding_handoff",
        decision=status,
        reason="Coding handoff prepared from a validated patch plan.",
        evidence={
            "schema_version": handoff["schema_version"],
            "pipeline_axis": handoff["pipeline_axis"],
            "target_files": handoff["target_files"],
            "blocking_issues": handoff["blocking_issues"],
            "validation_commands": handoff["validation_commands"],
        },
        user_input_used=user_approved,
        next_action=next_action,
    )
    return handoff


def _load_or_validate_patch_plan(competition: str, trial_id: str, *, user_approved: bool) -> dict[str, Any]:
    validation_path = trial_dir(competition, trial_id) / "patch_validation.json"
    if validation_path.exists() and not user_approved:
        return json.loads(validation_path.read_text(encoding="utf-8"))
    return validate_patch_plan(competition, trial_id, user_approved=user_approved)


def _context_files(competition: str, trial_id: str) -> list[str]:
    out_dir = trial_dir(competition, trial_id)
    candidates = [
        "code_patch_plan.json",
        "patch_validation.json",
        "config.yaml",
        "next_experiment.md",
        "model_candidates.json",
    ]
    return [f"experiments/{competition}/{trial_id}/{name}" for name in candidates if (out_dir / name).exists()]


def render_coding_agent_request(handoff: dict[str, Any], out_dir) -> str:
    next_experiment = read_text(out_dir / "next_experiment.md", default="").strip()
    lines = [
        f"# {handoff['trial_id']} Coding Agent Request",
        "",
        "## Objective",
        "",
        handoff["objective"],
        "",
        f"- competition: {handoff['competition']}",
        f"- trial_id: {handoff['trial_id']}",
        f"- request_id: {handoff['request_id']}",
        f"- schema_version: {handoff['schema_version']}",
        f"- strategy: {handoff.get('strategy')}",
        f"- pipeline_axis: {handoff.get('pipeline_axis')}",
        "",
        "## Input Context Files",
        "",
    ]
    lines.extend(f"- {item}" for item in handoff["context_files"] or ["None"])
    lines.extend(
        [
            "",
            "## Allowed Write Files",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in handoff["allowed_write_files"] or ["None"])
    lines.extend(["", "## Declared New Files", ""])
    lines.extend(f"- {item}" for item in handoff["create_files"] or ["None"])
    lines.extend(
        [
            "",
        "## Target Files",
        "",
        ]
    )
    lines.extend(f"- {item}" for item in handoff["target_files"] or ["None"])
    lines.extend(["", "## Required Changes", ""])
    config_changes = handoff.get("config_changes") or {}
    if config_changes:
        lines.extend(f"- `{key}` -> `{value}`" for key, value in config_changes.items())
    else:
        lines.append("- Follow the implementation steps from the patch plan.")
    lines.extend(["", "## Implementation Steps", ""])
    lines.extend(f"- {item}" for item in handoff["implementation_steps"] or ["No explicit implementation steps."])
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Preserve the validated config intent.",
            "- Do not edit submission artifacts.",
            "- Do not change protected axes unless a new approved patch plan says so.",
            "- Keep the change scoped to the listed target files unless the implementation cannot work otherwise.",
            "- Do not run training or submit to Kaggle from this coding step.",
            "",
            "## Forbidden Paths",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in handoff["forbidden_paths"])
    lines.extend(
        [
            "",
            "## Execution Constraints",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in handoff["execution_constraints"].items())
    lines.extend(
        [
            "",
            "## Validation Commands",
            "",
        ]
    )
    lines.extend(f"```powershell\n{item}\n```" for item in handoff["validation_commands"] or ["No validation command provided."])
    required_output = handoff["required_output"]
    lines.extend(
        [
            "",
            "## Required Result Contract",
            "",
            f"- json_file: {required_output['json_file']}",
            f"- markdown_file: {required_output['markdown_file']}",
            f"- status_values: {', '.join(required_output['status_values'])}",
            f"- next_action: {required_output['next_action']}",
            "- required_fields:",
        ]
    )
    lines.extend(f"  - {item}" for item in required_output["required_fields"])
    if next_experiment:
        lines.extend(["", "## Next Experiment Context", "", next_experiment])
    lines.append("")
    return "\n".join(lines)
