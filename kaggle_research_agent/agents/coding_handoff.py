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
    handoff = {
        "competition": competition,
        "trial_id": trial_id,
        "handoff_type": "coding_agent_request",
        "status": status,
        "strategy": plan.get("strategy"),
        "pipeline_axis": plan.get("pipeline_axis"),
        "target_files": plan.get("target_files", []),
        "config_changes": plan.get("config_changes", {}),
        "implementation_steps": plan.get("implementation_steps", []),
        "validation_commands": plan.get("validation_commands", []),
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


def render_coding_agent_request(handoff: dict[str, Any], out_dir) -> str:
    next_experiment = read_text(out_dir / "next_experiment.md", default="").strip()
    lines = [
        f"# {handoff['trial_id']} Coding Agent Request",
        "",
        "## Objective",
        "",
        f"- competition: {handoff['competition']}",
        f"- trial_id: {handoff['trial_id']}",
        f"- strategy: {handoff.get('strategy')}",
        f"- pipeline_axis: {handoff.get('pipeline_axis')}",
        "",
        "## Target Files",
        "",
    ]
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
            "",
            "## Validation Commands",
            "",
        ]
    )
    lines.extend(f"```powershell\n{item}\n```" for item in handoff["validation_commands"] or ["No validation command provided."])
    if next_experiment:
        lines.extend(["", "## Next Experiment Context", "", next_experiment])
    lines.append("")
    return "\n".join(lines)
