from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import paths, simple_yaml
from ..config_validator import validate_config
from ..paths import competition_configs_dir, configs_dir, trial_dir
from ..store import write_text
from .memory import log_decision


def validate_patch_plan(competition: str, trial_id: str, *, user_approved: bool = False) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    plan_path = out_dir / "code_patch_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing code patch plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    missing_targets = _missing_targets(plan.get("target_files", []))
    config_errors = _validate_trial_config(competition, trial_id)
    issues: list[str] = []
    issues.extend(f"missing_target:{item}" for item in missing_targets)
    issues.extend(f"config_error:{item}" for item in config_errors)

    if plan.get("requires_user_approval") and not user_approved:
        issues.append("user_approval_required")
    if not plan.get("validation_commands"):
        issues.append("missing_validation_command")
    if _touches_submission_artifacts(plan):
        issues.append("submission_artifact_target_forbidden")
    issues.extend(_protected_axis_violations(plan))

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked" if issues else "ready",
        "issues": issues,
        "missing_targets": missing_targets,
        "config_errors": config_errors,
        "requires_user_approval": bool(plan.get("requires_user_approval")),
        "user_approved": user_approved,
        "pipeline_axis": plan.get("pipeline_axis"),
        "protected_axes": plan.get("protected_axes", []),
        "target_files": plan.get("target_files", []),
    }
    write_text(out_dir / "patch_validation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "patch_validation.md", render_patch_validation(result))
    log_decision(
        competition,
        trial_id,
        decision_type="patch_validation",
        decision=result["status"],
        reason="Patch plan validation completed before code changes or local execution.",
        evidence={
            "issues": issues,
            "pipeline_axis": result["pipeline_axis"],
            "protected_axes": result["protected_axes"],
            "target_files": result["target_files"],
        },
        user_input_used=user_approved,
        next_action="apply-patch" if result["status"] == "ready" else "revise-patch-plan",
    )
    return result


def render_patch_validation(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Patch Validation",
        "",
        f"- status: {result['status']}",
        f"- pipeline_axis: {result.get('pipeline_axis')}",
        f"- requires_user_approval: {result['requires_user_approval']}",
        f"- user_approved: {result['user_approved']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Target Files", ""])
    lines.extend(f"- {item}" for item in result["target_files"] or ["None"])
    lines.append("")
    return "\n".join(lines)


def _missing_targets(target_files: list[str]) -> list[str]:
    root = paths.project_root()
    missing = []
    for item in target_files:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing.append(item)
    return missing


def _validate_trial_config(competition: str, trial_id: str) -> list[str]:
    config = simple_yaml.load(trial_dir(competition, trial_id) / "config.yaml", default={})
    allowed_path = competition_configs_dir(competition) / "allowed_space.yaml"
    if not allowed_path.exists():
        allowed_path = configs_dir() / "allowed_space.yaml"
    allowed = simple_yaml.load(allowed_path, default={})
    return validate_config(config, allowed) if allowed else []


def _protected_axis_violations(plan: dict[str, Any]) -> list[str]:
    protected = set(plan.get("protected_axes", []))
    changes = plan.get("config_changes", {})
    violations = []
    if "validation" in protected and any(str(key).startswith("cv.") for key in changes):
        violations.append("protected_axis_violation:validation")
    if "model_family" in protected and any(str(key) == "model.type" for key in changes):
        violations.append("protected_axis_violation:model_family")
    if "model_architecture" in protected and any(str(key).startswith("model.params.") for key in changes):
        violations.append("protected_axis_violation:model_architecture")
    if "pretraining_strategy" in protected and any(str(key).startswith("model.pretraining.") for key in changes):
        violations.append("protected_axis_violation:pretraining_strategy")
    return violations


def _touches_submission_artifacts(plan: dict[str, Any]) -> bool:
    forbidden = {"submission.csv", "submission_log.jsonl", "BEST_MARKER.md", "VERSION.md"}
    targets = " ".join(str(item) for item in plan.get("target_files", []))
    return any(item in targets for item in forbidden)
