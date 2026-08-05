"""Bridge from the loop to execution_core.

Everything downstream of a trial -- the metrics collector, the state DB, the
dashboard, the submission path -- reads artifacts from disk. So the bridge
writes exactly what the old runner wrote, at exactly the same paths, and none
of it needs to know which execution model produced them. What changes is only
how the numbers were arrived at.

Import direction is one-way on purpose: research_agent imports execution_core,
never the reverse. The new core stays testable without the old package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from execution_core import run_trial
from execution_core.contract import DEFAULT_HOLDOUT_RATIO, DEFAULT_SEED
from execution_core.metric_provisioning import provision_metric

from .agents.memory import log_decision
from .execution_profile import (
    contract_data_dir,
    contract_submission_template,
    load_execution_profile,
    validate_execution_profile,
)
from .paths import competition_dir, trial_dir
from .store import write_text
from .workspace_runner import _inspect_artifacts, _validate_artifacts


def run_contract_pipeline(
    competition: str,
    trial_id: str,
    *,
    run_now: bool = False,
    allow_constant_predictions: bool = False,
    generate_metric: Any | None = None,
) -> dict[str, Any]:
    """Run one trial under the contract model, in the old runner's shape."""
    validation = validate_execution_profile(competition)
    try:
        profile = load_execution_profile(competition)
    except (FileNotFoundError, ValueError):
        profile = {}

    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "planned",
        "run_now": run_now,
        "execution_model": "contract",
        "profile_status": validation["status"],
        "profile_issues": validation["issues"],
        "project_root": profile.get("project_root"),
        "commands": {},
        "command_results": [],
        "artifacts": {},
        "next_action": "rerun-with-run-now",
    }
    if validation["status"] != "ready":
        result["status"] = "blocked"
        result["next_action"] = "fix-execution-profile"
        return _finish(result)
    if not run_now:
        return _finish(result)

    metric = provision_metric(
        competition_dir(competition),
        target_keys=_target_keys(profile),
        generate=generate_metric,
    )
    result["metric_provisioning"] = {
        key: value for key, value in metric.items() if key != "spec"
    }
    if metric.get("status") != "ready":
        result["status"] = "blocked"
        result["failed_stage"] = "metric"
        result["issues"] = metric.get("issues", [])
        result["next_action"] = "fix-competition-metric"
        return _finish(result)

    trial = run_trial(
        profile["project_root"],
        str(profile["python"]),
        data_dir=contract_data_dir(profile),
        metric=metric["spec"],
        submission_template=contract_submission_template(profile),
        allow_constant_predictions=allow_constant_predictions,
        metric_confidence=metric.get("confidence"),
        # The validation structure is a legitimate improvement axis, so it is
        # configurable -- but it lives in the profile, outside the workspace,
        # where changing it is a deliberate act rather than a side effect of
        # rewriting a scorer.
        holdout_ratio=float(profile.get("holdout_ratio") or DEFAULT_HOLDOUT_RATIO),
        seed=int(profile.get("split_seed") or DEFAULT_SEED),
        timeout=int(profile.get("timeout_seconds") or 3600),
    )
    result["trial"] = {key: value for key, value in trial.items() if key != "child"}
    result["command_results"] = [
        {"stage": "contract", "index": 1, "command": "execution_core.run_trial",
         "returncode": 0 if trial["status"] == "completed" else 1}
    ]

    if trial["status"] != "completed":
        result["status"] = _failed_status(trial)
        result["failed_stage"] = trial.get("failed_stage") or "contract"
        result["failure"] = {"error": trial.get("error"), "issues": trial.get("issues")}
        result["issues"] = trial.get("issues", [])
        result["next_action"] = trial.get("next_action") or "fix-model-code"
        return _finish(result)

    result["artifacts"] = _inspect_artifacts(profile)
    artifact_issues = _validate_artifacts(result["artifacts"])
    if artifact_issues:
        result["status"] = "invalid_artifacts"
        result["artifact_issues"] = artifact_issues
        result["next_action"] = "fix-artifact-output"
        return _finish(result)

    result["status"] = "completed"
    result["next_action"] = "collect-metrics"
    return _finish(result)


def _failed_status(trial: dict[str, Any]) -> str:
    """Keep the core's own verdict rather than flattening it to "failed".

    A trial stopped because every prediction was identical is a different
    situation from one that crashed, and the loop decides differently on each.
    """
    if trial["status"] == "blocked_constant_predictor":
        return "blocked_constant_predictor"
    return "failed"


def _target_keys(profile: dict[str, Any]) -> list[str]:
    declared = profile.get("target_keys")
    return [str(key) for key in declared] if isinstance(declared, list) else []


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    out_dir = trial_dir(result["competition"], result["trial_id"])
    write_text(out_dir / "workspace_run.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_run.md", _render(result))
    log_decision(
        result["competition"],
        result["trial_id"],
        decision_type="workspace_pipeline",
        decision=result["status"],
        reason="Contract execution model: the framework ran fit, predict and scoring.",
        evidence={
            "execution_model": "contract",
            "trial": result.get("trial"),
            "metric_provisioning": result.get("metric_provisioning"),
            "artifacts": result.get("artifacts"),
        },
        user_input_used=result.get("run_now", False),
        next_action=result["next_action"],
    )
    return result


def _render(result: dict[str, Any]) -> str:
    trial = result.get("trial") or {}
    lines = [
        f"# {result['competition']} / {result['trial_id']} Workspace Run",
        "",
        f"- execution_model: contract",
        f"- status: {result['status']}",
        f"- run_now: {result['run_now']}",
        f"- project_root: {result['project_root']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Trial",
        "",
    ]
    for field in ("cv_score", "metric", "objective", "n_train", "holdout_count",
                  "n_test", "distinct_holdout_predictions", "fit_returned"):
        if trial.get(field) is not None:
            lines.append(f"- {field}: {trial[field]}")
    provisioning = result.get("metric_provisioning") or {}
    if provisioning:
        lines += [
            "",
            "## Metric",
            "",
            f"- source: {provisioning.get('source')}",
            f"- confidence: {provisioning.get('confidence')}",
            f"- spec_confidence: {provisioning.get('spec_confidence')}",
        ]
    if result.get("issues"):
        lines += ["", "## Issues", ""] + [f"- {issue}" for issue in result["issues"]]
    lines.append("")
    return "\n".join(lines)
