from __future__ import annotations

import json
from typing import Any

from ..paths import trial_dir
from ..store import write_text
from .experiment_runner import create_job, run_local_job
from .memory import log_decision
from .policy_gate import decide_execution


def run_after_validation(
    competition: str,
    trial_id: str,
    *,
    command: str | None = None,
    backend: str = "local",
    run_now: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    validation_path = out_dir / "validation_run.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation run: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if validation.get("status") != "passed":
        issues.append("validation_run_not_passed")

    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked" if issues else "ready",
        "issues": issues,
        "validation_status": validation.get("status"),
        "execution_decision": None,
        "backend": backend,
        "run_now": run_now,
        "run_command": command,
        "job_id": None,
        "job_status": None,
        "next_action": "fix-validation" if issues else "execute-trial",
    }

    if not issues:
        decision = decide_execution(
            competition,
            trial_id,
            backend=backend,
            run_now=run_now,
            command=command,
            log=True,
        )
        result["execution_decision"] = decision["decision"]
        result["next_action"] = decision["next_action"]
        _apply_execution_decision(result, competition, trial_id, command, backend)

    _write_result(competition, trial_id, result)
    return result


def render_post_validation_execution(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Post-Validation Execution",
        "",
        f"- status: {result['status']}",
        f"- validation_status: {result['validation_status']}",
        f"- execution_decision: {result.get('execution_decision')}",
        f"- backend: {result['backend']}",
        f"- run_now: {result['run_now']}",
        f"- job_status: {result.get('job_status')}",
        f"- next_action: {result['next_action']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    if result.get("run_command"):
        lines.extend(["", "## Run Command", "", "```powershell", result["run_command"], "```"])
    lines.append("")
    return "\n".join(lines)


def _apply_execution_decision(
    result: dict[str, Any],
    competition: str,
    trial_id: str,
    command: str | None,
    backend: str,
) -> None:
    decision = result["execution_decision"]
    if decision == "run_local":
        job = run_local_job(competition, trial_id, command=command)
        result["status"] = "executed" if job.get("status") == "done" else "execution_failed"
        result["job_id"] = job.get("job_id")
        result["job_status"] = job.get("status")
        result["next_action"] = "evaluate" if result["status"] == "executed" else "inspect-local-failure"
        return
    if decision in {"create_local_job", "create_colab_job"}:
        job_backend = "colab" if decision == "create_colab_job" else backend
        job = create_job(competition, trial_id, command=command, backend=job_backend)
        result["status"] = "job_created"
        result["job_id"] = job.get("job_id")
        result["job_status"] = job.get("status")
        result["next_action"] = "wait_for_metrics"
        return
    if decision == "evaluate_metrics":
        result["status"] = "ready_to_evaluate"
        result["next_action"] = "evaluate"
        return
    result["status"] = "blocked"


def _write_result(competition: str, trial_id: str, result: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "post_validation_execution.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "post_validation_execution.md", render_post_validation_execution(result))
    log_decision(
        competition,
        trial_id,
        decision_type="post_validation_execution",
        decision=result["status"],
        reason="Post-validation execution gate evaluated whether the trial can proceed to execution.",
        evidence={
            "validation_status": result["validation_status"],
            "execution_decision": result.get("execution_decision"),
            "issues": result["issues"],
            "job_status": result.get("job_status"),
        },
        next_action=result["next_action"],
    )
