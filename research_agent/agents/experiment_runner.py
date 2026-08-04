from __future__ import annotations


import json
import subprocess
from pathlib import Path
from typing import Any

from .. import simple_yaml
from ..paths import competition_jobs_dir, jobs_dir, trial_dir
from ..store import now_iso, write_text


def default_train_command(competition: str, trial_id: str) -> str:
    return f"python train.py --config experiments/{competition}/{trial_id}/config.yaml --output experiments/{competition}/{trial_id}"


def create_job(
    competition: str,
    trial_id: str,
    command: str | None = None,
    backend: str = "local",
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    config_path = out_dir / "config.yaml"
    if command is None:
        command = default_train_command(competition, trial_id)

    job = {
        "job_id": f"{competition}_{trial_id}",
        "competition": competition,
        "trial_id": trial_id,
        "backend": backend,
        "created_at": now_iso(),
        "status": "pending",
        "command": command,
        "config_path": str(config_path.as_posix()),
        "expected_outputs": [
            f"experiments/{competition}/{trial_id}/metrics.json",
            f"experiments/{competition}/{trial_id}/submission.csv",
        ],
    }
    competition_jobs_dir(competition).mkdir(parents=True, exist_ok=True)
    simple_yaml.dump(job, _job_path(competition, trial_id))
    write_text(out_dir / "job.md", render_job(job))
    return job


def create_colab_job(competition: str, trial_id: str, command: str | None = None) -> dict[str, Any]:
    return create_job(competition, trial_id, command=command, backend="colab")


def run_local_job(competition: str, trial_id: str, command: str | None = None) -> dict[str, Any]:
    job = create_job(competition, trial_id, command=command, backend="local")
    job_path = _job_path(competition, trial_id)
    job["status"] = "running"
    simple_yaml.dump(job, job_path)

    out_dir = trial_dir(competition, trial_id)
    log_path = out_dir / "local_run.log"
    completed = subprocess.run(
        job["command"],
        shell=True,
        cwd=jobs_dir().parent,
        text=True,
        capture_output=True,
    )
    log_path.write_text(
        "COMMAND:\n"
        + job["command"]
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )

    job["status"] = "done" if completed.returncode == 0 else "failed"
    job["returncode"] = completed.returncode
    job["log_path"] = str(log_path.as_posix())
    if job["status"] == "failed":
        failure = write_local_failure_artifact(
            competition,
            trial_id,
            command=job["command"],
            exit_code=completed.returncode,
            log_path=log_path,
        )
        job["failure_path"] = failure["artifact_path"]
    simple_yaml.dump(job, job_path)
    write_text(out_dir / "job.md", render_job(job))
    return job


def write_local_failure_artifact(
    competition: str,
    trial_id: str,
    *,
    command: str,
    exit_code: int,
    log_path: str | Path,
) -> dict[str, Any]:
    from .policy_gate import classify_local_failure

    out_dir = trial_dir(competition, trial_id)
    resolved_log_path = Path(log_path)
    classification = classify_local_failure(resolved_log_path, use_artifact=False)
    artifact_path = out_dir / "local_failure.json"
    failure = {
        "competition": competition,
        "trial_id": trial_id,
        "command": command,
        "exit_code": exit_code,
        "failure_type": classification["failure_type"],
        "matched_pattern": classification["matched_pattern"],
        "suggested_next_action": _suggested_failure_action(classification["failure_type"]),
        "log_path": str(resolved_log_path.as_posix()),
        "artifact_path": str(artifact_path.relative_to(jobs_dir().parent).as_posix()),
        "log_tail": _log_tail(resolved_log_path),
    }
    write_text(artifact_path, json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "local_failure.md", render_local_failure(failure))
    return failure


def render_local_failure(failure: dict[str, Any]) -> str:
    return f"""# {failure['trial_id']} Local Failure

- failure_type: {failure['failure_type']}
- matched_pattern: {failure['matched_pattern']}
- exit_code: {failure['exit_code']}
- suggested_next_action: {failure['suggested_next_action']}

## Command

```bash
{failure['command']}
```

## Log Tail

```text
{failure['log_tail']}
```
"""


def _job_path(competition: str, trial_id: str) -> Any:
    return competition_jobs_dir(competition) / f"{competition}_{trial_id}.yaml"


def render_job(job: dict[str, Any]) -> str:
    backend = job.get("backend", "colab")
    return f"""# {backend.title()} Job: {job['job_id']}

Status: {job['status']}
Backend: {backend}

Command:

```bash
{job['command']}
```

Expected outputs:

""" + "\n".join(f"- {item}" for item in job["expected_outputs"]) + "\n"
from .patch_validator import validate_patch_plan
from .result_analyst import diagnose_trial, evaluate_trial


def apply_patch_plan(
    competition: str,
    trial_id: str,
    run_command: str | None = None,
) -> dict[str, Any]:
    """Validate a prepared patch plan and optionally execute the trial locally."""

    out_dir = trial_dir(competition, trial_id)
    plan_path = out_dir / "code_patch_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing code patch plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    patch_validation = validate_patch_plan(competition, trial_id)
    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "strategy": plan.get("strategy"),
        "status": "blocked" if patch_validation["status"] == "blocked" else "ready",
        "missing_targets": patch_validation["missing_targets"],
        "validation_errors": patch_validation["config_errors"],
        "patch_validation_status": patch_validation["status"],
        "patch_validation_issues": patch_validation["issues"],
        "run_command": run_command,
        "job_status": None,
        "evaluation_recommendation": None,
        "diagnosis_strategy": None,
    }

    if result["status"] == "ready" and run_command:
        job = run_local_job(competition, trial_id, command=run_command)
        result["job_status"] = job.get("status")
        if job.get("status") == "done" and (out_dir / "metrics.json").exists():
            evaluation = evaluate_trial(competition, trial_id)
            diagnosis = diagnose_trial(competition, trial_id)
            result["status"] = "executed"
            result["evaluation_recommendation"] = evaluation["recommendation"]
            result["diagnosis_strategy"] = diagnosis["strategy_recommendation"]
        else:
            result["status"] = "execution_failed"

    write_text(out_dir / "code_edit_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "code_edit_result.md", render_code_edit_result(result))
    return result


def render_code_edit_result(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Code Edit Result",
        "",
        f"- status: {result['status']}",
        f"- strategy: {result['strategy']}",
        f"- job_status: {result['job_status']}",
        f"- patch_validation_status: {result.get('patch_validation_status')}",
        f"- evaluation_recommendation: {result['evaluation_recommendation']}",
        f"- diagnosis_strategy: {result['diagnosis_strategy']}",
        "",
        "## Missing Targets",
        "",
    ]
    lines.extend(f"- {item}" for item in result["missing_targets"] or ["None"])
    lines.extend(["", "## Validation Errors", ""])
    lines.extend(f"- {item}" for item in result["validation_errors"] or ["None"])
    lines.extend(["", "## Patch Validation Issues", ""])
    lines.extend(f"- {item}" for item in result.get("patch_validation_issues", []) or ["None"])
    if result.get("run_command"):
        lines.extend(["", "## Run Command", "", f"```bash\n{result['run_command']}\n```"])
    lines.append("")
    return "\n".join(lines)


def _log_tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def _suggested_failure_action(failure_type: str | None) -> str:
    if failure_type in {"resource_cpu_memory", "resource_gpu_missing"}:
        return "approve_colab"
    if failure_type == "missing_dependency":
        return "fix_dependency"
    if failure_type == "missing_file":
        return "fix_input_files"
    if failure_type == "permission_error":
        return "fix_permissions"
    return "inspect_code"


