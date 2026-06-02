from __future__ import annotations


from typing import Any

from ..config_validator import validate_config
from .experiment_runner import apply_patch_plan, create_job, run_local_job
from .memory import log_decision, remember_trial
from .pipeline_patch_planner import prepare_patch_plan
from .policy_gate import decide_execution, decide_human_review
from .research_planner import propose_next_experiment, propose_plan
from .result_analyst import diagnose_trial, evaluate_trial
from .review_pack import prepare_review_pack
from ..paths import competition_configs_dir, configs_dir, trial_dir
from .. import simple_yaml


def run_auto_research_loop(
    competition: str,
    start_trial_id: str = "trial_001",
    max_trials: int = 3,
    submit_policy: str = "never",
    stop_no_improvement: int = 3,
    run_now: bool = False,
    command: str | None = None,
    next_run_command: str | None = None,
) -> dict[str, Any]:
    if submit_policy not in {"never", "prepare_only"}:
        raise ValueError(f"Unsupported submit_policy: {submit_policy}")

    result: dict[str, Any] = {
        "competition": competition,
        "start_trial_id": start_trial_id,
        "max_trials": max_trials,
        "submit_policy": submit_policy,
        "stop_no_improvement": stop_no_improvement,
        "status": "running",
        "trials": [],
        "steps": [],
    }
    current_trial = start_trial_id
    no_improvement_count = 0

    for index in range(max_trials):
        next_trial = _increment_trial_id(current_trial) if index < max_trials - 1 else None
        cycle = run_cycle(
            competition,
            current_trial,
            create_job_request=True,
            run_now=run_now,
            command=command if index == 0 else next_run_command,
            next_trial_id=next_trial,
            prepare_next_patch=bool(next_trial),
            apply_next_patch=False,
        )
        result["trials"].append({"trial_id": current_trial, "steps": cycle["steps"], "cycle": cycle})
        result["steps"].append(f"cycled:{current_trial}")

        if cycle.get("config_errors"):
            result["status"] = "blocked_config"
            break

        memory = cycle.get("memory")
        if memory:
            no_improvement_count = 0 if memory.get("is_best") else no_improvement_count + 1
            if no_improvement_count >= stop_no_improvement:
                result["status"] = "stopped_no_improvement"
                break

        if not next_trial:
            result["status"] = "completed"
            break
        current_trial = next_trial
    else:
        result["status"] = "completed"

    if result["status"] == "running":
        result["status"] = "completed"
    result["no_improvement_count"] = no_improvement_count
    return result


def run_cycle(
    competition: str,
    trial_id: str,
    create_job_request: bool = True,
    backend: str = "local",
    run_now: bool = False,
    command: str | None = None,
    next_trial_id: str | None = None,
    prepare_next_patch: bool = False,
    apply_next_patch: bool = False,
    next_run_command: str | None = None,
) -> dict[str, Any]:
    """Run the conservative Level 1/2 cycle for one trial.

    The cycle plans and validates a trial. If metrics already exist, it also
    evaluates and records memory. Otherwise it creates or runs a job request.
    """

    result: dict[str, Any] = {"competition": competition, "trial_id": trial_id, "steps": []}

    plan = propose_plan(competition, trial_id)
    result["steps"].append("planned")
    result["plan"] = {"hypothesis": plan["hypothesis"]}

    errors = _validate_trial_config(competition, trial_id)
    result["config_errors"] = errors
    if errors:
        result["steps"].append("config_invalid")
        return result
    result["steps"].append("config_valid")

    metrics_path = trial_dir(competition, trial_id) / "metrics.json"
    if metrics_path.exists():
        report = evaluate_trial(competition, trial_id)
        result["steps"].append("evaluated")
        result["evaluation"] = report
        _diagnose_and_log(competition, trial_id, result)
        memory_row = remember_trial(competition, trial_id)
        result["memory"] = memory_row
        result["steps"].append("remembered")
        _plan_next_if_requested(
            competition,
            trial_id,
            next_trial_id,
            prepare_next_patch,
            apply_next_patch,
            next_run_command,
            result,
        )
        return result

    if create_job_request:
        execution = decide_execution(
            competition,
            trial_id,
            backend=backend,
            run_now=run_now,
            command=command,
            config_errors=errors,
        )
        result["execution_decision"] = execution
        result["steps"].append("execution_decided")

        if execution["decision"] == "ask_user":
            result["steps"].append("ask_user")
            return result
        if execution["decision"] == "wait_for_metrics":
            result["steps"].append("waiting_for_metrics")
            return result
        if run_now and execution["decision"] == "run_local":
            job = run_local_job(competition, trial_id, command=command)
            result["steps"].append("local_run_finished")
            if job.get("status") == "done" and metrics_path.exists():
                report = evaluate_trial(competition, trial_id)
                result["steps"].append("evaluated")
                result["evaluation"] = report
                _diagnose_and_log(competition, trial_id, result)
                memory_row = remember_trial(competition, trial_id)
                result["memory"] = memory_row
                result["steps"].append("remembered")
                _plan_next_if_requested(
                    competition,
                    trial_id,
                    next_trial_id,
                    prepare_next_patch,
                    apply_next_patch,
                    next_run_command,
                    result,
                )
        else:
            job_backend = "colab" if execution["decision"] == "create_colab_job" else backend
            job = create_job(competition, trial_id, command=command, backend=job_backend)
            result["steps"].append(f"{job_backend}_job_created")
        result["job"] = {"job_id": job["job_id"], "status": job["status"]}
    else:
        result["steps"].append("waiting_for_metrics")
    return result


def _diagnose_and_log(competition: str, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
    diagnosis = diagnose_trial(competition, trial_id)
    human_review = decide_human_review(competition, trial_id, diagnosis)
    if human_review["decision"] == "prepare_review_pack":
        result["review_pack"] = prepare_review_pack(competition, trial_id, diagnosis)
        result["steps"].append("review_pack_prepared")
    needs_review = diagnosis["needs_user_review"]
    log_decision(
        competition,
        trial_id,
        decision_type="diagnosis",
        decision="request_user_review" if needs_review else "continue",
        reason="Diagnosis completed after evaluation.",
        evidence={
            "cv_improved": diagnosis["cv_improved"],
            "issues": diagnosis["issues"],
            "strategy_recommendation": diagnosis["strategy_recommendation"],
        },
        user_input_used=False,
        next_action="request-review" if needs_review else "plan-next-trial",
    )
    result["steps"].append("diagnosed")
    result["diagnosis"] = diagnosis
    result["human_review"] = human_review
    return diagnosis


def _plan_next_if_requested(
    competition: str,
    source_trial_id: str,
    next_trial_id: str | None,
    prepare_next_patch: bool,
    apply_next_patch: bool,
    next_run_command: str | None,
    result: dict[str, Any],
) -> None:
    if not next_trial_id:
        return
    plan = propose_next_experiment(competition, source_trial_id, next_trial_id)
    result["next_experiment"] = plan
    result["steps"].append("next_experiment_planned")
    if prepare_next_patch:
        patch_plan = prepare_patch_plan(competition, source_trial_id, next_trial_id)
        result["patch_plan"] = patch_plan
        result["steps"].append("patch_plan_prepared")
    if apply_next_patch:
        if not prepare_next_patch:
            patch_plan = prepare_patch_plan(competition, source_trial_id, next_trial_id)
            result["patch_plan"] = patch_plan
            result["steps"].append("patch_plan_prepared")
        code_edit = apply_patch_plan(competition, next_trial_id, run_command=next_run_command)
        result["next_code_edit"] = code_edit
        result["steps"].append("next_patch_applied")


def _validate_trial_config(competition: str, trial_id: str) -> list[str]:
    config = simple_yaml.load(trial_dir(competition, trial_id) / "config.yaml", default={})
    allowed_path = competition_configs_dir(competition) / "allowed_space.yaml"
    if not allowed_path.exists():
        allowed_path = configs_dir() / "allowed_space.yaml"
    allowed = simple_yaml.load(allowed_path, default={})
    return validate_config(config, allowed)


def _increment_trial_id(trial_id: str) -> str:
    prefix, separator, suffix = trial_id.rpartition("_")
    if separator and suffix.isdigit():
        return f"{prefix}_{int(suffix) + 1:0{len(suffix)}d}"
    return f"{trial_id}_next"


