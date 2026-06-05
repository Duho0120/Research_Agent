from __future__ import annotations

from typing import Any, Literal

from ..agents.experiment_runner import apply_patch_plan, create_job, run_local_job
from ..agents.memory import log_decision, remember_trial
from ..agents.pipeline_patch_planner import prepare_patch_plan
from ..agents.policy_gate import decide_execution, decide_human_review
from ..agents.research_planner import propose_next_experiment, propose_plan
from ..agents.result_analyst import diagnose_trial, evaluate_trial
from ..agents.review_pack import prepare_review_pack
from ..config_validator import validate_config
from ..paths import competition_configs_dir, configs_dir, trial_dir
from .. import simple_yaml
from .state import ResearchGraphState


def plan_trial_node(state: ResearchGraphState) -> dict[str, Any]:
    plan = propose_plan(state["competition"], state["trial_id"])
    return {"plan": {"hypothesis": plan["hypothesis"]}, "steps": _append_step(state, "planned")}


def validate_config_node(state: ResearchGraphState) -> dict[str, Any]:
    errors = _validate_trial_config(state["competition"], state["trial_id"])
    step = "config_invalid" if errors else "config_valid"
    return {"config_errors": errors, "steps": _append_step(state, step), "status": "blocked_config" if errors else "running"}


def route_after_validation(state: ResearchGraphState) -> Literal["end", "check_metrics"]:
    return "end" if state.get("config_errors") else "check_metrics"


def check_metrics_node(state: ResearchGraphState) -> dict[str, Any]:
    metrics_exists = (trial_dir(state["competition"], state["trial_id"]) / "metrics.json").exists()
    return {"metrics_exists": metrics_exists}


def route_after_metrics(state: ResearchGraphState) -> Literal["evaluate", "decide_execution", "end"]:
    if state.get("metrics_exists"):
        return "evaluate"
    return "decide_execution" if state.get("create_job_request", True) else "end"


def evaluate_node(state: ResearchGraphState) -> dict[str, Any]:
    report = evaluate_trial(state["competition"], state["trial_id"])
    return {"evaluation": report, "steps": _append_step(state, "evaluated")}


def diagnose_node(state: ResearchGraphState) -> dict[str, Any]:
    competition = state["competition"]
    trial_id = state["trial_id"]
    diagnosis = diagnose_trial(competition, trial_id)
    human_review = decide_human_review(competition, trial_id, diagnosis)
    updates: dict[str, Any] = {
        "diagnosis": diagnosis,
        "human_review": human_review,
        "steps": _append_step(state, "diagnosed"),
    }
    if human_review["decision"] == "prepare_review_pack":
        updates["review_pack"] = prepare_review_pack(competition, trial_id, diagnosis)
        updates["steps"] = [*updates["steps"], "review_pack_prepared"]
    log_decision(
        competition,
        trial_id,
        decision_type="diagnosis",
        decision="request_user_review" if diagnosis["needs_user_review"] else "continue",
        reason="Diagnosis completed after evaluation.",
        evidence={
            "cv_improved": diagnosis["cv_improved"],
            "issues": diagnosis["issues"],
            "strategy_recommendation": diagnosis["strategy_recommendation"],
        },
        user_input_used=False,
        next_action="request-review" if diagnosis["needs_user_review"] else "plan-next-trial",
    )
    return updates


def remember_node(state: ResearchGraphState) -> dict[str, Any]:
    row = remember_trial(state["competition"], state["trial_id"])
    return {"memory": row, "steps": _append_step(state, "remembered")}


def route_after_remember(state: ResearchGraphState) -> Literal["plan_next", "end"]:
    return "plan_next" if state.get("next_trial_id") else "end"


def plan_next_node(state: ResearchGraphState) -> dict[str, Any]:
    competition = state["competition"]
    trial_id = state["trial_id"]
    next_trial_id = state["next_trial_id"]
    plan = propose_next_experiment(competition, trial_id, next_trial_id)
    steps = _append_step(state, "next_experiment_planned")
    updates: dict[str, Any] = {"next_experiment": plan, "steps": steps}
    if state.get("prepare_next_patch"):
        patch_plan = prepare_patch_plan(competition, trial_id, next_trial_id)
        updates["patch_plan"] = patch_plan
        updates["steps"] = [*steps, "patch_plan_prepared"]
    if state.get("apply_next_patch"):
        if not state.get("prepare_next_patch"):
            patch_plan = prepare_patch_plan(competition, trial_id, next_trial_id)
            updates["patch_plan"] = patch_plan
            updates["steps"] = [*steps, "patch_plan_prepared"]
        code_edit = apply_patch_plan(competition, next_trial_id, run_command=state.get("next_run_command"))
        updates["next_code_edit"] = code_edit
        updates["steps"] = [*updates["steps"], "next_patch_applied"]
    return updates


def decide_execution_node(state: ResearchGraphState) -> dict[str, Any]:
    execution = decide_execution(
        state["competition"],
        state["trial_id"],
        backend=state.get("backend", "local"),
        run_now=state.get("run_now", False),
        command=state.get("command"),
        config_errors=state.get("config_errors", []),
    )
    return {"execution_decision": execution, "steps": _append_step(state, "execution_decided")}


def route_after_execution(state: ResearchGraphState) -> Literal["ask_user", "wait", "run_local", "create_job"]:
    decision = state["execution_decision"]["decision"]
    if decision == "ask_user":
        return "ask_user"
    if decision == "wait_for_metrics":
        return "wait"
    if state.get("run_now") and decision == "run_local":
        return "run_local"
    return "create_job"


def ask_user_node(state: ResearchGraphState) -> dict[str, Any]:
    return {"status": "ask_user", "steps": _append_step(state, "ask_user")}


def wait_node(state: ResearchGraphState) -> dict[str, Any]:
    return {"status": "waiting_for_metrics", "steps": _append_step(state, "waiting_for_metrics")}


def run_local_node(state: ResearchGraphState) -> dict[str, Any]:
    job = run_local_job(state["competition"], state["trial_id"], command=state.get("command"))
    updates: dict[str, Any] = {"job": {"job_id": job["job_id"], "status": job["status"]}, "steps": _append_step(state, "local_run_finished")}
    if job.get("status") == "done" and (trial_dir(state["competition"], state["trial_id"]) / "metrics.json").exists():
        updates["metrics_exists"] = True
    else:
        updates["status"] = "waiting_for_metrics"
    return updates


def create_job_node(state: ResearchGraphState) -> dict[str, Any]:
    decision = state["execution_decision"]["decision"]
    backend = "colab" if decision == "create_colab_job" else state.get("backend", "local")
    job = create_job(state["competition"], state["trial_id"], command=state.get("command"), backend=backend)
    return {
        "job": {"job_id": job["job_id"], "status": job["status"], "backend": backend},
        "status": "waiting_for_metrics",
        "steps": _append_step(state, f"{backend}_job_created"),
    }


def route_after_local_run(state: ResearchGraphState) -> Literal["evaluate", "end"]:
    return "evaluate" if state.get("metrics_exists") else "end"


def finalize_node(state: ResearchGraphState) -> dict[str, Any]:
    status = state.get("status")
    if not status or status == "running":
        status = "completed"
    return {"status": status}


def _append_step(state: ResearchGraphState, step: str) -> list[str]:
    return [*state.get("steps", []), step]


def _validate_trial_config(competition: str, trial_id: str) -> list[str]:
    config = simple_yaml.load(trial_dir(competition, trial_id) / "config.yaml", default={})
    allowed_path = competition_configs_dir(competition) / "allowed_space.yaml"
    if not allowed_path.exists():
        allowed_path = configs_dir() / "allowed_space.yaml"
    allowed = simple_yaml.load(allowed_path, default={})
    return validate_config(config, allowed)
