from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..agents.memory import log_decision
from ..paths import competition_memory_dir
from ..paths import competition_jobs_dir, trial_dir
from ..policies import load_policy


def decide_execution(
    competition: str,
    trial_id: str,
    *,
    backend: str = "local",
    run_now: bool = False,
    command: str | None = None,
    config_errors: list[str] | None = None,
    log: bool = True,
) -> dict[str, Any]:
    policy = load_policy("execution_policy")
    out_dir = trial_dir(competition, trial_id)
    metrics_exists = (out_dir / "metrics.json").exists()
    local_failure = classify_local_failure(out_dir / "local_run.log")
    job = _existing_job(competition, trial_id)
    job_status = job.get("status") if job else None
    require_gpu = _config_text_mentions(out_dir / "config.yaml", "require_gpu: true")
    estimated_runtime = _read_estimated_runtime(out_dir / "config.yaml")

    evidence = {
        "default_backend": policy.get("default_backend"),
        "metrics_exists": metrics_exists,
        "backend": backend,
        "run_now": run_now,
        "run_command_available": bool(command),
        "job_status": job_status,
        "require_gpu": require_gpu,
        "estimated_runtime_minutes": estimated_runtime,
        "previous_local_failure_type": local_failure["failure_type"],
        "local_failure_artifact_path": local_failure.get("artifact_path"),
        "config_error_count": len(config_errors or []),
    }

    if metrics_exists:
        decision = _decision("evaluate_metrics", "Metrics already exist; skip execution.", evidence, "evaluate")
    elif config_errors:
        decision = _decision("blocked", "Config validation failed.", evidence, "fix-config")
    elif job_status in {"pending", "running"} and not run_now:
        decision = _decision("wait_for_metrics", "A job is already pending or running.", evidence, "wait")
    elif backend == "colab":
        if policy.get("ask_before_colab", True):
            decision = _decision("ask_user", "Colab was requested and policy requires approval.", evidence, "approve-colab")
        else:
            decision = _decision("create_colab_job", "Colab was explicitly requested.", evidence, "create-job")
    elif local_failure["failure_type"] in {"resource_cpu_memory", "resource_gpu_missing"}:
        decision = _decision("ask_user", "Previous local run failed due to resource limits.", evidence, "approve-colab")
    elif local_failure["failure_type"] == "missing_dependency":
        decision = _decision("ask_user", "Previous local run failed because a dependency is missing.", evidence, "fix-dependency")
    elif local_failure["failure_type"] in {"missing_file", "permission_error"}:
        decision = _decision("ask_user", "Previous local run failed because local inputs or permissions need attention.", evidence, "fix-local-environment")
    elif local_failure["failure_type"] == "unknown":
        decision = _decision("ask_user", "Previous local run failed and needs inspection before retrying.", evidence, "inspect-code")
    elif require_gpu:
        decision = _decision("ask_user", "Trial requires GPU according to config.", evidence, "approve-colab")
    elif estimated_runtime is not None and estimated_runtime > int(policy.get("estimated_runtime_minutes_over", 30)):
        decision = _decision("ask_user", "Estimated runtime exceeds local-first threshold.", evidence, "approve-execution")
    elif run_now and command:
        decision = _decision("run_local", "Trial has a local command and no blocking policy condition.", evidence, "run-local")
    elif policy.get("local_first", True):
        decision = _decision("create_local_job", "Local-first policy creates a local job request.", evidence, "create-job")
    else:
        decision = _decision("create_local_job", "Default execution path.", evidence, "create-job")

    if log:
        log_decision(
            competition,
            trial_id,
            decision_type="execution_backend",
            decision=decision["decision"],
            reason=decision["reason"],
            evidence=evidence,
            user_input_used=False,
            next_action=decision["next_action"],
        )
    return decision


def classify_local_failure(log_path: str | Path, *, use_artifact: bool = True) -> dict[str, Any]:
    path = Path(log_path)
    if use_artifact:
        artifact = path.with_name("local_failure.json")
        if artifact.exists():
            return _read_local_failure_artifact(artifact)
    if not path.exists():
        return {"failure_type": None, "matched_pattern": None, "artifact_path": None}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "returncode: 0" in text:
        return {"failure_type": None, "matched_pattern": None, "artifact_path": None}
    patterns = load_policy("execution_policy").get("local_failure_patterns", {})
    lower = text.casefold()
    for failure_type, candidates in patterns.items():
        for pattern in candidates:
            if str(pattern).casefold() in lower:
                return {"failure_type": failure_type, "matched_pattern": pattern, "artifact_path": None}
    return {"failure_type": "unknown", "matched_pattern": None, "artifact_path": None}


def decide_human_review(
    competition: str,
    trial_id: str,
    diagnosis: dict[str, Any],
    *,
    pipeline_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = load_policy("human_review_policy")
    issues = " ".join(diagnosis.get("issues", [])).casefold()
    questions = diagnosis.get("user_questions", [])
    strategy = diagnosis.get("strategy_recommendation")
    repeated_threshold = int(policy.get("repeated_failures_threshold", 3))
    recent_failures = int(diagnosis.get("consecutive_failures", 0) or 0)

    triggers: list[str] = []
    if diagnosis.get("needs_user_review") or questions:
        triggers.append("diagnosis_requested_review")
    if "segment" in issues or "concentrated" in issues:
        triggers.append("high_error_concentration")
    if "leakage" in issues or "cv/lb" in issues:
        triggers.append("validation_or_leakage_suspected")
    if "label" in issues and ("ambiguous" in issues or "boundary" in issues):
        triggers.append("label_boundary_ambiguous")
    blocking_classes = [str(item).casefold() for item in policy.get("blocking_safety_classes", [])]
    if "false negative" in issues and (
        "safety" in issues or any(class_name in issues for class_name in blocking_classes)
    ):
        triggers.append("safety_false_negative")
    if "missing" in issues and "required" in issues and any(
        term in issues for term in ["metric", "label", "input", "definition"]
    ):
        triggers.append("blocking_information_missing")
    if strategy == "strategy_escalation" or recent_failures >= repeated_threshold:
        triggers.append("strategy_shift_required")

    decision = "no_review"
    timing = "no_review"
    urgent = False
    mature = False
    next_action = "continue"
    if triggers:
        if pipeline_readiness is None:
            timing = "request_now"
        else:
            timing_policy = policy.get("review_timing", {})
            urgent_triggers = set(timing_policy.get("immediate_triggers", []))
            urgent = bool(urgent_triggers.intersection(triggers))
            minimum_trials = int(timing_policy.get("minimum_completed_trials_for_nonurgent_review", 2))
            mature = (
                bool(pipeline_readiness.get("execution_profile_ready"))
                and bool(pipeline_readiness.get("workspace_run_completed"))
                and bool(pipeline_readiness.get("metrics_collected"))
                and not bool(pipeline_readiness.get("pending_user_review"))
                and int(pipeline_readiness.get("completed_trial_count", 0)) >= minimum_trials
            )
            timing = "request_now" if urgent or mature else "defer"

        if timing == "request_now":
            decision = policy.get("default_review_action", "prepare_review_pack")
            next_action = "request_user_review" if decision == "request_review" else decision
        else:
            decision = "defer_review"
            next_action = "continue_until_pipeline_mature"

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "decision": decision,
        "timing": timing,
        "urgent": urgent,
        "pipeline_mature": mature,
        "pipeline_readiness": pipeline_readiness,
        "triggers": triggers,
        "questions": questions[: int(policy.get("max_questions_per_review", 3))],
        "next_action": next_action,
    }
    log_decision(
        competition,
        trial_id,
        decision_type="human_review",
        decision=decision,
        reason="Human review policy evaluated diagnosis.",
        evidence={
            "triggers": triggers,
            "question_count": len(questions),
            "timing": timing,
            "urgent": urgent,
            "pipeline_mature": mature,
            "pipeline_readiness": pipeline_readiness,
        },
        user_input_used=False,
        next_action=next_action,
    )
    return result


def should_call_llm(
    reason: str,
    *,
    competition: str | None = None,
    trial_id: str | None = None,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
) -> dict[str, Any]:
    policy = load_policy("token_policy")
    counted_calls = count_llm_calls_from_decision_log(competition, trial_id) if competition else {}
    resolved_trial_calls = (
        int(trial_llm_calls)
        if trial_llm_calls is not None
        else int(counted_calls.get("trial_llm_calls", 0))
    )
    resolved_strategy_calls = (
        int(strategy_calls_today)
        if strategy_calls_today is not None
        else int(counted_calls.get("strategy_calls_today", 0))
    )
    allowed_reasons = set(policy.get("call_llm_when", []))
    max_calls_per_trial = policy.get("max_llm_calls_per_trial", 4)
    max_calls_per_day = policy.get("max_strategy_calls_per_day", 20)
    within_budget = max_calls_per_trial is None or resolved_trial_calls < int(max_calls_per_trial)
    if within_budget and max_calls_per_day is not None:
        within_budget = resolved_strategy_calls < int(max_calls_per_day)
    return {
        "decision": "call_llm" if reason in allowed_reasons and within_budget else "skip_llm",
        "reason": reason,
        "within_budget": within_budget,
        "trial_llm_calls": resolved_trial_calls,
        "strategy_calls_today": resolved_strategy_calls,
        "counts_source": counted_calls.get("source", "manual") if competition else "manual",
        "summarize_logs_before_llm": bool(policy.get("summarize_logs_before_llm", True)),
        "avoid_raw_training_logs_in_prompt": bool(policy.get("avoid_raw_training_logs_in_prompt", True)),
    }


def log_llm_decision(
    competition: str,
    trial_id: str | None,
    reason: str,
    *,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
    prompt_summary_path: str | None = None,
) -> dict[str, Any]:
    decision = should_call_llm(
        reason,
        competition=competition,
        trial_id=trial_id,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
    )
    row = log_decision(
        competition,
        trial_id,
        decision_type="llm_call",
        decision=decision["decision"],
        reason=f"Token policy evaluated LLM reason: {reason}.",
        evidence={
            "llm_reason": reason,
            "trial_llm_calls": decision["trial_llm_calls"],
            "strategy_calls_today": decision["strategy_calls_today"],
            "counts_source": decision["counts_source"],
            "within_budget": decision["within_budget"],
            "summarize_logs_before_llm": decision["summarize_logs_before_llm"],
            "avoid_raw_training_logs_in_prompt": decision["avoid_raw_training_logs_in_prompt"],
            "prompt_summary_path": prompt_summary_path,
        },
        user_input_used=False,
        next_action="call-llm" if decision["decision"] == "call_llm" else "use-rule-based-path",
    )
    return {**decision, "log": row}


def count_llm_calls_from_decision_log(
    competition: str | None,
    trial_id: str | None = None,
    *,
    today: str | None = None,
) -> dict[str, Any]:
    if not competition:
        return {"trial_llm_calls": 0, "strategy_calls_today": 0, "source": "none"}
    path = competition_memory_dir(competition) / "decision_log.jsonl"
    if not path.exists():
        return {"trial_llm_calls": 0, "strategy_calls_today": 0, "source": "decision_log"}

    current_day = today or datetime.now(timezone.utc).date().isoformat()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    trial_count = _count_llm_call_events(
        row for row in rows if trial_id is not None and row.get("trial_id") == trial_id
    )
    today_count = _count_llm_call_events(
        row for row in rows if str(row.get("time", "")).startswith(current_day)
    )
    return {
        "trial_llm_calls": trial_count,
        "strategy_calls_today": today_count,
        "source": "decision_log",
        "decision_log_path": str(path.as_posix()),
        "date": current_day,
    }


def _count_llm_call_events(rows: Iterable[dict[str, Any]]) -> int:
    """Count API calls once even when the same token decision is logged repeatedly."""

    request_ids: set[str] = set()
    legacy_outcomes = 0
    preflight_decisions = 0
    for row in rows:
        evidence = row.get("evidence", {})
        evidence = evidence if isinstance(evidence, dict) else {}
        token_usage = evidence.get("token_usage", {})
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        request_id = str(token_usage.get("request_id") or "").strip()
        if request_id:
            request_ids.add(request_id)
            continue

        if row.get("decision_type") == "llm_call":
            if row.get("decision") == "call_llm":
                preflight_decisions += 1
            continue

        token_decision = evidence.get("token_decision", {})
        if isinstance(token_decision, dict) and token_decision.get("decision") == "call_llm":
            legacy_outcomes += 1

    completed_or_legacy_calls = len(request_ids) + legacy_outcomes
    return completed_or_legacy_calls if completed_or_legacy_calls else preflight_decisions


def _decision(decision: str, reason: str, evidence: dict[str, Any], next_action: str) -> dict[str, Any]:
    return {"decision": decision, "reason": reason, "evidence": evidence, "next_action": next_action}


def _read_local_failure_artifact(path: Path) -> dict[str, Any]:
    try:
        failure = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"failure_type": "unknown", "matched_pattern": None, "artifact_path": str(path.as_posix())}
    return {
        "failure_type": failure.get("failure_type") or "unknown",
        "matched_pattern": failure.get("matched_pattern"),
        "artifact_path": failure.get("artifact_path") or str(path.as_posix()),
        "suggested_next_action": failure.get("suggested_next_action"),
    }


def _existing_job(competition: str, trial_id: str) -> dict[str, Any] | None:
    from .. import simple_yaml

    path = competition_jobs_dir(competition) / f"{competition}_{trial_id}.yaml"
    if not path.exists():
        return None
    return simple_yaml.load(path, default={})


def _config_text_mentions(path: Path, needle: str) -> bool:
    return path.exists() and needle.casefold() in path.read_text(encoding="utf-8", errors="ignore").casefold()


def _read_estimated_runtime(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("estimated_runtime_minutes:"):
            value = stripped.split(":", 1)[1].strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None
