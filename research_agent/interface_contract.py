from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents.memory import record_user_feedback
from .feedback_policy import apply_feedback_response, mark_feedback_request_applied
from .interaction_contract import interaction_contract
from .user_insight_policy import interpret_user_insight, new_insight_id, register_user_insight
from .operations import build_operations_status, can_run_next_trial
from .state_db import list_pending_actions, resolve_pending_action
from .state_db_sync import sync_state_db
from .state_query import get_trial_detail


SCHEMA_VERSION = "1.0"


def sync_state(*, competition: str | None = None, db_path: Path | None = None) -> dict[str, Any]:
    result = sync_state_db(competition, db_path=db_path)
    return _envelope(
        action="sync_state",
        ok=result.get("status") == "completed",
        status=str(result.get("status") or "unknown"),
        message="State DB synchronized.",
        data={"sync": result},
        db_path=result.get("db_path"),
        synced=True,
    )


def list_experiments(*, sync: bool = True, db_path: Path | None = None) -> dict[str, Any]:
    source = _sync_if_needed(None, sync=sync, db_path=db_path)
    status = build_operations_status(db_path=db_path)
    experiments = [_experiment_summary(item) for item in status.get("experiments", [])]
    return _envelope(
        action="list_experiments",
        ok=True,
        status="completed",
        message="Experiment list loaded.",
        data={
            "experiment_count": len(experiments),
            "experiments": experiments,
        },
        db_path=status.get("db_path"),
        synced=source["synced"],
    )


def get_experiment(competition: str, *, sync: bool = True, db_path: Path | None = None) -> dict[str, Any]:
    source = _sync_if_needed(competition, sync=sync, db_path=db_path)
    status = build_operations_status(competition=competition, db_path=db_path)
    experiment = status.get("experiment") or {}
    if not experiment.get("competition"):
        return _envelope(
            action="get_experiment",
            ok=False,
            status="not_found",
            message="Experiment not found.",
            data={"competition": competition},
            db_path=status.get("db_path"),
            synced=source["synced"],
            errors=[{"code": "experiment_not_found", "message": f"Experiment `{competition}` was not found."}],
        )

    return _envelope(
        action="get_experiment",
        ok=True,
        status="completed",
        message="Experiment status loaded.",
        data={
            "experiment": _experiment_summary(experiment),
            "trials": [_trial_summary(row) for row in experiment.get("trials", [])],
            "pending_requests": _pending_requests(experiment.get("pending_actions", [])),
        },
        next_actions=_next_actions_from_operation(competition, experiment.get("operation") or {}),
        db_path=status.get("db_path"),
        synced=source["synced"],
    )


def get_trial(
    competition: str,
    trial_id: str,
    *,
    sync: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    source = _sync_if_needed(competition, sync=sync, db_path=db_path)
    detail = get_trial_detail(competition, trial_id, db_path)
    summary = detail.get("summary")
    if not summary:
        return _envelope(
            action="get_trial",
            ok=False,
            status="not_found",
            message="Trial not found.",
            data={"competition": competition, "trial_id": trial_id},
            db_path=detail.get("db_path"),
            synced=source["synced"],
            errors=[{"code": "trial_not_found", "message": f"Trial `{competition}/{trial_id}` was not found."}],
        )

    return _envelope(
        action="get_trial",
        ok=True,
        status=str(summary.get("status") or "unknown"),
        message="Trial detail loaded.",
        data={"trial": _trial_detail(detail)},
        db_path=detail.get("db_path"),
        synced=source["synced"],
    )


def preview_next_trial(competition: str, *, sync: bool = True, db_path: Path | None = None) -> dict[str, Any]:
    source = _sync_if_needed(competition, sync=sync, db_path=db_path)
    status = build_operations_status(competition=competition, db_path=db_path)
    experiment = status.get("experiment") or {}
    operation = experiment.get("operation") or {}
    if not experiment.get("competition"):
        return _envelope(
            action="preview_next_trial",
            ok=False,
            status="not_found",
            message="Experiment not found.",
            data={"competition": competition},
            db_path=status.get("db_path"),
            synced=source["synced"],
            errors=[{"code": "experiment_not_found", "message": f"Experiment `{competition}` was not found."}],
        )

    runnable = can_run_next_trial(operation)
    return _envelope(
        action="preview_next_trial",
        ok=runnable,
        status=str(operation.get("state") or "unknown"),
        message="Next trial preview is ready." if runnable else "Next trial is not ready to run.",
        data={
            "competition": competition,
            "can_run": runnable,
            "next_trial_id": operation.get("next_trial_id"),
            "operation_state": operation.get("state"),
            "next_action_label": operation.get("next_action_label"),
            "active_axis": _active_axis(operation),
            "commands": list(operation.get("commands") or []),
        },
        next_actions=_next_actions_from_operation(competition, operation),
        db_path=status.get("db_path"),
        synced=source["synced"],
        errors=[] if runnable else [{"code": "next_trial_not_ready", "message": operation.get("next_action_label")}],
    )


def list_pending_requests(
    competition: str | None = None,
    *,
    status: str = "pending",
    sync: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    source = _sync_if_needed(competition, sync=sync, db_path=db_path)
    requests = _pending_requests(list_pending_actions(competition, db_path, status=status))
    return _envelope(
        action="list_pending_requests",
        ok=True,
        status="completed",
        message="Pending requests loaded.",
        data={
            "competition": competition,
            "status_filter": status,
            "request_count": len(requests),
            "requests": requests,
        },
        db_path=db_path,
        synced=source["synced"],
    )


def get_pending_request(request_id: str, *, db_path: Path | None = None) -> dict[str, Any]:
    action = _find_pending_action(request_id, db_path=db_path)
    if not action:
        return _envelope(
            action="get_pending_request",
            ok=False,
            status="not_found",
            message="Pending request not found.",
            data={"request_id": request_id},
            db_path=db_path,
            synced=False,
            errors=[{"code": "pending_request_not_found", "message": f"Request `{request_id}` was not found."}],
        )
    return _envelope(
        action="get_pending_request",
        ok=True,
        status=str(action.get("status") or "unknown"),
        message="Pending request loaded.",
        data={"request": _pending_request(action)},
        db_path=db_path,
        synced=False,
    )


def respond_to_request(
    request_id: str,
    *,
    answers: dict[str, Any] | None = None,
    free_text: str = "",
    decision: str | None = None,
    follow_up_action: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    interaction = interaction_contract("human_review_response")
    action = _find_pending_action(request_id, db_path=db_path, statuses=("pending",))
    if not action:
        return _envelope(
            action="respond_to_request",
            ok=False,
            status="not_found",
            message="Pending request not found or already resolved.",
            data={"request_id": request_id, "interaction": interaction},
            db_path=db_path,
            synced=False,
            errors=[
                {
                    "code": "pending_request_not_found",
                    "message": f"Request `{request_id}` was not found in pending state.",
                }
            ],
        )

    competition = str(action.get("competition_id") or "")
    trial_id = action.get("trial_id")
    if not competition or not trial_id:
        return _envelope(
            action="respond_to_request",
            ok=False,
            status="blocked",
            message="Pending request does not have enough context to record feedback.",
            data={"request": _pending_request(action), "interaction": interaction},
            db_path=db_path,
            synced=False,
            errors=[{"code": "missing_feedback_context", "message": "competition_id and trial_id are required."}],
        )

    payload = action.get("payload") or {}
    selected_option = _selected_option(payload, answers or {})
    answer_text = _format_request_answer(answers or {}, free_text)
    feedback = record_user_feedback(
        competition,
        str(trial_id),
        topic=str(payload.get("topic") or action.get("action_type") or "pending_request"),
        question=str(payload.get("question") or action.get("message") or "User response requested."),
        user_feedback=answer_text,
        decision=decision or _derive_decision(answers or {}) or "user_response_recorded",
        follow_up_action=follow_up_action
        or str(
            (answers or {}).get("follow_up_action")
            or selected_option.get("follow_up_action")
            or payload.get("follow_up_action")
            or "consider_in_next_plan"
        ),
    )
    try:
        application = apply_feedback_response(
            competition,
            str(trial_id),
            request_id=request_id,
            feedback=feedback,
            payload=payload,
        )
    except Exception as error:
        return _envelope(
            action="respond_to_request",
            ok=False,
            status="apply_failed",
            message="답변은 기록했지만 실험 상태에 반영하지 못했습니다. 요청은 대기 상태로 유지됩니다.",
            data={
                "request": _pending_request(action),
                "feedback": feedback,
                "interaction": interaction,
            },
            db_path=db_path,
            synced=False,
            errors=[{"code": "feedback_application_failed", "message": str(error)}],
        )
    resolved = resolve_pending_action(request_id, db_path=db_path)
    mark_feedback_request_applied(competition, db_path=db_path)
    return _envelope(
        action="respond_to_request",
        ok=True,
        status="completed",
        message="사용자 답변을 다음 실험 계획에 반영했고 요청을 완료했습니다.",
        data={
            "request": _pending_request(resolved or action),
            "feedback": feedback,
            "application": application,
            "interaction": interaction,
        },
        next_actions=[
            {
                "id": "get_trial",
                "label": "응답이 반영된 trial 확인",
                "action": "get_trial",
                "params": {"competition": competition, "trial_id": trial_id},
            }
        ],
        db_path=db_path,
        synced=False,
    )


def submit_human_insight(
    competition: str,
    trial_id: str,
    *,
    insight: str,
    topic: str = "user_insight",
    question: str = "User-provided research insight.",
    decision: str = "user_insight_for_planner",
    follow_up_action: str = "planner_interpret_next_experiment",
    scope: str = "next_trial",
    priority: str = "normal",
    db_path: Path | None = None,
    allow_api: bool = False,
    insight_client: Any | None = None,
) -> dict[str, Any]:
    interaction = interaction_contract("user_insight")
    insight_id = new_insight_id(competition, trial_id, insight)
    interpretation = interpret_user_insight(
        insight,
        competition=competition,
        trial_id=trial_id,
        allow_api=allow_api,
        client=insight_client,
    )
    feedback = record_user_feedback(
        competition,
        trial_id,
        topic=topic,
        question=question,
        user_feedback=insight,
        decision=decision,
        follow_up_action=follow_up_action,
        scope=scope,
        priority=priority,
        insight_id=insight_id,
    )
    insight_record = register_user_insight(feedback, interpretation)
    return _envelope(
        action="submit_human_insight",
        ok=True,
        status="completed",
        message="Human insight recorded.",
        data={
            "competition": competition,
            "trial_id": trial_id,
            "feedback": feedback,
            "insight": insight_record,
            "interaction": interaction,
        },
        next_actions=[
            {
                "id": "preview_next_trial",
                "label": "사용자 의견 반영 후 다음 trial 미리보기",
                "action": "preview_next_trial",
                "params": {"competition": competition},
            }
        ],
        db_path=db_path,
        synced=False,
    )


def _envelope(
    *,
    action: str,
    ok: bool,
    status: str,
    message: str,
    data: dict[str, Any],
    db_path: str | Path | None,
    synced: bool,
    next_actions: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(ok),
        "action": action,
        "status": status,
        "message": message,
        "data": data,
        "next_actions": next_actions or [],
        "warnings": warnings or [],
        "errors": errors or [],
        "source": {
            "db_path": str(db_path) if db_path else None,
            "synced": bool(synced),
        },
    }


def _sync_if_needed(competition: str | None, *, sync: bool, db_path: Path | None) -> dict[str, Any]:
    if not sync:
        return {"synced": False, "result": None}
    result = sync_state_db(competition, db_path=db_path)
    return {"synced": result.get("status") == "completed", "result": result}


def _experiment_summary(experiment: dict[str, Any]) -> dict[str, Any]:
    competition = experiment.get("competition") or experiment
    operation = experiment.get("operation") or {}
    best = experiment.get("best_trial") or competition.get("best_trial") or {}
    latest = operation.get("latest_trial") or {}
    return {
        "competition": competition.get("competition_id"),
        "topic": competition.get("topic"),
        "platform": competition.get("platform"),
        "metric": competition.get("metric"),
        "objective": competition.get("objective"),
        "state": operation.get("state") or competition.get("status"),
        "trial_count": experiment.get("trial_count") or competition.get("trial_count") or 0,
        "best_trial": _trial_ref(best),
        "latest_trial": _trial_ref(latest),
        "active_axis": _active_axis(operation),
        "next_trial_id": operation.get("next_trial_id"),
        "next_action_label": operation.get("next_action_label"),
        "pending_request_count": len(experiment.get("pending_actions") or []),
        "total_tokens": experiment.get("total_tokens") or competition.get("total_tokens") or 0,
    }


def _trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "competition": trial.get("competition_id"),
        "trial_id": trial.get("trial_id"),
        "status": trial.get("status"),
        "metric": trial.get("metric"),
        "local_score": trial.get("local_score"),
        "lb_score": trial.get("lb_score"),
        "decision": trial.get("decision"),
        "change_axis": trial.get("primary_change_axis") or trial.get("change_axis"),
        "active_axis": trial.get("active_axis"),
        "axis_attempt_count": trial.get("axis_attempt_count"),
        "axis_attempt_limit": trial.get("axis_attempt_limit"),
        "is_best_local": bool(trial.get("is_best_local")),
        "is_best_lb": bool(trial.get("is_best_lb")),
    }


def _trial_detail(detail: dict[str, Any]) -> dict[str, Any]:
    summary = detail.get("summary") or {}
    trial = _trial_summary(summary)
    trial.update(
        {
            "recommended_base_trial": summary.get("recommended_base_trial")
            or summary.get("decision_recommended_base_trial"),
            "token_total": detail.get("token_total", 0),
            "user_artifacts": [_artifact(item) for item in detail.get("user_artifacts", [])],
            "submissions": list(detail.get("submissions") or []),
            "pending_requests": _pending_requests(detail.get("pending_actions", [])),
            "token_usage": list(detail.get("token_usage") or []),
        }
    )
    return trial


def _trial_ref(trial: dict[str, Any]) -> dict[str, Any] | None:
    if not trial:
        return None
    return {
        "trial_id": trial.get("trial_id"),
        "status": trial.get("status"),
        "score": trial.get("local_score"),
        "lb_score": trial.get("lb_score"),
    }


def _active_axis(operation: dict[str, Any]) -> dict[str, Any] | None:
    axis = operation.get("active_axis")
    if not axis:
        return None
    return {
        "name": axis,
        "attempt_count": operation.get("axis_attempt_count"),
        "attempt_limit": operation.get("axis_attempt_limit"),
    }


def _artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact_type = artifact.get("artifact_type")
    return {
        "type": artifact_type,
        "label": _artifact_label(str(artifact_type or "")),
        "path": artifact.get("path"),
        "is_user_facing": bool(artifact.get("is_user_facing", True)),
    }


def _artifact_label(artifact_type: str) -> str:
    labels = {
        "summary_card_ko": "한눈에 보기",
        "readme_ko": "사용자 안내",
        "plan_ko": "실험 계획서",
        "pipeline_structure_ko": "파이프라인 구조도",
        "code_pipeline_ko": "코드 설명",
        "result_ko": "실행 결과",
        "submission_ko": "제출 정보",
        "decision_ko": "판단 카드",
        "paths_ko": "경로 안내",
        "user_code": "코드 폴더",
    }
    return labels.get(artifact_type, artifact_type)


def _pending_requests(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_pending_request(action) for action in actions]


def _pending_request(action: dict[str, Any]) -> dict[str, Any]:
    payload = action.get("payload") or {}
    return {
        "request_id": action.get("action_id"),
        "competition": action.get("competition_id"),
        "trial_id": action.get("trial_id"),
        "type": action.get("action_type"),
        "status": action.get("status"),
        "priority": action.get("priority"),
        "title": payload.get("title") or action.get("action_type"),
        "message": action.get("message"),
        "topic": payload.get("topic"),
        "question": payload.get("question") or action.get("message"),
        "context_files": list(payload.get("context_files") or []),
        "questions": list(payload.get("questions") or []),
        "problem": payload.get("problem"),
        "evidence_snapshot": list(payload.get("evidence_snapshot") or []),
        "evidence_summary": list(payload.get("evidence_summary") or []),
        "interpretation": payload.get("interpretation"),
        "recommendation": payload.get("recommendation"),
        "why_user_needed": payload.get("why_user_needed"),
        "options": list(payload.get("options") or []),
        "default_if_no_response": payload.get("default_if_no_response"),
        "interaction_type": payload.get("interaction_type"),
        "interaction_label": payload.get("interaction_label"),
        "execution_supported": payload.get("execution_supported", True),
        "execution_note": payload.get("execution_note"),
        "policy": dict(payload.get("policy") or {}),
        "payload": payload,
        "created_at": action.get("created_at"),
        "resolved_at": action.get("resolved_at"),
    }


def _find_pending_action(
    request_id: str,
    *,
    db_path: Path | None,
    statuses: tuple[str, ...] = ("pending", "resolved"),
) -> dict[str, Any] | None:
    for status in statuses:
        for action in list_pending_actions(None, db_path, status=status):
            if action.get("action_id") == request_id:
                return action
    return None


def _format_request_answer(answers: dict[str, Any], free_text: str) -> str:
    parts = []
    if answers:
        parts.extend(f"- {key}: {value}" for key, value in sorted(answers.items()))
    if free_text.strip():
        parts.extend(["", free_text.strip()] if parts else [free_text.strip()])
    return "\n".join(parts).strip() or "No detailed response was provided."


def _derive_decision(answers: dict[str, Any]) -> str | None:
    for key in ("decision", "continue_axis", "approval", "overall_decision"):
        value = answers.get(key)
        if value:
            return str(value)
    return None


def _selected_option(payload: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    selected = str(answers.get("decision") or answers.get("approval") or "").strip()
    if not selected:
        return {}
    for option in payload.get("options") or []:
        if isinstance(option, dict) and str(option.get("value") or "") == selected:
            return option
    return {}


def _next_actions_from_operation(competition: str, operation: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    state = operation.get("state")
    latest = operation.get("latest_trial") or {}
    if state in {"ready_first_trial", "ready_next_trial"}:
        actions.extend(
            [
                {
                    "id": "preview_next_trial",
                    "label": "다음 trial 미리보기",
                    "action": "preview_next_trial",
                    "params": {"competition": competition},
                },
                {
                    "id": "run_next_trial",
                    "label": "다음 trial 실행",
                    "action": "run_next_trial",
                    "params": {"competition": competition, "trial_id": operation.get("next_trial_id")},
                },
            ]
        )
    elif state == "blocked" and latest.get("trial_id"):
        actions.append(
            {
                "id": "inspect_blocked_trial",
                "label": "중단 원인 확인",
                "action": "get_trial",
                "params": {"competition": competition, "trial_id": latest.get("trial_id")},
            }
        )
    elif state == "waiting_user":
        actions.append(
            {
                "id": "list_pending_requests",
                "label": "대기 요청 확인",
                "action": "list_pending_requests",
                "params": {"competition": competition},
            }
        )
    for command in operation.get("commands", []) or []:
        actions.append({"id": "cli_command", "type": "cli_command", "label": command, "command": command})
    return actions
