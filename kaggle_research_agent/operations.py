from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .state_query import get_experiment_status, get_trial_detail, list_experiment_statuses


def build_operations_status(
    *,
    competition: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    if competition:
        status = get_experiment_status(competition, db_path)
        return {
            "schema_version": "1.0",
            "mode": "single",
            "db_path": status["db_path"],
            "experiment": _with_operation_fields(status),
        }

    status = list_experiment_statuses(db_path)
    experiments = []
    for item in status["experiments"]:
        experiment_status = get_experiment_status(item["competition_id"], db_path)
        experiments.append(_with_operation_fields(experiment_status))
    return {
        "schema_version": "1.0",
        "mode": "list",
        "db_path": status["db_path"],
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def next_trial_id(trials: list[dict[str, Any]]) -> str:
    existing = {str(row.get("trial_id") or "") for row in trials}
    numbers = [_trial_number(str(row.get("trial_id") or "")) for row in trials]
    valid = [number for number in numbers if number is not None]
    next_number = max(valid) + 1 if valid else len(existing) + 1
    while True:
        candidate = f"trial_{next_number:03d}"
        if candidate not in existing:
            return candidate
        next_number += 1


def render_operations_status(status: dict[str, Any]) -> str:
    if status.get("mode") == "single":
        return render_single_operation(status.get("experiment") or {})
    lines = [
        "실험 운영 상태",
        f"- DB: {status.get('db_path')}",
        f"- 등록된 실험: {status.get('experiment_count', 0)}개",
        "",
    ]
    for index, experiment in enumerate(status.get("experiments", []), start=1):
        op = experiment.get("operation") or {}
        best = experiment.get("best_trial") or {}
        lines.append(
            f"{index}. {experiment.get('competition_id')} | {op.get('state')} | "
            f"trials={experiment.get('trial_count', 0)} | "
            f"best={best.get('trial_id') or '-'} / {best.get('local_score')} | "
            f"axis={_axis_label(op)} | next={op.get('next_trial_id') or '-'}"
        )
        lines.append(f"   다음 조치: {op.get('next_action_label')}")
    if not status.get("experiments"):
        lines.append("- 등록된 실험이 없습니다.")
    return "\n".join(lines)


def render_single_operation(experiment: dict[str, Any]) -> str:
    if not experiment.get("competition"):
        return "실험을 찾을 수 없습니다."

    competition = experiment["competition"]
    op = experiment.get("operation") or {}
    best = experiment.get("best_trial") or {}
    latest = op.get("latest_trial") or {}
    lines = [
        f"실험 운영 상세: {competition['competition_id']}",
        f"- 상태: {op.get('state')}",
        f"- 다음 조치: {op.get('next_action_label')}",
        f"- 다음 trial: {op.get('next_trial_id') or '-'}",
        f"- 주제: {competition.get('topic') or '-'}",
        f"- 플랫폼: {competition.get('platform') or '-'}",
        f"- 평가: {competition.get('metric') or '-'} / {competition.get('objective') or '-'}",
        f"- trial 수: {experiment.get('trial_count', 0)}",
        f"- 최신 trial: {latest.get('trial_id') or '-'} / status={latest.get('status') or '-'} / score={latest.get('local_score')}",
        f"- best trial: {best.get('trial_id') or '-'} / score={best.get('local_score')}",
        f"- 현재 개선축: {_axis_label(op)}",
        f"- 총 토큰: {experiment.get('total_tokens', 0)}",
        f"- 대기 요청: {len(experiment.get('pending_actions') or [])}개",
        "",
        "운영 명령",
    ]
    for command in op.get("commands", []):
        lines.append(f"- {command}")
    lines.extend(["", "최근 trial"])
    for trial in (experiment.get("trials") or [])[-5:]:
        axis = trial.get("primary_change_axis") or trial.get("change_axis") or "-"
        lines.append(
            f"- {trial.get('trial_id')}: status={trial.get('status') or '-'}, "
            f"score={trial.get('local_score')}, axis={axis}"
        )
    return "\n".join(lines)


def can_run_next_trial(operation: dict[str, Any]) -> bool:
    return operation.get("state") in {"ready_first_trial", "ready_next_trial"}


def _with_operation_fields(status: dict[str, Any]) -> dict[str, Any]:
    experiment = status.get("competition")
    if not experiment:
        return {
            "competition": None,
            "operation": {
                "state": "missing",
                "next_action_label": "실험 설정을 먼저 생성해야 합니다.",
                "next_trial_id": None,
                "commands": [],
            },
        }

    trials = list(status.get("trials") or [])
    latest = _latest_trial(trials)
    pending_actions = list(status.get("pending_actions") or [])
    next_id = next_trial_id(trials)
    competition_id = experiment["competition_id"]
    active_axis = latest.get("active_axis") or latest.get("primary_change_axis") or latest.get("change_axis")
    attempt_count = latest.get("axis_attempt_count")
    attempt_limit = latest.get("axis_attempt_limit")

    if pending_actions:
        state = "waiting_user"
        label = "사용자 승인/피드백 대기 요청을 먼저 확인하세요."
    elif not trials:
        state = "ready_first_trial"
        label = "첫 trial을 실행할 수 있습니다."
    elif str(latest.get("status") or "").lower() == "blocked":
        state = "blocked"
        label = "최신 trial이 중단되었습니다. show-trial로 원인을 확인하세요."
    elif str(experiment.get("status") or "").lower() in {"needs_data", "needs_review"}:
        state = str(experiment.get("status"))
        label = "실험 입력 자료 또는 데이터 준비가 필요합니다."
    else:
        state = "ready_next_trial"
        if _axis_refinement_in_progress(active_axis, attempt_count, attempt_limit):
            label = f"`{active_axis}` 개선축 안에서 다음 후보/파라미터 변형을 시도하세요."
        else:
            label = "다음 trial을 실행할 수 있습니다."

    operation = {
        "state": state,
        "next_action_label": label,
        "next_trial_id": next_id,
        "latest_trial": latest,
        "active_axis": active_axis,
        "axis_attempt_count": attempt_count,
        "axis_attempt_limit": attempt_limit,
        "commands": _commands_for_state(competition_id, state, next_id, latest),
    }
    result = dict(status)
    result["competition_id"] = competition_id
    result["operation"] = operation
    return result


def _commands_for_state(competition: str, state: str, next_id: str, latest: dict[str, Any]) -> list[str]:
    if state == "blocked" and latest:
        return [f"python -m research_agent.cli show-trial --competition {competition} --trial {latest.get('trial_id')} --sync"]
    if state == "waiting_user":
        return [f"python -m research_agent.cli status --competition {competition} --sync"]
    if state in {"ready_first_trial", "ready_next_trial"}:
        return [
            f"python -m research_agent.cli run-next-trial --competition {competition} --run-now --allow-api",
            f"python -m research_agent.cli run-next-trial --competition {competition} --dry-run",
        ]
    return [f"python -m research_agent.cli show-experiment --competition {competition} --sync"]


def _axis_label(operation: dict[str, Any]) -> str:
    axis = operation.get("active_axis") or "-"
    count = operation.get("axis_attempt_count")
    limit = operation.get("axis_attempt_limit")
    if count is None and limit is None:
        return str(axis)
    return f"{axis} ({count or 0}/{limit or '-'})"


def _axis_refinement_in_progress(axis: Any, count: Any, limit: Any) -> bool:
    if not axis:
        return False
    try:
        attempt_count = int(count or 0)
    except (TypeError, ValueError):
        attempt_count = 0
    try:
        attempt_limit = int(limit or 3)
    except (TypeError, ValueError):
        attempt_limit = 3
    return attempt_count < attempt_limit


def _latest_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return sorted(trials, key=lambda row: (_trial_number(str(row.get("trial_id") or "")) or -1, str(row.get("trial_id") or "")))[-1]


def _trial_number(trial_id: str) -> int | None:
    match = re.fullmatch(r"trial_(\d+)", trial_id)
    if not match:
        return None
    return int(match.group(1))
