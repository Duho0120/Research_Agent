from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state_db import (
    get_best_trial,
    get_trial_summary,
    initialize_state_db,
    list_pending_actions,
    state_db_connection,
)


def list_experiment_statuses(db_path: Path | None = None) -> dict[str, Any]:
    target_db = initialize_state_db(db_path)
    with state_db_connection(target_db) as connection:
        rows = connection.execute(
            """
            SELECT
                c.*,
                (
                    SELECT COUNT(*)
                    FROM trials t
                    WHERE t.competition_id = c.competition_id
                ) AS trial_count,
                (
                    SELECT SUM(COALESCE(u.total_tokens, 0))
                    FROM token_usage u
                    WHERE u.competition_id = c.competition_id
                ) AS total_tokens,
                (
                    SELECT COUNT(*)
                    FROM pending_actions p
                    WHERE p.competition_id = c.competition_id
                      AND p.status = 'pending'
                ) AS pending_action_count
            FROM competitions c
            ORDER BY c.updated_at DESC, c.competition_id
            """
        ).fetchall()

    experiments = []
    for row in rows:
        item = _decode_row(dict(row))
        best = get_best_trial(item["competition_id"], target_db)
        item["best_trial"] = _compact_trial(best)
        experiments.append(item)
    return {
        "db_path": str(target_db),
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def get_experiment_status(competition_id: str, db_path: Path | None = None) -> dict[str, Any]:
    target_db = initialize_state_db(db_path)
    with state_db_connection(target_db) as connection:
        competition = connection.execute(
            "SELECT * FROM competitions WHERE competition_id = ?",
            [competition_id],
        ).fetchone()
        trials = connection.execute(
            """
            SELECT
                t.*,
                s.metric,
                s.objective,
                s.local_score,
                s.lb_score,
                s.local_status,
                s.lb_status,
                s.is_best_local,
                s.is_best_lb,
                d.decision,
                d.change_axis,
                d.active_axis,
                d.axis_attempt_count,
                d.axis_attempt_limit
            FROM trials t
            LEFT JOIN trial_scores s
                ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
            LEFT JOIN trial_decisions d
                ON d.competition_id = t.competition_id AND d.trial_id = t.trial_id
            WHERE t.competition_id = ?
            ORDER BY t.trial_id
            """,
            [competition_id],
        ).fetchall()
        artifact_count = _count_rows(connection, "trial_artifacts", competition_id)
        submission_count = _count_rows(connection, "submissions", competition_id)
        token_total = connection.execute(
            "SELECT SUM(COALESCE(total_tokens, 0)) FROM token_usage WHERE competition_id = ?",
            [competition_id],
        ).fetchone()[0]

    return {
        "db_path": str(target_db),
        "competition": _decode_row(dict(competition)) if competition else None,
        "best_trial": _compact_trial(get_best_trial(competition_id, target_db)),
        "trial_count": len(trials),
        "trials": [_compact_trial(_decode_row(dict(row))) for row in trials],
        "pending_actions": list_pending_actions(competition_id, target_db),
        "artifact_count": artifact_count,
        "submission_count": submission_count,
        "total_tokens": int(token_total or 0),
    }


def get_trial_detail(competition_id: str, trial_id: str, db_path: Path | None = None) -> dict[str, Any]:
    target_db = initialize_state_db(db_path)
    summary = get_trial_summary(competition_id, trial_id, target_db)
    with state_db_connection(target_db) as connection:
        artifacts = connection.execute(
            """
            SELECT artifact_type, path, is_user_facing, created_at
            FROM trial_artifacts
            WHERE competition_id = ? AND trial_id = ?
            ORDER BY is_user_facing DESC, artifact_type, path
            """,
            [competition_id, trial_id],
        ).fetchall()
        token_usage = connection.execute(
            """
            SELECT provider, model, call_type, input_tokens, output_tokens, total_tokens, created_at
            FROM token_usage
            WHERE competition_id = ? AND trial_id = ?
            ORDER BY created_at, usage_id
            """,
            [competition_id, trial_id],
        ).fetchall()
        submissions = connection.execute(
            """
            SELECT platform, submission_file, status, lb_score, rank, submitted_at, requires_user_approval, updated_at
            FROM submissions
            WHERE competition_id = ? AND trial_id = ?
            ORDER BY updated_at DESC, submission_id DESC
            """,
            [competition_id, trial_id],
        ).fetchall()

    usage_rows = [_decode_row(dict(row)) for row in token_usage]
    return {
        "db_path": str(target_db),
        "summary": summary,
        "artifacts": [_decode_row(dict(row)) for row in artifacts],
        "user_artifacts": _sort_user_artifacts(
            [_decode_row(dict(row)) for row in artifacts if row["is_user_facing"]]
        ),
        "token_usage": usage_rows,
        "token_total": sum(int(row.get("total_tokens") or 0) for row in usage_rows),
        "submissions": [_decode_row(dict(row)) for row in submissions],
        "pending_actions": [
            action
            for action in list_pending_actions(competition_id, target_db)
            if action.get("trial_id") in {None, trial_id}
        ],
    }
def render_experiment_statuses(status: dict[str, Any]) -> str:
    lines = [
        "실험 상태 요약",
        f"- DB: {status['db_path']}",
        f"- 등록된 실험: {status['experiment_count']}개",
    ]
    for index, item in enumerate(status["experiments"], start=1):
        best = item.get("best_trial") or {}
        best_text = "-"
        if best:
            best_text = f"{best.get('trial_id')} / {best.get('local_score')}"
        lines.append(
            f"{index}. {item['competition_id']} | status={item.get('status') or '-'} "
            f"| trials={item.get('trial_count') or 0} | best={best_text} "
            f"| pending={item.get('pending_action_count') or 0}"
        )
    return "\n".join(lines)


def render_experiment_status(status: dict[str, Any]) -> str:
    competition = status.get("competition")
    if not competition:
        return "실험을 찾을 수 없습니다."
    best = status.get("best_trial") or {}
    lines = [
        f"실험 현황: {competition['competition_id']}",
        f"- 주제: {competition.get('topic') or '-'}",
        f"- 플랫폼: {competition.get('platform') or '-'}",
        f"- 평가: {competition.get('metric') or '-'} / {competition.get('objective') or '-'}",
        f"- Trial 수: {status['trial_count']}",
        f"- 현재 best: {best.get('trial_id') or '-'} / {best.get('local_score')}",
        f"- 총 토큰: {status['total_tokens']}",
        f"- 산출물 기록: {status['artifact_count']}개",
        f"- 제출 기록: {status['submission_count']}개",
        f"- 대기 요청: {len(status['pending_actions'])}개",
        "",
        "Trial 목록",
    ]
    for trial in status["trials"]:
        lines.append(
            f"- {trial['trial_id']}: status={trial.get('status') or '-'}, "
            f"score={trial.get('local_score')}, axis={trial.get('primary_change_axis') or trial.get('change_axis') or '-'}"
        )
    return "\n".join(lines)


def render_trial_detail(detail: dict[str, Any]) -> str:
    summary = detail.get("summary")
    if not summary:
        return "Trial을 찾을 수 없습니다."
    lines = [
        f"Trial 상세: {summary['competition_id']} / {summary['trial_id']}",
        f"- 상태: {summary.get('status') or '-'}",
        f"- 점수: {summary.get('metric') or '-'} = {summary.get('local_score')}",
        f"- LB 점수: {summary.get('lb_score')}",
        f"- 개선축: {summary.get('primary_change_axis') or summary.get('change_axis') or '-'}",
        f"- 판단: {summary.get('decision') or '-'}",
        f"- 기준 trial: {summary.get('recommended_base_trial') or summary.get('decision_recommended_base_trial') or '-'}",
        f"- 토큰 합계: {detail['token_total']}",
        "",
        "사용자가 확인할 파일",
    ]
    for artifact in detail["user_artifacts"]:
        lines.append(f"- {artifact['artifact_type']}: {artifact['path']}")
    if not detail["user_artifacts"]:
        lines.append("- 없음")
    lines.append("")
    lines.append("제출 기록")
    for submission in detail["submissions"]:
        lines.append(
            f"- {submission.get('status') or '-'}: {submission.get('submission_file') or '-'} "
            f"lb={submission.get('lb_score')} rank={submission.get('rank')}"
        )
    if not detail["submissions"]:
        lines.append("- 없음")
    return "\n".join(lines)


def to_pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _compact_trial(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "competition_id": row.get("competition_id"),
        "trial_id": row.get("trial_id"),
        "status": row.get("status"),
        "metric": row.get("metric"),
        "objective": row.get("objective"),
        "local_score": row.get("local_score"),
        "lb_score": row.get("lb_score"),
        "is_best_local": bool(row.get("is_best_local")),
        "is_best_lb": bool(row.get("is_best_lb")),
        "primary_change_axis": row.get("primary_change_axis"),
        "change_axis": row.get("change_axis"),
        "active_axis": row.get("active_axis"),
        "axis_attempt_count": row.get("axis_attempt_count"),
        "axis_attempt_limit": row.get("axis_attempt_limit"),
        "decision": row.get("decision"),
    }


def _sort_user_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "summary_card_ko": 0,
        "readme_ko": 1,
        "plan_ko": 2,
        "pipeline_structure_ko": 3,
        "code_pipeline_ko": 4,
        "result_ko": 5,
        "submission_ko": 6,
        "decision_ko": 7,
        "paths_ko": 8,
        "user_code": 9,
    }
    return sorted(artifacts, key=lambda item: (order.get(str(item.get("artifact_type")), 99), item.get("path") or ""))


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for field in ["is_best_local", "is_best_lb", "is_user_facing", "requires_user_approval"]:
        if field in decoded and decoded[field] is not None:
            decoded[field] = bool(decoded[field])
    return decoded


def _count_rows(connection: Any, table: str, competition_id: str) -> int:
    if table not in {"trial_artifacts", "submissions"}:
        raise ValueError(f"Unsupported state query table: {table}")
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE competition_id = ?",
            [competition_id],
        ).fetchone()[0]
    )
