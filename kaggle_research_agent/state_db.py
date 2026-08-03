from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from . import paths
from .store import now_iso


SCHEMA_VERSION = "4"


def default_db_path() -> Path:
    return paths.memory_dir() / "research_agent.sqlite3"


def connect_state_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def state_db_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect_state_db(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_state_db(db_path: Path | None = None) -> Path:
    path = db_path or default_db_path()
    with state_db_connection(path) as connection:
        connection.executescript(_schema_sql())
        _ensure_schema_columns(connection)
        connection.execute(
            """
            INSERT INTO schema_meta (key, value, updated_at)
            VALUES ('schema_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (SCHEMA_VERSION, now_iso()),
        )
    return path


def upsert_competition(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        existing = _fetch_one(connection, "SELECT created_at FROM competitions WHERE competition_id = ?", [competition_id])
        created_at = existing.get("created_at") if existing else record.get("created_at") or timestamp
        values = {
            "competition_id": competition_id,
            "platform": record.get("platform"),
            "topic": record.get("topic"),
            "metric": record.get("metric"),
            "objective": record.get("objective"),
            "status": record.get("status"),
            "workspace_path": _string_or_none(record.get("workspace_path")),
            "created_at": created_at,
            "updated_at": record.get("updated_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO competitions (
                competition_id, platform, topic, metric, objective, status, workspace_path, created_at, updated_at
            )
            VALUES (
                :competition_id, :platform, :topic, :metric, :objective, :status, :workspace_path, :created_at, :updated_at
            )
            ON CONFLICT(competition_id) DO UPDATE SET
                platform = excluded.platform,
                topic = excluded.topic,
                metric = excluded.metric,
                objective = excluded.objective,
                status = excluded.status,
                workspace_path = excluded.workspace_path,
                updated_at = excluded.updated_at
            """,
            values,
        )
    return values


def upsert_trial(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        existing = _fetch_one(
            connection,
            "SELECT created_at FROM trials WHERE competition_id = ? AND trial_id = ?",
            [competition_id, trial_id],
        )
        created_at = existing.get("created_at") if existing else record.get("created_at") or timestamp
        values = {
            "competition_id": competition_id,
            "trial_id": trial_id,
            "status": record.get("status"),
            "source_trial_id": record.get("source_trial_id"),
            "recommended_base_trial": record.get("recommended_base_trial"),
            "plan_type": record.get("plan_type"),
            "plan_summary": record.get("plan_summary"),
            "primary_change_axis": record.get("primary_change_axis"),
            "created_at": created_at,
            "updated_at": record.get("updated_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO trials (
                competition_id, trial_id, status, source_trial_id, recommended_base_trial,
                plan_type, plan_summary, primary_change_axis, created_at, updated_at
            )
            VALUES (
                :competition_id, :trial_id, :status, :source_trial_id, :recommended_base_trial,
                :plan_type, :plan_summary, :primary_change_axis, :created_at, :updated_at
            )
            ON CONFLICT(competition_id, trial_id) DO UPDATE SET
                status = excluded.status,
                source_trial_id = excluded.source_trial_id,
                recommended_base_trial = excluded.recommended_base_trial,
                plan_type = excluded.plan_type,
                plan_summary = excluded.plan_summary,
                primary_change_axis = excluded.primary_change_axis,
                updated_at = excluded.updated_at
            """,
            values,
        )
    return values


def upsert_trial_score(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_trial(connection, competition_id, trial_id)
        values = {
            "competition_id": competition_id,
            "trial_id": trial_id,
            "metric": record.get("metric"),
            "objective": record.get("objective"),
            "local_score": _float_or_none(record.get("local_score")),
            "lb_score": _float_or_none(record.get("lb_score")),
            "local_status": record.get("local_status"),
            "lb_status": record.get("lb_status"),
            "is_best_local": _bool_int(record.get("is_best_local")),
            "is_best_lb": _bool_int(record.get("is_best_lb")),
            "updated_at": record.get("updated_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO trial_scores (
                competition_id, trial_id, metric, objective, local_score, lb_score,
                local_status, lb_status, is_best_local, is_best_lb, updated_at
            )
            VALUES (
                :competition_id, :trial_id, :metric, :objective, :local_score, :lb_score,
                :local_status, :lb_status, :is_best_local, :is_best_lb, :updated_at
            )
            ON CONFLICT(competition_id, trial_id) DO UPDATE SET
                metric = excluded.metric,
                objective = excluded.objective,
                local_score = excluded.local_score,
                lb_score = excluded.lb_score,
                local_status = excluded.local_status,
                lb_status = excluded.lb_status,
                is_best_local = excluded.is_best_local,
                is_best_lb = excluded.is_best_lb,
                updated_at = excluded.updated_at
            """,
            values,
        )
    return _decode_bool_fields(values, ["is_best_local", "is_best_lb"])


def upsert_trial_decision(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_trial(connection, competition_id, trial_id)
        values = {
            "competition_id": competition_id,
            "trial_id": trial_id,
            "decision": record.get("decision"),
            "change_axis": record.get("change_axis"),
            "active_axis": record.get("active_axis"),
            "axis_attempt_count": _int_or_none(record.get("axis_attempt_count")),
            "axis_attempt_limit": _int_or_none(record.get("axis_attempt_limit")),
            "recommended_base_trial": record.get("recommended_base_trial"),
            "rejected_axes_json": _json_text(record.get("rejected_axes", [])),
            "rejected_candidates_json": _json_text(record.get("rejected_candidates", [])),
            "planner_constraints_json": _json_text(record.get("planner_constraints", [])),
            "updated_at": record.get("updated_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO trial_decisions (
                competition_id, trial_id, decision, change_axis, active_axis,
                axis_attempt_count, axis_attempt_limit, recommended_base_trial,
                rejected_axes_json, rejected_candidates_json, planner_constraints_json, updated_at
            )
            VALUES (
                :competition_id, :trial_id, :decision, :change_axis, :active_axis,
                :axis_attempt_count, :axis_attempt_limit, :recommended_base_trial,
                :rejected_axes_json, :rejected_candidates_json, :planner_constraints_json, :updated_at
            )
            ON CONFLICT(competition_id, trial_id) DO UPDATE SET
                decision = excluded.decision,
                change_axis = excluded.change_axis,
                active_axis = excluded.active_axis,
                axis_attempt_count = excluded.axis_attempt_count,
                axis_attempt_limit = excluded.axis_attempt_limit,
                recommended_base_trial = excluded.recommended_base_trial,
                rejected_axes_json = excluded.rejected_axes_json,
                rejected_candidates_json = excluded.rejected_candidates_json,
                planner_constraints_json = excluded.planner_constraints_json,
                updated_at = excluded.updated_at
            """,
            values,
        )
    result = dict(values)
    result["rejected_axes"] = _json_value(result.pop("rejected_axes_json"))
    result["rejected_candidates"] = _json_value(result.pop("rejected_candidates_json"))
    result["planner_constraints"] = _json_value(result.pop("planner_constraints_json"))
    return result


def upsert_trial_artifact(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    artifact_type = _required(record, "artifact_type")
    path = _required(record, "path")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_trial(connection, competition_id, trial_id)
        values = {
            "competition_id": competition_id,
            "trial_id": trial_id,
            "artifact_type": artifact_type,
            "path": path,
            "is_user_facing": _bool_int(record.get("is_user_facing")),
            "created_at": record.get("created_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO trial_artifacts (
                competition_id, trial_id, artifact_type, path, is_user_facing, created_at
            )
            VALUES (
                :competition_id, :trial_id, :artifact_type, :path, :is_user_facing, :created_at
            )
            ON CONFLICT(competition_id, trial_id, artifact_type, path) DO UPDATE SET
                is_user_facing = excluded.is_user_facing
            """,
            values,
        )
    return _decode_bool_fields(values, ["is_user_facing"])


def record_token_usage(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        values = {
            "source_key": record.get("source_key"),
            "competition_id": competition_id,
            "trial_id": record.get("trial_id"),
            "provider": record.get("provider"),
            "model": record.get("model"),
            "call_type": record.get("call_type"),
            "input_tokens": _int_or_none(record.get("input_tokens")),
            "output_tokens": _int_or_none(record.get("output_tokens")),
            "total_tokens": _int_or_none(record.get("total_tokens")),
            "created_at": record.get("created_at") or timestamp,
        }
        cursor = connection.execute(
            """
            INSERT INTO token_usage (
                source_key, competition_id, trial_id, provider, model, call_type,
                input_tokens, output_tokens, total_tokens, created_at
            )
            VALUES (
                :source_key, :competition_id, :trial_id, :provider, :model, :call_type,
                :input_tokens, :output_tokens, :total_tokens, :created_at
            )
            ON CONFLICT(source_key) DO UPDATE SET
                competition_id = excluded.competition_id,
                trial_id = excluded.trial_id,
                provider = excluded.provider,
                model = excluded.model,
                call_type = excluded.call_type,
                input_tokens = excluded.input_tokens,
                output_tokens = excluded.output_tokens,
                total_tokens = excluded.total_tokens,
                created_at = excluded.created_at
            """,
            values,
        )
        values["usage_id"] = cursor.lastrowid
    return values


def upsert_submission(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_trial(connection, competition_id, trial_id)
        values = {
            "competition_id": competition_id,
            "trial_id": trial_id,
            "platform": record.get("platform"),
            "submission_file": _string_or_none(record.get("submission_file")) or "",
            "status": record.get("status"),
            "lb_score": _float_or_none(record.get("lb_score")),
            "rank": _int_or_none(record.get("rank")),
            "submitted_at": record.get("submitted_at"),
            "requires_user_approval": _bool_int(record.get("requires_user_approval")),
            "updated_at": record.get("updated_at") or timestamp,
        }
        connection.execute(
            """
            INSERT INTO submissions (
                competition_id, trial_id, platform, submission_file, status,
                lb_score, rank, submitted_at, requires_user_approval, updated_at
            )
            VALUES (
                :competition_id, :trial_id, :platform, :submission_file, :status,
                :lb_score, :rank, :submitted_at, :requires_user_approval, :updated_at
            )
            ON CONFLICT(competition_id, trial_id, submission_file) DO UPDATE SET
                platform = excluded.platform,
                status = excluded.status,
                lb_score = excluded.lb_score,
                rank = excluded.rank,
                submitted_at = excluded.submitted_at,
                requires_user_approval = excluded.requires_user_approval,
                updated_at = excluded.updated_at
            """,
            values,
        )
    return _decode_bool_fields(values, ["requires_user_approval"])


def create_pending_action(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    action_id = str(record.get("action_id") or uuid.uuid4())
    timestamp = now_iso()
    values = {
        "action_id": action_id,
        "competition_id": competition_id,
        "trial_id": record.get("trial_id"),
        "action_type": _required(record, "action_type"),
        "status": record.get("status") or "pending",
        "priority": _int_or_none(record.get("priority")) or 0,
        "message": record.get("message"),
        "payload_json": _json_text(record.get("payload", {})),
        "created_at": record.get("created_at") or timestamp,
        "resolved_at": record.get("resolved_at"),
    }
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        connection.execute(
            """
            INSERT INTO pending_actions (
                action_id, competition_id, trial_id, action_type, status, priority,
                message, payload_json, created_at, resolved_at
            )
            VALUES (
                :action_id, :competition_id, :trial_id, :action_type, :status, :priority,
                :message, :payload_json, :created_at, :resolved_at
            )
            ON CONFLICT(action_id) DO UPDATE SET
                status = excluded.status,
                priority = excluded.priority,
                message = excluded.message,
                payload_json = excluded.payload_json,
                resolved_at = excluded.resolved_at
            """,
            values,
        )
    result = dict(values)
    result["payload"] = _json_value(result.pop("payload_json"))
    return result


def resolve_pending_action(action_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        connection.execute(
            "UPDATE pending_actions SET status = 'resolved', resolved_at = ? WHERE action_id = ?",
            [timestamp, action_id],
        )
        row = _fetch_one(connection, "SELECT * FROM pending_actions WHERE action_id = ?", [action_id])
    return _decode_row(row) if row else None


def record_review_evidence(record: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    trial_id = _required(record, "trial_id")
    trigger = _required(record, "trigger")
    timestamp = now_iso()
    values = {
        "competition_id": competition_id,
        "trial_id": trial_id,
        "trial_number": _int_or_none(record.get("trial_number")) or 0,
        "trigger": trigger,
        "axis": record.get("axis"),
        "fingerprint": _required(record, "fingerprint"),
        "evidence_json": _json_text(record.get("evidence", {})),
        "created_at": record.get("created_at") or timestamp,
    }
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        connection.execute(
            """
            INSERT INTO review_evidence (
                competition_id, trial_id, trial_number, trigger, axis,
                fingerprint, evidence_json, created_at
            )
            VALUES (
                :competition_id, :trial_id, :trial_number, :trigger, :axis,
                :fingerprint, :evidence_json, :created_at
            )
            ON CONFLICT(competition_id, trial_id, trigger) DO UPDATE SET
                trial_number = excluded.trial_number,
                axis = excluded.axis,
                fingerprint = excluded.fingerprint,
                evidence_json = excluded.evidence_json
            """,
            values,
        )
    result = dict(values)
    result["evidence"] = _json_value(result.pop("evidence_json"))
    return result


def list_review_evidence(
    competition_id: str,
    *,
    minimum_trial_number: int = 0,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM review_evidence
            WHERE competition_id = ? AND trial_number >= ?
            ORDER BY trial_number, created_at
            """,
            [competition_id, minimum_trial_number],
        ).fetchall()
    return [_decode_row(dict(row)) for row in rows]


def get_review_policy_state(
    competition_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        row = _fetch_one(
            connection,
            "SELECT * FROM review_policy_state WHERE competition_id = ?",
            [competition_id],
        )
    return _decode_row(row) if row else None


def upsert_review_policy_state(
    record: dict[str, Any],
    db_path: Path | None = None,
) -> dict[str, Any]:
    competition_id = _required(record, "competition_id")
    values = {
        "competition_id": competition_id,
        "pending_action_id": record.get("pending_action_id"),
        "last_request_trial": _int_or_none(record.get("last_request_trial")),
        "last_fingerprint": record.get("last_fingerprint"),
        "cooldown_until_trial": _int_or_none(record.get("cooldown_until_trial")),
        "updated_at": record.get("updated_at") or now_iso(),
    }
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        connection.execute(
            """
            INSERT INTO review_policy_state (
                competition_id, pending_action_id, last_request_trial,
                last_fingerprint, cooldown_until_trial, updated_at
            )
            VALUES (
                :competition_id, :pending_action_id, :last_request_trial,
                :last_fingerprint, :cooldown_until_trial, :updated_at
            )
            ON CONFLICT(competition_id) DO UPDATE SET
                pending_action_id = excluded.pending_action_id,
                last_request_trial = excluded.last_request_trial,
                last_fingerprint = excluded.last_fingerprint,
                cooldown_until_trial = excluded.cooldown_until_trial,
                updated_at = excluded.updated_at
            """,
            values,
        )
    return values


def create_chat_session(
    competition_id: str,
    db_path: Path | None = None,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    competition_id = str(competition_id).strip()
    if not competition_id:
        raise ValueError("Missing required field: competition_id")
    timestamp = now_iso()
    session_id = f"chat_{uuid.uuid4().hex}"
    values = {
        "session_id": session_id,
        "competition_id": competition_id,
        "title": str(title or "새 대화").strip() or "새 대화",
        "status": "active",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        connection.execute(
            """
            UPDATE chat_sessions
            SET status = 'archived', updated_at = ?
            WHERE competition_id = ? AND status = 'active'
            """,
            (timestamp, competition_id),
        )
        connection.execute(
            """
            INSERT INTO chat_sessions (
                session_id, competition_id, title, status, created_at, updated_at
            )
            VALUES (
                :session_id, :competition_id, :title, :status, :created_at, :updated_at
            )
            """,
            values,
        )
    return values


def get_active_chat_session(
    competition_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        row = _fetch_one(
            connection,
            """
            SELECT *
            FROM chat_sessions
            WHERE competition_id = ? AND status = 'active'
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            [competition_id],
        )
    return _decode_row(row) if row else None


def list_chat_sessions(
    competition_id: str,
    db_path: Path | None = None,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT
                s.*,
                COUNT(m.message_id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.session_id
            WHERE s.competition_id = ?
            GROUP BY s.session_id
            ORDER BY
                CASE WHEN s.status = 'active' THEN 0 ELSE 1 END,
                s.updated_at DESC,
                s.created_at DESC
            LIMIT ?
            """,
            (competition_id, safe_limit),
        ).fetchall()
    return [_decode_row(dict(row)) for row in rows]


def list_chat_messages(
    competition_id: str,
    session_id: str,
    db_path: Path | None = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 500))
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT *
            FROM (
                SELECT *
                FROM chat_messages
                WHERE competition_id = ? AND session_id = ?
                ORDER BY message_id DESC
                LIMIT ?
            )
            ORDER BY message_id ASC
            """,
            (competition_id, session_id, safe_limit),
        ).fetchall()
    return [_decode_row(dict(row)) for row in rows]


def record_chat_exchange(
    competition_id: str,
    *,
    trial_id: str | None,
    question: str,
    answer: str,
    session_id: str | None = None,
    assistant_metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    competition_id = str(competition_id).strip()
    question = str(question).strip()
    answer = str(answer).strip()
    if not competition_id:
        raise ValueError("Missing required field: competition_id")
    if not question:
        raise ValueError("Missing required field: question")
    if not answer:
        raise ValueError("Missing required field: answer")

    timestamp = now_iso()
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        _ensure_competition(connection, competition_id)
        session = None
        if session_id:
            session = _fetch_one(
                connection,
                """
                SELECT * FROM chat_sessions
                WHERE session_id = ? AND competition_id = ?
                """,
                [session_id, competition_id],
            )
            if session is None:
                raise ValueError("Chat session does not belong to the selected experiment.")
        else:
            session = _fetch_one(
                connection,
                """
                SELECT * FROM chat_sessions
                WHERE competition_id = ? AND status = 'active'
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                [competition_id],
            )

        if session is None:
            session_id = f"chat_{uuid.uuid4().hex}"
            session = {
                "session_id": session_id,
                "competition_id": competition_id,
                "title": "새 대화",
                "status": "active",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    session_id, competition_id, title, status, created_at, updated_at
                )
                VALUES (
                    :session_id, :competition_id, :title, :status, :created_at, :updated_at
                )
                """,
                session,
            )
        else:
            session_id = str(session["session_id"])

        connection.execute(
            """
            UPDATE chat_sessions
            SET status = 'archived', updated_at = ?
            WHERE competition_id = ? AND session_id <> ? AND status = 'active'
            """,
            (timestamp, competition_id, session_id),
        )
        title = _chat_title(question)
        connection.execute(
            """
            UPDATE chat_sessions
            SET
                status = 'active',
                title = CASE WHEN title IS NULL OR title = '' OR title = '새 대화' THEN ? ELSE title END,
                updated_at = ?
            WHERE session_id = ?
            """,
            (title, timestamp, session_id),
        )
        user_cursor = connection.execute(
            """
            INSERT INTO chat_messages (
                session_id, competition_id, trial_id, role, content, metadata_json, created_at
            )
            VALUES (?, ?, ?, 'user', ?, '{}', ?)
            """,
            (session_id, competition_id, trial_id, question, timestamp),
        )
        assistant_cursor = connection.execute(
            """
            INSERT INTO chat_messages (
                session_id, competition_id, trial_id, role, content, metadata_json, created_at
            )
            VALUES (?, ?, ?, 'assistant', ?, ?, ?)
            """,
            (
                session_id,
                competition_id,
                trial_id,
                answer,
                _json_text(assistant_metadata or {}),
                timestamp,
            ),
        )
        saved_session = _fetch_one(
            connection,
            "SELECT * FROM chat_sessions WHERE session_id = ?",
            [session_id],
        )
    return {
        "session": _decode_row(saved_session or session),
        "user_message_id": int(user_cursor.lastrowid),
        "assistant_message_id": int(assistant_cursor.lastrowid),
    }


def list_competitions(db_path: Path | None = None) -> list[dict[str, Any]]:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute("SELECT * FROM competitions ORDER BY updated_at DESC, competition_id").fetchall()
    return [_decode_row(dict(row)) for row in rows]


def delete_competition(competition_id: str, db_path: Path | None = None) -> None:
    """Delete a competition row; ON DELETE CASCADE removes all its trials,
    scores, decisions, artifacts, submissions, pending actions, review
    evidence, and chat history in the same statement.
    """
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        connection.execute("DELETE FROM competitions WHERE competition_id = ?", [competition_id])


def list_trials(competition_id: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            "SELECT * FROM trials WHERE competition_id = ? ORDER BY trial_id",
            [competition_id],
        ).fetchall()
    return [_decode_row(dict(row)) for row in rows]


def get_trial_summary(competition_id: str, trial_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        row = _fetch_one(
            connection,
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
                d.axis_attempt_limit,
                d.recommended_base_trial AS decision_recommended_base_trial
            FROM trials t
            LEFT JOIN trial_scores s
                ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
            LEFT JOIN trial_decisions d
                ON d.competition_id = t.competition_id AND d.trial_id = t.trial_id
            WHERE t.competition_id = ? AND t.trial_id = ?
            """,
            [competition_id, trial_id],
        )
    return _decode_row(row) if row else None


def get_best_trial(competition_id: str, db_path: Path | None = None, *, prefer_lb: bool = False) -> dict[str, Any] | None:
    score_column = "lb_score" if prefer_lb else "local_score"
    best_column = "is_best_lb" if prefer_lb else "is_best_local"
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            f"""
            SELECT t.*, s.metric, s.objective, s.local_score, s.lb_score, s.is_best_local, s.is_best_lb
            FROM trials t
            JOIN trial_scores s ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
            WHERE t.competition_id = ?
              AND s.{score_column} IS NOT NULL
            ORDER BY s.{best_column} DESC, t.trial_id DESC
            """,
            [competition_id],
        ).fetchall()
    decoded = [_decode_row(dict(row)) for row in rows]
    if not decoded:
        return None
    objective = str((decoded[0].get("objective") or "maximize")).lower()
    reverse = objective != "minimize"
    return sorted(
        decoded,
        key=lambda row: (
            row.get(score_column),
            bool(row.get(best_column)),
            str(row.get("trial_id") or ""),
        ),
        reverse=reverse,
    )[0]


def list_pending_actions(
    competition_id: str | None = None, db_path: Path | None = None, *, status: str = "pending"
) -> list[dict[str, Any]]:
    clauses = ["status = ?"]
    params: list[Any] = [status]
    if competition_id:
        clauses.append("competition_id = ?")
        params.append(competition_id)
    with state_db_connection(db_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            f"SELECT * FROM pending_actions WHERE {' AND '.join(clauses)} ORDER BY priority DESC, created_at ASC",
            params,
        ).fetchall()
    return [_decode_row(dict(row)) for row in rows]


def _schema_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS competitions (
        competition_id TEXT PRIMARY KEY,
        platform TEXT,
        topic TEXT,
        metric TEXT,
        objective TEXT,
        status TEXT,
        workspace_path TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS trials (
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        status TEXT,
        source_trial_id TEXT,
        recommended_base_trial TEXT,
        plan_type TEXT,
        plan_summary TEXT,
        primary_change_axis TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (competition_id, trial_id),
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS trial_scores (
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        metric TEXT,
        objective TEXT,
        local_score REAL,
        lb_score REAL,
        local_status TEXT,
        lb_status TEXT,
        is_best_local INTEGER NOT NULL DEFAULT 0,
        is_best_lb INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (competition_id, trial_id),
        FOREIGN KEY (competition_id, trial_id) REFERENCES trials(competition_id, trial_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS trial_artifacts (
        artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        path TEXT NOT NULL,
        is_user_facing INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE (competition_id, trial_id, artifact_type, path),
        FOREIGN KEY (competition_id, trial_id) REFERENCES trials(competition_id, trial_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS trial_decisions (
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        decision TEXT,
        change_axis TEXT,
        active_axis TEXT,
        axis_attempt_count INTEGER,
        axis_attempt_limit INTEGER,
        recommended_base_trial TEXT,
        rejected_axes_json TEXT NOT NULL DEFAULT '[]',
        rejected_candidates_json TEXT NOT NULL DEFAULT '[]',
        planner_constraints_json TEXT NOT NULL DEFAULT '[]',
        updated_at TEXT NOT NULL,
        PRIMARY KEY (competition_id, trial_id),
        FOREIGN KEY (competition_id, trial_id) REFERENCES trials(competition_id, trial_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS token_usage (
        usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_key TEXT UNIQUE,
        competition_id TEXT NOT NULL,
        trial_id TEXT,
        provider TEXT,
        model TEXT,
        call_type TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS submissions (
        submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        platform TEXT,
        submission_file TEXT NOT NULL DEFAULT '',
        status TEXT,
        lb_score REAL,
        rank INTEGER,
        submitted_at TEXT,
        requires_user_approval INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL,
        UNIQUE (competition_id, trial_id, submission_file),
        FOREIGN KEY (competition_id, trial_id) REFERENCES trials(competition_id, trial_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS pending_actions (
        action_id TEXT PRIMARY KEY,
        competition_id TEXT NOT NULL,
        trial_id TEXT,
        action_type TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS review_evidence (
        evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        competition_id TEXT NOT NULL,
        trial_id TEXT NOT NULL,
        trial_number INTEGER NOT NULL,
        trigger TEXT NOT NULL,
        axis TEXT,
        fingerprint TEXT NOT NULL,
        evidence_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE (competition_id, trial_id, trigger),
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS review_policy_state (
        competition_id TEXT PRIMARY KEY,
        pending_action_id TEXT,
        last_request_trial INTEGER,
        last_fingerprint TEXT,
        cooldown_until_trial INTEGER,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        competition_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '새 대화',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        competition_id TEXT NOT NULL,
        trial_id TEXT,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        FOREIGN KEY (competition_id) REFERENCES competitions(competition_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_trials_competition_status
        ON trials(competition_id, status);
    CREATE INDEX IF NOT EXISTS idx_scores_competition_local
        ON trial_scores(competition_id, local_score);
    CREATE INDEX IF NOT EXISTS idx_scores_competition_lb
        ON trial_scores(competition_id, lb_score);
    CREATE INDEX IF NOT EXISTS idx_pending_competition_status
        ON pending_actions(competition_id, status, priority);
    CREATE INDEX IF NOT EXISTS idx_review_evidence_competition_trial
        ON review_evidence(competition_id, trial_number, trigger);
    CREATE INDEX IF NOT EXISTS idx_chat_sessions_competition_updated
        ON chat_sessions(competition_id, updated_at);
    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_message
        ON chat_messages(session_id, message_id);
    """


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(_schema_sql())
    _ensure_schema_columns(connection)
    connection.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES ('schema_version', ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (SCHEMA_VERSION, now_iso()),
    )


def _ensure_schema_columns(connection: sqlite3.Connection) -> None:
    trial_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(trials)").fetchall()
    }
    if "plan_summary" not in trial_columns:
        connection.execute("ALTER TABLE trials ADD COLUMN plan_summary TEXT")


def _ensure_competition(connection: sqlite3.Connection, competition_id: str) -> None:
    timestamp = now_iso()
    connection.execute(
        """
        INSERT INTO competitions (competition_id, status, created_at, updated_at)
        VALUES (?, 'discovered', ?, ?)
        ON CONFLICT(competition_id) DO NOTHING
        """,
        (competition_id, timestamp, timestamp),
    )


def _ensure_trial(connection: sqlite3.Connection, competition_id: str, trial_id: str) -> None:
    _ensure_competition(connection, competition_id)
    timestamp = now_iso()
    connection.execute(
        """
        INSERT INTO trials (competition_id, trial_id, status, created_at, updated_at)
        VALUES (?, ?, 'discovered', ?, ?)
        ON CONFLICT(competition_id, trial_id) DO NOTHING
        """,
        (competition_id, trial_id, timestamp, timestamp),
    )


def _fetch_one(connection: sqlite3.Connection, query: str, params: list[Any]) -> dict[str, Any] | None:
    row = connection.execute(query, params).fetchone()
    return dict(row) if row else None


def _required(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required field: {key}")
    return str(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_int(value: Any) -> int:
    return 1 if bool(value) else 0


def _json_text(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _decode_bool_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    decoded = dict(row)
    for field in fields:
        if field in decoded:
            decoded[field] = bool(decoded[field])
    return decoded


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    for field in ["is_best_local", "is_best_lb", "is_user_facing", "requires_user_approval"]:
        if field in decoded and decoded[field] is not None:
            decoded[field] = bool(decoded[field])
    for field in [
        "payload_json",
        "evidence_json",
        "rejected_axes_json",
        "rejected_candidates_json",
        "planner_constraints_json",
        "metadata_json",
    ]:
        if field in decoded:
            decoded[field.removesuffix("_json")] = _json_value(decoded.pop(field))
    return decoded


def _chat_title(question: str) -> str:
    compact = " ".join(str(question).split())
    return compact if len(compact) <= 48 else compact[:47].rstrip() + "…"
