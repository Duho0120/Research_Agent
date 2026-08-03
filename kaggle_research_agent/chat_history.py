from __future__ import annotations

from pathlib import Path
from typing import Any

from .experiment_qa import ResponseClient, answer_experiment_question
from .state_db import (
    create_chat_session,
    get_active_chat_session,
    list_chat_messages,
    list_chat_sessions,
    record_chat_exchange,
)


def chat_history_snapshot(
    competition: str,
    *,
    session_id: str | None = None,
    db_path: Path | None = None,
    message_limit: int = 100,
) -> dict[str, Any]:
    sessions = list_chat_sessions(competition, db_path)
    selected = _select_session(sessions, session_id)
    if session_id and selected is None:
        raise ValueError("Chat session does not belong to the selected experiment.")
    messages = (
        list_chat_messages(
            competition,
            str(selected["session_id"]),
            db_path,
            limit=message_limit,
        )
        if selected
        else []
    )
    return {
        "competition": competition,
        "active_session": selected,
        "sessions": sessions,
        "messages": messages,
    }


def start_new_chat(
    competition: str,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    session = create_chat_session(competition, db_path)
    return chat_history_snapshot(
        competition,
        session_id=str(session["session_id"]),
        db_path=db_path,
    )


def answer_chat_question(
    competition: str,
    trial_id: str | None,
    question: str,
    *,
    session_id: str | None = None,
    db_path: Path | None = None,
    client: ResponseClient | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    active = _resolve_session(competition, session_id, db_path)
    history = (
        list_chat_messages(
            competition,
            str(active["session_id"]),
            db_path,
            limit=6,
        )
        if active
        else []
    )
    result = answer_experiment_question(
        competition,
        trial_id,
        question,
        conversation=history,
        client=client,
        use_llm=use_llm,
    )
    rendered_answer = format_chat_answer(result)
    recorded = record_chat_exchange(
        competition,
        trial_id=trial_id,
        question=question,
        answer=rendered_answer,
        session_id=str(active["session_id"]) if active else None,
        assistant_metadata={
            "mode": result.get("mode"),
            "mode_label": result.get("mode_label"),
            "sources": list(result.get("sources") or []),
            "warning": result.get("warning"),
            "interaction": result.get("interaction"),
        },
        db_path=db_path,
    )
    snapshot = chat_history_snapshot(
        competition,
        session_id=str(recorded["session"]["session_id"]),
        db_path=db_path,
    )
    return {
        **result,
        "rendered_answer": rendered_answer,
        "session": recorded["session"],
        "history": snapshot,
    }


def format_chat_answer(result: dict[str, Any]) -> str:
    mode = result.get("mode_label") or result.get("mode") or "unknown"
    answer = str(result.get("answer") or "").strip()
    warning = str(result.get("warning") or "").strip()
    parts = [f"[{mode}]"]
    if warning:
        parts.append(warning)
    if answer:
        parts.append(answer)
    return "\n".join(parts)


def _resolve_session(
    competition: str,
    session_id: str | None,
    db_path: Path | None,
) -> dict[str, Any] | None:
    if session_id:
        sessions = list_chat_sessions(competition, db_path, limit=100)
        selected = _select_session(sessions, session_id)
        if not selected or str(selected.get("session_id")) != str(session_id):
            raise ValueError("Chat session does not belong to the selected experiment.")
        return selected
    return get_active_chat_session(competition, db_path)


def _select_session(
    sessions: list[dict[str, Any]],
    session_id: str | None,
) -> dict[str, Any] | None:
    if session_id:
        return next(
            (item for item in sessions if str(item.get("session_id")) == str(session_id)),
            None,
        )
    return next(
        (item for item in sessions if str(item.get("status")) == "active"),
        sessions[0] if sessions else None,
    )
