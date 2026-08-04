from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..paths import trial_dir
from .state import ResearchGraphState


GRAPH_STATE_FILE = "graph_state.json"
NODE_EVENTS_FILE = "node_events.jsonl"


def wrap_graph_node(node_name: str, fn: Callable[[ResearchGraphState], dict[str, Any]]) -> Callable[[ResearchGraphState], dict[str, Any]]:
    def wrapped(state: ResearchGraphState) -> dict[str, Any]:
        record_node_event(state, node_name, "started")
        try:
            updates = fn(state)
        except Exception as exc:
            failed_state = _merge_state(state, {"status": "failed", "current_node": node_name, "failed_node": node_name})
            record_node_event(failed_state, node_name, "failed", message=str(exc))
            write_graph_state(failed_state)
            raise

        merged = _merge_state(
            state,
            {
                **updates,
                "current_node": node_name,
                "last_completed_node": node_name,
                "graph_state_file": _relative_graph_path(state, GRAPH_STATE_FILE),
                "node_events_file": _relative_graph_path(state, NODE_EVENTS_FILE),
            },
        )
        record_node_event(merged, node_name, "completed", status=merged.get("status"))
        write_graph_state(merged)
        return {
            **updates,
            "current_node": node_name,
            "last_completed_node": node_name,
            "graph_state_file": _relative_graph_path(state, GRAPH_STATE_FILE),
            "node_events_file": _relative_graph_path(state, NODE_EVENTS_FILE),
        }

    return wrapped


def record_node_event(
    state: ResearchGraphState,
    node_name: str,
    event: str,
    *,
    status: str | None = None,
    message: str | None = None,
) -> None:
    out_dir = _graph_out_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": _utc_now(),
        "competition": state.get("competition"),
        "trial_id": state.get("trial_id"),
        "node": node_name,
        "event": event,
        "status": status if status is not None else state.get("status", "running"),
        "steps": state.get("steps", []),
    }
    if message:
        payload["message"] = message
    with (out_dir / NODE_EVENTS_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def write_graph_state(state: ResearchGraphState) -> Path:
    out_dir = _graph_out_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / GRAPH_STATE_FILE
    payload = {
        "schema_version": "1.0",
        "updated_at": _utc_now(),
        "competition": state.get("competition"),
        "trial_id": state.get("trial_id"),
        "status": state.get("status", "running"),
        "current_node": state.get("current_node"),
        "last_completed_node": state.get("last_completed_node"),
        "failed_node": state.get("failed_node"),
        "steps": state.get("steps", []),
        "state": dict(state),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _merge_state(state: ResearchGraphState, updates: dict[str, Any]) -> ResearchGraphState:
    merged: ResearchGraphState = dict(state)
    merged.update(updates)
    return merged


def _graph_out_dir(state: ResearchGraphState) -> Path:
    return trial_dir(str(state["competition"]), str(state["trial_id"]))


def _relative_graph_path(state: ResearchGraphState, filename: str) -> str:
    return f"experiments/{state['competition']}/{state['trial_id']}/{filename}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
