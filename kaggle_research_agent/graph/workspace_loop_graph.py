from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Literal, TypedDict


PLANNED_STATUSES = {
    "planned",
    "planned_with_deferred_review",
    "planned_with_pending_review",
}
HUMAN_REVIEW_WAIT_STATUSES = {"blocked_human_review", "awaiting_human_review"}


class WorkspaceLoopGraphState(TypedDict, total=False):
    competition: str
    selected_trials: list[str]
    trial_index: int
    settings: dict[str, Any]
    status: str
    phase: str
    current_trial: str | None
    successor_trial: str | None
    last_completed_trial: str | None
    next_trial: str | None
    rows: list[dict[str, Any]]
    steps: list[str]
    plan_result: dict[str, Any]
    trial_result: dict[str, Any]
    next_plan_result: dict[str, Any]
    failure_stage: str
    error: str | None
    pause_requested: bool
    graph_runtime: str
    graph_thread_id: str


@dataclass(frozen=True)
class WorkspaceLoopCallbacks:
    prepare_trial: Callable[[Any, str], dict[str, Any]]
    run_trial: Callable[[Any, str], dict[str, Any]]
    plan_successor: Callable[[Any, str, str], dict[str, Any]]
    sync_state: Callable[[str], Any]
    save_state: Callable[..., dict[str, Any]]
    pause_requested: Callable[[], bool]
    human_review_resolved: Callable[[str, str], bool] = lambda _competition, _trial_id: True


def build_workspace_loop_graph(callbacks: WorkspaceLoopCallbacks, *, checkpointer: Any | None = None):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is required for the workspace research loop.") from exc

    def initialize_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        selected = list(state["selected_trials"])
        index = int(state.get("trial_index", 0))
        trial_id = selected[index]
        callbacks.save_state(
            competition=state["competition"],
            status="running",
            phase="planning",
            current_trial=None,
            next_trial=trial_id,
            graph_runtime="langgraph",
            graph_thread_id=state.get("graph_thread_id"),
        )
        return {
            "status": "running",
            "phase": "planning",
            "current_trial": trial_id,
            "next_trial": trial_id,
            "successor_trial": _next_trial_id(trial_id),
            "steps": _append_step(state, f"initialize:{trial_id}"),
        }

    def prepare_trial_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        trial_id = str(state["current_trial"])
        settings = dict(state.get("settings") or {})
        if not settings.get("code_writer", False):
            result = {"status": "planned", "trial_id": trial_id, "skipped": True}
        else:
            result = callbacks.prepare_trial(_namespace(settings), trial_id)
        if result.get("status") not in PLANNED_STATUSES:
            return {
                "plan_result": _result_summary(result),
                "failure_stage": "planning",
                "error": str(result.get("status") or "planning_failed"),
                "steps": _append_step(state, f"prepare_failed:{trial_id}"),
            }
        return {
            "plan_result": _result_summary(result),
            "phase": "planned",
            "steps": _append_step(state, f"prepared:{trial_id}"),
        }

    def route_after_prepare(state: WorkspaceLoopGraphState) -> Literal["execute", "fail"]:
        return "fail" if state.get("failure_stage") else "execute"

    def execute_trial_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        trial_id = str(state["current_trial"])
        result = callbacks.run_trial(_namespace(state.get("settings") or {}), trial_id)
        rows = [*state.get("rows", []), {"trial_id": trial_id, "status": result.get("status")}]
        if result.get("status") != "completed":
            return {
                "trial_result": _result_summary(result),
                "rows": rows,
                "failure_stage": "blocked",
                "error": str(result.get("status") or "trial_failed"),
                "steps": _append_step(state, f"execute_failed:{trial_id}"),
            }
        return {
            "trial_result": _result_summary(result),
            "rows": rows,
            "last_completed_trial": trial_id,
            "phase": "organizing_artifacts",
            "steps": _append_step(state, f"executed:{trial_id}"),
        }

    def route_after_execute(state: WorkspaceLoopGraphState) -> Literal["plan_successor", "fail"]:
        return "fail" if state.get("failure_stage") else "plan_successor"

    def plan_successor_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        trial_id = str(state["current_trial"])
        successor = str(state["successor_trial"])
        result = callbacks.plan_successor(_namespace(state.get("settings") or {}), trial_id, successor)
        if result.get("status") in HUMAN_REVIEW_WAIT_STATUSES:
            return {
                "next_plan_result": _result_summary(result),
                "status": "awaiting_human_review",
                "phase": "awaiting_human_review",
                "next_trial": successor,
                "error": None,
                "steps": _append_step(state, f"awaiting_human_review:{trial_id}"),
            }
        if result.get("status") not in PLANNED_STATUSES:
            return {
                "next_plan_result": _result_summary(result),
                "failure_stage": "planning_next",
                "error": str(result.get("status") or "planning_next_failed"),
                "next_trial": successor,
                "steps": _append_step(state, f"successor_plan_failed:{successor}"),
            }
        return {
            "next_plan_result": _result_summary(result),
            "phase": "planned",
            "next_trial": successor,
            "steps": _append_step(state, f"successor_planned:{successor}"),
        }

    def route_after_successor_plan(state: WorkspaceLoopGraphState) -> Literal["sync", "await_review", "fail"]:
        if state.get("phase") == "awaiting_human_review":
            return "await_review"
        return "fail" if state.get("failure_stage") else "sync"

    def sync_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        callbacks.sync_state(state["competition"])
        callbacks.save_state(
            status="running",
            phase="planned",
            current_trial=None,
            last_completed_trial=state.get("last_completed_trial"),
            next_trial=state.get("next_trial"),
            error=None,
            graph_runtime="langgraph",
            graph_thread_id=state.get("graph_thread_id"),
        )
        return {"phase": "planned", "steps": _append_step(state, "state_synced")}

    def pause_gate_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        requested = callbacks.pause_requested()
        return {
            "pause_requested": requested,
            "steps": _append_step(state, "pause_requested" if requested else "continue_requested"),
        }

    def route_after_pause(state: WorkspaceLoopGraphState) -> Literal["pause", "advance"]:
        return "pause" if state.get("pause_requested") else "advance"

    def advance_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        next_index = int(state.get("trial_index", 0)) + 1
        return {
            "trial_index": next_index,
            "current_trial": None,
            "failure_stage": "",
            "error": None,
            "steps": _append_step(state, "trial_advanced"),
        }

    def route_after_advance(state: WorkspaceLoopGraphState) -> Literal["initialize", "complete"]:
        return "initialize" if int(state["trial_index"]) < len(state["selected_trials"]) else "complete"

    def pause_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        callbacks.save_state(
            status="paused",
            phase="planned",
            current_trial=None,
            last_completed_trial=state.get("last_completed_trial"),
            next_trial=state.get("next_trial"),
            pid=None,
            graph_runtime="langgraph",
            graph_thread_id=state.get("graph_thread_id"),
        )
        return {"status": "paused", "current_trial": None, "steps": _append_step(state, "paused")}

    def await_human_review_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        from langgraph.types import interrupt

        callbacks.sync_state(state["competition"])
        callbacks.save_state(
            status="awaiting_human_review",
            phase="awaiting_human_review",
            current_trial=None,
            last_completed_trial=state.get("last_completed_trial"),
            next_trial=state.get("next_trial"),
            error=None,
            pid=None,
            graph_runtime="langgraph",
            graph_thread_id=state.get("graph_thread_id"),
        )
        resume_payload = interrupt(
            {
                "type": "human_review",
                "competition": state["competition"],
                "source_trial_id": state.get("last_completed_trial"),
                "next_trial_id": state.get("next_trial"),
            }
        )
        source_trial = str(state.get("last_completed_trial") or state.get("current_trial") or "")
        if not callbacks.human_review_resolved(state["competition"], source_trial):
            interrupt(
                {
                    "type": "human_review",
                    "status": "feedback_not_applied",
                    "competition": state["competition"],
                    "source_trial_id": source_trial,
                    "resume_payload": resume_payload,
                }
            )
        return {
            "status": "running",
            "phase": "planning_next",
            "steps": _append_step(state, "human_review_resumed"),
        }

    def complete_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        next_trial = _next_trial_id(state["selected_trials"][-1])
        callbacks.save_state(
            status="completed",
            phase="planned",
            current_trial=None,
            next_trial=next_trial,
            pause_requested=False,
            error=None,
            pid=None,
            graph_runtime="langgraph",
            graph_thread_id=state.get("graph_thread_id"),
        )
        return {
            "status": "completed",
            "phase": "planned",
            "current_trial": None,
            "next_trial": next_trial,
            "steps": _append_step(state, "completed"),
        }

    def fail_node(state: WorkspaceLoopGraphState) -> dict[str, Any]:
        stage = str(state.get("failure_stage") or "blocked")
        callbacks.sync_state(state["competition"])
        updates: dict[str, Any] = {
            "status": "failed",
            "phase": stage,
            "current_trial": None,
            "next_trial": state.get("next_trial") or state.get("current_trial"),
            "error": state.get("error"),
            "pid": None,
            "graph_runtime": "langgraph",
            "graph_thread_id": state.get("graph_thread_id"),
        }
        if stage == "planning_next":
            updates["last_completed_trial"] = state.get("last_completed_trial")
            updates["next_trial"] = state.get("successor_trial")
        callbacks.save_state(**updates)
        return {
            "status": "failed",
            "phase": stage,
            "current_trial": None,
            "next_trial": updates.get("next_trial"),
            "steps": _append_step(state, f"failed:{stage}"),
        }

    graph = StateGraph(WorkspaceLoopGraphState)
    graph.add_node("initialize_trial", initialize_node)
    graph.add_node("prepare_trial", prepare_trial_node)
    graph.add_node("execute_trial", execute_trial_node)
    graph.add_node("plan_successor", plan_successor_node)
    graph.add_node("sync_state", sync_node)
    graph.add_node("pause_gate", pause_gate_node)
    graph.add_node("advance_trial", advance_node)
    graph.add_node("pause", pause_node)
    graph.add_node("await_human_review", await_human_review_node)
    graph.add_node("complete", complete_node)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "initialize_trial")
    graph.add_edge("initialize_trial", "prepare_trial")
    graph.add_conditional_edges("prepare_trial", route_after_prepare, {"execute": "execute_trial", "fail": "fail"})
    graph.add_conditional_edges(
        "execute_trial",
        route_after_execute,
        {"plan_successor": "plan_successor", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "plan_successor",
        route_after_successor_plan,
        {"sync": "sync_state", "await_review": "await_human_review", "fail": "fail"},
    )
    graph.add_edge("sync_state", "pause_gate")
    graph.add_conditional_edges("pause_gate", route_after_pause, {"pause": "pause", "advance": "advance_trial"})
    graph.add_conditional_edges(
        "advance_trial",
        route_after_advance,
        {"initialize": "initialize_trial", "complete": "complete"},
    )
    graph.add_edge("pause", END)
    graph.add_edge("await_human_review", "plan_successor")
    graph.add_edge("complete", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer)


# Every trial cycles through ~7 graph nodes (initialize -> prepare_trial ->
# execute_trial -> plan_successor -> sync_state -> pause_gate -> advance_trial),
# and the pause/human-review branches add a couple more hops. LangGraph's
# default recursion_limit (25) only covers ~3 trials in one run/resume call,
# so requesting more than that reliably hits GraphRecursionError even though
# nothing is actually stuck looping. Size the limit to the batch instead of
# relying on the fixed default.
STEPS_PER_TRIAL = 10
RECURSION_LIMIT_BUFFER = 10


def _recursion_limit_for_trial_count(trial_count: int) -> int:
    return max(25, trial_count * STEPS_PER_TRIAL + RECURSION_LIMIT_BUFFER)


def run_workspace_loop_graph(
    initial_state: WorkspaceLoopGraphState,
    callbacks: WorkspaceLoopCallbacks,
    *,
    checkpointer: Any | None = None,
    thread_id: str,
) -> dict[str, Any]:
    graph = build_workspace_loop_graph(callbacks, checkpointer=checkpointer)
    trial_count = len(initial_state.get("selected_trials") or [])
    config: dict[str, Any] = {"recursion_limit": _recursion_limit_for_trial_count(trial_count)}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": thread_id}
    return dict(graph.invoke(initial_state, config=config))


def workspace_loop_thread_has_pending_work(
    callbacks: WorkspaceLoopCallbacks,
    *,
    checkpointer: Any,
    thread_id: str,
) -> bool:
    """Report whether resuming this thread would actually execute anything.

    A thread whose graph already reached END has no next node. Resuming it
    makes LangGraph replay the stored terminal state immediately without
    running a single node -- so no callback fires, no state is written, and
    the caller just gets the old terminal result back. A loop that resumes
    such a thread can never make progress no matter how many times it is
    restarted, so callers must start a fresh thread instead.
    """
    graph = build_workspace_loop_graph(callbacks, checkpointer=checkpointer)
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return bool(getattr(snapshot, "next", ()))


def resume_workspace_loop_graph(
    callbacks: WorkspaceLoopCallbacks,
    *,
    checkpointer: Any,
    thread_id: str,
    resume_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from langgraph.types import Command

    graph = build_workspace_loop_graph(callbacks, checkpointer=checkpointer)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get_tuple(config)
    if checkpoint is None:
        raise RuntimeError(f"LangGraph checkpoint not found for thread: {thread_id}")
    channel_values = (checkpoint.checkpoint or {}).get("channel_values", {}) if checkpoint.checkpoint else {}
    selected_trials = channel_values.get("selected_trials") or []
    trial_index = int(channel_values.get("trial_index") or 0)
    remaining_trials = max(1, len(selected_trials) - trial_index)
    config["recursion_limit"] = _recursion_limit_for_trial_count(remaining_trials)
    command: Any = Command(resume=resume_value) if resume_value is not None else None
    return dict(graph.invoke(command, config=config))


@contextmanager
def workspace_loop_checkpointer(path: Path) -> Iterator[tuple[Any, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver(), "memory"
        return

    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver, "sqlite"


def _namespace(settings: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**settings)


def _append_step(state: WorkspaceLoopGraphState, step: str) -> list[str]:
    return [*state.get("steps", []), step]


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("status", "competition", "trial_id", "source_trial_id", "next_trial_id")
    return {key: result.get(key) for key in keys if result.get(key) is not None}


def _next_trial_id(trial_id: str) -> str:
    prefix, _, number = str(trial_id).rpartition("_")
    try:
        return f"{prefix}_{int(number) + 1:03d}"
    except ValueError:
        return "trial_002"
