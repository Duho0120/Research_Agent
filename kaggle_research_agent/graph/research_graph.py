from __future__ import annotations

from typing import Any

from .nodes import (
    ask_user_node,
    check_metrics_node,
    create_job_node,
    decide_execution_node,
    diagnose_node,
    evaluate_node,
    finalize_node,
    plan_next_node,
    plan_trial_node,
    remember_node,
    route_after_execution,
    route_after_local_run,
    route_after_metrics,
    route_after_remember,
    route_after_validation,
    run_local_node,
    safe_execution_chain_node,
    validate_config_node,
    wait_node,
)
from .state import ResearchGraphState


def build_research_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Install `langgraph` to use run-graph-cycle.") from exc

    graph = StateGraph(ResearchGraphState)
    graph.add_node("plan_trial", plan_trial_node)
    graph.add_node("validate_config", validate_config_node)
    graph.add_node("check_metrics", check_metrics_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("remember", remember_node)
    graph.add_node("plan_next", plan_next_node)
    graph.add_node("decide_execution", decide_execution_node)
    graph.add_node("safe_chain", safe_execution_chain_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("wait", wait_node)
    graph.add_node("run_local", run_local_node)
    graph.add_node("create_job", create_job_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan_trial")
    graph.add_edge("plan_trial", "validate_config")
    graph.add_conditional_edges("validate_config", route_after_validation, {"check_metrics": "check_metrics", "end": "finalize"})
    graph.add_conditional_edges(
        "check_metrics",
        route_after_metrics,
        {"evaluate": "evaluate", "safe_chain": "safe_chain", "decide_execution": "decide_execution", "end": "finalize"},
    )
    graph.add_edge("evaluate", "diagnose")
    graph.add_edge("diagnose", "remember")
    graph.add_conditional_edges("remember", route_after_remember, {"plan_next": "plan_next", "end": "finalize"})
    graph.add_edge("plan_next", "finalize")
    graph.add_conditional_edges(
        "decide_execution",
        route_after_execution,
        {"ask_user": "ask_user", "wait": "wait", "run_local": "run_local", "create_job": "create_job"},
    )
    graph.add_edge("safe_chain", "finalize")
    graph.add_edge("ask_user", "finalize")
    graph.add_edge("wait", "finalize")
    graph.add_conditional_edges("run_local", route_after_local_run, {"evaluate": "evaluate", "end": "finalize"})
    graph.add_edge("create_job", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_graph_cycle(
    competition: str,
    trial_id: str,
    *,
    create_job_request: bool = True,
    backend: str = "local",
    run_now: bool = False,
    command: str | None = None,
    next_trial_id: str | None = None,
    prepare_next_patch: bool = False,
    apply_next_patch: bool = False,
    next_run_command: str | None = None,
    run_safe_chain: bool = False,
    safe_chain_mock_response_file: str | None = None,
    safe_chain_allow_api: bool = False,
    safe_chain_model: str = "gpt-5",
) -> dict[str, Any]:
    initial: ResearchGraphState = {
        "competition": competition,
        "trial_id": trial_id,
        "create_job_request": create_job_request,
        "backend": backend,
        "run_now": run_now,
        "command": command,
        "next_trial_id": next_trial_id,
        "prepare_next_patch": prepare_next_patch,
        "apply_next_patch": apply_next_patch,
        "next_run_command": next_run_command,
        "run_safe_chain": run_safe_chain,
        "safe_chain_mock_response_file": safe_chain_mock_response_file,
        "safe_chain_allow_api": safe_chain_allow_api,
        "safe_chain_model": safe_chain_model,
        "steps": [],
        "status": "running",
    }
    return dict(build_research_graph().invoke(initial))
