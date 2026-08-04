from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from .. import paths
from ..agents.code_writer_adapter import FileResponseClient
from ..demo_one_cycle import (
    _build_planning_context_pack_if_needed,
    _finish,
    _load_accepted_workspace_code_writer,
    _load_ready_demo_plan,
    _write_continuation_context,
    create_demo_experiment_plan,
    load_demo_context,
    prepare_demo_submission,
    record_demo_cycle_result,
    render_demo_context,
    resolve_demo_model_policy,
)
from ..paths import trial_dir
from ..store import now_iso, write_text
from ..workspace_code_writer import run_workspace_code_writer
from ..workspace_coding_handoff import prepare_workspace_coding_handoff
from ..workspace_metrics_collector import collect_workspace_metrics
from ..workspace_runner import run_workspace_pipeline
from .events import wrap_graph_node


class DemoCycleGraphState(TypedDict, total=False):
    competition: str
    trial_id: str
    model: Any
    provider: Any
    allow_api: bool
    mock_plan_file: Any
    mock_response_file: Any
    run_now: bool
    trial_llm_calls: Any
    strategy_calls_today: Any
    status: str
    next_action: str
    issues: list[str]
    steps: list[str]
    graph_options: dict[str, Any]
    context: dict[str, Any]
    model_policy: dict[str, Any]
    plan: dict[str, Any]
    handoff: dict[str, Any]
    code_writer: dict[str, Any]
    workspace_run: dict[str, Any]
    metrics_collection: dict[str, Any]
    record: dict[str, Any]
    submission_manifest: dict[str, Any]
    final_result: dict[str, Any]


def build_demo_cycle_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Install `langgraph` to use demo-graph-cycle.") from exc

    graph = StateGraph(DemoCycleGraphState)
    graph.add_node("load_context", wrap_graph_node("load_context", load_context_node))
    graph.add_node("plan_experiment", wrap_graph_node("plan_experiment", plan_experiment_node))
    graph.add_node("write_code", wrap_graph_node("write_code", write_code_node))
    graph.add_node("run_local", wrap_graph_node("run_local", run_local_node))
    graph.add_node("collect_metrics", wrap_graph_node("collect_metrics", collect_metrics_node))
    graph.add_node("record_result", wrap_graph_node("record_result", record_result_node))
    graph.add_node("prepare_submission", wrap_graph_node("prepare_submission", prepare_submission_node))
    graph.add_node("finalize", wrap_graph_node("finalize", finalize_node))

    graph.add_edge(START, "load_context")
    graph.add_conditional_edges("load_context", route_after_context, {"plan": "plan_experiment", "finalize": "finalize"})
    graph.add_conditional_edges("plan_experiment", route_after_plan, {"code": "write_code", "finalize": "finalize"})
    graph.add_conditional_edges("write_code", route_after_code, {"run": "run_local", "finalize": "finalize"})
    graph.add_conditional_edges("run_local", route_after_run, {"collect": "collect_metrics", "finalize": "finalize"})
    graph.add_conditional_edges("collect_metrics", route_after_collect, {"record": "record_result", "finalize": "finalize"})
    graph.add_edge("record_result", "prepare_submission")
    graph.add_edge("prepare_submission", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_demo_graph_cycle(
    competition: str,
    trial_id: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    allow_api: bool = False,
    mock_plan_file: str | None = None,
    mock_response_file: str | None = None,
    run_now: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
    low_cost_user_summary: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "model": model,
        "provider": provider,
        "allow_api": allow_api,
        "mock_plan_file": mock_plan_file,
        "mock_response_file": mock_response_file,
        "run_now": run_now,
        "trial_llm_calls": trial_llm_calls,
        "strategy_calls_today": strategy_calls_today,
        "low_cost_user_summary": low_cost_user_summary,
    }
    write_text(out_dir / "demo_graph_options.json", json.dumps(options, ensure_ascii=False, indent=2) + "\n")
    initial: DemoCycleGraphState = {
        "competition": competition,
        "trial_id": trial_id,
        "model": model,
        "provider": provider,
        "allow_api": allow_api,
        "mock_plan_file": mock_plan_file,
        "mock_response_file": mock_response_file,
        "run_now": run_now,
        "trial_llm_calls": trial_llm_calls,
        "strategy_calls_today": strategy_calls_today,
        "low_cost_user_summary": low_cost_user_summary,
        "graph_options": options,
        "status": "running",
        "steps": [],
        "issues": [],
    }
    state = dict(build_demo_cycle_graph().invoke(initial))
    return state.get("final_result", state)


def load_context_node(state: DemoCycleGraphState) -> dict[str, Any]:
    competition = state["competition"]
    trial_id = state["trial_id"]
    out_dir = trial_dir(competition, trial_id)
    options = _graph_options(state)
    write_text(out_dir / "demo_graph_options.json", json.dumps(options, ensure_ascii=False, indent=2) + "\n")
    model_policy = resolve_demo_model_policy(model_override=options.get("model"), provider_override=options.get("provider"))
    context = load_demo_context(competition, trial_id)
    context["model_policy"] = model_policy
    context["planning_context_pack"] = _build_planning_context_pack_if_needed(competition, trial_id, context)
    write_text(out_dir / "demo_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_context.md", render_demo_context(context))
    if context["status"] != "ready":
        return {
            "context": context,
            "model_policy": model_policy,
            "graph_options": options,
            "status": "blocked",
            "issues": context["issues"],
            "next_action": "fix-execution-profile",
            "steps": _append_step(state, "F-01"),
        }
    return {
        "context": context,
        "model_policy": model_policy,
        "graph_options": options,
        "status": "running",
        "next_action": "create-demo-experiment-plan",
        "steps": _append_step(state, "F-01"),
    }


def route_after_context(state: DemoCycleGraphState) -> Literal["plan", "finalize"]:
    return "plan" if state.get("context", {}).get("status") == "ready" else "finalize"


def plan_experiment_node(state: DemoCycleGraphState) -> dict[str, Any]:
    out_dir = trial_dir(state["competition"], state["trial_id"])
    options = _graph_options(state)
    context = state.get("context") or _load_json_object(out_dir / "demo_context.json")
    model_policy = state.get("model_policy") or resolve_demo_model_policy(
        model_override=options.get("model"),
        provider_override=options.get("provider"),
    )
    plan = None if options.get("mock_plan_file") else _load_ready_demo_plan(out_dir)
    if plan is None:
        plan = create_demo_experiment_plan(
            state["competition"],
            state["trial_id"],
            context,
            model_config=model_policy["experiment_planning"],
            allow_api=bool(options.get("allow_api")),
            mock_plan_file=options.get("mock_plan_file"),
            trial_llm_calls=options.get("trial_llm_calls"),
            strategy_calls_today=options.get("strategy_calls_today"),
        )
    else:
        plan["resumed_from_existing_artifact"] = True
    if plan["status"] != "ready":
        return {
            "plan": plan,
            "status": "blocked",
            "issues": plan["issues"],
            "next_action": plan["next_action"],
            "steps": _append_step(state, "F-02"),
        }
    return {
        "plan": plan,
        "context": context,
        "model_policy": model_policy,
        "graph_options": options,
        "status": "running",
        "next_action": "prepare-workspace-handoff",
        "steps": _append_step(state, "F-02"),
    }


def route_after_plan(state: DemoCycleGraphState) -> Literal["code", "finalize"]:
    return "code" if state.get("plan", {}).get("status") == "ready" else "finalize"


def write_code_node(state: DemoCycleGraphState) -> dict[str, Any]:
    competition = state["competition"]
    trial_id = state["trial_id"]
    options = _graph_options(state)
    model_policy = state.get("model_policy") or resolve_demo_model_policy(
        model_override=options.get("model"),
        provider_override=options.get("provider"),
    )
    _write_continuation_context(competition, trial_id)
    handoff = prepare_workspace_coding_handoff(competition, trial_id)
    if handoff["status"] != "ready":
        return {
            "handoff": handoff,
            "status": "blocked",
            "issues": handoff.get("blocking_issues", []),
            "next_action": handoff["next_action"],
            "steps": [*_append_step(state, "F-03"), "F-03:handoff"],
        }

    code_writer = None if options.get("mock_response_file") else _load_accepted_workspace_code_writer(competition, trial_id)
    if code_writer is None:
        client = FileResponseClient(options["mock_response_file"]) if options.get("mock_response_file") else None
        code_writer = run_workspace_code_writer(
            competition,
            trial_id,
            client=client,
            model=model_policy["workspace_code_writing"]["model"],
            provider=model_policy["workspace_code_writing"]["provider"],
            allow_api=bool(options.get("allow_api")),
            trial_llm_calls=options.get("trial_llm_calls"),
            strategy_calls_today=options.get("strategy_calls_today"),
        )
    if code_writer["status"] != "accepted":
        return {
            "handoff": handoff,
            "code_writer": code_writer,
            "status": "blocked",
            "issues": code_writer.get("issues") or code_writer.get("blocking_issues", []),
            "next_action": code_writer["next_action"],
            "steps": _append_step(state, "F-03"),
        }
    return {
        "handoff": handoff,
        "code_writer": code_writer,
        "model_policy": model_policy,
        "graph_options": options,
        "status": "running",
        "next_action": "run-workspace-pipeline",
        "steps": _append_step(state, "F-03"),
    }


def route_after_code(state: DemoCycleGraphState) -> Literal["run", "finalize"]:
    return "run" if state.get("code_writer", {}).get("status") == "accepted" else "finalize"


def run_local_node(state: DemoCycleGraphState) -> dict[str, Any]:
    options = _graph_options(state)
    workspace_run = run_workspace_pipeline(state["competition"], state["trial_id"], run_now=bool(options.get("run_now")))
    if workspace_run["status"] != "completed":
        return {
            "workspace_run": workspace_run,
            "status": "blocked",
            "issues": workspace_run.get("issues", []),
            "next_action": workspace_run["next_action"],
            "steps": _append_step(state, "F-04"),
        }
    return {
        "workspace_run": workspace_run,
        "graph_options": options,
        "status": "running",
        "next_action": "collect-workspace-metrics",
        "steps": _append_step(state, "F-04"),
    }


def route_after_run(state: DemoCycleGraphState) -> Literal["collect", "finalize"]:
    return "collect" if state.get("workspace_run", {}).get("status") == "completed" else "finalize"


def collect_metrics_node(state: DemoCycleGraphState) -> dict[str, Any]:
    metrics_collection = collect_workspace_metrics(state["competition"], state["trial_id"])
    if metrics_collection["status"] != "collected":
        return {
            "metrics_collection": metrics_collection,
            "status": "blocked",
            "issues": metrics_collection.get("issues", []),
            "next_action": metrics_collection["next_action"],
            "steps": _append_step(state, "F-06:collect"),
        }
    return {
        "metrics_collection": metrics_collection,
        "status": "running",
        "next_action": "record-demo-result",
        "steps": _append_step(state, "F-06:collect"),
    }


def route_after_collect(state: DemoCycleGraphState) -> Literal["record", "finalize"]:
    return "record" if state.get("metrics_collection", {}).get("status") == "collected" else "finalize"


def record_result_node(state: DemoCycleGraphState) -> dict[str, Any]:
    out_dir = trial_dir(state["competition"], state["trial_id"])
    context = state.get("context") or _load_json_object(out_dir / "demo_context.json")
    plan = state.get("plan") or _load_json_object(out_dir / "demo_experiment_plan.json")
    code_writer = state.get("code_writer") or _load_json_object(out_dir / "workspace_coding_result_validation.json")
    workspace_run = state.get("workspace_run") or _load_json_object(out_dir / "workspace_run.json")
    metrics_collection = state.get("metrics_collection") or _load_json_object(out_dir / "metrics_collection.json")
    record = record_demo_cycle_result(
        state["competition"],
        state["trial_id"],
        context=context,
        plan=plan,
        code_writer=code_writer,
        workspace_run=workspace_run,
        metrics_collection=metrics_collection,
    )
    return {
        "context": context,
        "plan": plan,
        "code_writer": code_writer,
        "workspace_run": workspace_run,
        "metrics_collection": metrics_collection,
        "record": record,
        "status": "running",
        "next_action": "prepare-submission-manifest",
        "steps": _append_step(state, "F-06"),
    }


def prepare_submission_node(state: DemoCycleGraphState) -> dict[str, Any]:
    manifest = prepare_demo_submission(
        state["competition"],
        state["trial_id"],
        record=state.get("record"),
        notes="Graph demo cycle completed local execution. Submit only after user approval.",
    )
    status = "completed" if manifest.get("status") == "ready" else "blocked"
    return {
        "submission_manifest": manifest,
        "status": status,
        "issues": manifest.get("checks", []) if status == "blocked" else [],
        "next_action": "review-submit-manifest" if status == "completed" else "fix-submit-manifest-checks",
        "steps": [*_append_step(state, "P-01"), "done"] if status == "completed" else _append_step(state, "P-01"),
    }


def finalize_node(state: DemoCycleGraphState) -> dict[str, Any]:
    status = state.get("status")
    if not status or status == "running":
        status = "completed"
    out_dir = trial_dir(state["competition"], state["trial_id"])
    options = _graph_options(state)
    context = state.get("context") or _load_json_object(out_dir / "demo_context.json")
    handoff = state.get("handoff") or _load_json_object(out_dir / "workspace_coding_handoff.json")
    result = {
        "competition": state["competition"],
        "trial_id": state["trial_id"],
        "status": status,
        "run_now": bool(options.get("run_now")),
        "steps": state.get("steps", []),
        "context": context,
        "model_policy": state.get("model_policy"),
        "plan": state.get("plan") or _load_json_object(out_dir / "demo_experiment_plan.json"),
        "handoff": handoff,
        "code_writer": state.get("code_writer") or _load_json_object(out_dir / "workspace_coding_result_validation.json"),
        "workspace_run": state.get("workspace_run") or _load_json_object(out_dir / "workspace_run.json"),
        "metrics_collection": state.get("metrics_collection") or _load_json_object(out_dir / "metrics_collection.json"),
        "record": state.get("record") or _load_json_object(out_dir / "demo_cycle_record.json"),
        "submission_manifest": state.get("submission_manifest") or _load_json_object(out_dir / "submit_manifest.json"),
        "issues": state.get("issues", []),
        "next_action": state.get("next_action") or "review-submit-manifest",
        "graph_execution": {
            "enabled": True,
            "graph_state_file": f"experiments/{state['competition']}/{state['trial_id']}/graph_state.json",
            "node_events_file": f"experiments/{state['competition']}/{state['trial_id']}/node_events.jsonl",
            "graph_rag_manifest_file": f"experiments/{state['competition']}/{state['trial_id']}/graph_rag_manifest.json",
        },
    }
    result["graph_rag_manifest"] = _write_graph_rag_manifest(out_dir, result)
    final = _finish(
        state["competition"],
        state["trial_id"],
        result,
        low_cost_user_summary=bool(options.get("low_cost_user_summary")),
        allow_api=bool(options.get("allow_api")),
    )
    write_text(
        trial_dir(state["competition"], state["trial_id"]) / "demo_graph_cycle.json",
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
    )
    return {"status": status, "final_result": final}


def _append_step(state: DemoCycleGraphState, step: str) -> list[str]:
    return [*state.get("steps", []), step]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _graph_options_from_state(state: DemoCycleGraphState) -> dict[str, Any]:
    return {
        "model": state.get("model"),
        "provider": state.get("provider"),
        "allow_api": bool(state.get("allow_api")),
        "mock_plan_file": state.get("mock_plan_file"),
        "mock_response_file": state.get("mock_response_file"),
        "run_now": bool(state.get("run_now")),
        "trial_llm_calls": state.get("trial_llm_calls"),
        "strategy_calls_today": state.get("strategy_calls_today"),
        "low_cost_user_summary": bool(state.get("low_cost_user_summary")),
    }


def _graph_options(state: DemoCycleGraphState) -> dict[str, Any]:
    if isinstance(state.get("graph_options"), dict):
        return dict(state["graph_options"])
    return _load_json_object(trial_dir(state["competition"], state["trial_id"]) / "demo_graph_options.json")


def _write_graph_rag_manifest(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": "1.0",
        "built_at": now_iso(),
        "competition": result["competition"],
        "trial_id": result["trial_id"],
        "status": result["status"],
        "graph": {
            "orchestrator": "langgraph_state_graph",
            "nodes": [
                "load_context",
                "plan_experiment",
                "write_code",
                "run_local",
                "collect_metrics",
                "record_result",
                "prepare_submission",
                "finalize",
            ],
            "steps": result.get("steps", []),
            "graph_state_file": result.get("graph_execution", {}).get("graph_state_file"),
            "node_events_file": result.get("graph_execution", {}).get("node_events_file"),
        },
        "rag_contexts": _graph_rag_contexts(result),
        "manifest_file": f"experiments/{result['competition']}/{result['trial_id']}/graph_rag_manifest.json",
        "summary_file": f"experiments/{result['competition']}/{result['trial_id']}/graph_rag_manifest.md",
    }
    write_text(out_dir / "graph_rag_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "graph_rag_manifest.md", _render_graph_rag_manifest(manifest))
    return manifest


def _graph_rag_contexts(result: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    planning = (result.get("context") or {}).get("planning_context_pack")
    if isinstance(planning, dict) and planning:
        contexts.append(_compact_graph_rag_context("experiment_planning", planning))
    coding = (result.get("handoff") or {}).get("retrieval_context")
    if isinstance(coding, dict) and coding:
        contexts.append(_compact_graph_rag_context("workspace_code_writing", coding))
    return contexts


def _compact_graph_rag_context(task: str, context: dict[str, Any]) -> dict[str, Any]:
    context_pack_file = context.get("context_pack_file")
    context_pack_md_file = context.get("context_pack_md_file")
    retrieval_manifest_file = context.get("retrieval_manifest_file")
    return {
        "task": context.get("task") or task,
        "query": context.get("query"),
        "document_count": context.get("document_count"),
        "skipped": bool(context.get("skipped")),
        "skip_reason": context.get("skip_reason"),
        "policy": context.get("policy"),
        "context_pack_file": context_pack_file,
        "context_pack_md_file": context_pack_md_file,
        "retrieval_manifest_file": retrieval_manifest_file,
        "files_exist": {
            "context_pack_file": _relative_file_exists(context_pack_file),
            "context_pack_md_file": _relative_file_exists(context_pack_md_file),
            "retrieval_manifest_file": _relative_file_exists(retrieval_manifest_file),
        },
        "documents": [
            {
                "source_path": doc.get("source_path"),
                "source_kind": doc.get("source_kind"),
                "trial_id": doc.get("trial_id"),
                "score": doc.get("score"),
            }
            for doc in context.get("documents", [])
            if isinstance(doc, dict)
        ],
    }


def _relative_file_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return (paths.project_root() / value).is_file()


def _render_graph_rag_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Graph/RAG Manifest - {manifest['competition']} / {manifest['trial_id']}",
        "",
        f"- status: {manifest['status']}",
        f"- graph_state_file: `{manifest['graph']['graph_state_file']}`",
        f"- node_events_file: `{manifest['graph']['node_events_file']}`",
        f"- steps: {', '.join(manifest['graph'].get('steps', []))}",
        "",
        "## RAG Contexts",
        "",
    ]
    for context in manifest.get("rag_contexts", []):
        lines.extend(
            [
                f"### {context.get('task')}",
                "",
                f"- documents: {context.get('document_count')}",
                f"- skipped: {context.get('skipped')}",
                f"- skip_reason: {context.get('skip_reason')}",
                f"- context_pack: `{context.get('context_pack_md_file')}`",
                f"- manifest: `{context.get('retrieval_manifest_file')}`",
                f"- files_exist: {context.get('files_exist')}",
                "",
            ]
        )
        for doc in context.get("documents", [])[:8]:
            lines.append(
                f"- `{doc.get('source_path')}` | kind={doc.get('source_kind')} | "
                f"trial={doc.get('trial_id') or '-'} | score={doc.get('score')}"
            )
        lines.append("")
    if not manifest.get("rag_contexts"):
        lines.append("- No RAG context pack was recorded.")
    return "\n".join(lines)
