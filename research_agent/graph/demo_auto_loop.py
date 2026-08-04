from __future__ import annotations

import json
from typing import Any

from ..paths import competition_memory_dir
from ..retrieval.index_builder import build_document_index
from ..store import load_state, now_iso, save_state, write_text
from .demo_cycle_graph import run_demo_graph_cycle


def run_demo_graph_auto_loop(
    competition: str,
    *,
    start_trial_id: str = "trial_001",
    max_trials: int = 3,
    stop_no_improvement: int = 3,
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
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "competition": competition,
        "start_trial_id": start_trial_id,
        "max_trials": max_trials,
        "stop_no_improvement": stop_no_improvement,
        "started_at": now_iso(),
        "status": "running",
        "trials": [],
        "best_trial": None,
        "no_improvement_count": 0,
        "next_action": "run-next-demo-graph-cycle",
    }
    current_trial = start_trial_id

    for index in range(max_trials):
        cycle = run_demo_graph_cycle(
            competition,
            current_trial,
            model=model,
            provider=provider,
            allow_api=allow_api,
            mock_plan_file=mock_plan_file,
            mock_response_file=mock_response_file,
            run_now=run_now,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
            low_cost_user_summary=low_cost_user_summary,
        )
        best_update = _update_demo_best_trial(competition, cycle)
        build_document_index(competition)
        trial_row = {
            "trial_id": current_trial,
            "status": cycle.get("status"),
            "local_score": (cycle.get("record") or {}).get("local_score"),
            "metric": (cycle.get("record") or {}).get("metric"),
            "objective": (cycle.get("record") or {}).get("objective"),
            "is_best": best_update.get("is_best", False),
            "best_trial": best_update.get("best_trial"),
            "next_action": cycle.get("next_action"),
            "graph_state_file": (cycle.get("graph_execution") or {}).get("graph_state_file"),
            "node_events_file": (cycle.get("graph_execution") or {}).get("node_events_file"),
        }
        result["trials"].append(trial_row)
        result["best_trial"] = best_update.get("best_trial") or result.get("best_trial")
        result["next_action"] = cycle.get("next_action")

        if cycle.get("status") != "completed":
            result["status"] = "blocked"
            result["blocked_trial"] = current_trial
            result["issues"] = cycle.get("issues", [])
            break

        if best_update.get("is_best"):
            result["no_improvement_count"] = 0
        else:
            result["no_improvement_count"] = int(result.get("no_improvement_count", 0)) + 1

        if result["no_improvement_count"] >= stop_no_improvement:
            result["status"] = "stopped_no_improvement"
            result["next_action"] = "inspect-best-trial-or-change-strategy"
            break

        if index == max_trials - 1:
            result["status"] = "completed"
            result["next_action"] = "inspect-demo-auto-loop-summary"
            break

        current_trial = _increment_trial_id(current_trial)
    else:
        result["status"] = "completed"
        result["next_action"] = "inspect-demo-auto-loop-summary"

    result["finished_at"] = now_iso()
    _write_demo_auto_loop_summary(competition, result)
    return result


def _update_demo_best_trial(competition: str, cycle: dict[str, Any]) -> dict[str, Any]:
    record = cycle.get("record") if isinstance(cycle.get("record"), dict) else {}
    score = record.get("local_score")
    if cycle.get("status") != "completed" or score is None:
        return {"is_best": False, "best_trial": _load_demo_best_trial(competition)}

    objective = record.get("objective") or "maximize"
    current = _load_demo_best_trial(competition)
    is_best = _is_better(score, current.get("local_score") if current else None, objective)
    if not is_best:
        return {"is_best": False, "best_trial": current}

    best = {
        "competition": competition,
        "trial_id": cycle["trial_id"],
        "local_score": score,
        "metric": record.get("metric"),
        "objective": objective,
        "updated_at": now_iso(),
        "record_path": f"experiments/{competition}/{cycle['trial_id']}/demo_cycle_record.json",
        "user_view": f"runs/{competition}/{cycle['trial_id']}",
    }
    memory = competition_memory_dir(competition)
    write_text(memory / "demo_best_trial.json", json.dumps(best, ensure_ascii=False, indent=2) + "\n")
    write_text(memory / "demo_best_trial.md", _render_demo_best_trial(best))
    _update_state_best_trial(competition, best)
    return {"is_best": True, "best_trial": best}


def _load_demo_best_trial(competition: str) -> dict[str, Any] | None:
    path = competition_memory_dir(competition) / "demo_best_trial.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_better(score: float, best_score: Any, objective: str) -> bool:
    if not isinstance(best_score, (int, float)):
        return True
    return score < best_score if objective == "minimize" else score > best_score


def _update_state_best_trial(competition: str, best: dict[str, Any]) -> None:
    state = load_state(competition)
    current_state = state.setdefault("current_state", {})
    current_state["best_trial"] = {
        "trial_id": best["trial_id"],
        "cv_score": best["local_score"],
        "lb_score": None,
        "source": "demo_graph_auto_loop",
    }
    current_state["active_trial"] = best["trial_id"]
    save_state(competition, state)


def _write_demo_auto_loop_summary(competition: str, result: dict[str, Any]) -> None:
    memory = competition_memory_dir(competition)
    write_text(memory / "demo_graph_auto_loop.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(memory / "demo_graph_auto_loop.md", _render_demo_auto_loop_summary(result))


def _render_demo_auto_loop_summary(result: dict[str, Any]) -> str:
    lines = [
        f"# Demo Graph Auto Loop - {result['competition']}",
        "",
        f"- status: {result['status']}",
        f"- start_trial_id: {result['start_trial_id']}",
        f"- max_trials: {result['max_trials']}",
        f"- no_improvement_count: {result['no_improvement_count']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Best Trial",
        "",
    ]
    best = result.get("best_trial") or {}
    if best:
        lines.extend(
            [
                f"- trial_id: {best.get('trial_id')}",
                f"- score: {best.get('local_score')}",
                f"- metric: {best.get('metric')}",
                f"- user_view: `{best.get('user_view')}`",
            ]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Trials", ""])
    for row in result.get("trials", []):
        lines.append(
            f"- {row.get('trial_id')}: status={row.get('status')}, "
            f"score={row.get('local_score')}, is_best={row.get('is_best')}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_demo_best_trial(best: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Demo Best Trial - {best['competition']}",
            "",
            f"- trial_id: {best['trial_id']}",
            f"- local_score: {best['local_score']}",
            f"- metric: {best.get('metric')}",
            f"- objective: {best.get('objective')}",
            f"- updated_at: {best['updated_at']}",
            f"- user_view: `{best['user_view']}`",
            "",
        ]
    )


def _increment_trial_id(trial_id: str) -> str:
    prefix, separator, suffix = trial_id.rpartition("_")
    if separator and suffix.isdigit():
        return f"{prefix}_{int(suffix) + 1:0{len(suffix)}d}"
    return f"{trial_id}_next"
