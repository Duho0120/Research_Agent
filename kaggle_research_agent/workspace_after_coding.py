from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .paths import trial_dir
from .store import write_text
from .workspace_metrics_collector import collect_workspace_metrics
from .workspace_result_cycle import process_workspace_result
from .workspace_runner import run_workspace_pipeline


def run_workspace_after_coding(
    competition: str,
    trial_id: str,
    *,
    run_now: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    validation = _load_coding_validation(out_dir)
    if not validation or validation.get("status") != "accepted":
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "run_now": run_now,
                "issues": ["workspace_coding_result_not_accepted"],
                "coding_result_validation": validation or {},
                "workspace_run": None,
                "metrics_collection": None,
                "workspace_result_cycle": None,
                "next_action": "validate-workspace-coding-result",
            },
        )

    workspace_run = run_workspace_pipeline(competition, trial_id, run_now=run_now)
    if not run_now:
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "ready_to_run",
                "run_now": run_now,
                "issues": [],
                "coding_result_validation": validation,
                "workspace_run": workspace_run,
                "metrics_collection": None,
                "workspace_result_cycle": None,
                "next_action": "rerun-with-run-now",
            },
        )

    if workspace_run.get("status") != "completed":
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"workspace_run_{workspace_run.get('status')}",
                "run_now": run_now,
                "issues": [f"workspace_run_not_completed:{workspace_run.get('status')}"],
                "coding_result_validation": validation,
                "workspace_run": workspace_run,
                "metrics_collection": None,
                "workspace_result_cycle": None,
                "next_action": workspace_run.get("next_action", "fix-workspace-run"),
            },
        )

    metrics_collection = collect_workspace_metrics(competition, trial_id)
    if metrics_collection.get("status") != "collected":
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"metrics_{metrics_collection.get('status')}",
                "run_now": run_now,
                "issues": metrics_collection.get("issues", []),
                "coding_result_validation": validation,
                "workspace_run": workspace_run,
                "metrics_collection": metrics_collection,
                "workspace_result_cycle": None,
                "next_action": metrics_collection.get("next_action", "fix-metrics-collection"),
            },
        )

    workspace_result_cycle = process_workspace_result(competition, trial_id)
    status = "completed" if workspace_result_cycle.get("status") != "blocked" else "result_cycle_blocked"
    return _finish(
        out_dir,
        {
            "competition": competition,
            "trial_id": trial_id,
            "status": status,
            "run_now": run_now,
            "issues": workspace_result_cycle.get("issues", []),
            "coding_result_validation": validation,
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
            "workspace_result_cycle": workspace_result_cycle,
            "next_action": workspace_result_cycle.get("next_action", "plan-next-workspace-trial"),
        },
    )


def _load_coding_validation(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "workspace_coding_result_validation.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finish(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    write_text(out_dir / "workspace_after_coding_cycle.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_after_coding_cycle.md", _render_after_coding(result))
    log_decision(
        result["competition"],
        result["trial_id"],
        decision_type="workspace_after_coding_cycle",
        decision=result["status"],
        reason="Workspace after-coding cycle evaluated code validation, execution, metrics, and result-cycle stages.",
        evidence={
            "run_now": result["run_now"],
            "issues": result.get("issues", []),
            "workspace_run_status": (result.get("workspace_run") or {}).get("status"),
            "metrics_collection_status": (result.get("metrics_collection") or {}).get("status"),
            "workspace_result_cycle_status": (result.get("workspace_result_cycle") or {}).get("status"),
        },
        user_input_used=result["run_now"],
        next_action=result["next_action"],
    )
    return result


def _render_after_coding(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['competition']} / {result['trial_id']} Workspace After-Coding Cycle",
        "",
        f"- status: {result['status']}",
        f"- run_now: {result['run_now']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Stage Status",
        "",
        f"- coding_result_validation: {(result.get('coding_result_validation') or {}).get('status')}",
        f"- workspace_run: {(result.get('workspace_run') or {}).get('status')}",
        f"- metrics_collection: {(result.get('metrics_collection') or {}).get('status')}",
        f"- workspace_result_cycle: {(result.get('workspace_result_cycle') or {}).get('status')}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("issues", []) or ["None"])
    lines.append("")
    return "\n".join(lines)
