from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .execution_profile import load_execution_profile, validate_execution_profile
from .paths import trial_dir
from .store import load_state, write_text
from .trial_artifacts import read_trial_json


def collect_workspace_metrics(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked",
        "workspace_run_status": None,
        "profile_status": None,
        "source_metrics_path": None,
        "score_source": None,
        "cv_score": None,
        "metric": None,
        "objective": None,
        "issues": [],
        "review_questions": [],
        "next_action": "inspect-metrics-collection",
    }

    workspace_run = read_trial_json(out_dir, "workspace_run.json")
    if not workspace_run:
        result["issues"] = ["invalid_or_missing_workspace_run"]
        return _finish(out_dir, result)
    result["workspace_run_status"] = workspace_run.get("status")
    if result["workspace_run_status"] != "completed":
        result["issues"] = ["workspace_run_not_completed"]
        result["next_action"] = "complete-workspace-run"
        return _finish(out_dir, result)

    validation = validate_execution_profile(competition)
    result["profile_status"] = validation["status"]
    if validation["status"] != "ready":
        result["issues"] = ["execution_profile_not_ready", *validation["issues"]]
        result["next_action"] = "fix-execution-profile"
        return _finish(out_dir, result)

    profile = load_execution_profile(competition)
    try:
        relative_path = profile["artifacts"]["metrics"][0]
        source_path = Path(profile["project_root"]) / relative_path
    except (KeyError, IndexError, TypeError):
        result["issues"] = ["metrics_artifact_not_declared"]
        result["next_action"] = "fix-execution-profile"
        return _finish(out_dir, result)
    result["source_metrics_path"] = str(source_path.resolve())
    if source_path.suffix.casefold() != ".json":
        result["issues"] = ["unsupported_metrics_format"]
        result["next_action"] = "provide-json-metrics"
        return _finish(out_dir, result)
    try:
        source_metrics = json.loads(source_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        result["issues"] = ["invalid_metrics_json"]
        result["next_action"] = "fix-metrics-artifact"
        return _finish(out_dir, result)
    if not isinstance(source_metrics, dict):
        result["issues"] = ["metrics_json_must_be_object"]
        result["next_action"] = "fix-metrics-artifact"
        return _finish(out_dir, result)

    state = load_state(competition)
    competition_state = state.get("competition", {})
    try:
        cv_score, score_source = _resolve_score(source_metrics, profile, competition_state)
    except (AttributeError, KeyError, TypeError, ValueError):
        result["status"] = "needs_review"
        result["issues"] = ["missing_or_invalid_primary_score"]
        result["review_questions"] = [
            "Which JSON key should be used as the primary validation score? Add it as metrics_contract.source_key."
        ]
        result["next_action"] = "confirm-metrics-source-key"
        return _finish(out_dir, result)

    objective = source_metrics.get("objective")
    if objective not in {"maximize", "minimize"}:
        objective = competition_state.get("objective", "maximize")

    canonical = dict(source_metrics)
    canonical.update(
        {
            "trial_id": trial_id,
            "metric": source_metrics.get("metric") or competition_state.get("metric", "unknown"),
            "cv_score": cv_score,
            "objective": objective,
            "source_metrics_path": str(source_path.resolve()),
        }
    )
    result.update(
        {
            "status": "collected",
            "score_source": score_source,
            "cv_score": canonical["cv_score"],
            "metric": canonical["metric"],
            "objective": canonical["objective"],
            "issues": [],
            "next_action": "evaluate-trial",
        }
    )
    write_text(out_dir / "metrics.json", json.dumps(canonical, ensure_ascii=False, indent=2) + "\n")
    return _finish(out_dir, result)


def _finish(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    _write_result(out_dir, result)
    log_decision(
        result["competition"],
        result["trial_id"],
        decision_type="workspace_metrics_collection",
        decision=result["status"],
        reason="External JSON metrics collection was evaluated against the explicit metrics contract.",
        evidence={
            "source_metrics_path": result["source_metrics_path"],
            "score_source": result["score_source"],
            "cv_score": result["cv_score"],
            "issues": result["issues"],
        },
        next_action=result["next_action"],
    )
    return result


def _resolve_score(
    source_metrics: dict[str, Any],
    profile: dict[str, Any],
    competition_state: dict[str, Any],
) -> tuple[float | int, str]:
    direct = source_metrics.get("cv_score")
    if _is_finite_number(direct):
        return direct, "cv_score"

    source_key = profile.get("metrics_contract", {}).get("source_key")
    if source_key:
        return _resolve_nested_score(source_metrics, source_key)

    metric = str(competition_state.get("metric") or profile.get("metric") or "").strip().casefold()
    if metric:
        for candidate in (f"validation_{metric}", f"val_{metric}", f"valid_{metric}"):
            value = source_metrics.get(candidate)
            if _is_finite_number(value):
                return value, candidate

    raise KeyError("metrics_contract.source_key")


def _resolve_nested_score(source_metrics: dict[str, Any], source_key: str) -> tuple[float | int, str]:
    value: Any = source_metrics
    for part in source_key.split("."):
        value = value[part]
    if not _is_finite_number(value):
        raise ValueError(f"Metric source is not a finite number: {source_key}")
    return value, source_key


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _write_result(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "metrics_collection.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {result['competition']} / {result['trial_id']} Metrics Collection",
        "",
        f"- status: {result['status']}",
        f"- source_metrics_path: {result['source_metrics_path']}",
        f"- score_source: {result['score_source']}",
        f"- cv_score: {result['cv_score']}",
        f"- metric: {result['metric']}",
        f"- objective: {result['objective']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Review Questions", ""])
    lines.extend(f"- {item}" for item in result["review_questions"] or ["None"])
    lines.append("")
    write_text(out_dir / "metrics_collection.md", "\n".join(lines))
