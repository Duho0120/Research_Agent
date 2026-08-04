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
    plan_metadata = _plan_metadata(out_dir)
    canonical.update(
        {
            "competition": competition,
            "trial": trial_id,
            "trial_id": trial_id,
            "base_trial": plan_metadata.get("base_trial"),
            "change_axis": plan_metadata.get("change_axis"),
            "metric": source_metrics.get("metric") or competition_state.get("metric", "unknown"),
            "cv_score": cv_score,
            "objective": objective,
            "source_metrics_path": str(source_path.resolve()),
        }
    )
    canonical["pipeline_summary"] = _canonical_pipeline_summary(
        canonical.get("pipeline_summary"),
        competition=competition,
        trial_id=trial_id,
        plan_metadata=plan_metadata,
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


def _plan_metadata(out_dir: Path) -> dict[str, Any]:
    plan = read_trial_json(out_dir, "demo_experiment_plan.json")
    return {
        "base_trial": plan.get("source_trial_id"),
        "change_axis": plan.get("primary_change_axis") or None,
        "plan_title": plan.get("plan_title"),
    }


def _canonical_pipeline_summary(
    value: Any,
    *,
    competition: str,
    trial_id: str,
    plan_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(value) if isinstance(value, dict) else {}
    summary.update(
        {
            "competition": competition,
            "trial": trial_id,
            "trial_id": trial_id,
            "base_trial": plan_metadata.get("base_trial"),
            "change_axis": plan_metadata.get("change_axis"),
        }
    )
    return summary


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
        try:
            return _resolve_nested_score(source_metrics, source_key)
        except (KeyError, TypeError, ValueError):
            # The configured key was a good guess for a previous trial's metrics
            # shape, but a different code-writer run can reasonably name its
            # score field differently. Fall through to the metric-name-based
            # detection below instead of failing outright.
            pass

    # Match on a punctuation- and case-insensitive form of the metric name.
    # A metric like "R-Hit@1cm" gets written to metrics.json under whatever
    # spelling the code writer picked that run ("R-Hit@1cm", "r_hit_at_1cm",
    # "val_rhit_1cm", ...), so comparing raw strings made the collector
    # demand a human-configured source_key on nearly every new metric.
    metric = str(competition_state.get("metric") or profile.get("metric") or "").strip()
    if metric:
        normalized_index = _normalized_metric_index(source_metrics)

        def lookup(name: str) -> tuple[str, float | int] | None:
            for variant in _metric_name_variants(name):
                match = normalized_index.get(variant)
                if match:
                    return match
            return None

        if source_key:
            match = lookup(source_key)
            if match:
                return match[1], match[0]
        for prefix in ("validation_", "val_", "valid_", ""):
            match = lookup(f"{prefix}{metric}")
            if match:
                return match[1], match[0]
        value = source_metrics.get("value")
        if _is_finite_number(value) and _metric_name_variants(str(source_metrics.get("metric") or "")) & _metric_name_variants(metric):
            return value, "value"

    raise KeyError("metrics_contract.source_key")


def _metric_name_variants(name: str) -> set[str]:
    """Punctuation- and case-insensitive spellings a metric name may take.

    Metric names like "R-Hit@1cm" or "precision@10" get written into
    metrics.json under whichever spelling the code writer chose that run --
    "R-Hit@1cm", "r_hit_at_1cm" (with "@" spelled out), or "val_RHit1cm"
    (with "@" dropped). Comparing raw strings matched none of those, so a
    human had to configure metrics_contract.source_key for nearly every new
    metric. Both "@" readings are generated so all of those spellings match.
    """
    text = str(name).casefold()
    return {
        "".join(character for character in text.replace("@", replacement) if character.isalnum())
        for replacement in ("at", "")
    }


def _normalized_metric_index(source_metrics: dict[str, Any]) -> dict[str, tuple[str, float | int]]:
    """Map each normalized spelling of a numeric key -> (original key, value).

    Non-numeric entries are skipped so a descriptive field such as
    {"metric": "R-Hit@1cm"} -- which holds the metric's *name*, not its
    score -- is never mistaken for the score itself.
    """
    index: dict[str, tuple[str, float | int]] = {}
    for key, value in source_metrics.items():
        if not _is_finite_number(value):
            continue
        for variant in _metric_name_variants(key):
            index.setdefault(variant, (key, value))
    return index


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
