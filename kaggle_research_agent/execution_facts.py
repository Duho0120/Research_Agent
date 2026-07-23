from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .paths import trial_dir
from .store import write_text


def resolve_trial_plan(competition: str, trial_id: str) -> dict[str, Any]:
    """Resolve the authoritative plan, preferring structured incremental artifacts."""

    out_dir = trial_dir(competition, trial_id)
    demo_plan = _read_trial_json(out_dir, "demo_experiment_plan.json")
    effective = _read_trial_json(out_dir, "effective_plan.json")
    effective_delta = effective.get("current_delta") if isinstance(effective.get("current_delta"), dict) else {}
    delta = _read_trial_json(out_dir, "delta_plan.json")

    plan: dict[str, Any] = {}
    for source in [demo_plan, effective_delta, delta]:
        for key, value in source.items():
            if value not in (None, "", [], {}):
                plan[key] = value

    handoff = _read_trial_json(out_dir, "workspace_coding_handoff.json")
    if not plan.get("source_trial_id"):
        plan["source_trial_id"] = (
            handoff.get("recommended_base_trial")
            or handoff.get("code_base_trial_id")
            or handoff.get("source_trial_id")
        )
    if not plan.get("primary_change_axis"):
        plan["primary_change_axis"] = _next_experiment_strategy(out_dir)
    if not plan.get("plan_type"):
        plan["plan_type"] = "continuation_delta_plan" if plan.get("source_trial_id") else "initial_pipeline_plan"
    return plan


def write_executed_trial_facts(
    competition: str,
    trial_id: str,
    *,
    pipeline_structure: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Persist rule-based facts derived from the executed code and structured plan."""

    out_dir = trial_dir(competition, trial_id)
    plan = resolve_trial_plan(competition, trial_id)
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    model = _stage_details(pipeline_structure, "model_definition")
    preprocessing = _stage_details(pipeline_structure, "preprocessing")
    validation = _stage_details(pipeline_structure, "data_split_cv")
    features = _stage_details(pipeline_structure, "feature_representation")
    submission = _stage_details(pipeline_structure, "test_inference_output")
    consistency_issues = list(pipeline_structure.get("consistency_issues") or [])

    facts = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "source_trial_id": plan.get("source_trial_id"),
        "plan_type": plan.get("plan_type"),
        "primary_change_axis": plan.get("primary_change_axis"),
        "candidate": plan.get("candidate") or {},
        "change_details": plan.get("change_details") or [],
        "keep_unchanged": plan.get("keep_unchanged") or [],
        "model": model,
        "preprocessing": preprocessing,
        "validation": validation,
        "features": features,
        "submission": submission,
        "scores": {
            "metric": summary.get("metric") or metrics.get("metric"),
            "objective": summary.get("objective") or metrics.get("objective"),
            "local": summary.get("local_score") or metrics.get("validation_accuracy") or metrics.get("local_score"),
            "submission": summary.get("submitted_lb_score"),
            "rank": summary.get("submitted_rank"),
        },
        "consistency_issues": consistency_issues,
        "sources": {
            "plan": _plan_source(out_dir),
            "model": "internal/pipeline_structure.json:model_definition",
            "scores": "metrics.json+submission_log",
        },
    }
    write_text(
        out_dir / "internal" / "executed_trial_facts.json",
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n",
    )
    return facts


def load_executed_trial_facts(competition: str, trial_id: str) -> dict[str, Any]:
    return _read_trial_json(trial_dir(competition, trial_id), "executed_trial_facts.json")


def _stage_details(structure: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in structure.get("stages", []):
        if isinstance(stage, dict) and stage.get("id") == stage_id:
            details = stage.get("structured_details")
            return dict(details) if isinstance(details, dict) else {}
    return {}


def _plan_source(out_dir: Path) -> str:
    candidates = [
        ("delta_plan.json", out_dir / "delta_plan.json"),
        ("internal/effective_plan.json", out_dir / "internal" / "effective_plan.json"),
        ("demo_experiment_plan.json", out_dir / "demo_experiment_plan.json"),
        ("internal/demo_experiment_plan.json", out_dir / "internal" / "demo_experiment_plan.json"),
        ("next_experiment.md", out_dir / "next_experiment.md"),
    ]
    return next((name for name, path in candidates if path.is_file()), "derived_default")


def _next_experiment_strategy(out_dir: Path) -> str | None:
    try:
        text = (out_dir / "next_experiment.md").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    match = re.search(r"(?ims)^##\s+Strategy\s*$\s*([A-Za-z0-9_-]+)\s*$", text)
    return match.group(1).strip() if match else None


def _read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    for path in [out_dir / name, out_dir / "internal" / name]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}
