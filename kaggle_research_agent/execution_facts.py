from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .execution_plan_snapshot import load_execution_plan_snapshot
from .paths import trial_dir
from .store import write_text


def resolve_trial_plan(competition: str, trial_id: str) -> dict[str, Any]:
    """Resolve the plan that produced the executed code.

    A retry may create a new proposal in the same trial directory after code has
    already run. The coding handoff is the execution boundary, so its base trial
    is used to reject those self-referential or successor-only plans.
    """

    out_dir = trial_dir(competition, trial_id)
    snapshot = load_execution_plan_snapshot(competition, trial_id)
    snapshot_plan = snapshot.get("plan") if isinstance(snapshot.get("plan"), dict) else {}
    # The snapshot's markdown parser only understands one plan-document layout
    # (demo_one_cycle's "## Primary Change Axis" style). A plan rendered in a
    # different layout (e.g. research_planner's "## Strategy" style) still
    # produces a non-empty but axis-less snapshot, which must not be trusted as
    # a complete plan or the richer fallback below (which already knows how to
    # recover primary_change_axis from either layout) never runs.
    if snapshot_plan.get("primary_change_axis"):
        plan = _normalize_plan_candidate(snapshot_plan, trial_id)
        plan["_resolved_plan_sources"] = ["internal/execution_plan_snapshot.json"]
        return plan

    handoff = _read_trial_json(out_dir, "workspace_coding_handoff.json")
    expected_source = _handoff_source_trial(handoff)
    candidates = _current_plan_candidates(out_dir)
    compatible = [
        (source, candidate)
        for source, candidate in candidates
        if _matches_execution(candidate, trial_id, expected_source)
    ]
    if expected_source:
        revisions = _revision_plan_candidates(out_dir, trial_id, expected_source)
        coarse = [item for item in compatible if item[0] == "next_experiment.json"]
        detailed = [item for item in compatible if item[0] != "next_experiment.json"]
        compatible = [*coarse, *revisions, *detailed]

    plan: dict[str, Any] = {}
    sources: list[str] = []
    for source_name, candidate in compatible:
        normalized = _normalize_plan_candidate(candidate, trial_id)
        for key, value in normalized.items():
            if value not in (None, "", [], {}):
                plan[key] = value
        sources.append(source_name)

    if not plan.get("source_trial_id"):
        plan["source_trial_id"] = expected_source
    if not plan.get("trial_id"):
        plan["trial_id"] = trial_id
    if not plan.get("plan_title"):
        candidate = plan.get("candidate") if isinstance(plan.get("candidate"), dict) else {}
        plan["plan_title"] = candidate.get("name")
    if not plan.get("primary_change_axis"):
        plan["primary_change_axis"] = _next_experiment_strategy(out_dir)
    if not plan.get("plan_type"):
        plan["plan_type"] = "continuation_delta_plan" if plan.get("source_trial_id") else "initial_pipeline_plan"
    plan["_resolved_plan_sources"] = sources or ["workspace_coding_handoff.json"]
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
            "plan": _plan_source(out_dir, plan),
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


def _plan_source(out_dir: Path, plan: dict[str, Any]) -> str:
    resolved = plan.get("_resolved_plan_sources")
    if isinstance(resolved, list) and resolved:
        return ",".join(str(item) for item in resolved)
    candidates = [
        ("delta_plan.json", out_dir / "delta_plan.json"),
        ("internal/effective_plan.json", out_dir / "internal" / "effective_plan.json"),
        ("demo_experiment_plan.json", out_dir / "demo_experiment_plan.json"),
        ("internal/demo_experiment_plan.json", out_dir / "internal" / "demo_experiment_plan.json"),
        ("next_experiment.md", out_dir / "next_experiment.md"),
    ]
    return next((name for name, path in candidates if path.is_file()), "derived_default")


def _current_plan_candidates(out_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    effective = _read_trial_json(out_dir, "effective_plan.json")
    effective_delta = effective.get("current_delta") if isinstance(effective.get("current_delta"), dict) else {}
    return [
        ("next_experiment.json", _read_trial_json(out_dir, "next_experiment.json")),
        ("demo_experiment_plan.json", _read_trial_json(out_dir, "demo_experiment_plan.json")),
        ("internal/effective_plan.json:current_delta", effective_delta),
        ("delta_plan.json", _read_trial_json(out_dir, "delta_plan.json")),
    ]


def _revision_plan_candidates(
    out_dir: Path,
    trial_id: str,
    expected_source: str,
) -> list[tuple[str, dict[str, Any]]]:
    revision_root = out_dir / "internal" / "plan_revisions"
    if not revision_root.is_dir():
        return []
    matches: list[tuple[str, dict[str, Any]]] = []
    for revision_dir in sorted(revision_root.iterdir(), reverse=True):
        if not revision_dir.is_dir():
            continue
        revision_matches: list[tuple[str, dict[str, Any]]] = []
        for name in ("next_experiment.json", "demo_experiment_plan.json", "delta_plan.json"):
            candidate = _read_json(revision_dir / name)
            if _matches_execution(candidate, trial_id, expected_source):
                revision_matches.append(
                    (f"internal/plan_revisions/{revision_dir.name}/{name}", candidate)
                )
        if revision_matches:
            matches.extend(revision_matches)
            break
    return matches


def _matches_execution(
    candidate: dict[str, Any],
    trial_id: str,
    expected_source: str | None,
) -> bool:
    if not candidate:
        return False
    target = str(candidate.get("trial_id") or candidate.get("next_trial_id") or "").strip()
    if target and target != trial_id:
        return False
    source = str(candidate.get("source_trial_id") or "").strip()
    if source == trial_id:
        return False
    if expected_source:
        return source == expected_source
    return True


def _normalize_plan_candidate(candidate: dict[str, Any], trial_id: str) -> dict[str, Any]:
    normalized = dict(candidate)
    normalized.setdefault("trial_id", normalized.get("next_trial_id") or trial_id)
    if not normalized.get("primary_change_axis") and normalized.get("strategy"):
        normalized["primary_change_axis"] = normalized["strategy"]
    if not normalized.get("change_details") and normalized.get("changes"):
        normalized["change_details"] = normalized["changes"]
    candidate_details = normalized.get("candidate") if isinstance(normalized.get("candidate"), dict) else {}
    if not normalized.get("plan_title"):
        normalized["plan_title"] = candidate_details.get("name") or normalized.get("strategy")
    return normalized


def _handoff_source_trial(handoff: dict[str, Any]) -> str | None:
    value = (
        handoff.get("recommended_base_trial")
        or handoff.get("code_base_trial_id")
        or handoff.get("source_trial_id")
    )
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _next_experiment_strategy(out_dir: Path) -> str | None:
    try:
        text = (out_dir / "next_experiment.md").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    match = re.search(r"(?ims)^##\s+Strategy\s*$\s*([A-Za-z0-9_-]+)\s*$", text)
    return match.group(1).strip() if match else None


def _read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    for path in [out_dir / name, out_dir / "internal" / name]:
        value = _read_json(path)
        if value:
            return value
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
