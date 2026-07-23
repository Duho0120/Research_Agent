from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .code_snapshot import load_trial_code_snapshot
from .paths import trial_dir
from .store import write_text


AXIS_IMPACT_MAP: dict[str, dict[str, list[str]]] = {
    "feature": {
        "stages": ["feature_representation", "preprocessing", "training", "test_inference_output"],
        "symbols": ["build_pipeline", "train_validate", "make_submission"],
    },
    "preprocess": {
        "stages": ["preprocessing", "feature_representation", "training", "test_inference_output"],
        "symbols": ["build_pipeline", "train_validate", "make_submission"],
    },
    "model": {
        "stages": ["model_definition", "training", "evaluation", "test_inference_output"],
        "symbols": ["build_pipeline", "train_validate", "make_submission"],
    },
    "ensemble": {
        "stages": ["model_definition", "training", "evaluation", "test_inference_output"],
        "symbols": ["build_pipeline", "train_validate", "make_submission"],
    },
    "validation": {
        "stages": ["data_split_cv", "training", "evaluation"],
        "symbols": ["train_validate", "run_experiment"],
    },
    "hyperparameter": {
        "stages": ["model_definition", "training", "evaluation"],
        "symbols": ["build_pipeline", "train_validate"],
    },
}


def build_base_summary(competition: str, source_trial_id: str | None) -> dict[str, Any]:
    if not source_trial_id:
        return {}
    source = trial_dir(competition, source_trial_id)
    metrics = _read_json(source / "metrics.json")
    memory = _read_trial_json(source, "trial_memory_card.json")
    decision = _read_trial_json(source, "decision_card.json")
    structure = _read_json(source / "internal" / "pipeline_structure.json")
    plan = _read_trial_json(source, "demo_experiment_plan.json")
    stages = structure.get("stages", []) if isinstance(structure.get("stages"), list) else []
    return {
        "schema_version": "1.0",
        "competition": competition,
        "base_trial_id": source_trial_id,
        "scores": {
            "local": _first_value(metrics, memory, decision, keys=("cv_score", "validation_accuracy", "local_score")),
            "submit": _first_value(memory, decision, keys=("lb_score", "submitted_lb_score")),
        },
        "plan": {
            "type": plan.get("plan_type"),
            "title": plan.get("plan_title"),
            "change_axis": plan.get("primary_change_axis") or memory.get("change_axis"),
        },
        "pipeline": {
            "metric": structure.get("metric") or metrics.get("metric"),
            "objective": structure.get("objective") or metrics.get("objective"),
            "stages": [_compact_stage(stage) for stage in stages if stage.get("included")],
        },
        "decision": {
            "recommended_base_trial": decision.get("recommended_base_trial") or source_trial_id,
            "active_axis": decision.get("active_axis"),
            "axis_attempt_count": decision.get("axis_attempt_count"),
            "axis_attempt_limit": decision.get("axis_attempt_limit"),
            "rejected_axes": decision.get("rejected_axes", []),
            "rejected_candidates": decision.get("rejected_candidates", []),
        },
        "code": {
            "symbols": _snapshot_symbols(source),
        },
    }


def write_base_summary(competition: str, trial_id: str, source_trial_id: str | None) -> dict[str, Any]:
    summary = build_base_summary(competition, source_trial_id)
    if summary:
        write_text(
            trial_dir(competition, trial_id) / "base_summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
    return summary


def enrich_delta_plan(plan: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(plan)
    axis_text = str(enriched.get("primary_change_axis") or "").lower()
    fallback_text = " ".join(
        str(item)
        for item in [
            enriched.get("primary_change_axis"),
            enriched.get("plan_title"),
            enriched.get("candidate"),
            enriched.get("change_details"),
        ]
    ).lower()
    inferred_stages: list[str] = []
    inferred_symbols: list[str] = []
    matched_axis = any(marker in axis_text for marker in AXIS_IMPACT_MAP)
    impact_text = axis_text if matched_axis else fallback_text
    for marker, impact in AXIS_IMPACT_MAP.items():
        if marker not in impact_text:
            continue
        inferred_stages.extend(impact["stages"])
        inferred_symbols.extend(impact["symbols"])
    enriched["affected_stages"] = _unique(
        _string_list(enriched.get("affected_stages")) or inferred_stages or ["training", "evaluation"]
    )
    enriched["required_code_symbols"] = _unique(
        _string_list(enriched.get("required_code_symbols")) or inferred_symbols or ["build_pipeline", "train_validate"]
    )
    enriched["expected_metadata_changes"] = _unique(
        _string_list(enriched.get("expected_metadata_changes"))
        or ["trial_id", "source_trial_id", "model", "features", "split"]
    )
    return enriched


def write_effective_trial_artifacts(
    competition: str,
    trial_id: str,
    *,
    pipeline_structure: dict[str, Any],
) -> dict[str, str]:
    out_dir = trial_dir(competition, trial_id)
    plan = _read_trial_json(out_dir, "demo_experiment_plan.json")
    handoff = _read_trial_json(out_dir, "workspace_coding_handoff.json")
    if not plan:
        plan = _legacy_plan_from_artifacts(out_dir, handoff)
    source_trial_id = str(
        plan.get("source_trial_id")
        or handoff.get("recommended_base_trial")
        or handoff.get("code_base_trial_id")
        or handoff.get("source_trial_id")
        or ""
    ) or None
    if source_trial_id and not plan.get("source_trial_id"):
        plan["source_trial_id"] = source_trial_id
    base_effective = _load_base_effective_plan(competition, source_trial_id)
    delta = _read_json(out_dir / "delta_plan.json")
    if delta and not delta.get("schema_version"):
        delta = dict(delta)
        delta["affected_stages"] = []
        delta["required_code_symbols"] = []
        delta["expected_metadata_changes"] = []
        delta = enrich_delta_plan(delta)
        delta["schema_version"] = "1.0"
        write_text(out_dir / "delta_plan.json", json.dumps(delta, ensure_ascii=False, indent=2) + "\n")
    if not delta and source_trial_id:
        delta = enrich_delta_plan(_delta_from_plan(plan))
        delta["schema_version"] = "1.0"
        write_text(out_dir / "delta_plan.json", json.dumps(delta, ensure_ascii=False, indent=2) + "\n")
    effective = _merge_effective_plan(
        competition,
        trial_id,
        plan=plan,
        base_effective=base_effective,
        delta=delta,
    )
    internal = out_dir / "internal"
    write_text(internal / "effective_plan.json", json.dumps(effective, ensure_ascii=False, indent=2) + "\n")

    base_structure = _read_json(trial_dir(competition, source_trial_id) / "internal" / "pipeline_structure.json") if source_trial_id else {}
    changes = _pipeline_changes(base_structure, pipeline_structure)
    if source_trial_id and not base_structure:
        changes = [
            {
                "stage_id": stage_id,
                "before": {"status": "base_structure_unavailable"},
                "after": _comparable_stage(
                    next(
                        (stage for stage in pipeline_structure.get("stages", []) if stage.get("id") == stage_id),
                        {},
                    )
                ),
            }
            for stage_id in _string_list(delta.get("affected_stages"))
        ]
    pipeline_delta = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "source_trial_id": source_trial_id,
        "changes": changes,
    }
    write_text(
        internal / "pipeline_delta_applied.json",
        json.dumps(pipeline_delta, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "effective_plan_file": "internal/effective_plan.json",
        "pipeline_delta_file": "internal/pipeline_delta_applied.json",
    }


def _compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": stage.get("id"),
        "actual_applied": _string_list(stage.get("actual_applied"))[:6],
        "structured_details": stage.get("structured_details", {}),
        "code_locations": _string_list(stage.get("code_locations"))[:5],
    }


def _snapshot_symbols(source: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for relative, text in load_trial_code_snapshot(source):
        if not relative.endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if names:
            result[relative] = names
    return result


def _load_base_effective_plan(competition: str, source_trial_id: str | None) -> dict[str, Any]:
    if not source_trial_id:
        return {}
    source = trial_dir(competition, source_trial_id)
    effective = _read_json(source / "internal" / "effective_plan.json")
    if effective:
        return effective
    plan = _read_trial_json(source, "demo_experiment_plan.json")
    return {
        "baseline_trial_id": source_trial_id,
        "baseline_plan": plan,
        "change_history": [],
    }


def _merge_effective_plan(
    competition: str,
    trial_id: str,
    *,
    plan: dict[str, Any],
    base_effective: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    if not plan.get("source_trial_id"):
        return {
            "schema_version": "1.0",
            "competition": competition,
            "trial_id": trial_id,
            "baseline_trial_id": trial_id,
            "baseline_plan": plan,
            "current_delta": None,
            "change_history": [],
        }
    history = list(base_effective.get("change_history", []))
    history.append(
        {
            "trial_id": trial_id,
            "source_trial_id": plan.get("source_trial_id"),
            "primary_change_axis": delta.get("primary_change_axis") or plan.get("primary_change_axis"),
            "candidate": delta.get("candidate") or plan.get("candidate"),
            "change_details": delta.get("change_details") or plan.get("change_details", []),
        }
    )
    return {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "baseline_trial_id": base_effective.get("baseline_trial_id") or plan.get("source_trial_id"),
        "baseline_plan": base_effective.get("baseline_plan", {}),
        "current_delta": delta,
        "change_history": history,
    }


def _delta_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: plan.get(key)
        for key in [
            "plan_type",
            "source_trial_id",
            "primary_change_axis",
            "candidate",
            "change_details",
            "keep_unchanged",
            "affected_stages",
            "required_code_symbols",
            "expected_metadata_changes",
            "success_criteria",
            "failure_decision",
        ]
    }


def _legacy_plan_from_artifacts(out_dir: Path, handoff: dict[str, Any]) -> dict[str, Any]:
    markdown = _read_text(out_dir / "next_experiment.md")
    strategy = _markdown_section(markdown, "Strategy").strip()
    changes = _markdown_bullets(_markdown_section(markdown, "Changes"))
    guardrails = _markdown_bullets(_markdown_section(markdown, "Guardrails"))
    result = _read_trial_json(out_dir, "workspace_coding_result.json")
    source = handoff.get("recommended_base_trial") or handoff.get("code_base_trial_id") or handoff.get("source_trial_id")
    return {
        "plan_type": "continuation_delta_plan" if source else "initial_pipeline_plan",
        "source_trial_id": source,
        "plan_title": result.get("summary") or f"{strategy or 'pipeline'} improvement",
        "primary_change_axis": strategy,
        "candidate": {
            "name": strategy or "legacy_change",
            "description": result.get("summary") or "",
            "implementation_hint": "",
        },
        "change_details": changes,
        "keep_unchanged": guardrails,
        "success_criteria": [],
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return ""


def _markdown_section(text: str, heading: str) -> str:
    target = f"## {heading}".lower()
    collected: list[str] = []
    active = False
    for line in text.splitlines():
        if line.startswith("## "):
            if active:
                break
            active = line.strip().lower() == target
            continue
        if active:
            collected.append(line)
    return "\n".join(collected).strip()


def _markdown_bullets(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _pipeline_changes(base: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    base_stages = {str(stage.get("id")): stage for stage in base.get("stages", []) if isinstance(stage, dict)}
    current_stages = {str(stage.get("id")): stage for stage in current.get("stages", []) if isinstance(stage, dict)}
    changes: list[dict[str, Any]] = []
    for stage_id in sorted(set(base_stages) | set(current_stages)):
        before = _comparable_stage(base_stages.get(stage_id, {}))
        after = _comparable_stage(current_stages.get(stage_id, {}))
        if before != after:
            changes.append({"stage_id": stage_id, "before": before, "after": after})
    return changes


def _comparable_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "included": stage.get("included"),
        "actual_applied": stage.get("actual_applied", []),
        "structured_details": stage.get("structured_details", {}),
    }


def _read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    root = _read_json(out_dir / name)
    return root or _read_json(out_dir / "internal" / name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_value(*sources: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None and value != "":
                return value
    return None


def _string_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    return [str(item).strip() for item in values if str(item).strip()]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
