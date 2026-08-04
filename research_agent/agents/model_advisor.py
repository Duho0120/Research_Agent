from __future__ import annotations

import json
from typing import Any

from ..paths import competition_dir, trial_dir
from ..store import load_state, write_text


def advise_model_candidates(competition: str, trial_id: str) -> dict[str, Any]:
    """Recommend model-family candidates without changing the experiment plan."""

    out_dir = trial_dir(competition, trial_id)
    profile = _load_data_profile(competition)
    metrics = _load_optional_json(out_dir / "metrics.json")
    pipeline_plan = _load_optional_json(out_dir / "pipeline_improvement_plan.json")
    state = load_state(competition)

    task_type = profile.get("task_type", "unknown")
    primary_axis = pipeline_plan.get("primary_axis")
    protected_axes = pipeline_plan.get("protected_axes", [])
    failures = int(state.get("current_state", {}).get("consecutive_failures", 0) or 0)
    model_change_protected = primary_axis == "validation" or "model_family" in protected_axes
    recommendation_scope = _recommendation_scope(primary_axis, model_change_protected, failures, metrics)

    candidates = _candidate_models(task_type, recommendation_scope)
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "task_type": task_type,
        "recommendation_scope": recommendation_scope,
        "model_change_protected": model_change_protected,
        "candidates": candidates,
        "guardrails": _guardrails(recommendation_scope, task_type, protected_axes),
        "rationale": _rationale(task_type, recommendation_scope, primary_axis, failures, metrics),
        "evidence_used": {
            "data_profile_status": profile.get("status"),
            "target_candidates": profile.get("target_candidates", []),
            "primary_pipeline_axis": primary_axis,
            "protected_axes": protected_axes,
            "consecutive_failures": failures,
            "cv_score": metrics.get("cv_score"),
            "lb_score": metrics.get("lb_score"),
            "prediction_correlation_with_best": metrics.get("prediction_correlation_with_best"),
        },
    }
    write_text(out_dir / "model_candidates.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "model_candidates.md", render_model_candidates(result))
    return result


def render_model_candidates(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Model Candidates",
        "",
        f"- task_type: {result['task_type']}",
        f"- recommendation_scope: {result['recommendation_scope']}",
        f"- model_change_protected: {result['model_change_protected']}",
        "",
        "## Candidates",
        "",
    ]
    for candidate in result["candidates"]:
        lines.extend(
            [
                f"### {candidate['model_family']}",
                "",
                f"- priority: {candidate['priority']}",
                f"- training_strategy: {candidate['training_strategy']}",
                f"- when_to_use: {candidate['when_to_use']}",
                f"- risk: {candidate['risk']}",
                "",
            ]
        )
    lines.extend(["## Guardrails", ""])
    lines.extend(f"- {item}" for item in result["guardrails"])
    lines.extend(["", "## Rationale", "", result["rationale"], "", "## Evidence Used", "", "```json"])
    lines.append(json.dumps(result["evidence_used"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def _load_data_profile(competition: str) -> dict[str, Any]:
    path = competition_dir(competition) / "data_profile.json"
    if not path.exists():
        return {"competition": competition, "status": "missing", "task_type": "unknown", "target_candidates": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recommendation_scope(
    primary_axis: str | None,
    model_change_protected: bool,
    failures: int,
    metrics: dict[str, Any],
) -> str:
    if model_change_protected:
        return "defer_model_change"
    if primary_axis in {"model_family", "model_architecture", "pretraining_strategy"}:
        return "model_family_exploration"
    if failures >= 3 or (metrics.get("prediction_correlation_with_best") or 0) >= 0.995:
        return "model_family_exploration"
    return "baseline_or_controlled_refinement"


def _candidate_models(task_type: str, scope: str) -> list[dict[str, Any]]:
    if task_type == "image":
        candidates = [
            {
                "model_family": "self_supervised_vision_backbone",
                "priority": "high" if scope == "model_family_exploration" else "medium",
                "training_strategy": "pretrained_finetune",
                "when_to_use": "Use when image generalization or representation quality is the main bottleneck.",
                "risk": "Can hide validation or preprocessing issues if introduced too early.",
            },
            {
                "model_family": "convnext_or_efficientnet",
                "priority": "medium",
                "training_strategy": "pretrained_finetune",
                "when_to_use": "Use as a strong supervised image baseline with stable training behavior.",
                "risk": "May be less diverse if the current best already uses a similar supervised backbone.",
            },
        ]
    elif task_type == "tabular":
        candidates = [
            {
                "model_family": "gradient_boosted_trees",
                "priority": "high",
                "training_strategy": "train_from_scratch",
                "when_to_use": "Use as the default strong tabular baseline after target and split are confirmed.",
                "risk": "Feature leakage and split mistakes can look like model gains.",
            },
            {
                "model_family": "regularized_linear_or_simple_tree",
                "priority": "medium",
                "training_strategy": "train_from_scratch",
                "when_to_use": "Use as a sanity-check baseline for metric and data pipeline verification.",
                "risk": "May underfit, but helps detect pipeline issues.",
            },
        ]
    elif task_type == "text":
        candidates = [
            {
                "model_family": "pretrained_text_encoder",
                "priority": "high" if scope == "model_family_exploration" else "medium",
                "training_strategy": "pretrained_finetune",
                "when_to_use": "Use when raw text semantics dominate feature quality.",
                "risk": "Requires careful validation and token-budget/runtime planning.",
            }
        ]
    else:
        candidates = [
            {
                "model_family": "simple_baseline_first",
                "priority": "high",
                "training_strategy": "train_from_scratch",
                "when_to_use": "Use until task type, target, and submission format are confirmed.",
                "risk": "Model advice is weak before data profile questions are resolved.",
            }
        ]

    if scope == "defer_model_change":
        for candidate in candidates:
            candidate["priority"] = "deferred"
    return candidates


def _guardrails(scope: str, task_type: str, protected_axes: list[str]) -> list[str]:
    guardrails = ["Keep validation fixed unless the selected pipeline axis is validation."]
    if scope == "defer_model_change":
        guardrails.append("Defer model-family changes until validation concerns are resolved.")
    if task_type in {"image", "text"}:
        guardrails.append("Prefer pretrained fine-tuning before full from-scratch training unless data scale justifies it.")
    if protected_axes:
        guardrails.append("Do not change protected axes in the same trial: " + ", ".join(protected_axes) + ".")
    return guardrails


def _rationale(
    task_type: str,
    scope: str,
    primary_axis: str | None,
    failures: int,
    metrics: dict[str, Any],
) -> str:
    parts = [f"Selected `{scope}` for a `{task_type}` task."]
    if primary_axis:
        parts.append(f"Current pipeline axis is `{primary_axis}`.")
    if failures >= 3:
        parts.append("Repeated failures justify considering a more diverse model family.")
    if metrics.get("prediction_correlation_with_best") is not None:
        parts.append(f"Prediction correlation with best is {metrics['prediction_correlation_with_best']}.")
    return " ".join(parts)
