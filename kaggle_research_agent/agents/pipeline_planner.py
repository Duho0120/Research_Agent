from __future__ import annotations

import json
from typing import Any

from ..policies import load_policy
from ..store import load_state, write_text
from ..paths import competition_submissions_dir, trial_dir
from .result_analyst import diagnose_trial


def plan_pipeline_improvement(competition: str, trial_id: str) -> dict[str, Any]:
    """Choose the next pipeline improvement axis from trial evidence."""

    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    submission = _latest_submission(competition, trial_id)
    effective_lb_score = metrics.get("lb_score")
    if effective_lb_score is None and submission:
        effective_lb_score = submission.get("submitted_lb_score")
    effective_rank = metrics.get("rank")
    if effective_rank is None and submission:
        effective_rank = submission.get("submitted_rank")
    state = load_state(competition)
    diagnosis = diagnose_trial(competition, trial_id)
    policy = load_policy("pipeline_improvement_policy")

    primary_axis, secondary_axes, protected_axes, requires_human_review = _select_axes(
        metrics=metrics,
        diagnosis=diagnosis,
        consecutive_failures=int(state.get("current_state", {}).get("consecutive_failures", 0) or 0),
        protect_model_changes=bool(policy.get("protect_model_changes_when_validation_suspected", True)),
    )
    plan = {
        "competition": competition,
        "trial_id": trial_id,
        "primary_axis": primary_axis,
        "secondary_axes": secondary_axes,
        "protected_axes": protected_axes,
        "requires_human_review": requires_human_review,
        "candidate_actions": _candidate_actions(primary_axis),
        "success_criteria": _success_criteria(primary_axis),
        "do_not_change": _do_not_change(primary_axis, protected_axes),
        "rationale": _rationale(primary_axis, diagnosis, metrics),
        "evidence_used": {
            "cv_score": metrics.get("cv_score"),
            "lb_score": effective_lb_score,
            "rank": effective_rank,
            "leaderboard_source": "submission_log" if submission and metrics.get("lb_score") is None else "metrics",
            "issues": diagnosis.get("issues", []),
            "strategy_recommendation": diagnosis.get("strategy_recommendation"),
            "segment_errors_present": bool(metrics.get("segment_errors")),
            "prediction_correlation_with_best": metrics.get("prediction_correlation_with_best"),
        },
        "next_trial_rule": "Change one primary pipeline axis and keep protected axes fixed.",
    }
    write_text(out_dir / "pipeline_improvement_plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "pipeline_improvement_plan.md", render_pipeline_improvement_plan(plan))
    return plan


def _latest_submission(competition: str, trial_id: str) -> dict[str, Any] | None:
    path = competition_submissions_dir(competition) / "submission_log.jsonl"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("trial_id") == trial_id:
            rows.append(row)
    return rows[-1] if rows else None


def render_pipeline_improvement_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan['trial_id']} Pipeline Improvement Plan",
        "",
        "## Primary Axis",
        "",
        plan["primary_axis"],
        "",
        "## Secondary Axes",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["secondary_axes"] or ["None"])
    lines.extend(["", "## Protected Axes", ""])
    lines.extend(f"- {item}" for item in plan["protected_axes"] or ["None"])
    lines.extend(["", "## Candidate Actions", ""])
    lines.extend(f"- {item}" for item in plan["candidate_actions"])
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(f"- {item}" for item in plan["success_criteria"])
    lines.extend(["", "## Do Not Change", ""])
    lines.extend(f"- {item}" for item in plan["do_not_change"])
    lines.extend(
        [
            "",
            "## Human Review",
            "",
            f"- requires_human_review: {plan['requires_human_review']}",
            "",
            "## Rationale",
            "",
            plan["rationale"],
            "",
            "## Evidence Used",
            "",
            "```json",
            json.dumps(plan["evidence_used"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _select_axes(
    *,
    metrics: dict[str, Any],
    diagnosis: dict[str, Any],
    consecutive_failures: int,
    protect_model_changes: bool,
) -> tuple[str, list[str], list[str], bool]:
    issues = " ".join(diagnosis.get("issues", [])).casefold()
    corr = metrics.get("prediction_correlation_with_best")
    protected_axes: list[str] = []

    if "cv/lb" in issues or metrics.get("leakage_warning"):
        if protect_model_changes:
            protected_axes.extend(["model_family", "model_architecture", "pretraining_strategy"])
        return "validation", ["data_split", "leakage_check"], protected_axes, True

    if metrics.get("segment_errors"):
        return "error_analysis", ["preprocessing", "sampling", "human_review"], ["model_family"], True

    if consecutive_failures >= 3 or (corr is not None and corr >= 0.995):
        return "model_family", ["model_architecture", "pretraining_strategy"], ["validation"], True

    if metrics.get("class_imbalance") or metrics.get("class_errors"):
        return "sampling", ["loss_metric_alignment", "augmentation"], ["validation", "model_family"], False

    if metrics.get("threshold_sweep_needed") or "threshold" in issues:
        return "loss_metric_alignment", ["post_processing"], ["validation", "model_family"], False

    if metrics.get("data_quality_issues"):
        return "preprocessing", ["feature_engineering"], ["validation", "model_family"], True

    return "hyperparameter", ["training_recipe", "feature_engineering"], ["validation", "model_family"], False


def _candidate_actions(axis: str) -> list[str]:
    actions = {
        "validation": [
            "Audit data split assumptions before changing model family.",
            "Compare random, stratified, group, or time-aware data split candidates.",
            "Check leakage-sensitive columns or duplicated entities.",
        ],
        "error_analysis": [
            "Prepare representative error cases for human review.",
            "Break down errors by class, fold, group, scenario, view, or segment.",
            "Decide whether the concentrated errors imply preprocessing, sampling, or feature changes.",
        ],
        "model_family": [
            "Change the model family only after keeping validation fixed.",
            "Compare one diverse candidate against the current best.",
            "Review whether pretrained, partial fine-tuning, or full training is justified by the modality and data size.",
        ],
        "sampling": [
            "Adjust class or segment sampling weights.",
            "Test hard-example or minority-class sampling without changing validation.",
        ],
        "loss_metric_alignment": [
            "Align the training loss or thresholds with the competition metric.",
            "Run a threshold or calibration sweep before model-family changes.",
        ],
        "preprocessing": [
            "Fix data quality, normalization, missing values, or modality-specific preprocessing.",
            "Keep model family fixed while measuring preprocessing impact.",
        ],
        "hyperparameter": [
            "Tune one training parameter such as learning rate, regularization, batch size, or early stopping.",
            "Keep data split, model family, and major feature assumptions fixed.",
        ],
    }
    return actions.get(axis, ["Make one controlled pipeline change and preserve attribution."])


def _success_criteria(axis: str) -> list[str]:
    if axis == "validation":
        return ["CV/LB disagreement is explained or reduced.", "Future trial comparisons become more trustworthy."]
    if axis == "error_analysis":
        return ["A concrete error pattern is identified.", "The next trial has a narrower, evidence-backed hypothesis."]
    if axis == "model_family":
        return ["Predictions are meaningfully different from the current best.", "CV improves beyond known seed variance."]
    return ["The selected axis improves CV or diagnostic quality without changing protected axes."]


def _do_not_change(axis: str, protected_axes: list[str]) -> list[str]:
    items = [f"Do not change {item} in the same trial." for item in protected_axes]
    if axis != "validation":
        items.append("Do not change validation unless this is explicitly a validation review.")
    return items or ["Do not mix unrelated pipeline changes in the same trial."]


def _rationale(axis: str, diagnosis: dict[str, Any], metrics: dict[str, Any]) -> str:
    issues = diagnosis.get("issues", [])
    parts = [f"Selected `{axis}` as the primary pipeline improvement axis."]
    if issues:
        parts.append("Diagnosis issues: " + "; ".join(issues))
    if metrics.get("segment_errors"):
        parts.append("Segment-level errors indicate that aggregate metrics are not enough.")
    if metrics.get("prediction_correlation_with_best") is not None:
        parts.append(f"Prediction correlation with best: {metrics.get('prediction_correlation_with_best')}.")
    return " ".join(parts)
