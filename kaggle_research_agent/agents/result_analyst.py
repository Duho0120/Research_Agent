from __future__ import annotations


import json
from pathlib import Path
from typing import Any

from .. import simple_yaml
from ..paths import competition_configs_dir
from ..paths import trial_dir
from ..store import load_state, read_text, write_text


def evaluate_trial(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = load_state(competition)
    objective = metrics.get("objective") or state.get("competition", {}).get("objective", "maximize")
    best = state.get("current_state", {}).get("best_trial")
    best_score = _best_score_before_trial(best, trial_id, "cv_score")
    cv_score = metrics.get("cv_score")
    lb_score = metrics.get("lb_score")
    corr = metrics.get("prediction_correlation_with_best")

    cv_improved = _improved(cv_score, best_score, objective) if best_score is not None else True
    diversity_ok = corr is None or corr < 0.995
    leakage_risk = bool(metrics.get("leakage_warning", False))

    recommendation = "accept_as_candidate" if cv_improved and diversity_ok and not leakage_risk else "record_but_do_not_promote"
    if leakage_risk:
        recommendation = "investigate_leakage"
    elif not diversity_ok:
        recommendation = "consider_only_if_score_gain_is_large"

    report = {
        "trial_id": trial_id,
        "objective": objective,
        "cv_score": cv_score,
        "lb_score": lb_score,
        "best_cv_before": best_score,
        "cv_improved": cv_improved,
        "diversity_ok": diversity_ok,
        "leakage_risk": leakage_risk,
        "recommendation": recommendation,
        "notes": metrics.get("notes", ""),
    }
    write_text(out_dir / "evaluation.md", render_evaluation(report))
    write_text(out_dir / "reflection.md", render_reflection(report, read_text(out_dir / "plan.md")))
    return report


def _improved(score: float | None, best: float | None, objective: str) -> bool:
    if score is None or best is None:
        return False
    return score < best if objective == "minimize" else score > best


def render_evaluation(report: dict[str, Any]) -> str:
    return f"""# {report['trial_id']} Evaluation

## Scores

- objective: {report['objective']}
- cv_score: {report['cv_score']}
- lb_score: {report['lb_score']}
- best_cv_before: {report['best_cv_before']}

## Checks

- cv_improved: {report['cv_improved']}
- diversity_ok: {report['diversity_ok']}
- leakage_risk: {report['leakage_risk']}

## Recommendation

{report['recommendation']}

## Notes

{report['notes']}
"""


def render_reflection(report: dict[str, Any], plan_text: str) -> str:
    outcome = "improved locally" if report["cv_improved"] else "did not improve locally"
    return f"""# {report['trial_id']} Reflection

## Outcome

This trial {outcome}. Recommendation: `{report['recommendation']}`.

## Interpretation

The result should be compared against seed variance and, if submitted, public LB movement.

## Memory Update Candidate

- Trial {report['trial_id']} produced CV={report['cv_score']} and LB={report['lb_score']}.
- Keep validation assumptions unchanged unless repeated CV/LB disagreement appears.
"""



import json
from typing import Any

from ..paths import trial_dir
from ..store import load_recent_trials, load_state, write_text


def diagnose_trial(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = load_state(competition)
    recent = load_recent_trials(competition)

    objective = metrics.get("objective") or state.get("competition", {}).get("objective", "maximize")
    best = state.get("current_state", {}).get("best_trial")
    best_score = _best_score_before_trial(best, trial_id, "cv_score")
    best_lb_score = _best_score_before_trial(best, trial_id, "lb_score")
    if best_score is None:
        best_score = _best_recent_score(recent, "cv_score", objective)
    if best_lb_score is None:
        best_lb_score = _best_recent_score(recent, "lb_score", objective)
    cv_score = metrics.get("cv_score")
    lb_score = metrics.get("lb_score")
    corr = metrics.get("prediction_correlation_with_best")
    consecutive_failures = int(state.get("current_state", {}).get("consecutive_failures", 0))

    raw_cv_improved = _improved(cv_score, best_score, objective) if best_score is not None else True
    has_direction_conflict = _leaderboard_tracking_enabled(competition) and (
        cv_score is not None
        and lb_score is not None
        and _direction_conflict(cv_score, lb_score, best_score, best_lb_score, objective)
    )
    cv_improved = raw_cv_improved and not has_direction_conflict

    issues: list[str] = []
    improvement_candidates: list[str] = []
    user_questions: list[str] = []

    if not cv_improved:
        issues.append("CV did not improve against the current best trial.")
        improvement_candidates.append("Review whether the current method is saturated before another small tweak.")
    if has_direction_conflict:
        issues.append("CV/LB movement may be inconsistent.")
        user_questions.append(
            "\uc0ac\uc6a9\uc790\uc5d0\uac8c validation split \ub610\ub294 submission strategy "
            "\ubcc0\uacbd\uc774 \ud544\uc694\ud55c\uc9c0 \ud655\uc778\uc744 \uc694\uccad\ud569\ub2c8\ub2e4."
        )
    if corr is not None and corr >= 0.995:
        issues.append("Predictions are highly correlated with the current best trial.")
        improvement_candidates.append("Prefer diversity or model-family changes over another similar submission.")
    if metrics.get("leakage_warning"):
        issues.append("Leakage warning is present in metrics.")
        user_questions.append(
            "\uc0ac\uc6a9\uc790\uc5d0\uac8c leakage \uac00\ub2a5\uc131\uc774 \uc788\ub294 feature "
            "\ub610\ub294 data split \ud655\uc778\uc744 \uc694\uccad\ud569\ub2c8\ub2e4."
        )
    if metrics.get("segment_errors"):
        issues.append("Errors are concentrated in one or more segments/groups/folds/features/patterns.")
        user_questions.append(
            "\uc0ac\uc6a9\uc790\uc5d0\uac8c \uc9d1\uc911 \uc624\ub958 \uad6c\uac04\uc5d0 \ub300\ud55c "
            "\ub3c4\uba54\uc778 \ud310\ub2e8\uc744 \uc694\uccad\ud569\ub2c8\ub2e4."
        )
    if consecutive_failures >= 3:
        issues.append("Recent failures suggest strategy escalation is needed.")
        improvement_candidates.append("Prepare model-family, architecture, ensemble, or SOTA exploration candidates.")

    if not improvement_candidates:
        improvement_candidates.append("Continue controlled refinement while keeping validation stable.")

    needs_user_review = bool(user_questions) or consecutive_failures >= 3
    strategy_recommendation = "strategy_escalation" if consecutive_failures >= 3 else "continue_refinement"

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "objective": objective,
        "cv_score": cv_score,
        "lb_score": lb_score,
        "best_cv_before": best_score,
        "best_lb_before": best_lb_score,
        "cv_improved": cv_improved,
        "issues": issues,
        "improvement_candidates": improvement_candidates,
        "user_questions": user_questions,
        "needs_user_review": needs_user_review,
        "strategy_recommendation": strategy_recommendation,
        "consecutive_failures": consecutive_failures,
        "segment_errors": metrics.get("segment_errors") or {},
        "overall_error_rate": metrics.get("overall_error_rate"),
        "leakage_warning": bool(metrics.get("leakage_warning")),
        "leakage_features": (
            metrics.get("leakage_features")
            or metrics.get("suspected_leakage_features")
            or []
        ),
        "representative_error_cases": metrics.get("representative_error_cases") or [],
        "recent_trial_count": len(recent),
    }
    write_text(out_dir / "diagnosis.md", render_diagnosis(result))
    return result


def _improved(score: float | None, best: float | None, objective: str) -> bool:
    if score is None or best is None:
        return False
    return score < best if objective == "minimize" else score > best


def _best_score_before_trial(best: Any, trial_id: str, score_key: str) -> float | None:
    if not isinstance(best, dict):
        return None
    if best.get("trial_id") == trial_id:
        return None
    value = best.get(score_key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _best_recent_score(recent: list[dict[str, Any]], score_key: str, objective: str) -> float | None:
    values = [
        row.get(score_key)
        for row in recent
        if isinstance(row, dict)
        and isinstance(row.get(score_key), (int, float))
        and not isinstance(row.get(score_key), bool)
    ]
    if not values:
        return None
    return min(values) if objective == "minimize" else max(values)


def _direction_conflict(
    score: float,
    lb_score: float,
    best_score: float | None,
    best_lb_score: float | None,
    objective: str,
) -> bool:
    if best_score is None or best_lb_score is None:
        return False
    if objective == "minimize":
        return score < best_score and lb_score >= best_lb_score
    return score > best_score and lb_score <= best_lb_score


def _leaderboard_tracking_enabled(competition: str) -> bool:
    path = competition_configs_dir(competition) / "research_policy.yaml"
    if not path.exists():
        return False
    policy = simple_yaml.load(path, default={})
    return bool(policy.get("leaderboard_tracking", {}).get("enabled", False))


def render_diagnosis(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Diagnosis",
        "",
        "## Scores",
        "",
        f"- objective: {result['objective']}",
        f"- cv_score: {result['cv_score']}",
        f"- lb_score: {result['lb_score']}",
        f"- best_cv_before: {result['best_cv_before']}",
        f"- best_lb_before: {result['best_lb_before']}",
        f"- cv_improved: {result['cv_improved']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["No major issues detected."])
    lines.extend(["", "## Improvement Candidates", ""])
    lines.extend(f"- {item}" for item in result["improvement_candidates"])
    lines.extend(["", "## User Review", "", f"- needs_user_review: {result['needs_user_review']}"])
    lines.extend(f"- {item}" for item in result["user_questions"] or ["No user question required."])
    lines.extend(["", "## Strategy Recommendation", "", result["strategy_recommendation"], ""])
    return "\n".join(lines)


