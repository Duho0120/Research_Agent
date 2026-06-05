from __future__ import annotations

import json
from typing import Any

from ..paths import competition_dir, trial_dir
from ..policies import load_policy
from ..store import load_state, read_text, write_text
from .result_analyst import diagnose_trial


def build_research_protocol(
    competition: str,
    trial_id: str,
    next_trial_id: str | None = None,
) -> dict[str, Any]:
    """Build the operating protocol snapshot before planning the next code experiment."""

    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = load_state(competition)
    profile = _load_data_profile(competition)
    diagnosis = diagnose_trial(competition, trial_id)
    policy = load_policy("research_operating_policy")

    objective = metrics.get("objective") or state.get("competition", {}).get("objective", "maximize")
    best = state.get("current_state", {}).get("best_trial", {})
    cv_score = metrics.get("cv_score")
    lb_score = metrics.get("lb_score")
    best_cv = best.get("cv_score") if isinstance(best, dict) else None
    best_lb = best.get("lb_score") if isinstance(best, dict) else None

    risk_flags = _risk_flags(metrics, diagnosis, profile, state, objective, policy, trial_id)
    risk_level = _risk_level(risk_flags)
    recommended_strategy = _recommended_strategy(risk_flags, diagnosis, metrics)
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "next_trial_id": next_trial_id,
        "current_state": {
            "objective": objective,
            "platform": state.get("competition", {}).get("platform", "unknown"),
            "active_trial": state.get("current_state", {}).get("active_trial"),
            "best_trial": best,
            "consecutive_failures": state.get("current_state", {}).get("consecutive_failures", 0),
            "validation_suspected": state.get("current_state", {}).get("validation_suspected", False),
        },
        "evidence": {
            "cv_score": cv_score,
            "lb_score": lb_score,
            "best_cv_before": best_cv,
            "best_lb_before": best_lb,
            "cv_improved": diagnosis.get("cv_improved"),
            "diagnosis_issues": diagnosis.get("issues", []),
            "task_type": profile.get("task_type", "unknown"),
            "train_rows": profile.get("train_rows"),
            "subjects": profile.get("subjects"),
            "target_columns": profile.get("target_columns", []),
        },
        "risk": {
            "level": risk_level,
            "flags": risk_flags,
            "summary": _risk_summary(risk_flags),
        },
        "candidate_actions": _candidate_actions(recommended_strategy, risk_flags, policy),
        "recommended_next_trial": {
            "trial_id": next_trial_id,
            "strategy": recommended_strategy,
            "reason": _strategy_reason(recommended_strategy, risk_flags),
        },
        "do_not_change": _do_not_change(recommended_strategy, risk_flags),
        "need_user_check": _need_user_check(risk_flags, state, profile),
        "execution_plan": _execution_plan(recommended_strategy, risk_flags, next_trial_id),
        "required_output_sections": policy.get("required_output_sections", []),
    }
    write_text(out_dir / "research_protocol.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "research_protocol.md", render_research_protocol(result))
    return result


def render_research_protocol(protocol: dict[str, Any]) -> str:
    lines = [
        f"# {protocol['trial_id']} Research Protocol",
        "",
        "## Current State",
        "",
        "```json",
        json.dumps(protocol["current_state"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(protocol["evidence"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Risk",
        "",
        f"- level: {protocol['risk']['level']}",
    ]
    lines.extend(f"- {item}" for item in protocol["risk"]["flags"] or ["none"])
    lines.extend(["", "## Candidate Actions", ""])
    for lane, actions in protocol["candidate_actions"].items():
        lines.append(f"### {lane}")
        lines.extend(f"- {item}" for item in actions)
        lines.append("")
    lines.extend(
        [
            "## Recommended Next Trial",
            "",
            f"- trial_id: {protocol['recommended_next_trial']['trial_id']}",
            f"- strategy: {protocol['recommended_next_trial']['strategy']}",
            f"- reason: {protocol['recommended_next_trial']['reason']}",
            "",
            "## Do Not Change",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["do_not_change"])
    lines.extend(["", "## Need User Check", ""])
    lines.extend(f"- {item}" for item in protocol["need_user_check"] or ["No immediate user check required."])
    lines.extend(["", "## Execution Plan", ""])
    lines.extend(f"- {item}" for item in protocol["execution_plan"])
    lines.append("")
    return "\n".join(lines)


def _load_data_profile(competition: str) -> dict[str, Any]:
    path = competition_dir(competition) / "data_profile.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _risk_flags(
    metrics: dict[str, Any],
    diagnosis: dict[str, Any],
    profile: dict[str, Any],
    state: dict[str, Any],
    objective: str,
    policy: dict[str, Any],
    trial_id: str,
) -> list[str]:
    flags: list[str] = []
    issues = " ".join(diagnosis.get("issues", [])).casefold()
    lb_score = metrics.get("lb_score")
    best = state.get("current_state", {}).get("best_trial", {})
    best_lb = best.get("lb_score") if isinstance(best, dict) else None
    cv_score = metrics.get("cv_score")
    best_cv = best.get("cv_score") if isinstance(best, dict) else None

    if "cv/lb" in issues:
        flags.append("cv_lb_conflict")
    if _local_improved(cv_score, best_cv, objective) and lb_score is not None and best_lb is not None:
        if (objective == "minimize" and lb_score >= best_lb) or (objective == "maximize" and lb_score <= best_lb):
            flags.extend(["cv_lb_conflict", "public_anchor_preserved"])
    is_current_best_trial = isinstance(best, dict) and best.get("trial_id") == trial_id
    if (_local_improved(cv_score, best_cv, objective) or is_current_best_trial) and lb_score is None:
        flags.append("local_best_public_unknown")
    if metrics.get("leakage_warning"):
        flags.append("leakage_suspected")
    if state.get("current_state", {}).get("validation_suspected"):
        flags.append("validation_suspected")
    if metrics.get("prediction_correlation_with_best", 0) >= 0.995:
        flags.append("low_prediction_diversity")
    if _small_data(profile, policy):
        flags.append("small_data_or_subject_count")
    if state.get("competition", {}).get("platform") not in {None, "", "kaggle"}:
        flags.append("external_platform_manual_submission")
    return _unique(flags)


def _local_improved(score: float | None, best: float | None, objective: str) -> bool:
    if score is None or best is None:
        return False
    return score < best if objective == "minimize" else score > best


def _small_data(profile: dict[str, Any], policy: dict[str, Any]) -> bool:
    train_rows = profile.get("train_rows")
    subjects = profile.get("subjects")
    row_threshold = policy.get("small_data_train_rows_threshold", 1000)
    subject_threshold = policy.get("small_subject_count_threshold", 20)
    return (isinstance(train_rows, int) and train_rows < row_threshold) or (
        isinstance(subjects, int) and subjects < subject_threshold
    )


def _risk_level(flags: list[str]) -> str:
    high = {"cv_lb_conflict", "leakage_suspected", "public_anchor_preserved"}
    medium = {"local_best_public_unknown", "validation_suspected", "small_data_or_subject_count"}
    if any(flag in high for flag in flags):
        return "high"
    if any(flag in medium for flag in flags):
        return "medium"
    return "low"


def _recommended_strategy(flags: list[str], diagnosis: dict[str, Any], metrics: dict[str, Any]) -> str:
    if "cv_lb_conflict" in flags or "leakage_suspected" in flags:
        return "validation_review"
    if "local_best_public_unknown" in flags:
        return "safe_submission_or_holdout_confirmation"
    if metrics.get("segment_errors") or diagnosis.get("needs_user_review"):
        return "error_analysis_human_review"
    if "low_prediction_diversity" in flags:
        return "diverse_candidate_search"
    return "controlled_refinement"


def _candidate_actions(strategy: str, flags: list[str], policy: dict[str, Any]) -> dict[str, list[str]]:
    default_checks = policy.get("default_probability_checks", [])
    if strategy == "validation_review":
        return {
            "safe": ["Audit validation split and leakage assumptions.", *default_checks],
            "main": ["Try conservative calibration or blend repair without changing model family."],
            "aggressive": ["Delay architecture/model-family changes until validation conflict is explained."],
        }
    if strategy == "safe_submission_or_holdout_confirmation":
        return {
            "safe": ["Record or request leaderboard/holdout evidence for the local best."],
            "main": ["Prepare a conservative next trial anchored to the trusted public baseline."],
            "aggressive": ["Postpone model-family changes until public evidence exists."],
        }
    if strategy == "error_analysis_human_review":
        return {
            "safe": ["Prepare error slices and review questions."],
            "main": ["Change one data, feature, sampling, or calibration axis based on the error pattern."],
            "aggressive": ["Consider model-family changes only after the error pattern is understood."],
        }
    return {
        "safe": ["Make one small controlled change and keep validation fixed."],
        "main": ["Use pipeline improvement planning to choose one primary axis."],
        "aggressive": ["Consider model or architecture changes only after repeated saturation evidence."],
    }


def _do_not_change(strategy: str, flags: list[str]) -> list[str]:
    items = ["Do not mix multiple primary improvement axes in one trial."]
    if strategy in {"validation_review", "safe_submission_or_holdout_confirmation"} or "validation_suspected" in flags:
        items.append("Do not change model family before resolving public evidence.")
    if "public_anchor_preserved" in flags or "local_best_public_unknown" in flags:
        items.append("Do not replace the trusted public baseline with local-only evidence.")
    if "small_data_or_subject_count" in flags:
        items.append("Do not trust high-capacity changes without strong validation evidence.")
    return items


def _need_user_check(flags: list[str], state: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    if "local_best_public_unknown" in flags:
        checks.append("Record or request leaderboard evidence before promoting the local best.")
    if "cv_lb_conflict" in flags:
        checks.append("Ask whether the validation split or submission strategy should change.")
    if "external_platform_manual_submission" in flags:
        checks.append("Confirm platform submission limits and how leaderboard evidence will be recorded.")
    if profile.get("target_columns"):
        checks.append("Confirm target semantics before changing target dependencies or classifier chains.")
    return checks


def _execution_plan(strategy: str, flags: list[str], next_trial_id: str | None) -> list[str]:
    plan = [
        "Write or update pipeline_improvement_plan before code changes.",
        "Create patch plan and validate it before coding handoff.",
        "Run validation commands before training or job creation.",
    ]
    if next_trial_id:
        plan.insert(0, f"Use `{next_trial_id}` as the next trial id.")
    if strategy == "safe_submission_or_holdout_confirmation":
        plan.append("Record manual or platform leaderboard result before aggressive follow-up.")
    if "cv_lb_conflict" in flags:
        plan.append("Run validation review before any model-family change.")
    return plan


def _risk_summary(flags: list[str]) -> str:
    if not flags:
        return "No major protocol risks detected."
    return "Protocol risks detected: " + ", ".join(flags) + "."


def _strategy_reason(strategy: str, flags: list[str]) -> str:
    if flags:
        return f"Selected because risk flags are present: {', '.join(flags)}."
    return f"Selected `{strategy}` because no higher-priority protocol risk was detected."


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
