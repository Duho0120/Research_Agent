from __future__ import annotations

import json
from typing import Any

from .. import simple_yaml
from ..paths import competition_configs_dir, competition_dir, trial_dir
from ..policies import load_policy
from ..store import load_state, load_trial_index, write_text
from .result_analyst import diagnose_trial


def build_research_protocol(
    competition: str,
    trial_id: str,
    next_trial_id: str | None = None,
) -> dict[str, Any]:
    """Build a small, domain-neutral research decision before code changes."""

    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = load_state(competition)
    profile = _load_data_profile(competition)
    diagnosis = diagnose_trial(competition, trial_id)
    global_policy = load_policy("research_operating_policy")
    competition_policy = _load_competition_policy(competition)

    objective = metrics.get("objective") or state.get("competition", {}).get("objective", "maximize")
    best = state.get("current_state", {}).get("best_trial", {})
    score = metrics.get("cv_score")
    best_score = _best_score_before_trial(competition, best, trial_id, "cv_score", objective)
    issues = list(diagnosis.get("issues", []))
    optional_evidence: dict[str, Any] = {}

    leaderboard_conflict = _apply_optional_leaderboard_check(
        competition,
        metrics,
        best,
        objective,
        competition_policy,
        issues,
        optional_evidence,
        current_trial_id=trial_id,
    )
    strategy = _choose_strategy(
        diagnosis,
        state,
        leaderboard_conflict=leaderboard_conflict,
        leaderboard_affects_strategy=bool(
            competition_policy.get("leaderboard_tracking", {}).get("affects_strategy", False)
        ),
    )

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "next_trial_id": next_trial_id,
        "current_state": {
            "objective": objective,
            "active_trial": state.get("current_state", {}).get("active_trial"),
            "best_trial": best,
            "consecutive_failures": state.get("current_state", {}).get("consecutive_failures", 0),
        },
        "evidence": {
            "score": score,
            "best_score_before": best_score,
            "improved": True if best_score is None and score is not None else _improved(score, best_score, objective),
            "task_type": profile.get("task_type", "unknown"),
            "diagnosis_issues": diagnosis.get("issues", []),
        },
        "issues": _unique(issues),
        "candidate_actions": _candidate_actions(strategy),
        "recommended_action": {
            "trial_id": next_trial_id,
            "strategy": strategy,
            "reason": _strategy_reason(strategy, issues),
        },
        "constraints": _constraints(strategy),
        "user_questions": _user_questions(diagnosis, competition_policy, leaderboard_conflict),
        "execution_plan": _execution_plan(next_trial_id),
        "optional_evidence": optional_evidence,
        "enabled_extensions": _enabled_extensions(competition_policy),
        "required_output_sections": global_policy.get("required_output_sections", []),
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
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["issues"] or ["No major issue detected."])
    lines.extend(["", "## Candidate Actions", ""])
    lines.extend(f"- {item}" for item in protocol["candidate_actions"])
    lines.extend(
        [
            "",
            "## Recommended Action",
            "",
            f"- trial_id: {protocol['recommended_action']['trial_id']}",
            f"- strategy: {protocol['recommended_action']['strategy']}",
            f"- reason: {protocol['recommended_action']['reason']}",
            "",
            "## Constraints",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["constraints"])
    lines.extend(["", "## User Questions", ""])
    lines.extend(f"- {item}" for item in protocol["user_questions"] or ["No immediate user question required."])
    lines.extend(["", "## Execution Plan", ""])
    lines.extend(f"- {item}" for item in protocol["execution_plan"])
    if protocol["optional_evidence"]:
        lines.extend(
            [
                "",
                "## Optional Competition Evidence",
                "",
                "```json",
                json.dumps(protocol["optional_evidence"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _load_data_profile(competition: str) -> dict[str, Any]:
    path = competition_dir(competition) / "data_profile.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_competition_policy(competition: str) -> dict[str, Any]:
    path = competition_configs_dir(competition) / "research_policy.yaml"
    return simple_yaml.load(path, default={}) if path.exists() else {}


def _apply_optional_leaderboard_check(
    competition: str,
    metrics: dict[str, Any],
    best: dict[str, Any],
    objective: str,
    competition_policy: dict[str, Any],
    issues: list[str],
    optional_evidence: dict[str, Any],
    *,
    current_trial_id: str,
) -> bool:
    settings = competition_policy.get("leaderboard_tracking", {})
    if not settings.get("enabled", False):
        return False
    leaderboard_score = metrics.get(settings.get("score_field", "lb_score"))
    score_field = settings.get("score_field", "lb_score")
    best_leaderboard_score = _best_score_before_trial(competition, best, current_trial_id, score_field, objective)
    best_cv_score = _best_score_before_trial(competition, best, current_trial_id, "cv_score", objective)
    optional_evidence["leaderboard_score"] = leaderboard_score
    optional_evidence["best_leaderboard_score"] = best_leaderboard_score
    conflict = (
        _improved(metrics.get("cv_score"), best_cv_score, objective)
        and leaderboard_score is not None
        and best_leaderboard_score is not None
        and not _improved(leaderboard_score, best_leaderboard_score, objective)
    )
    if conflict:
        issues.append("Local and leaderboard movement disagree.")
    return conflict


def _choose_strategy(
    diagnosis: dict[str, Any],
    state: dict[str, Any],
    *,
    leaderboard_conflict: bool,
    leaderboard_affects_strategy: bool,
) -> str:
    if leaderboard_conflict and leaderboard_affects_strategy:
        return "validation_review"
    if diagnosis.get("needs_user_review"):
        return "human_review"
    if state.get("current_state", {}).get("validation_suspected"):
        return "validation_review"
    if int(state.get("current_state", {}).get("consecutive_failures", 0) or 0) >= 3:
        return "strategy_change"
    return "controlled_improvement"


def _candidate_actions(strategy: str) -> list[str]:
    if strategy == "human_review":
        return ["Prepare the evidence needed for a focused user decision."]
    if strategy == "validation_review":
        return ["Review validation assumptions before changing the pipeline."]
    if strategy == "strategy_change":
        return ["Select one new improvement axis and keep the remaining assumptions fixed."]
    return [
        "Choose one primary improvement axis from the latest evidence.",
        "Keep unrelated pipeline assumptions unchanged for attribution.",
    ]


def _constraints(strategy: str) -> list[str]:
    constraints = [
        "Change only one primary improvement axis in the next trial.",
        "Do not bypass code validation or protected-file rules.",
    ]
    if strategy == "validation_review":
        constraints.append("Do not make a large model change until validation is reviewed.")
    return constraints


def _user_questions(
    diagnosis: dict[str, Any],
    competition_policy: dict[str, Any],
    leaderboard_conflict: bool,
) -> list[str]:
    questions = list(diagnosis.get("user_questions", []))
    settings = competition_policy.get("leaderboard_tracking", {})
    if leaderboard_conflict and settings.get("ask_user_on_conflict", True):
        questions.append("Should validation or submission assumptions be reviewed before the next trial?")
    return _unique(questions)


def _execution_plan(next_trial_id: str | None) -> list[str]:
    steps = []
    if next_trial_id:
        steps.append(f"Use `{next_trial_id}` as the next trial id.")
    steps.extend(
        [
            "Create the pipeline improvement plan.",
            "Validate the patch plan before code writing.",
            "Run validation commands before training.",
            "Evaluate and store the result before planning another trial.",
        ]
    )
    return steps


def _enabled_extensions(policy: dict[str, Any]) -> list[str]:
    enabled = []
    if policy.get("leaderboard_tracking", {}).get("enabled", False):
        enabled.append("leaderboard_tracking")
    return enabled


def _improved(score: float | None, best: float | None, objective: str) -> bool:
    if score is None or best is None:
        return False
    return score < best if objective == "minimize" else score > best


def _best_score_before_trial(
    competition: str,
    best: Any,
    trial_id: str,
    score_key: str,
    objective: str,
) -> float | None:
    if isinstance(best, dict) and best.get("trial_id") != trial_id:
        value = best.get(score_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    if not (isinstance(best, dict) and trial_id and best.get("trial_id") == trial_id):
        return None
    # `state.current_state.best_trial` already points at this same trial (it just
    # became the new best, e.g. via agents/memory.py:remember_trial, which runs
    # before this protocol is built for it). Recompute the best score among all
    # OTHER recorded trials instead of returning None here, or a CV/LB conflict
    # can never be detected for the exact trial where it matters most.
    return _best_other_trial_score(competition, trial_id, score_key, objective)


def _best_other_trial_score(
    competition: str,
    trial_id: str,
    score_key: str,
    objective: str,
) -> float | None:
    values = [
        row.get(score_key)
        for row in load_trial_index(competition)
        if row.get("trial_id") != trial_id
    ]
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not values:
        return None
    return min(values) if objective == "minimize" else max(values)


def _strategy_reason(strategy: str, issues: list[str]) -> str:
    if issues:
        return f"Selected `{strategy}` from the current diagnosis: {'; '.join(_unique(issues))}"
    return f"Selected `{strategy}` because no higher-priority issue was detected."


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
