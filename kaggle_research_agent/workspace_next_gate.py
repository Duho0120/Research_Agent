from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.memory import log_decision, request_user_review
from .agents.research_planner import propose_next_experiment
from .paths import trial_dir
from .store import write_text
from .trial_decision import load_latest_decision_context
from .user_insight_policy import build_next_trial_user_insight_override


CONTINUABLE_STATUSES = {"completed", "completed_review_deferred", "already_processed"}


def plan_next_workspace_trial(
    competition: str,
    source_trial_id: str,
    next_trial_id: str,
    *,
    allow_api: bool = False,
    insight_client: Any | None = None,
) -> dict[str, Any]:
    source_dir = trial_dir(competition, source_trial_id)
    cycle = _load_result_cycle(source_dir)
    if not cycle:
        return _finish(
            source_dir,
            {
                "competition": competition,
                "source_trial_id": source_trial_id,
                "next_trial_id": next_trial_id,
                "status": "blocked_missing_result_cycle",
                "continuation_mode": "must_wait",
                "pending_human_review": False,
                "issues": ["missing_or_invalid_workspace_result_cycle"],
                "next_action": "process-workspace-result",
            },
        )

    source_status = cycle.get("status")
    if source_status == "blocked":
        return _finish(
            source_dir,
            {
                "competition": competition,
                "source_trial_id": source_trial_id,
                "next_trial_id": next_trial_id,
                "status": "blocked_source_result",
                "continuation_mode": "must_wait",
                "pending_human_review": False,
                "issues": cycle.get("issues", []) or ["workspace_result_cycle_blocked"],
                "source_result_status": source_status,
                "next_action": cycle.get("next_action", "resolve-source-result"),
            },
        )

    if source_status == "awaiting_human_review":
        _register_review_request(competition, source_trial_id, cycle)
        if _is_blocking_review(cycle):
            return _finish(
                source_dir,
                {
                    "competition": competition,
                    "source_trial_id": source_trial_id,
                    "next_trial_id": next_trial_id,
                    "status": "blocked_human_review",
                    "continuation_mode": "must_wait",
                    "pending_human_review": True,
                    "issues": _review_issues(cycle),
                    "source_result_status": source_status,
                    "review_request_path": str((source_dir / "user_review_request.md").as_posix()),
                    "next_action": "wait-for-user-feedback",
                },
            )
        return _plan_with_context(
            competition,
            source_trial_id,
            next_trial_id,
            source_dir,
            cycle,
            status="planned_with_pending_review",
            continuation_mode="continue_with_caution",
            pending_human_review=True,
            allow_api=allow_api,
            insight_client=insight_client,
        )

    if source_status in CONTINUABLE_STATUSES:
        pending = source_status == "completed_review_deferred"
        return _plan_with_context(
            competition,
            source_trial_id,
            next_trial_id,
            source_dir,
            cycle,
            status="planned_with_deferred_review" if pending else "planned",
            continuation_mode="continue_with_caution" if pending else "can_continue",
            pending_human_review=pending,
            allow_api=allow_api,
            insight_client=insight_client,
        )

    return _finish(
        source_dir,
        {
            "competition": competition,
            "source_trial_id": source_trial_id,
            "next_trial_id": next_trial_id,
            "status": "blocked_unknown_result_status",
            "continuation_mode": "must_wait",
            "pending_human_review": False,
            "issues": [f"unsupported_workspace_result_status:{source_status}"],
            "source_result_status": source_status,
            "next_action": "inspect-workspace-result-cycle",
        },
    )


def _plan_with_context(
    competition: str,
    source_trial_id: str,
    next_trial_id: str,
    source_dir: Path,
    cycle: dict[str, Any],
    *,
    status: str,
    continuation_mode: str,
    pending_human_review: bool,
    allow_api: bool = False,
    insight_client: Any | None = None,
) -> dict[str, Any]:
    decision_context = load_latest_decision_context(competition)
    user_insight_override = build_next_trial_user_insight_override(
        competition,
        source_trial_id,
        next_trial_id,
        allow_api=allow_api,
        insight_client=insight_client,
    )
    plan = propose_next_experiment(
        competition,
        source_trial_id,
        next_trial_id,
        user_insight_override=user_insight_override,
    )
    if user_insight_override:
        decision_context["user_insight_override"] = user_insight_override
        if user_insight_override.get("status") == "active":
            decision_context["active_axis"] = user_insight_override.get("active_axis")
            decision_context["axis_attempt_count"] = user_insight_override.get("axis_attempt_count")
            decision_context["axis_attempt_limit"] = user_insight_override.get("axis_attempt_limit")
            decision_context["recommended_base_trial"] = user_insight_override.get("base_trial_id")
            decision_context["previous_active_axis_status"] = user_insight_override.get("previous_active_axis_status")
            decision_context["planner_constraints"] = list(decision_context.get("planner_constraints") or []) + [
                "User insight has next_trial scope and overrides the previous active axis for this plan.",
                "Use submitted leaderboard score/public score to choose the base trial.",
                "Treat the previous active axis as paused/superseded_by_user_insight, not rejected.",
            ]
            _append_user_insight_policy_to_plan(trial_dir(competition, next_trial_id), user_insight_override)
    context = {
        "competition": competition,
        "source_trial_id": source_trial_id,
        "recommended_base_trial": decision_context.get("recommended_base_trial"),
        "next_trial_id": next_trial_id,
        "continuation_mode": continuation_mode,
        "pending_human_review": pending_human_review,
        "review_source_trial": source_trial_id if pending_human_review else None,
        "source_result_status": cycle.get("status"),
        "human_review": cycle.get("human_review", {}),
        "blocked_topics": _blocked_topics(cycle) if continuation_mode == "must_wait" else [],
        "allowed_topics": _allowed_topics(continuation_mode),
        "decision_context": decision_context,
    }
    next_dir = trial_dir(competition, next_trial_id)
    write_text(next_dir / "continuation_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(next_dir / "continuation_context.md", _render_continuation_context(context))
    return _finish(
        source_dir,
        {
            "competition": competition,
            "source_trial_id": source_trial_id,
            "next_trial_id": next_trial_id,
            "status": status,
            "continuation_mode": continuation_mode,
            "pending_human_review": pending_human_review,
            "issues": [],
            "source_result_status": cycle.get("status"),
            "next_experiment": {
                "next_trial_id": plan["next_trial_id"],
                "strategy": plan["strategy"],
                "requires_user_review_before_submit": plan["requires_user_review_before_submit"],
                "user_insight_override": user_insight_override,
            },
            "continuation_context": context,
            "next_action": "prepare-next-trial",
        },
    )


def _append_user_insight_policy_to_plan(next_dir: Path, override: dict[str, Any]) -> None:
    path = next_dir / "next_experiment.md"
    try:
        text = path.read_text(encoding="utf-8").rstrip()
    except FileNotFoundError:
        return
    user_feedback = override.get("user_insight") or ""
    lines = [
        text,
        "",
        "## User Insight Override Policy",
        "",
        f"- status: {override.get('status')}",
        f"- insight_id: {override.get('insight_id')}",
        f"- user_insight_axis: {override.get('active_axis')}",
        f"- base_trial_by_submission_score: {override.get('base_trial_id')}",
        f"- axis_attempt_count: {override.get('axis_attempt_count')}/{override.get('axis_attempt_limit')}",
        f"- previous_active_axis_status: {override.get('previous_active_axis_status') or 'none'}",
        f"- user_insight: {user_feedback}",
        f"- interpretation: {json.dumps(override.get('interpretation') or {}, ensure_ascii=False)}",
        "- instruction: prioritize this user insight axis for the next trial; keep the previous axis paused, not rejected.",
        "",
    ]
    write_text(path, "\n".join(lines))


def _load_result_cycle(source_dir: Path) -> dict[str, Any] | None:
    path = source_dir / "workspace_result_cycle.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


def _register_review_request(competition: str, source_trial_id: str, cycle: dict[str, Any]) -> None:
    diagnosis = cycle.get("diagnosis")
    if isinstance(diagnosis, dict):
        request_user_review(competition, source_trial_id, diagnosis)


def _is_blocking_review(cycle: dict[str, Any]) -> bool:
    human_review = cycle.get("human_review", {})
    if not isinstance(human_review, dict):
        return False
    if human_review.get("urgent") is True:
        return True
    triggers = set(human_review.get("triggers", []))
    return bool(
        triggers
        & {
            "validation_or_leakage_suspected",
            "blocking_information_missing",
            "safety_false_negative",
        }
    )


def _review_issues(cycle: dict[str, Any]) -> list[str]:
    diagnosis = cycle.get("diagnosis", {})
    if isinstance(diagnosis, dict) and diagnosis.get("issues"):
        return list(diagnosis["issues"])
    return ["pending_human_review_blocks_next_experiment"]


def _blocked_topics(cycle: dict[str, Any]) -> list[str]:
    triggers = cycle.get("human_review", {}).get("triggers", [])
    topics = []
    if "validation_or_leakage_suspected" in triggers:
        topics.append("validation_or_leakage")
    if "blocking_information_missing" in triggers:
        topics.append("missing_required_information")
    if "safety_false_negative" in triggers:
        topics.append("safety_critical_error")
    return topics


def _allowed_topics(continuation_mode: str) -> list[str]:
    if continuation_mode == "can_continue":
        return ["next_experiment_planning", "code_patch_planning", "local_execution_after_validation"]
    if continuation_mode == "continue_with_caution":
        return [
            "controlled_refinement",
            "low_cost_validation",
            "baseline_cleanup",
            "non_submission_trial_planning",
        ]
    return []


def _finish(source_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    write_text(source_dir / "workspace_next_gate.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(source_dir / "workspace_next_gate.md", _render_gate_result(result))
    log_decision(
        result["competition"],
        result["source_trial_id"],
        decision_type="workspace_next_gate",
        decision=result["status"],
        reason="Workspace next-trial continuation gate evaluated the result-cycle status and review urgency.",
        evidence={
            "continuation_mode": result.get("continuation_mode"),
            "pending_human_review": result.get("pending_human_review"),
            "source_result_status": result.get("source_result_status"),
            "issues": result.get("issues", []),
        },
        next_action=result["next_action"],
    )
    return result


def _render_gate_result(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['source_trial_id']} Workspace Next Gate",
        "",
        f"- status: {result['status']}",
        f"- continuation_mode: {result['continuation_mode']}",
        f"- pending_human_review: {result['pending_human_review']}",
        f"- next_trial_id: {result['next_trial_id']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in result.get("issues", []) or ["No blocking issue."])
    lines.append("")
    return "\n".join(lines)


def _render_continuation_context(context: dict[str, Any]) -> str:
    lines = [
        f"# {context['next_trial_id']} Continuation Context",
        "",
        f"- source_trial_id: {context['source_trial_id']}",
        f"- continuation_mode: {context['continuation_mode']}",
        f"- pending_human_review: {context['pending_human_review']}",
        "",
        "## Allowed Topics",
        "",
    ]
    lines.extend(f"- {item}" for item in context["allowed_topics"])
    lines.extend(["", "## Blocked Topics", ""])
    lines.extend(f"- {item}" for item in context["blocked_topics"] or ["No blocked topic for this continuation."])
    lines.append("")
    return "\n".join(lines)
