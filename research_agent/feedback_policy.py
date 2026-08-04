from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .feedback_decision_support import (
    AXIS_INTERACTION_TYPES,
    build_decision_support,
    pending_action_payload,
)
from .state_db import (
    create_pending_action,
    get_review_policy_state,
    list_pending_actions,
    list_review_evidence,
    record_review_evidence,
    upsert_review_policy_state,
)
from .paths import trial_dir
from .store import write_text


URGENT_TRIGGERS = {
    "explicit_leakage",
    "label_boundary_ambiguous",
    "safety_false_negative",
    "blocking_information_missing",
}
AUTONOMOUS_INTERACTION = "autonomous_action"
EVIDENCE_WINDOW = 6
REQUEST_SCORE_THRESHOLD = 4.0
COOLDOWN_TRIALS = 4


def evaluate_feedback_policy(
    competition: str,
    trial_id: str,
    diagnosis: dict[str, Any],
    *,
    pipeline_readiness: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Decide whether user feedback is worth interrupting the research loop for."""

    support = build_decision_support(diagnosis)
    axis = str(support.get("axis") or "human_review")
    interaction_type = AXIS_INTERACTION_TYPES.get(axis, "domain_question")
    trial_number = _trial_number(trial_id)
    triggers = _detect_triggers(diagnosis)
    fingerprint = _fingerprint(axis, triggers)

    for trigger in triggers:
        record_review_evidence(
            {
                "competition_id": competition,
                "trial_id": trial_id,
                "trial_number": trial_number,
                "trigger": trigger,
                "axis": axis,
                "fingerprint": fingerprint,
                "evidence": _compact_evidence(diagnosis, support),
            },
            db_path,
        )

    evidence_rows = list_review_evidence(
        competition,
        minimum_trial_number=max(0, trial_number - EVIDENCE_WINDOW + 1),
        db_path=db_path,
    )
    counts = _trigger_counts(evidence_rows)
    evidence_summary = _evidence_summary(evidence_rows, triggers)
    pending = list_pending_actions(competition, db_path, status="pending")
    policy_state = get_review_policy_state(competition, db_path) or {}
    urgent = sorted(URGENT_TRIGGERS.intersection(triggers))
    score, score_reasons = _decision_value_score(
        diagnosis,
        triggers,
        counts,
        interaction_type=interaction_type,
    )

    action = "no_review"
    reason = "no_material_uncertainty"
    blocking = False
    if pending:
        action = "suppressed_pending"
        reason = "evidence_merged_with_existing_pending_request"
    elif urgent:
        action = "request_now"
        reason = "critical_evidence_requires_user_confirmation"
        blocking = True
    elif interaction_type == AUTONOMOUS_INTERACTION:
        action = "autonomous_continue"
        reason = "agent_has_a_safe_autonomous_action"
    elif _same_question_in_cooldown(policy_state, trial_number, fingerprint):
        action = "suppressed_cooldown"
        reason = "same_question_is_in_cooldown"
    elif score >= REQUEST_SCORE_THRESHOLD:
        action = "request_now"
        reason = "user_answer_has_high_expected_decision_value"
        blocking = True
    elif triggers:
        action = "accumulate"
        reason = "evidence_not_yet_actionable"

    timing = "request_now" if action == "request_now" else ("defer" if action == "accumulate" else "no_review")
    return {
        "competition": competition,
        "trial_id": trial_id,
        "decision": "prepare_review_pack" if action == "request_now" else action,
        "action": action,
        "timing": timing,
        "urgent": bool(urgent),
        "blocking": blocking,
        "pipeline_mature": bool(pipeline_readiness and pipeline_readiness.get("metrics_collected")),
        "pipeline_readiness": pipeline_readiness,
        "axis": axis,
        "interaction_type": interaction_type,
        "triggers": triggers,
        "fingerprint": fingerprint,
        "score": score,
        "score_threshold": REQUEST_SCORE_THRESHOLD,
        "score_reasons": score_reasons,
        "evidence_summary": evidence_summary,
        "questions": list(diagnosis.get("user_questions") or [])[:3],
        "next_action": "request_user_review" if action == "request_now" else "continue",
    }


def create_feedback_request(
    competition: str,
    trial_id: str,
    diagnosis: dict[str, Any],
    policy_decision: dict[str, Any],
    *,
    review_pack: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    existing = list_pending_actions(competition, db_path, status="pending")
    if existing:
        return existing[0]

    support = build_decision_support(diagnosis)
    support["evidence_summary"] = list(policy_decision.get("evidence_summary") or [])
    support["policy_score"] = policy_decision.get("score")
    support["policy_score_reasons"] = list(policy_decision.get("score_reasons") or [])
    options = list(support.get("options") or [])
    options.append(
        {
            "value": "use_default",
            "label": "기본 추천안으로 계속",
            "impact": str(support.get("default_if_no_response") or "에이전트의 기본 추천안을 적용합니다."),
            "follow_up_action": "apply_default_recommendation",
        }
    )
    support["options"] = options

    context_files = [
        {"label": "결과 분석", "path": f"experiments/{competition}/{trial_id}/diagnosis.md"},
        {"label": "실험 점수", "path": f"experiments/{competition}/{trial_id}/metrics.json"},
        {"label": "판단 근거", "path": f"experiments/{competition}/{trial_id}/review_pack/decision_support.json"},
    ]
    payload = pending_action_payload(support, context_files=context_files)
    payload.update(
        {
            "evidence_summary": support["evidence_summary"],
            "policy": {
                "action": policy_decision.get("action"),
                "score": policy_decision.get("score"),
                "threshold": policy_decision.get("score_threshold"),
                "score_reasons": policy_decision.get("score_reasons"),
                "triggers": policy_decision.get("triggers"),
                "fingerprint": policy_decision.get("fingerprint"),
            },
            "review_pack": review_pack or {},
            "apply_scope": "next_trial",
        }
    )
    if payload.get("questions"):
        payload["questions"][0]["choices"] = [
            {"value": item["value"], "label": item["label"]}
            for item in options
        ]

    action_id = _action_id(competition, trial_id, str(policy_decision.get("fingerprint") or "review"))
    action = create_pending_action(
        {
            "action_id": action_id,
            "competition_id": competition,
            "trial_id": trial_id,
            "action_type": "human_review",
            "priority": 100 if policy_decision.get("urgent") else 50,
            "message": support.get("title"),
            "payload": payload,
        },
        db_path,
    )
    pack_dir = trial_dir(competition, trial_id) / "review_pack"
    write_text(
        pack_dir / "decision_support.json",
        json.dumps(support, ensure_ascii=False, indent=2) + "\n",
    )
    manifest_path = pack_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        manifest["pending_action_id"] = action_id
        manifest["policy_fingerprint"] = policy_decision.get("fingerprint")
        write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    trial_number = _trial_number(trial_id)
    upsert_review_policy_state(
        {
            "competition_id": competition,
            "pending_action_id": action_id,
            "last_request_trial": trial_number,
            "last_fingerprint": policy_decision.get("fingerprint"),
            "cooldown_until_trial": trial_number + COOLDOWN_TRIALS,
        },
        db_path,
    )
    return action


def mark_feedback_request_applied(
    competition: str,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    state = get_review_policy_state(competition, db_path) or {"competition_id": competition}
    state["pending_action_id"] = None
    return upsert_review_policy_state(state, db_path)


def apply_feedback_response(
    competition: str,
    trial_id: str,
    *,
    request_id: str,
    feedback: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Apply recorded feedback to the source trial before resolving the request."""

    source_dir = trial_dir(competition, trial_id)
    cycle_path = source_dir / "workspace_result_cycle.json"
    cycle = _read_json(cycle_path)
    execution_supported = bool(payload.get("execution_supported", True))
    follow_up_action = str(feedback.get("follow_up_action") or "consider_in_next_plan")
    application_mode = "recorded_for_next_plan"
    if not execution_supported or follow_up_action == "record_constraint_only":
        application_mode = "constraint_recorded_only"

    application = {
        "status": "applied",
        "request_id": request_id,
        "competition": competition,
        "source_trial_id": trial_id,
        "apply_scope": str(payload.get("apply_scope") or feedback.get("scope") or "next_trial"),
        "application_mode": application_mode,
        "decision": feedback.get("decision"),
        "follow_up_action": follow_up_action,
        "execution_supported": execution_supported,
        "next_action": "resume-successor-planning",
    }
    write_text(
        source_dir / "feedback_application.json",
        json.dumps(application, ensure_ascii=False, indent=2) + "\n",
    )

    if cycle:
        cycle["status"] = "completed_feedback_applied"
        cycle["next_action"] = "plan-next-experiment"
        cycle["feedback_application"] = application
        human_review = cycle.get("human_review")
        if isinstance(human_review, dict):
            human_review["status"] = "resolved"
            human_review["resolved_request_id"] = request_id
        write_text(cycle_path, json.dumps(cycle, ensure_ascii=False, indent=2) + "\n")
    return application


def _detect_triggers(diagnosis: dict[str, Any]) -> list[str]:
    issue_items = diagnosis.get("issues", [])
    issue_items = issue_items if isinstance(issue_items, list) else [issue_items]
    structured_values: list[Any] = []
    for field in ("issue_codes", "trigger_codes"):
        value = diagnosis.get(field)
        structured_values.extend(value if isinstance(value, list) else [value] if value else [])
    structured_codes = {
        str(value).strip().casefold().replace("-", "_")
        for value in structured_values
        if value
    }
    for item in issue_items:
        if isinstance(item, dict):
            for key in ("code", "type", "category"):
                if item.get(key):
                    structured_codes.add(str(item[key]).strip().casefold().replace("-", "_"))
    issues = " ".join(
        str(item.get("message") or item.get("problem") or item)
        if isinstance(item, dict)
        else str(item)
        for item in issue_items
    ).casefold()
    triggers: list[str] = []
    if structured_codes & {"explicit_leakage", "target_leakage", "data_leakage"} or "leakage" in issues or "누수" in issues:
        triggers.append("explicit_leakage")
    if structured_codes & {"cv_lb_conflict", "validation_mismatch", "validation_split_issue"} or "cv/lb" in issues or "validation split" in issues or "검증 불일치" in issues:
        triggers.append("cv_lb_conflict")
    if structured_codes & {"concentrated_error", "segment_error"} or "concentrated" in issues or "segment" in issues or "구간 집중" in issues:
        triggers.append("concentrated_error")
    if structured_codes & {"label_boundary_ambiguous", "ambiguous_label"} or (
        ("label" in issues or "레이블" in issues) and ("ambiguous" in issues or "boundary" in issues or "모호" in issues)
    ):
        triggers.append("label_boundary_ambiguous")
    if structured_codes & {"safety_false_negative", "critical_false_negative"} or (
        ("false negative" in issues or "거짓 음성" in issues) and ("safety" in issues or "안전" in issues)
    ):
        triggers.append("safety_false_negative")
    if structured_codes & {"blocking_information_missing", "required_information_missing"} or (
        ("missing" in issues and "required" in issues) or "필수 정보 누락" in issues
    ):
        triggers.append("blocking_information_missing")
    return list(dict.fromkeys(triggers))


def _decision_value_score(
    diagnosis: dict[str, Any],
    triggers: list[str],
    counts: dict[str, int],
    *,
    interaction_type: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for trigger in triggers:
        count = counts.get(trigger, 0)
        if trigger == "cv_lb_conflict" and count >= 2:
            score += 2.0
            reasons.append(f"CV/LB 불일치가 최근 {count}회 반복됨 +2.0")
        elif trigger == "concentrated_error" and count >= 3:
            score += 2.0
            reasons.append(f"같은 유형의 집중 오류가 최근 {count}회 반복됨 +2.0")
    axis_attempt_count = int(diagnosis.get("axis_attempt_count") or 0)
    if axis_attempt_count >= 3:
        score += 1.75
        reasons.append(f"현재 개선축 시도 {axis_attempt_count}/3 이상 +1.75")
    if interaction_type != AUTONOMOUS_INTERACTION and triggers:
        score += 2.0
        reasons.append("사용자만 확정할 수 있는 도메인·제약 판단 +2.0")
    autonomous_options = int(diagnosis.get("autonomous_options") or 0)
    if autonomous_options >= 2:
        score -= 0.75
        reasons.append("안전한 자율 대안이 2개 이상 있음 -0.75")
    return round(max(0.0, score), 2), reasons


def _compact_evidence(diagnosis: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis": support.get("axis"),
        "problem": support.get("problem"),
        "cv_score": diagnosis.get("cv_score"),
        "best_cv_before": diagnosis.get("best_cv_before"),
        "lb_score": diagnosis.get("lb_score"),
        "best_lb_before": diagnosis.get("best_lb_before"),
        "consecutive_failures": diagnosis.get("consecutive_failures"),
        "segment_errors": diagnosis.get("segment_errors") or {},
        "overall_error_rate": diagnosis.get("overall_error_rate"),
        "leakage_features": diagnosis.get("leakage_features") or [],
        "representative_error_cases": list(
            diagnosis.get("representative_error_cases") or []
        )[:3],
        "issues": list(diagnosis.get("issues") or [])[:5],
    }


def _trigger_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        trigger = str(row.get("trigger") or "")
        counts[trigger] = counts.get(trigger, 0) + 1
    return counts


def _evidence_summary(rows: list[dict[str, Any]], triggers: list[str]) -> list[str]:
    summaries: list[str] = []
    for trigger in triggers:
        matched = [row for row in rows if row.get("trigger") == trigger]
        if matched:
            trial_ids = ", ".join(str(row.get("trial_id")) for row in matched)
            summaries.append(f"{trigger}: 최근 {len(matched)}회 ({trial_ids})")
    return summaries


def _same_question_in_cooldown(state: dict[str, Any], trial_number: int, fingerprint: str) -> bool:
    return bool(
        state.get("last_fingerprint") == fingerprint
        and int(state.get("cooldown_until_trial") or 0) >= trial_number
    )


def _trial_number(trial_id: str) -> int:
    try:
        return int(str(trial_id).rpartition("_")[2])
    except ValueError:
        return 0


def _fingerprint(axis: str, triggers: list[str]) -> str:
    raw = json.dumps({"axis": axis, "triggers": sorted(triggers)}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _action_id(competition: str, trial_id: str, fingerprint: str) -> str:
    safe = hashlib.sha256(f"{competition}:{trial_id}:{fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"review_{safe}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
