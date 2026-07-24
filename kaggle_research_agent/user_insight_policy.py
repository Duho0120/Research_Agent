from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from . import paths
from .agents.code_writer_adapter import create_llm_client, provider_log_name
from .agents.memory import log_token_usage
from .paths import competition_memory_dir, competition_submissions_dir
from .policies import load_policy, resolve_model_for_call
from .store import write_text
from .trial_decision import MAX_AXIS_NO_IMPROVE_ATTEMPTS, load_latest_decision_context


class InsightClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


AXIS_KEYWORDS = {
    "model_ensemble": ["ensemble", "앙상블", "blend", "stack", "voting", "soft vote", "hard vote"],
    "validation_review": ["validation", "cv", "split", "leak", "overfit", "검증", "누수", "과적합"],
    "feature_engineering": ["feature", "title", "family", "cabin", "ticket", "파생변수", "피처", "특성"],
    "preprocessing": ["preprocess", "imputer", "imputation", "encoding", "encoder", "scaler", "normalization", "전처리"],
    "hyperparameter_tuning": ["hyperparameter", "parameter", "learning_rate", "max_depth", "regularization", "하이퍼파라미터"],
    "model_family": ["model", "forest", "boost", "xgboost", "lightgbm", "catboost", "모델"],
    "submission_strategy": ["submit", "submission", "leaderboard", "lb", "제출"],
}


def new_insight_id(competition: str, trial_id: str, text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha1(f"{competition}|{trial_id}|{text}|{stamp}".encode("utf-8")).hexdigest()[:8]
    return f"insight_{stamp}_{digest}"


def interpret_user_insight(
    text: str,
    *,
    competition: str | None = None,
    trial_id: str | None = None,
    allow_api: bool = False,
    client: InsightClient | None = None,
) -> dict[str, Any]:
    lowered = text.casefold()
    matched = {
        axis: [token for token in tokens if token.casefold() in lowered]
        for axis, tokens in AXIS_KEYWORDS.items()
    }
    matched = {axis: tokens for axis, tokens in matched.items() if tokens}
    if "model_ensemble" in matched:
        return _interpretation(
            "model_ensemble",
            text,
            confidence=0.98,
            source="rules",
            matched=matched["model_ensemble"],
        )
    if len(matched) == 1:
        axis = next(iter(matched))
        return _interpretation(axis, text, confidence=0.95, source="rules", matched=matched[axis])
    if matched:
        axis = max(matched, key=lambda item: (len(matched[item]), -list(AXIS_KEYWORDS).index(item)))
        rule_result = _interpretation(axis, text, confidence=0.6, source="rules_ambiguous", matched=matched[axis])
    else:
        rule_result = _interpretation("user_insight", text, confidence=0.25, source="rules_unclassified", matched=[])
    rule_result["needs_llm"] = True
    if not allow_api and client is None:
        return rule_result
    return _interpret_with_low_cost_llm(
        text,
        fallback=rule_result,
        competition=competition,
        trial_id=trial_id,
        client=client,
    )


def strategy_for_axis(axis: str) -> str:
    return {
        "model_ensemble": "model_ensemble",
        "validation_review": "validation_review",
        "feature_engineering": "feature_engineering",
        "preprocessing": "preprocessing",
        "hyperparameter_tuning": "hyperparameter_tuning",
        "model_family": "model_family_change",
        "submission_strategy": "submission_strategy",
    }.get(axis, "controlled_refinement")


def register_user_insight(feedback: dict[str, Any], interpretation: dict[str, Any]) -> dict[str, Any]:
    competition = str(feedback.get("competition") or "")
    trial_id = str(feedback.get("trial_id") or "")
    insight_id = str(feedback.get("insight_id") or new_insight_id(competition, trial_id, str(feedback.get("user_feedback") or "")))
    path = competition_memory_dir(competition) / "user_insights.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        if row.get("status") in {"pending", "planned", "applied", "evaluated", "application_mismatch"}:
            row["status"] = "superseded"
            row["superseded_by"] = insight_id
    record = {
        "schema_version": "1.0",
        "insight_id": insight_id,
        "competition": competition,
        "source_trial_id": trial_id,
        "raw_text": feedback.get("user_feedback"),
        "created_at": feedback.get("time"),
        "status": "pending",
        "scope": feedback.get("scope") or "next_trial",
        "interpretation": interpretation,
        "axis": interpretation.get("axis"),
        "target_trial": None,
        "base_trial_id": None,
        "baseline_lb_score": None,
        "attempt": 0,
        "attempt_limit": MAX_AXIS_NO_IMPROVE_ATTEMPTS,
        "applied_trials": [],
        "result": None,
    }
    rows.append(record)
    _write_jsonl(path, rows)
    return record


def migrate_legacy_user_insights(competition: str) -> dict[str, Any]:
    feedback_path = competition_memory_dir(competition) / "user_feedback.jsonl"
    feedback_rows = _read_jsonl(feedback_path)
    migrated: list[str] = []
    for feedback in feedback_rows:
        if feedback.get("topic") != "user_insight" or feedback.get("scope") != "next_trial":
            continue
        if feedback.get("insight_id"):
            continue
        source_trial_id = str(feedback.get("trial_id") or "")
        raw_text = str(feedback.get("user_feedback") or "")
        digest = hashlib.sha1(
            f"{competition}|{source_trial_id}|{feedback.get('time')}|{raw_text}".encode("utf-8")
        ).hexdigest()[:12]
        insight_id = f"insight_legacy_{digest}"
        feedback["competition"] = competition
        feedback["insight_id"] = insight_id
        interpretation = interpret_user_insight(raw_text)
        record = register_user_insight(feedback, interpretation)
        target_trial = _next_trial_id(source_trial_id)
        if target_trial:
            context = _load_json(paths.project_root() / "experiments" / competition / target_trial / "continuation_context.json")
            decision = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
            override = decision.get("user_insight_override") if isinstance(decision.get("user_insight_override"), dict) else {}
            update_user_insight_record(
                competition,
                insight_id,
                status="planned",
                target_trial=target_trial,
                base_trial_id=override.get("base_trial_id"),
                baseline_lb_score=_score(override.get("baseline_lb_score"), override.get("base_lb_score")),
                attempt=int(override.get("axis_attempt_count") or 1),
            )
            facts = _load_json(
                paths.project_root() / "experiments" / competition / target_trial / "internal" / "executed_trial_facts.json"
            )
            if facts:
                mark_user_insight_applied(competition, target_trial, facts)
                submitted = lb_score_for_trial(competition, target_trial)
                if submitted is not None:
                    evaluate_user_insight_submission(
                        competition,
                        target_trial,
                        submitted,
                        objective=str((facts.get("scores") or {}).get("objective") or "maximize"),
                    )
        migrated.append(str(record["insight_id"]))
    if migrated:
        _write_jsonl(feedback_path, feedback_rows)
    return {"competition": competition, "migrated": migrated, "count": len(migrated)}


def latest_user_insight_record(competition: str, source_trial_id: str | None = None) -> dict[str, Any] | None:
    rows = [
        row
        for row in _read_jsonl(competition_memory_dir(competition) / "user_insights.jsonl")
        if row.get("status") not in {"superseded", "completed", "exhausted"}
    ]
    if source_trial_id:
        exact = [row for row in rows if row.get("source_trial_id") == source_trial_id]
        if exact:
            return exact[-1]
    return rows[-1] if rows else None


def user_insight_record_by_id(competition: str, insight_id: str) -> dict[str, Any] | None:
    for row in reversed(_read_jsonl(competition_memory_dir(competition) / "user_insights.jsonl")):
        if row.get("insight_id") == insight_id:
            return row
    return None


def update_user_insight_record(competition: str, insight_id: str, **updates: Any) -> dict[str, Any] | None:
    path = competition_memory_dir(competition) / "user_insights.jsonl"
    rows = _read_jsonl(path)
    target = None
    for row in rows:
        if row.get("insight_id") == insight_id:
            row.update(updates)
            target = row
    if target is not None:
        _write_jsonl(path, rows)
    return target


def build_next_trial_user_insight_override(
    competition: str,
    source_trial_id: str,
    next_trial_id: str,
    *,
    allow_api: bool = False,
    insight_client: InsightClient | None = None,
) -> dict[str, Any] | None:
    insight = latest_next_trial_user_insight(competition, source_trial_id)
    if not insight:
        return None

    record = latest_user_insight_record(competition, source_trial_id)
    feedback_insight_id = str(insight.get("insight_id") or "")
    if record is None and feedback_insight_id:
        terminal = user_insight_record_by_id(competition, feedback_insight_id)
        if terminal and terminal.get("status") in {"superseded", "completed", "exhausted"}:
            return None
    if record is None:
        interpretation = interpret_user_insight(
            str(insight.get("user_feedback") or ""),
            competition=competition,
            trial_id=source_trial_id,
            allow_api=allow_api,
            client=insight_client,
        )
        feedback_record = dict(insight)
        feedback_record.setdefault("competition", competition)
        feedback_record.setdefault("trial_id", source_trial_id)
        feedback_record.setdefault("scope", "next_trial")
        record = register_user_insight(feedback_record, interpretation)
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    if interpretation.get("needs_llm") and allow_api:
        interpretation = interpret_user_insight(
            str(record.get("raw_text") or insight.get("user_feedback") or ""),
            competition=competition,
            trial_id=source_trial_id,
            allow_api=True,
            client=insight_client,
        )
        record = update_user_insight_record(
            competition,
            str(record["insight_id"]),
            interpretation=interpretation,
            axis=interpretation.get("axis"),
        ) or record
    axis = str(interpretation.get("axis") or axis_from_user_insight(str(insight.get("user_feedback") or "")))
    insight_id = str(record.get("insight_id") or insight.get("insight_id") or "")
    prior = _latest_prior_user_insight_context(competition, axis, insight_id=insight_id)
    best = best_submitted_trial_by_lb(competition)
    objective = str((best or {}).get("objective") or "maximize").lower()
    if objective not in {"maximize", "minimize"}:
        objective = "maximize"

    source_score = lb_score_for_trial(competition, source_trial_id)
    prior_baseline_score = _score((prior or {}).get("baseline_lb_score"), (prior or {}).get("base_lb_score"))
    best_score = _score((best or {}).get("lb_score"))
    source_improved = (
        _is_better(source_score, prior_baseline_score, objective)
        if prior_baseline_score is not None
        else False
    )
    prior_count = int((prior or {}).get("axis_attempt_count") or 0)
    prior_base_trial = str((prior or {}).get("base_trial_id") or "")
    base_trial_id = source_trial_id if source_improved else str((best or {}).get("trial_id") or prior_base_trial or source_trial_id)
    baseline_lb_score = source_score if source_improved else (prior_baseline_score if prior_baseline_score is not None else best_score)
    attempt_count = 1 if source_improved or prior_count <= 0 else min(prior_count + 1, MAX_AXIS_NO_IMPROVE_ATTEMPTS)

    exhausted = (
        prior_count >= MAX_AXIS_NO_IMPROVE_ATTEMPTS
        and not source_improved
        and bool(prior)
    )
    if exhausted:
        update_user_insight_record(competition, insight_id, status="exhausted", attempt=prior_count)
        return {
            "status": "exhausted",
            "reason": "user_insight_axis_attempt_limit_reached",
            "user_feedback": insight,
            "active_axis": axis,
            "axis_attempt_count": prior_count,
            "axis_attempt_limit": MAX_AXIS_NO_IMPROVE_ATTEMPTS,
            "base_trial_id": base_trial_id,
            "next_trial_id": next_trial_id,
            "insight_id": insight_id,
        }

    decision_context = load_latest_decision_context(competition)
    previous_axis = decision_context.get("active_axis")
    override = {
        "status": "active",
        "reason": "user_insight_override",
        "competition": competition,
        "source_trial_id": source_trial_id,
        "next_trial_id": next_trial_id,
        "user_feedback": insight,
        "user_insight": insight.get("user_feedback"),
        "insight_id": insight_id,
        "interpretation": interpretation,
        "active_axis": axis,
        "axis_attempt_count": attempt_count,
        "axis_attempt_limit": MAX_AXIS_NO_IMPROVE_ATTEMPTS,
        "base_trial_id": base_trial_id,
        "best_trial_by_submission": best,
        "source_lb_score": source_score,
        "base_lb_score": lb_score_for_trial(competition, base_trial_id),
        "baseline_lb_score": baseline_lb_score,
        "previous_active_axis": previous_axis,
        "previous_active_axis_status": (
            "paused/superseded_by_user_insight" if previous_axis and previous_axis != axis else None
        ),
        "source_improved_submission_score": source_improved,
        "policy": [
            "When next_trial user insight exists, choose the best submitted trial by LB/public score as base.",
            "Prioritize the user insight axis over any existing active axis for the next trial.",
            "Do not reject the previous active axis; mark it paused/superseded_by_user_insight.",
            "Start a new user-insight axis at attempt 1/3.",
            "If the next submitted score improves, keep the same user-insight axis.",
            "If it does not improve, try variants of the same user-insight axis up to 3 attempts.",
        ],
    }
    update_user_insight_record(
        competition,
        insight_id,
        status="planned",
        target_trial=next_trial_id,
        base_trial_id=base_trial_id,
        baseline_lb_score=baseline_lb_score,
        attempt=attempt_count,
    )
    return override


def latest_next_trial_user_insight(competition: str, source_trial_id: str | None = None) -> dict[str, Any] | None:
    path = competition_memory_dir(competition) / "user_feedback.jsonl"
    rows = _read_jsonl(path)
    scoped = [
        row
        for row in rows
        if row.get("topic") == "user_insight" and row.get("scope") == "next_trial" and row.get("user_feedback")
    ]
    if not scoped:
        return None
    if source_trial_id:
        for row in reversed(scoped):
            if row.get("trial_id") == source_trial_id:
                return row
    return scoped[-1]


def axis_from_user_insight(text: str) -> str:
    return str(interpret_user_insight(text).get("axis") or "user_insight")


def best_submitted_trial_by_lb(competition: str) -> dict[str, Any] | None:
    rows = submitted_score_rows(competition)
    rows = [row for row in rows if _score(row.get("lb_score")) is not None]
    if not rows:
        return None
    objective = str(rows[-1].get("objective") or "maximize").lower()
    if objective == "minimize":
        return min(rows, key=lambda row: float(row["lb_score"]))
    return max(rows, key=lambda row: float(row["lb_score"]))


def lb_score_for_trial(competition: str, trial_id: str | None) -> float | None:
    if not trial_id:
        return None
    candidates = [row for row in submitted_score_rows(competition) if row.get("trial_id") == trial_id]
    if not candidates:
        return None
    scored = [_score(row.get("lb_score")) for row in candidates]
    scored = [value for value in scored if value is not None]
    return scored[-1] if scored else None


def submitted_score_rows(competition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_manual_metrics_rows(competition))
    rows.extend(_manual_summary_rows(competition))
    rows.extend(_submission_log_rows(competition))
    rows.extend(_trial_submission_rows(competition))
    return _dedupe_rows(rows)


def _manual_metrics_rows(competition: str) -> list[dict[str, Any]]:
    base = paths.project_root() / "demo_workspaces" / competition / "manual_trials"
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("trial_*/metrics.json")):
        value = _load_json(path)
        if not value:
            continue
        trial_id = str(value.get("trial_id") or path.parent.name)
        rows.append(
            {
                "trial_id": trial_id,
                "local_score": _score(value.get("local_score"), value.get("cv_score")),
                "lb_score": _score(value.get("kaggle_lb_score"), value.get("lb_score")),
                "objective": value.get("objective") or "maximize",
                "source": str(path.as_posix()),
            }
        )
    return rows


def _manual_summary_rows(competition: str) -> list[dict[str, Any]]:
    path = paths.project_root() / "demo_workspaces" / competition / "manual_trials" / "summary.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "trial_id": row.get("trial_id"),
                    "local_score": _score(row.get("local_score")),
                    "lb_score": _score(row.get("kaggle_lb_score"), row.get("publicScore")),
                    "objective": "maximize",
                    "source": str(path.as_posix()),
                }
            )
    return rows


def _submission_log_rows(competition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(competition_submissions_dir(competition) / "submission_log.jsonl"):
        rows.append(
            {
                "trial_id": row.get("trial_id"),
                "local_score": _score(row.get("cv_score"), row.get("local_score")),
                "lb_score": _score(row.get("submitted_lb_score"), row.get("lb_score")),
                "objective": row.get("objective") or "maximize",
                "source": "submission_log",
            }
        )
    return rows


def _trial_submission_rows(competition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial_path in sorted((paths.project_root() / "experiments" / competition).glob("trial_*")):
        trial_id = trial_path.name
        metrics = _load_json(trial_path / "metrics.json")
        run = _load_json(trial_path / "submission_run.json")
        score = _score(
            run.get("submitted_lb_score"),
            run.get("lb_score"),
            (run.get("recorded_submission") or {}).get("submitted_lb_score")
            if isinstance(run.get("recorded_submission"), dict)
            else None,
            _parse_submitted_lb_from_markdown(trial_path / "submission_result.md"),
        )
        if score is None:
            continue
        rows.append(
            {
                "trial_id": trial_id,
                "local_score": _score(metrics.get("cv_score"), metrics.get("local_score")),
                "lb_score": score,
                "objective": metrics.get("objective") or run.get("objective") or "maximize",
                "source": str(trial_path.as_posix()),
            }
        )
    return rows


def _latest_prior_user_insight_context(
    competition: str,
    axis: str,
    *,
    insight_id: str | None = None,
) -> dict[str, Any] | None:
    base = paths.project_root() / "experiments" / competition
    contexts: list[dict[str, Any]] = []
    for path in sorted(base.glob("trial_*/continuation_context.json")):
        context = _load_json(path)
        decision = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
        override = decision.get("user_insight_override") if isinstance(decision.get("user_insight_override"), dict) else {}
        same_insight = not insight_id or not override.get("insight_id") or override.get("insight_id") == insight_id
        if override.get("active_axis") == axis and override.get("status") == "active" and same_insight:
            contexts.append(override)
    return contexts[-1] if contexts else None


def mark_user_insight_applied(competition: str, trial_id: str, facts: dict[str, Any]) -> dict[str, Any] | None:
    record = _insight_for_target_trial(competition, trial_id)
    if not record:
        return None
    axis = str(record.get("axis") or "")
    actual_axis = str(facts.get("primary_change_axis") or "")
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    contract = _verification_contract(interpretation, axis)
    base_facts = _load_base_execution_facts(competition, record)
    verification = _evaluate_verification_contract(
        contract,
        phase="execution",
        candidate=facts,
        base=base_facts,
    )
    applied = verification["passed"]
    applied_trials = list(record.get("applied_trials") or [])
    if trial_id not in applied_trials and applied:
        applied_trials.append(trial_id)
    return update_user_insight_record(
        competition,
        str(record["insight_id"]),
        status="applied" if applied else "application_mismatch",
        applied_trials=applied_trials,
        application_check={
            "trial_id": trial_id,
            "expected_axis": axis,
            "actual_axis": actual_axis,
            "applied": applied,
            "verification": verification,
        },
    )


def evaluate_user_insight_submission(
    competition: str,
    trial_id: str,
    submitted_lb_score: float | None,
    *,
    objective: str = "maximize",
) -> dict[str, Any] | None:
    record = _insight_for_target_trial(competition, trial_id)
    if not record or submitted_lb_score is None:
        return record
    baseline = _score(record.get("baseline_lb_score"))
    improved = _is_better(_score(submitted_lb_score), baseline, objective) if baseline is not None else None
    attempt = int(record.get("attempt") or 0)
    status = "completed" if improved else ("exhausted" if attempt >= MAX_AXIS_NO_IMPROVE_ATTEMPTS else "evaluated")
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    verification = _evaluate_verification_contract(
        _verification_contract(interpretation, str(record.get("axis") or "")),
        phase="submission",
        candidate={"result": {"submitted_lb_score": submitted_lb_score}},
        base={},
    )
    return update_user_insight_record(
        competition,
        str(record["insight_id"]),
        status=status,
        result={
            "trial_id": trial_id,
            "baseline_lb_score": baseline,
            "submitted_lb_score": submitted_lb_score,
            "objective": objective,
            "improved": improved,
            "verification": verification,
        },
    )


def validate_user_insight_code_result(
    competition: str,
    trial_id: str,
    coding_result: dict[str, Any],
) -> list[str]:
    record = _insight_for_target_trial(competition, trial_id)
    if not record:
        return []
    axis = str(record.get("axis") or "")
    interpretation = record.get("interpretation") if isinstance(record.get("interpretation"), dict) else {}
    contract = _verification_contract(interpretation, axis)
    verification = _evaluate_verification_contract(
        contract,
        phase="code",
        candidate={"serialized": json.dumps(coding_result, ensure_ascii=False).casefold()},
        base={},
    )
    if verification["passed"]:
        return []
    return [
        f"user_insight_not_implemented:{axis}:{rule_id}"
        for rule_id in verification["failed_rule_ids"]
    ]


def _insight_for_target_trial(competition: str, trial_id: str) -> dict[str, Any] | None:
    rows = _read_jsonl(competition_memory_dir(competition) / "user_insights.jsonl")
    matches = [row for row in rows if row.get("target_trial") == trial_id and row.get("status") != "superseded"]
    return matches[-1] if matches else None


def user_insight_target_trial_ids(competition: str) -> set[str]:
    """Return trials that were planned from an active or completed user insight."""

    rows = _read_jsonl(competition_memory_dir(competition) / "user_insights.jsonl")
    return {
        str(row["target_trial"])
        for row in rows
        if row.get("target_trial") and row.get("status") != "superseded"
    }


def _default_verification_contract(axis: str) -> dict[str, Any]:
    """Return phase-aware checks without coupling validation to one user example."""

    common_axis_rule = {
        "id": "planned_axis_executed",
        "phase": "execution",
        "field": "primary_change_axis",
        "operator": "equals",
        "value": strategy_for_axis(axis),
    }
    contracts: dict[str, dict[str, Any]] = {
        "model_ensemble": {
            "expected_changes": ["model"],
            "protected_components": ["features", "validation", "submission"],
            "rules": [
                {
                    "id": "model_change_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": [
                        "votingclassifier",
                        "stackingclassifier",
                        "votingregressor",
                        "stackingregressor",
                        "voting=",
                        "blend",
                    ],
                },
                common_axis_rule,
                {
                    "id": "executed_model_is_ensemble",
                    "phase": "execution",
                    "field": "model.estimator",
                    "operator": "contains_any",
                    "values": ["voting", "stacking", "ensemble", "blend"],
                },
                {
                    "id": "ensemble_has_multiple_members",
                    "phase": "execution",
                    "field": "model.members",
                    "operator": "min_items",
                    "value": 2,
                },
            ],
        },
        "model_family": {
            "expected_changes": ["model"],
            "protected_components": ["features", "validation", "submission"],
            "rules": [
                {
                    "id": "model_delta_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": ["classifier(", "regressor(", "model =", "estimator="],
                },
                {
                    "id": "model_definition_changed",
                    "phase": "execution",
                    "field": "model",
                    "operator": "changed_from_base",
                },
                common_axis_rule,
            ],
        },
        "feature_engineering": {
            "expected_changes": ["features", "preprocessing"],
            "protected_components": ["validation", "submission"],
            "rules": [
                {
                    "id": "feature_delta_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": ["feature", "derive", "transform", "assign(", "column"],
                },
                {
                    "id": "feature_definition_changed",
                    "phase": "execution",
                    "field": "features",
                    "operator": "changed_from_base",
                },
                common_axis_rule,
            ],
        },
        "validation_review": {
            "expected_changes": ["validation"],
            "protected_components": ["submission"],
            "rules": [
                {
                    "id": "validation_delta_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": ["train_test_split", "cross_val", "kfold", "test_size", "stratify", "groupkfold"],
                },
                {
                    "id": "validation_configuration_changed",
                    "phase": "execution",
                    "field": "validation",
                    "operator": "changed_from_base",
                },
                common_axis_rule,
            ],
        },
        "preprocessing": {
            "expected_changes": ["preprocessing"],
            "protected_components": ["validation", "submission"],
            "rules": [
                {
                    "id": "preprocessing_delta_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": ["imputer", "encoder", "scaler", "preprocess", "transform"],
                },
                {
                    "id": "preprocessing_changed",
                    "phase": "execution",
                    "field": "preprocessing",
                    "operator": "changed_from_base",
                },
                common_axis_rule,
            ],
        },
        "hyperparameter_tuning": {
            "expected_changes": ["model"],
            "protected_components": ["features", "validation", "submission"],
            "rules": [
                {
                    "id": "parameter_delta_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": [
                        "max_depth",
                        "learning_rate",
                        "n_estimators",
                        "regularization",
                        "solver",
                        "penalty",
                    ],
                },
                {
                    "id": "model_parameters_changed",
                    "phase": "execution",
                    "field": "model.parameters",
                    "operator": "changed_from_base",
                },
                common_axis_rule,
            ],
        },
        "submission_strategy": {
            "expected_changes": ["submission"],
            "protected_components": [],
            "rules": [
                common_axis_rule,
                {
                    "id": "submission_evidence_recorded",
                    "phase": "submission",
                    "field": "result.submitted_lb_score",
                    "operator": "recorded",
                },
            ],
        },
        "user_insight": {
            "expected_changes": ["planner_defined_delta"],
            "protected_components": [],
            "rules": [common_axis_rule],
        },
    }
    return contracts.get(axis, contracts["user_insight"])


def _verification_contract(interpretation: dict[str, Any], axis: str) -> dict[str, Any]:
    contract = interpretation.get("verification_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("rules"), list):
        return _default_verification_contract(axis)
    return contract


def _evaluate_verification_contract(
    contract: dict[str, Any],
    *,
    phase: str,
    candidate: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failed: list[str] = []
    for raw_rule in contract.get("rules", []):
        if not isinstance(raw_rule, dict) or raw_rule.get("phase") != phase:
            continue
        rule_id = str(raw_rule.get("id") or "unnamed_rule")
        actual = _nested_value(candidate, str(raw_rule.get("field") or ""))
        reference = _nested_value(base, str(raw_rule.get("field") or ""))
        passed = _evaluate_rule(raw_rule, actual, reference)
        checks.append(
            {
                "rule_id": rule_id,
                "field": raw_rule.get("field"),
                "operator": raw_rule.get("operator"),
                "passed": passed,
                "actual": actual,
                "reference": reference if raw_rule.get("operator") == "changed_from_base" else None,
            }
        )
        if not passed:
            failed.append(rule_id)
    if phase == "execution":
        for component in contract.get("protected_components", []):
            field = {
                "submission_format": "submission",
                "validated_pipeline_until_evidence_supports_change": "",
            }.get(str(component), str(component))
            if not field:
                continue
            actual = _nested_value(candidate, field)
            reference = _nested_value(base, field)
            if reference in (None, "", [], {}):
                continue
            rule_id = f"protected_component_unchanged:{component}"
            passed = actual == reference
            checks.append(
                {
                    "rule_id": rule_id,
                    "field": field,
                    "operator": "unchanged_from_base",
                    "passed": passed,
                    "actual": actual,
                    "reference": reference,
                }
            )
            if not passed:
                failed.append(rule_id)
    return {
        "phase": phase,
        "passed": not failed,
        "failed_rule_ids": failed,
        "checks": checks,
        "deferred_phases": sorted(
            {
                str(rule.get("phase"))
                for rule in contract.get("rules", [])
                if isinstance(rule, dict) and rule.get("phase") not in {None, phase}
            }
        ),
    }


def _evaluate_rule(rule: dict[str, Any], actual: Any, reference: Any) -> bool:
    operator = str(rule.get("operator") or "")
    if operator == "equals":
        return actual == rule.get("value")
    if operator == "contains_any":
        text = json.dumps(actual, ensure_ascii=False).casefold()
        return any(str(value).casefold() in text for value in rule.get("values", []))
    if operator == "min_items":
        return isinstance(actual, (list, tuple, set, dict)) and len(actual) >= int(rule.get("value") or 0)
    if operator == "changed_from_base":
        return actual not in (None, "", [], {}) and reference not in (None, "", [], {}) and actual != reference
    if operator == "recorded":
        return actual not in (None, "", [], {})
    return False


def _nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _load_base_execution_facts(competition: str, record: dict[str, Any]) -> dict[str, Any]:
    base_trial_id = str(record.get("base_trial_id") or "")
    if not base_trial_id:
        return {}
    return _load_json(
        paths.project_root()
        / "experiments"
        / competition
        / base_trial_id
        / "internal"
        / "executed_trial_facts.json"
    )


def _next_trial_id(trial_id: str) -> str | None:
    match = re.fullmatch(r"trial_(\d+)", trial_id)
    if not match:
        return None
    return f"trial_{int(match.group(1)) + 1:03d}"


def _interpretation(
    axis: str,
    text: str,
    *,
    confidence: float,
    source: str,
    matched: list[str],
) -> dict[str, Any]:
    intents = {
        "model_ensemble": {
            "change": "Build a 2-3 member ensemble using diverse model families or calibrated variants.",
            "keep_unchanged": ["features", "validation", "submission_format"],
            "verification": "The executed estimator must be an ensemble with at least two members.",
        },
        "validation_review": {
            "change": "Audit or revise validation while keeping unrelated pipeline stages fixed.",
            "keep_unchanged": ["submission_format"],
            "verification": "Validation configuration must differ and be recorded.",
        },
        "feature_engineering": {
            "change": "Add or revise the feature logic named by the user.",
            "keep_unchanged": ["validation", "submission_format"],
            "verification": "Executed feature metadata must record the new feature logic.",
        },
        "preprocessing": {
            "change": "Change the requested preprocessing step while preserving unrelated stages.",
            "keep_unchanged": ["validation", "submission_format"],
            "verification": "Executed preprocessing metadata must differ from the base trial.",
        },
        "hyperparameter_tuning": {
            "change": "Change only the requested model or training parameters.",
            "keep_unchanged": ["features", "validation", "submission_format"],
            "verification": "Executed model parameters must differ from the base trial.",
        },
        "model_family": {
            "change": "Change the estimator family while preserving features and validation.",
            "keep_unchanged": ["features", "validation", "submission_format"],
            "verification": "Executed estimator must differ from the base estimator.",
        },
        "submission_strategy": {
            "change": "Prioritize submission-score evidence and CV/LB alignment.",
            "keep_unchanged": ["validated_pipeline_until_evidence_supports_change"],
            "verification": "Submission result and comparison baseline must be recorded.",
        },
        "user_insight": {
            "change": text,
            "keep_unchanged": ["validation", "submission_format"],
            "verification": "Planner must convert the free text into one concrete delta.",
        },
    }
    return {
        "axis": axis,
        "confidence": confidence,
        "source": source,
        "matched_keywords": matched,
        "needs_llm": confidence < 0.7,
        "implementation_intent": intents[axis],
        "verification_contract": _default_verification_contract(axis),
    }


def _interpret_with_low_cost_llm(
    text: str,
    *,
    fallback: dict[str, Any],
    competition: str | None,
    trial_id: str | None,
    client: InsightClient | None,
) -> dict[str, Any]:
    policy = load_policy("model_policy")
    model_selection = resolve_model_for_call(
        "user_insight_interpretation",
        policy=policy,
        model_env_var="RESEARCH_AGENT_INSIGHT_MODEL",
    )
    provider = str(model_selection.get("provider") or "openai")
    model = str(model_selection.get("model"))
    request = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Classify one ML research insight. Return only JSON with axis, "
                    "implementation_intent.change, implementation_intent.keep_unchanged, "
                    "implementation_intent.verification, and optionally a verification_contract "
                    "using only phase(code|execution|submission), field, and operators "
                    "(contains_any|changed_from_base|equals|min_items|recorded)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_insight": text,
                        "available_axes": list(AXIS_KEYWORDS) + ["user_insight"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "max_output_tokens": 350,
    }
    try:
        active_client = client or create_llm_client(provider)
        response = active_client.create_response(request)
        parsed = _parse_llm_json(response)
        axis = str(parsed.get("axis") or "")
        if axis not in {*AXIS_KEYWORDS, "user_insight"}:
            return fallback
        result = _interpretation(axis, text, confidence=0.8, source="low_cost_llm", matched=[])
        if isinstance(parsed.get("implementation_intent"), dict):
            result["implementation_intent"].update(parsed["implementation_intent"])
        if isinstance(parsed.get("verification_contract"), dict) and isinstance(
            parsed["verification_contract"].get("rules"), list
        ):
            result["verification_contract"] = parsed["verification_contract"]
        result["needs_llm"] = False
        usage = response.get("usage") if isinstance(response, dict) else None
        if competition and trial_id and isinstance(usage, dict):
            result["token_usage"] = log_token_usage(
                competition,
                trial_id,
                provider=provider_log_name(provider),
                model=model,
                call_type="user_insight_interpretation",
                usage=usage,
                request_id=response.get("id"),
            )
        return result
    except Exception as error:
        fallback = dict(fallback)
        fallback["llm_error"] = str(error)
        return fallback


def _parse_llm_json(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list):
        for item in output:
            for content in item.get("content", []) if isinstance(item, dict) else []:
                text = content.get("text") if isinstance(content, dict) else None
                if isinstance(text, str):
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
                    value = json.loads(cleaned)
                    return value if isinstance(value, dict) else {}
    return {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for row in rows:
        trial_id = str(row.get("trial_id") or "").strip()
        score = _score(row.get("lb_score"))
        if not trial_id or score is None:
            continue
        key = (trial_id, score)
        if key in seen:
            continue
        seen.add(key)
        copy = dict(row)
        copy["trial_id"] = trial_id
        copy["lb_score"] = score
        result.append(copy)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_submitted_lb_from_markdown(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    match = re.search(r"submitted_lb_score:\s*([0-9.]+)", text)
    return _score(match.group(1)) if match else None


def _score(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool) or value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_better(current: float | None, reference: float | None, objective: str) -> bool:
    if current is None or reference is None:
        return False
    return current < reference if objective == "minimize" else current > reference
