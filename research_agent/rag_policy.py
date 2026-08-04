from __future__ import annotations

from typing import Any


USER_FEEDBACK_TRIGGERS = {
    "user_feedback",
    "human_feedback",
    "review_feedback",
    "feedback_text",
    "user_notes",
    "review_notes",
    "human_review_response",
    "user_feedback_path",
    "human_review_path",
}

BROAD_CONTEXT_TRIGGERS = {
    "force_rag",
    "use_rag",
    "external_research_needed",
    "literature_search_needed",
    "domain_research_needed",
    "needs_broad_context",
    "no_viable_axis",
    "all_axes_exhausted",
    "unexplained_failure",
    "ambiguous_error",
    "human_review_needed",
    "pending_human_review",
}


def evaluate_rag_policy(
    context: dict[str, Any],
    *,
    task: str,
    is_first_trial: bool,
) -> dict[str, Any]:
    """Decide whether retrieval should be used for an agent step.

    RAG is intentionally selective. Continuation trials should usually rely on
    structured decision context plus the selected base trial/code snapshot.
    Retrieval is reserved for bootstrapping, user feedback, human review, and
    broad-context situations that cannot be resolved from trial metadata.
    """

    if is_first_trial:
        return _decision(True, "first_trial_context_bootstrap", task)

    feedback_key = _find_trigger(context, USER_FEEDBACK_TRIGGERS)
    if feedback_key:
        return _decision(True, "user_feedback_or_human_review_present", task, feedback_key)

    broad_key = _find_trigger(context, BROAD_CONTEXT_TRIGGERS)
    if broad_key:
        return _decision(True, "broad_context_trigger_present", task, broad_key)

    decision_context = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
    feedback_key = _find_trigger(decision_context, USER_FEEDBACK_TRIGGERS)
    if feedback_key:
        return _decision(True, "user_feedback_or_human_review_present", task, f"decision_context.{feedback_key}")

    broad_key = _find_trigger(decision_context, BROAD_CONTEXT_TRIGGERS)
    if broad_key:
        return _decision(True, "broad_context_trigger_present", task, f"decision_context.{broad_key}")

    return _decision(False, "continuation_uses_structured_memory_and_code_snapshot", task)


def _decision(use_rag: bool, reason: str, task: str, trigger: str | None = None) -> dict[str, Any]:
    result = {
        "use_rag": use_rag,
        "task": task,
        "reason": reason,
    }
    if trigger:
        result["trigger"] = trigger
    return result


def _find_trigger(context: dict[str, Any], keys: set[str]) -> str | None:
    for key in keys:
        if _is_present(context.get(key)):
            return key
    return None


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized and normalized not in {"false", "no", "none", "null", "0"})
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    return True
