from __future__ import annotations

from copy import deepcopy
from typing import Any


_INTERACTION_CONTRACTS: dict[str, dict[str, Any]] = {
    "experiment_question": {
        "channel": "experiment_question",
        "access": "read_only",
        "scope": "selected_experiment",
        "research_state_mutations": [],
        "operational_side_effects": ["token_usage_telemetry_when_llm_used"],
        "requires_explicit_submit": False,
        "description": "Reads experiment evidence and cannot change plans, code, scores, or research decisions.",
    },
    "user_insight": {
        "channel": "user_insight",
        "access": "write_intent",
        "scope": "next_trial",
        "research_state_mutations": ["user_feedback", "user_insight"],
        "operational_side_effects": [],
        "requires_explicit_submit": True,
        "description": "Records an explicit user insight for interpretation during the next planning step.",
    },
    "human_review_response": {
        "channel": "human_review_response",
        "access": "resolve_request",
        "scope": "requested_trial",
        "research_state_mutations": ["human_feedback", "pending_request_status"],
        "operational_side_effects": [],
        "requires_explicit_submit": True,
        "requires_pending_request": True,
        "description": "Records a response only for an existing pending Human Review request and resolves that request.",
    },
}


def interaction_contract(channel: str) -> dict[str, Any]:
    try:
        return deepcopy(_INTERACTION_CONTRACTS[channel])
    except KeyError as error:
        raise ValueError(f"Unknown interaction channel: {channel}") from error
