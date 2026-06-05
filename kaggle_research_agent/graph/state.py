from __future__ import annotations

from typing import Any, TypedDict


class ResearchGraphState(TypedDict, total=False):
    competition: str
    trial_id: str
    create_job_request: bool
    backend: str
    run_now: bool
    command: str | None
    next_trial_id: str | None
    prepare_next_patch: bool
    apply_next_patch: bool
    next_run_command: str | None
    run_safe_chain: bool
    safe_chain_mock_response_file: str | None
    safe_chain_allow_api: bool
    safe_chain_model: str
    steps: list[str]
    status: str
    plan: dict[str, Any]
    config_errors: list[str]
    metrics_exists: bool
    evaluation: dict[str, Any]
    diagnosis: dict[str, Any]
    human_review: dict[str, Any]
    review_pack: dict[str, Any]
    memory: dict[str, Any]
    execution_decision: dict[str, Any]
    job: dict[str, Any]
    safe_execution_chain: dict[str, Any]
    next_experiment: dict[str, Any]
    patch_plan: dict[str, Any]
    next_code_edit: dict[str, Any]
