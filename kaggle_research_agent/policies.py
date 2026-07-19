from __future__ import annotations

from typing import Any

from . import simple_yaml
from .paths import policies_dir


DEFAULT_POLICIES: dict[str, dict[str, Any]] = {
    "token_policy": {
        "max_llm_calls_per_trial": 4,
        "max_strategy_calls_per_day": 20,
        "summarize_logs_before_llm": True,
        "use_rules_before_llm": True,
        "avoid_raw_training_logs_in_prompt": True,
        "require_user_approval_for_expensive_steps": True,
        "call_llm_when": [
            "new_best_trial",
            "three_failures_in_a_row",
            "validation_suspected",
            "high_error_concentration",
            "human_review_needed",
            "experiment_planning",
            "strategy_shift_required",
            "code_writing",
        ],
    },
    "execution_policy": {
        "default_backend": "local",
        "local_first": True,
        "ask_before_colab": True,
        "estimated_runtime_minutes_over": 30,
        "colab_when": {
            "require_gpu": True,
            "estimated_runtime_minutes_over": 30,
            "local_device_missing": True,
            "local_previous_run_failed_due_to_resource": True,
        },
        "local_failure_patterns": {
            "resource_cpu_memory": ["CUDA out of memory", "out of memory", "MemoryError"],
            "resource_gpu_missing": ["CUDA is not available", "No CUDA", "GPU not found"],
            "missing_file": ["No such file", "FileNotFoundError"],
            "missing_dependency": ["ModuleNotFoundError", "ImportError"],
            "permission_error": ["PermissionError", "Access is denied"],
        },
    },
    "human_review_policy": {
        "review_when": {
            "high_error_concentration": True,
            "label_boundary_ambiguous": True,
            "validation_or_leakage_suspected": True,
            "safety_false_negative": True,
            "strategy_shift_required": True,
            "repeated_failures": True,
        },
        "repeated_failures_threshold": 3,
        "blocking_safety_classes": ["Fall"],
        "question_types": [
            "label_question",
            "validation_question",
            "feature_question",
            "execution_approval",
            "submission_approval",
            "strategy_shift_question",
            "visual_semantic_question",
            "data_quality_question",
        ],
        "max_questions_per_review": 3,
        "default_review_action": "prepare_review_pack",
        "review_timing": {
            "minimum_completed_trials_for_nonurgent_review": 2,
            "immediate_triggers": [
                "validation_or_leakage_suspected",
                "label_boundary_ambiguous",
                "safety_false_negative",
                "blocking_information_missing",
            ],
        },
    },
    "pipeline_improvement_policy": {
        "axes": [
            "validation",
            "preprocessing",
            "feature_engineering",
            "augmentation",
            "sampling",
            "loss_metric_alignment",
            "hyperparameter",
            "training_recipe",
            "model_architecture",
            "model_family",
            "pretraining_strategy",
            "post_processing",
            "ensemble_submission",
            "compute_backend",
            "error_analysis",
            "human_review",
        ],
        "one_primary_axis_per_trial": True,
        "protect_model_changes_when_validation_suspected": True,
        "strategy_map": {
            "validation": "validation_review",
            "model_architecture": "sota_architecture_attempt",
            "model_family": "model_family_change",
            "pretraining_strategy": "sota_architecture_attempt",
            "hyperparameter": "controlled_refinement",
            "training_recipe": "controlled_refinement",
            "feature_engineering": "controlled_refinement",
            "augmentation": "controlled_refinement",
            "sampling": "controlled_refinement",
            "loss_metric_alignment": "controlled_refinement",
            "post_processing": "controlled_refinement",
            "error_analysis": "controlled_refinement",
        },
    },
    "research_operating_policy": {
        "required_output_sections": [
            "current_state",
            "evidence",
            "issues",
            "candidate_actions",
            "recommended_action",
            "constraints",
            "user_questions",
            "execution_plan",
        ],
        "one_primary_axis_per_trial": True,
        "require_validation_before_execution": True,
        "require_user_approval_for_expensive_or_external_actions": True,
    },
    "model_policy": {
        "high_cost": {
            "provider": "openai",
            "api": "responses",
            "model": "gpt-5.5",
            "call_types": [
                "experiment_planning",
                "code_writing",
                "workspace_code_writing",
                "complex_diagnosis",
                "research_strategy",
            ],
        },
        "low_cost": {
            "provider": "openai",
            "api": "responses",
            "model": "gpt-5.6-luna",
            "call_types": [
                "status_summary",
                "log_summary",
                "short_note_rewrite",
                "simple_context_summary",
            ],
        },
        "fallback": {
            "provider": "openai",
            "api": "responses",
            "model": "gpt-5.6-luna",
        },
    },
    "rag_policy": {
        "prefer_memory_cards": True,
        "avoid_user_view_for_llm": True,
        "tasks": {
            "experiment_planning": {
                "max_documents": 5,
                "max_chars_per_document": 900,
                "prompt_chars_per_document": 650,
                "preferred_source_kinds": [
                    "competition_data_card",
                    "data_profile",
                    "decision_card",
                    "trial_memory_card",
                    "trial_metrics",
                    "trial_record",
                    "trial_plan",
                    "pipeline_structure",
                    "competition_metric",
                    "competition_overview",
                ],
                "demote_source_kinds": [
                    "user_summary",
                    "user_plan",
                    "user_pipeline_structure",
                    "user_code_pipeline",
                    "user_result",
                ],
            },
            "workspace_code_writing": {
                "max_documents": 4,
                "max_chars_per_document": 1000,
                "prompt_chars_per_document": 700,
                "preferred_source_kinds": [
                    "competition_data_card",
                    "data_profile",
                    "decision_card",
                    "trial_memory_card",
                    "next_experiment",
                    "workspace_context_snapshot",
                    "pipeline_structure",
                    "trial_metrics",
                    "coding_result",
                ],
                "demote_source_kinds": [
                    "user_summary",
                    "user_plan",
                    "user_pipeline_structure",
                    "user_code_pipeline",
                    "user_result",
                ],
            },
        },
    },
    "artifact_policy": {
        "save_submission": True,
        "save_metrics": True,
        "save_code_snapshot": True,
        "save_pipeline_summary": True,
        "save_model": {
            "default": False,
            "allowed_when": [
                "required_for_separate_predict_command",
                "best_trial",
                "submitted_trial",
                "expensive_to_retrain",
                "ensemble_candidate",
                "user_requested",
            ],
            "require_reason": True,
            "prefer_small_serialized_artifact": True,
            "cleanup_non_best_models": True,
        },
        "notes": [
            "A trial should not persist trained model artifacts by default.",
            "Submission, metrics, code snapshot, and pipeline summary are the primary memory artifacts.",
            "If a model artifact is persisted, the plan/code should record why it is needed.",
        ],
    },
}


def load_policy(name: str) -> dict[str, Any]:
    if name not in DEFAULT_POLICIES:
        raise ValueError(f"Unknown policy: {name}")
    path = policies_dir() / f"{name}.yaml"
    loaded = simple_yaml.load(path, default=None)
    if not loaded:
        return dict(DEFAULT_POLICIES[name])
    return _deep_merge(DEFAULT_POLICIES[name], loaded)


def select_model_for_call(call_type: str, *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    model_policy = policy or load_policy("model_policy")
    for tier in ("high_cost", "low_cost"):
        section = model_policy.get(tier, {})
        if call_type in section.get("call_types", []):
            return {
                "tier": tier,
                "provider": section.get("provider"),
                "api": section.get("api"),
                "model": section.get("model"),
                "call_type": call_type,
            }
    fallback = model_policy.get("fallback", {})
    return {
        "tier": "fallback",
        "provider": fallback.get("provider"),
        "api": fallback.get("api"),
        "model": fallback.get("model"),
        "call_type": call_type,
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
