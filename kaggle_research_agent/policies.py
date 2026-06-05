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
            "risk",
            "candidate_actions",
            "recommended_next_trial",
            "do_not_change",
            "need_user_check",
            "execution_plan",
        ],
        "preserve_public_anchor_on_cv_lb_conflict": True,
        "require_public_evidence_before_promoting_local_best": True,
        "small_data_train_rows_threshold": 1000,
        "small_subject_count_threshold": 20,
        "candidate_lanes": ["safe", "main", "aggressive"],
        "default_probability_checks": [
            "Compare against the trusted baseline before submission.",
            "Check per-target or per-segment movement when available.",
            "Keep validation fixed unless the protocol selects validation review.",
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
