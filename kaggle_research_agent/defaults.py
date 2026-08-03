from __future__ import annotations


DEFAULT_STATE = {
    "competition": {
        "name": "",
        "metric": "unknown",
        "objective": "maximize",
        "submission_limit_per_day": None,
    },
    "current_state": {
        "active_trial": None,
        "best_trial": None,
        "consecutive_failures": 0,
        "submissions_today": 0,
        "validation_suspected": False,
    },
    "strategy": {
        "current_focus": "build reliable baseline",
        "promising_directions": [],
        "forbidden_directions": [],
    },
}


DEFAULT_ALLOWED_SPACE = {
    "model": {
        "type": ["lightgbm", "xgboost", "catboost", "tabular_nn"],
        "params": {
            "learning_rate": {"min": 0.005, "max": 0.2},
            "num_leaves": {"min": 16, "max": 256},
            "max_depth": {"min": 3, "max": 12},
        },
    },
    "features": {
        "use_frequency_encoding": [True, False],
        "use_target_encoding": [True, False],
        "use_interactions": [True, False],
        "use_missing_indicators": [True, False],
    },
    "cv": {
        "n_splits": [5, 10],
        "seed": {"min": 1, "max": 9999},
    },
}


DEFAULT_CONFIG = {
    "model": {
        "type": "lightgbm",
        "params": {
            "learning_rate": 0.03,
            "num_leaves": 64,
            "max_depth": 8,
        },
    },
    "features": {
        "use_frequency_encoding": False,
        "use_target_encoding": False,
        "use_interactions": False,
        "use_missing_indicators": True,
    },
    "cv": {
        "n_splits": 5,
        "seed": 42,
    },
}


DEFAULT_RULES = """# Research Rules

- One trial should test one primary hypothesis.
- Do not change validation strategy and model family in the same trial unless the Orchestrator Agent explicitly marks a validation review.
- If CV improves but LB worsens in two consecutive submissions, suspect validation mismatch.
- If three trials in a row fail to improve, shift strategy instead of making smaller tweaks.
- Do not submit when the expected gain is smaller than known seed variance.
- Treat sudden large score gains as possible leakage until checked.
"""


DEFAULT_NOTES = """# Research Notes

## Current Understanding

- No experiments have been completed yet.

## Promising Directions

- Establish a reliable baseline and validation procedure first.

## Failed Directions

- None yet.
"""
