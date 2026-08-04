# trial_002 Workspace Coding Agent Request

## Objective

Implement the next workspace experiment within the Execution Profile write scope.

- competition: 236716
- trial_id: trial_002
- request_id: 236716:trial_002:workspace-coding
- project_root: C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\236716
- continuation_mode: can_continue
- source_trial_id: trial_001
- code_base_trial_id: trial_001
- pending_human_review: False
- edit_mode: patch_only

## Input Context Files

- experiments/236716/trial_002/next_experiment.md
- experiments/236716/trial_002/continuation_context.json
- experiments/236716/trial_002/continuation_context.md
- experiments/236716/trial_002/workspace_context_snapshot.md

## RAG Context Pack

- task: workspace_code_writing
- documents: 0
- skipped: True
- skip_reason: continuation_uses_structured_memory_and_code_snapshot
- context_pack: `None`
- manifest: `None`

## Data Card Summary

```json
{
  "task_type": "unknown",
  "target_column": null,
  "id_column": null,
  "submission_prediction_column": null,
  "train_file": null,
  "test_file": null,
  "sample_submission_file": "sample_submission.csv",
  "dataset_layout": "per_sample_files",
  "train_dir": "train/",
  "test_dir": "test/",
  "directory_datasets": [
    {
      "name": "test/",
      "role": "test",
      "file_count": 10000,
      "filename_pattern": "TEST_#####.csv",
      "example_files": [
        "TEST_00001.csv",
        "TEST_00002.csv",
        "TEST_00003.csv"
      ],
      "per_file_columns": [
        "timestep_ms",
        "x",
        "y",
        "z"
      ],
      "sample_id_source": "filename_stem",
      "id_matched_files": [
        "sample_submission.csv:id"
      ],
      "notes": [
        "Each CSV in this directory is ONE sample; the sample id is the filename stem.",
        "Load every file in the directory and derive features per sample -- do not expect a single flat table."
      ]
    },
    {
      "name": "train/",
      "role": "train",
      "file_count": 10000,
      "filename_pattern": "TRAIN_#####.csv",
      "example_files": [
        "TRAIN_00001.csv",
        "TRAIN_00002.csv",
        "TRAIN_00003.csv"
      ],
      "per_file_columns": [
        "timestep_ms",
        "x",
        "y",
        "z"
      ],
      "sample_id_source": "filename_stem",
      "id_matched_files": [
        "train_labels.csv:id"
      ],
      "notes": [
        "Each CSV in this directory is ONE sample; the sample id is the filename stem.",
        "Load every file in the directory and derive features per sample -- do not expect a single flat table."
      ]
    }
  ],
  "include_features_first": [],
  "defer_features_first": [],
  "exclude_columns": [],
  "preferred_model_families": [
    "logistic_regression_or_linear_classifier_for_binary_classification",
    "small_random_forest_or_gradient_boosted_tree_if_available"
  ],
  "avoid_first_trial": [
    "gaussian_naive_bayes_for_mixed_numeric_and_categorical_data",
    "raw_high_cardinality_text_or_identifier_one_hot_features",
    "broad_schema_discovery_when_target_id_and_columns_are_known"
  ]
}
```

## Allowed External Write Paths

- src/
- tests/
- train_step.py
- predict_step.py
- test_step.py
- workspace_config.json

## Edit Policy

- mode: patch_only
- prefer_patch_updates: True
- allow_full_file_updates: False
- restore_base_before_patch: True
- base_code_source: experiments/236716/trial_001/internal/code_snapshot
- When base_code_source is present, treat that source trial code as the authoritative starting point.
- For patch mode, return `patch_updates` with exact `find` text and replacement text.
- Do not return whole-file `file_updates` in patch mode unless the policy explicitly allows it.

## Forbidden External Paths

- data/
- outputs/metrics.json
- outputs/submission.csv

## Execution Constraints

- Do not run training.
- Do not submit to any competition platform.
- Do not edit data, metrics, submission, or output artifacts.
- Do not write outside the allowed external write paths.
- If a base trial code snapshot is declared, do not preserve rejected changes from later failed trials.
- Never fabricate, synthesize, or hardcode placeholder train/test data as a fallback when an expected file (e.g. data/train.csv, data/test.csv) is missing. A trial that raises a clear error is always correct over one that silently substitutes made-up data to produce a plausible-looking metric or submission -- a fabricated result is worse than a visible failure because it hides the real problem.
- The actual data file/folder layout for this competition is listed under 'Data Card Summary' below (and in the competition's data_notes.md, if provided) -- read code against those real paths, not against a conventional train.csv/test.csv name you assume exists. If the data is split across many per-sample files or separate feature/label files, write code that loads and joins them accordingly.

## Validation Commands

```powershell
{python} test_step.py
```

## Metrics Output Contract

- path: outputs/metrics.json
- score_key: cv_score
- required_keys:
  - cv_score
  - metric
  - objective
- notes:
  - Training code must write a finite numeric cv_score to the metrics artifact.
  - metric should match the competition metric name when known.
  - objective must be maximize or minimize.
  - Additional diagnostic keys such as validation_accuracy are allowed, but cv_score is the canonical score.
  - Every metrics and pipeline-summary value must be JSON serializable. Convert numpy scalars, callables, estimators, paths, and other objects to primitive values or stable strings before json.dumps.

## Artifact Policy

- Metrics, submission, code snapshot, and pipeline summary are the primary trial memory.
- Do not persist trained model/checkpoint artifacts by default.
- Persist a model only when the policy allows it and record the reason in your summary or metrics metadata.
```json
{
  "save_submission": true,
  "save_metrics": true,
  "save_code_snapshot": true,
  "save_pipeline_summary": true,
  "save_model": {
    "default": false,
    "allowed_when": [
      "required_for_separate_predict_command",
      "best_trial",
      "submitted_trial",
      "expensive_to_retrain",
      "ensemble_candidate",
      "user_requested"
    ],
    "require_reason": true,
    "prefer_small_serialized_artifact": true,
    "cleanup_non_best_models": true
  },
  "notes": [
    "A trial should not persist trained model artifacts by default.",
    "Submission, metrics, code snapshot, and pipeline summary are the primary memory artifacts.",
    "If a model artifact is persisted, the plan/code should record why it is needed."
  ]
}
```

## Required Result Contract

- json_file: experiments/236716/trial_002/workspace_coding_result.json
- markdown_file: experiments/236716/trial_002/workspace_coding_result.md
- status_values: completed, blocked, failed
- next_action: validate-workspace-code-change
- required_fields:
  - status
  - summary
  - changed_files
  - validation_results
  - blocking_issues

## Next Experiment

# trial_002 Next Experiment

## Strategy

controlled_refinement

## Rationale

Selected `controlled_refinement` after 0 consecutive failure(s). Latest submission movement: score_delta=None, rank_delta=None.

## Changes

- Make one controlled improvement based on the latest diagnosis.
- Keep model family and validation fixed for attribution.

## Guardrails

- Keep validation unchanged unless this is explicitly a validation review.
- When submission tracking is enabled, submit every completed trial so leaderboard evidence is recorded.
- Record current and submitted leaderboard score/rank if a submission is made.

## Research Protocol

- protocol_strategy: controlled_improvement

### Issues

- CV did not improve against the current best trial.

### Constraints

- Change only one primary improvement axis in the next trial.
- Do not bypass code validation or protected-file rules.

### User Questions

- No immediate user question required.

## Submit Gate

- requires_user_review_before_submit: False
- Preserve source and next-trial artifacts before any leaderboard submission.

## Evidence Used

```json
{
  "consecutive_failures": 0,
  "strategy_hint": null,
  "issues": [],
  "latest_submission": {
    "submission_id": "236716_trial_001_trial_001_auto",
    "competition": "236716",
    "trial_id": "trial_001",
    "version_name": "trial_001_auto",
    "submitted_at": "2026-07-31T00:55:16.980061+00:00",
    "submission_file": "demo_workspaces/236716/outputs/submission.csv",
    "cv_score": 0.591,
    "previous_lb_score": null,
    "previous_rank": null,
    "submitted_lb_score": 0.6006,
    "submitted_rank": null,
    "score_delta": null,
    "rank_delta": null,
    "best_reference_score": null,
    "is_best": true,
    "notes": "DACON leaderboard score read back from submission history (sub_id 1506746)"
  },
  "latest_user_feedback": null,
  "user_insight_override": null,
  "pipeline_improvement": {
    "competition": "236716",
    "trial_id": "trial_001",
    "primary_axis": "hyperparameter",
    "secondary_axes": [
      "training_recipe",
      "feature_engineering"
    ],
    "protected_axes": [
      "validation",
      "model_family"
    ],
    "requires_human_review": false,
    "candidate_actions": [
      "Tune one training parameter such as learning rate, regularization, batch size, or early stopping.",
      "Keep data split, model family, and major feature assumptions fixed."
    ],
    "success_criteria": [
      "The selected axis improves CV or diagnostic quality without changing protected axes."
    ],
    "do_not_change": [
      "Do not change validation in the same trial.",
      "Do not change model_family in the same trial.",
      "Do not change validation unless this is explicitly a validation review."
    ],
    "rationale": "Selected `hyperparameter` as the primary pipeline improvement axis. Diagnosis issues: CV did not improve against the current best trial.",
    "evidence_used": {
      "cv_score": 0.591,
      "lb_score": 0.6006,
      "rank": null,
      "leaderboard_source": "metrics",
      "issues": [
        "CV did not improve against the current best trial."
      ],
      "strategy_recommendation": "continue_refinement",
      "segment_errors_present": false,
      "prediction_correlation_with_best": null
    },
    "next_trial_rule": "Change one primary pipeline axis and keep protected axes fixed."
  },
  "research_protocol": {
    "issues": [
      "CV did not improve against the current best trial."
    ],
    "recommended_action": {
      "trial_id": "trial_002",
      "strategy": "controlled_improvement",
      "reason": "Selected `controlled_improvement` from the current diagnosis: CV did not improve against the current best trial."
    },
    "user_questions": [],
    "constraints": [
      "Change only one primary improvement axis in the next trial.",
      "Do not bypass code validation or protected-file rules."
    ],
    "enabled_extensions": []
  }
}
```
