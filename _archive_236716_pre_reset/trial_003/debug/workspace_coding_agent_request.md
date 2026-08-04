# trial_003 Workspace Coding Agent Request

## Objective

Implement the next workspace experiment within the Execution Profile write scope.

- competition: 236716
- trial_id: trial_003
- request_id: 236716:trial_003:workspace-coding
- project_root: C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\236716
- continuation_mode: can_continue
- source_trial_id: trial_002
- code_base_trial_id: trial_001
- pending_human_review: False
- edit_mode: patch_only

## Input Context Files

- experiments/236716/trial_003/delta_plan.json
- experiments/236716/trial_003/delta_plan.md
- experiments/236716/trial_003/workspace_context_snapshot.md

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

- json_file: experiments/236716/trial_003/workspace_coding_result.json
- markdown_file: experiments/236716/trial_003/workspace_coding_result.md
- status_values: completed, blocked, failed
- next_action: validate-workspace-code-change
- required_fields:
  - status
  - summary
  - changed_files
  - validation_results
  - blocking_issues

## Next Experiment

# trial_003 Demo Experiment Plan

- status: ready
- plan_type: continuation_delta_plan
- source_trial_id: trial_002
- title: Rollback to trial_001 and align submission schema to id,x,y,z
- next_action: prepare-workspace-handoff

## Objective

Maximize R-Hit@1cm by fixing output schema to match sample_submission

## Rationale

- Previous best/local score used: trial_001 cv=1.000
- trial_002 changed axis=controlled_refinement and regressed (cv=0.591, LB=0.3876) — axis rejected
- sample_submission expects columns [id,x,y,z]; trial_002 emitted [id,target], likely harming LB despite similar logic
- Rollback to recommended base (trial_001) and change exactly one axis: submission_schema_alignment

## Primary Change Axis

submission_schema_alignment

## Keep Unchanged

- Train File: train_labels.csv
- Sample Submission: sample_submission.csv
- Id Column: id
- Method: random_holdout_by_id
- Test Size: 0.1
- Random State: 42
- Stratify: False
- 전처리: none
- Family: RuleBasedConstantVelocity
- Training: none (stateless)
- Name: R-Hit@1cm
- Objective: maximize

## Change Details

- Test Inference Output: Change submission prediction columns from ['target'] to ['x','y','z']
- Test Inference Output: Write outputs/submission.csv with column order exactly: id,x,y,z; enforce float32 for x,y,z
- Predict Step.Py: Update _read_sample_submission to expect/propagate ['id','x','y','z'] header
- Predict Step.Py: Modify prediction assembly to return a dict/array with keys/cols ['x','y','z'] from the constant-velocity estimate
- Predict Step.Py: If underlying predictor returns a 3-vector, map directly to (x,y,z); assert shape == (3,) per id
- Test Step.Py: Ensure evaluation paths read ground-truth/predictions as 3D and compute Euclidean distance via _euclid for R-Hit@1cm
- Workspace Config.Json: Update submission prediction_columns metadata to ['x','y','z']

## Code Change Targets

- predict_step.py
- test_step.py
- workspace_config.json

## Implementation Notes

- No model persistence; rule-based predictor remains stateless
- Preserve data loading, split (random_holdout_by_id, test_size=0.1, random_state=42), and model (RuleBasedConstantVelocity)
- Ensure predict_step writes `outputs/submission.csv` with columns [id,x,y,z] in that order, float dtype, no NaN/inf
- Derive (x,y,z) from the same last/constant-velocity 3D state used in evaluation; do not collapse to a single target
- Mirror schema in test_step evaluation paths to compute distance in 3D (already uses _euclid)

## Success Criteria

- outputs/submission.csv header exactly ['id','x','y','z']; row count matches sample_submission
- All x,y,z are finite floats; no NaN/inf; dtype float32 or float64
- Local cv_score matches trial_001 behavior (no drop from schema-only change)
- Internal schema validator passes (no unexpected columns)

## Failure Decision

- If header/order or dtype checks fail: fix column mapping in predict_step and re-run without altering model/split
- If cv worsens unexpectedly: revert code diff and re-verify I/O only; do not change model or split in this axis

## Expected Outputs

- outputs/metrics.json (cv_score unchanged vs base if data consistent)
- outputs/submission.csv with columns: id,x,y,z
- pipeline_summary.json (updated test_inference_output.prediction_columns)
- code snapshot (modified predict_step.py, test_step.py, workspace_config.json)

## Issues

- None
