# trial_001 Workspace Coding Agent Request

## Objective

Implement the next workspace experiment within the Execution Profile write scope.

- competition: 236716
- trial_id: trial_001
- request_id: 236716:trial_001:workspace-coding
- project_root: C:\Users\ASUS\Desktop\Research_Agent\demo_workspaces\236716
- continuation_mode: can_continue
- source_trial_id: None
- code_base_trial_id: trial_001
- pending_human_review: False
- edit_mode: full_file_allowed

## Input Context Files

- experiments/236716/trial_001/next_experiment.md
- experiments/236716/trial_001/continuation_context.json
- experiments/236716/trial_001/continuation_context.md
- experiments/236716/trial_001/workspace_context_snapshot.md
- experiments/236716/trial_001/context_pack_workspace_code_writing.md
- experiments/236716/trial_001/retrieval_manifest_workspace_code_writing.json

## RAG Context Pack

- task: workspace_code_writing
- documents: 3
- skipped: False
- skip_reason: None
- context_pack: `experiments/236716/trial_001/context_pack_workspace_code_writing.md`
- manifest: `experiments/236716/trial_001/retrieval_manifest_workspace_code_writing.json`

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

- mode: full_file_allowed
- prefer_patch_updates: False
- allow_full_file_updates: True
- restore_base_before_patch: False
- base_code_source: None
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

- json_file: experiments/236716/trial_001/workspace_coding_result.json
- markdown_file: experiments/236716/trial_001/workspace_coding_result.md
- status_values: completed, blocked, failed
- next_action: validate-workspace-code-change
- required_fields:
  - status
  - summary
  - changed_files
  - validation_results
  - blocking_issues

## Next Experiment

# trial_001 Demo Experiment Plan

- status: ready
- plan_type: initial_pipeline_plan
- source_trial_id: None
- title: Baseline: Constant-Velocity (+80ms) using last two timesteps per id
- next_action: prepare-workspace-handoff

## Objective

Maximize R-Hit@1cm via a reproducible rule-based baseline; verify submission format (id,x,y,z) matches sample_submission.csv.

## Rationale

- Data profile only confirms sample_submission schema: id,x,y,z; target schema otherwise unknown.
- Baseline guardrails prioritize simple, stable first trial; avoid complex models and high-cardinality features.
- Prior trial memory for this workspace indicates a constant-velocity +80ms heuristic (t0 and t-40ms).

## Pipeline Blueprint

- Inputs: data/train.* (if available): per-id time-ordered positions
- Inputs: data/test.*: per-id recent history to t0 and t-40ms
- Inputs: sample_submission.csv: canonical id list and output columns [id,x,y,z]
- Output Schema: id
- Output Schema: x
- Output Schema: y
- Output Schema: z
- Method: random_holdout_by_id
- Validation Size: 0.1
- Random Seed: 42
- 전처리: Group by id; sort by time column (prefer one of: ['t','time','timestamp_ms']).
- 전처리: Extract last two observations per id: P(t0) and P(t-40ms) for x,y,z.
- 전처리: If dt != 40ms, compute exact dt_ms between the last two rows; compute v = (P(t0)-P(prev)) / (dt_ms/1000).
- 전처리: Fallback: if only one observation available, predict P(t0) unchanged.
- Type: RuleBasedConstantVelocity
- Fit Required: False
- Inference: P_pred = P(t0) + v * 0.08 (seconds)
- Name: R-Hit@1cm
- Definition: mean( ||pred - true||_2 <= 0.01 ) over validation targets
- Prediction Procedure: Load sample_submission.csv to get id order.
- Prediction Procedure: For each id, derive P(t0), P(prev), dt_ms from test context; compute v and P_pred.
- Prediction Procedure: Assemble outputs with columns [id,x,y,z] in sample_submission order.
- Train: {python} train_step.py
- Test: {python} test_step.py
- Predict: {python} predict_step.py

## Code Change Targets

- File: predict_step.py
- Changes: Read sample_submission.csv for id list and column order.
- Changes: Load test context (auto-detect columns: ['id', 'x','y','z'] and time as one of ['t','time','timestamp_ms']).
- Changes: Per id: sort by time; take last two rows; compute v and P_pred for x,y,z; fallback to last observed if <2 rows.
- Changes: Write outputs/submission.csv with columns [id,x,y,z]; validate no NaNs/inf and row count matches sample_submission.
- File: test_step.py
- Changes: Load train data if present; split by id with validation_size=0.1, seed=42.
- Changes: From validation histories, construct targets at t0+80ms and contexts up to t0.
- Changes: Apply same constant-velocity predictor; compute R-Hit@1cm and basic counts.
- Changes: Save outputs/metrics.json with {'R-Hit@1cm': float, 'n_ids': int}.
- File: train_step.py
- Changes: No model training; generate outputs/pipeline_summary.json with data columns detected, split params, and heuristic details.
- Changes: Do not persist any model artifact (artifact_policy).

## Implementation Notes

- Time detection: prefer 'timestamp_ms'; else 'time' (assumed ms); else 't'. Ensure numeric dtype.
- Unit safety: dt_s = max((t0 - t_prev)/1000.0, 1e-6) to avoid divide-by-zero; clip absurd velocities if needed (optional cap at |v|<=100 m/s).
- Vector ops with numpy/pandas; avoid saving any trained model artifact.
- If test file already has explicit columns for last two snapshots (e.g., x0,y0,z0 and xm40,ym40,zm40), bypass grouping and compute directly.
- Always drive prediction row order from sample_submission.csv.

## Success Criteria

- outputs/submission.csv exactly matches sample_submission columns and id order; no NaN/inf; row count equals sample_submission.
- outputs/metrics.json exists and includes R-Hit@1cm (if train data available).
- outputs/pipeline_summary.json exists and records split params and heuristic.
- Reproducible run with seed=42; no dependency on external services.

## Expected Outputs

- outputs/metrics.json
- outputs/submission.csv
- outputs/pipeline_summary.json
- outputs/code_snapshot.txt

## Issues

- None
