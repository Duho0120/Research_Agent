# trial_001 Execution Plan Snapshot

- status: pending
- source: workspace_coding_agent_request.md:Next Experiment
- request_id: 236716:trial_001:workspace-coding
- source_trial_id: None
- primary_change_axis: None
- plan_title: Baseline: Constant-Velocity (+80ms) using last two timesteps per id

## Plan Delivered To Code Writer

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
