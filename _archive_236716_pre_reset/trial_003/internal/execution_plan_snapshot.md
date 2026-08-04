# trial_003 Execution Plan Snapshot

- status: finalized
- source: workspace_coding_agent_request.md:Next Experiment
- request_id: 236716:trial_003:workspace-coding
- source_trial_id: trial_002
- primary_change_axis: submission_schema_alignment
- plan_title: Rollback to trial_001 and align submission schema to id,x,y,z

## Plan Delivered To Code Writer

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
