# trial_004 Demo Experiment Plan

- status: ready
- plan_type: delta_patch
- source_trial_id: trial_003
- title: Submission schema: mirror sample (id,x,y,z)
- next_action: prepare-workspace-handoff

## Objective

Create and run one local experiment for 236716.

## Rationale

Use the available Execution Profile and local metric artifact to verify the loop.

## Primary Change Axis

submission_schema_alignment

## Keep Unchanged

- data_split_cv
- model_definition
- preprocessing
- training

## Change Details

- Modify predict_step._read_sample_submission to return expected output columns (exclude 'id'). In predict_step.main, after generating predictions, coerce to (N, len(expected_cols)) via np.asarray and pad/slice as needed, then assemble submission with id + expected_cols. In test_step.main, mirror the same write path to ensure local test output uses the same columns. Cast to float32 and save to outputs/submission.csv.

## Code Change Targets

- predict_step.py
- test_step.py
- workspace_config.json

## Implementation Notes

- None

## Success Criteria

- outputs/submission.csv columns exactly match sample_submission (id,x,y,z)
- No change in local CV score behavior
- Submission file accepted by evaluator without schema errors

## Failure Decision

- If schema mismatch persists or runtime errors occur when writing submission, mark this attempt as failed for the axis and prepare an alternative mapping strategy in the next attempt within the same axis.

## Expected Outputs

- outputs/submission.csv with columns: id,x,y,z
- Logs indicating detected expected_cols from sample_submission

## Issues

- None
