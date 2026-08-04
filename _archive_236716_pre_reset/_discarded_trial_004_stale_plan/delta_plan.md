# trial_004 Delta Patch Plan

- base/source trial: trial_003
- active axis: submission_schema_alignment
- candidate: MirrorSampleSchemaColumns
- description: Build submission by mirroring sample_submission header (expecting id,x,y,z) and map model outputs to those columns.
- implementation_hint: In predict_step.main, read sample_submission to get expected_cols (exclude 'id'). Ensure predictions shaped (N, D) map to expected_cols; if D < len(expected_cols), pad missing dims with 0.0; if D > len, slice. Write DataFrame with ['id']+expected_cols. Keep model/split/preproc unchanged.

## Do Not Repeat

- submission_schema_alignment: Inference | Output | Predict | Update

## Change Details

- Modify predict_step._read_sample_submission to return expected output columns (exclude 'id'). In predict_step.main, after generating predictions, coerce to (N, len(expected_cols)) via np.asarray and pad/slice as needed, then assemble submission with id + expected_cols. In test_step.main, mirror the same write path to ensure local test output uses the same columns. Cast to float32 and save to outputs/submission.csv.

## Keep Unchanged

- data_split_cv
- model_definition
- preprocessing
- training

## Code Change Targets

- predict_step.py
- test_step.py
- workspace_config.json

## Success Criteria

- outputs/submission.csv columns exactly match sample_submission (id,x,y,z)
- No change in local CV score behavior
- Submission file accepted by evaluator without schema errors
