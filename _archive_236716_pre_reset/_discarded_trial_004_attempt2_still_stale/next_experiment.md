# trial_004 Demo Experiment Plan

- status: ready
- plan_type: delta_patch
- source_trial_id: trial_001
- title: Mirror sample_submission schema in submission.csv
- next_action: prepare-workspace-handoff

## Objective

Create and run one local experiment for 236716.

## Rationale

Use the available Execution Profile and local metric artifact to verify the loop.

## Primary Change Axis

submission_schema_alignment

## Keep Unchanged

- data_load
- preprocessing
- data_split_cv
- model_definition
- training

## Change Details

- predict_step._read_sample_submission: return (df, id_col=first column, pred_cols=columns[1:]).
- predict_step.main: construct submission = sample_df.copy(); fill submission[pred_cols] from predictions; if fewer dims than pred_cols, pad remaining with 0.0; ensure float32; drop any 'target'; preserve column order.
- test_step.main: align local write to use sample_submission header and same fill logic.

## Code Change Targets

- predict_step.py
- test_step.py

## Implementation Notes

- None

## Success Criteria

- outputs/submission.csv columns exactly equal sample_submission.csv columns and order
- no schema/key errors during local run
- server accepts submission without schema rejection

## Failure Decision

- If submission header mismatch persists or file write fails, mark this candidate failed and keep axis for next attempt

## Expected Outputs

- outputs/submission.csv with columns [id, x, y, z] and float values (no 'target' column)

## Issues

- None
