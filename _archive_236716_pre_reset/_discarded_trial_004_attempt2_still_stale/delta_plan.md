# trial_004 Delta Patch Plan

- base/source trial: trial_001
- active axis: submission_schema_alignment
- candidate: SampleSubmission Schema Mirror
- description: Make submission.csv mirror sample_submission.csv columns (e.g., id,x,y,z) exactly, replacing the current id,target layout.
- implementation_hint: In predict_step.py, read sample_submission to get header; build submission with same column order and dtypes; map model outputs to pred_cols, pad missing dims with 0.0; write CSV with index=False.

## Do Not Repeat

- submission_schema_alignment: Inference | Output | Predict | Update

## Change Details

- predict_step._read_sample_submission: return (df, id_col=first column, pred_cols=columns[1:]).
- predict_step.main: construct submission = sample_df.copy(); fill submission[pred_cols] from predictions; if fewer dims than pred_cols, pad remaining with 0.0; ensure float32; drop any 'target'; preserve column order.
- test_step.main: align local write to use sample_submission header and same fill logic.

## Keep Unchanged

- data_load
- preprocessing
- data_split_cv
- model_definition
- training

## Code Change Targets

- predict_step.py
- test_step.py

## Success Criteria

- outputs/submission.csv columns exactly equal sample_submission.csv columns and order
- no schema/key errors during local run
- server accepts submission without schema rejection
