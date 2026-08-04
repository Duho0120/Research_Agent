# trial_004 Delta Patch Plan

- base/source trial: trial_003
- active axis: submission_schema_alignment
- candidate: submission_schema_alignment: MergeToTemplate | EnforceOrder+Float6
- description: Write submission by merging predictions onto sample_submission to preserve id coverage/order and enforce exact columns [id,x,y,z] with 6-decimal floats.
- implementation_hint: In predict_step.py: in main(), after building preds DataFrame with columns ['id','x','y','z'], read template = _read_sample_submission(); out = template[['id']].merge(preds, on='id', how='left'); out = out[['id','x','y','z']]; for c in ['x','y','z']: out[c] = out[c].astype('float32'); out.to_csv(SUBMISSION_PATH, index=False, float_format='%.6f')

## Do Not Repeat

- submission_schema_alignment: Inference | Output | Predict | Update

## Change Details

- Adjust predict_step.main to merge predictions with sample_submission for id alignment, enforce exact column order ['id','x','y','z'], cast to float32, and save CSV with float_format='%.6f' and no index.

## Keep Unchanged

- data_split_cv
- model_definition
- preprocessing

## Code Change Targets

- predict_step.py

## Success Criteria

- outputs/submission.csv has exactly columns id,x,y,z in order with all test ids
- No LB schema errors; score loads successfully
- Local R-Hit@1cm >= 0.591
