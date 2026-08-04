# Trial Memory Card: trial_003

- plan_type: continuation_delta_plan
- source_trial_id: trial_002
- change_axis: submission_schema_alignment
- candidate_label: submission_schema_alignment: Inference | Output | Predict | Update
- local_score: 0.591
- local_status: flat
- local_delta: 0.0
- previous_local_status: flat
- previous_local_delta: 0.0
- lb_score: 0.6006
- decision: continue_axis_refinement
- active_axis: submission_schema_alignment
- axis_attempt_count: 1/3
- recommended_base_trial: trial_001
- model_type: RuleBasedConstantVelocity
- model: {'estimator': 'RuleBasedConstantVelocity', 'parameters': {}, 'pretrained': False, 'pipeline_container': None}
- split: {'method': 'random_holdout_by_id', 'test_size': 0.1, 'random_state': 42, 'stratify': False}

## Change Details

- Test Inference Output: Change submission prediction columns from ['target'] to ['x','y','z']
- Test Inference Output: Write outputs/submission.csv with column order exactly: id,x,y,z; enforce float32 for x,y,z
- Predict Step.Py: Update _read_sample_submission to expect/propagate ['id','x','y','z'] header
- Predict Step.Py: Modify prediction assembly to return a dict/array with keys/cols ['x','y','z'] from the constant-velocity estimate

## Kept Unchanged

- Train File: train_labels.csv
- Sample Submission: sample_submission.csv
- Id Column: id
- Method: random_holdout_by_id
- Test Size: 0.1

## Features

- feature_columns: []
- numeric_features: []
- categorical_features: []

## Rejected Axes

- controlled_refinement

## Rejected Candidates

- controlled_refinement: controlled_refinement
- submission_schema_alignment: Inference | Output | Predict | Update

## Active Axis Rejected Candidates

- submission_schema_alignment: Inference | Output | Predict | Update

## Next Guidance

`submission_schema_alignment` has not improved yet, but the axis is not rejected. Start from best/base trial `trial_001` and apply another candidate or parameter variant within the same axis.
