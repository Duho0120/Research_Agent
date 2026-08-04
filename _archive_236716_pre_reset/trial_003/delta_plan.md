# trial_003 Delta Patch Plan

- base/source trial: trial_002
- active axis: submission_schema_alignment
- candidate: Rollback to trial_001 and align submission schema to id,x,y,z
- description: 
- implementation_hint: 

## Do Not Repeat

- None

## Change Details

- Test Inference Output: Change submission prediction columns from ['target'] to ['x','y','z']
- Test Inference Output: Write outputs/submission.csv with column order exactly: id,x,y,z; enforce float32 for x,y,z
- Predict Step.Py: Update _read_sample_submission to expect/propagate ['id','x','y','z'] header
- Predict Step.Py: Modify prediction assembly to return a dict/array with keys/cols ['x','y','z'] from the constant-velocity estimate
- Predict Step.Py: If underlying predictor returns a 3-vector, map directly to (x,y,z); assert shape == (3,) per id
- Test Step.Py: Ensure evaluation paths read ground-truth/predictions as 3D and compute Euclidean distance via _euclid for R-Hit@1cm
- Workspace Config.Json: Update submission prediction_columns metadata to ['x','y','z']

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

## Code Change Targets

- predict_step.py
- test_step.py
- workspace_config.json

## Success Criteria

- outputs/submission.csv header exactly ['id','x','y','z']; row count matches sample_submission
- All x,y,z are finite floats; no NaN/inf; dtype float32 or float64
- Local cv_score matches trial_001 behavior (no drop from schema-only change)
- Internal schema validator passes (no unexpected columns)
