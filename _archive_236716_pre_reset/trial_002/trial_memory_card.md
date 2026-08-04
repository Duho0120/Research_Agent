# Trial Memory Card: trial_002

- plan_type: continuation_delta_plan
- source_trial_id: trial_001
- change_axis: controlled_refinement
- candidate_label: controlled_refinement: controlled_refinement
- local_score: 0.591
- local_status: flat
- local_delta: 0.0
- previous_local_status: flat
- previous_local_delta: 0.0
- lb_score: 0.3876
- decision: reject_or_hold
- active_axis: None
- axis_attempt_count: 1/3
- recommended_base_trial: trial_001
- model_type: RuleBasedConstantVelocity
- model: {'estimator': 'RuleBasedConstantVelocity', 'parameters': {}, 'pretrained': False, 'pipeline_container': None}
- split: {'method': 'random_holdout_by_id', 'test_size': 0.1, 'random_state': 42, 'stratify': False}

## Change Details

- Make one controlled improvement based on the latest diagnosis.
- Keep model family and validation fixed for attribution.

## Kept Unchanged

- None

## Features

- feature_columns: []
- numeric_features: []
- categorical_features: []

## Rejected Axes

- controlled_refinement

## Rejected Candidates

- controlled_refinement: controlled_refinement

## Active Axis Rejected Candidates

- controlled_refinement: controlled_refinement

## Next Guidance

Treat `controlled_refinement` as exhausted for now. Start from `trial_001` and try a different single axis.
