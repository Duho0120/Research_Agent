# Trial Decision Card: trial_003

- decision: continue_axis_refinement
- change_axis: submission_schema_alignment
- source_trial_id: trial_002
- recommended_base_trial: trial_001
- local_score: 0.591
- local_status: flat
- local_delta: 0.0
- previous_local_status: flat
- previous_local_delta: 0.0
- active_axis: submission_schema_alignment
- axis_attempt_count: 1
- axis_attempt_limit: 3
- lb_score: 0.6006
- lb_status: flat
- lb_delta: 0.0
- previous_lb_status: improved
- previous_lb_delta: 0.213

## Next Guidance

`submission_schema_alignment` has not improved yet, but the axis is not rejected. Start from best/base trial `trial_001` and apply another candidate or parameter variant within the same axis.

## Planner Constraints

- Use `trial_001` as the base trial unless the user explicitly overrides it.
- Change exactly one primary improvement axis in the next trial.
- Keep split/model/preprocessing fixed unless selected as the primary axis.
- Keep the primary improvement axis as `submission_schema_alignment` for the next trial.
- Use `trial_001` as the code/pipeline base even if the active axis was attempted on another trial.
- Try a different candidate or parameter variant inside the same axis; do not switch to a new axis yet.
- Do not preserve the failed candidate unless the new variant explicitly requires it.

## Rejected Axes

- controlled_refinement

## Rejected Candidates

- controlled_refinement: controlled_refinement
- submission_schema_alignment: Inference | Output | Predict | Update

## Active Axis Rejected Candidates

- submission_schema_alignment: Inference | Output | Predict | Update

## Accepted Axes

- None
