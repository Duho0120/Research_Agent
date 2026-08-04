# Trial Decision Card: trial_002

- decision: reject_or_hold
- change_axis: controlled_refinement
- source_trial_id: trial_001
- recommended_base_trial: trial_001
- local_score: 0.591
- local_status: flat
- local_delta: 0.0
- previous_local_status: flat
- previous_local_delta: 0.0
- active_axis: None
- axis_attempt_count: 1
- axis_attempt_limit: 3
- lb_score: 0.3876
- lb_status: regressed
- lb_delta: -0.213
- previous_lb_status: regressed
- previous_lb_delta: -0.213

## Next Guidance

Treat `controlled_refinement` as exhausted for now. Start from `trial_001` and try a different single axis.

## Planner Constraints

- Use `trial_001` as the base trial unless the user explicitly overrides it.
- Change exactly one primary improvement axis in the next trial.
- Keep split/model/preprocessing fixed unless selected as the primary axis.
- FACT: `controlled_refinement` regressed the score by more than 25% -- far beyond tuning noise. The idea itself is wrong for this data; do not retry it with different parameters, and do not retry it under a different axis name.
- Do not keep stacking on rejected axis `controlled_refinement`.
- If the rejected axis was implemented in the last trial, roll back to the recommended base before applying a new change.

## Rejected Axes

- controlled_refinement

## Rejected Candidates

- controlled_refinement: controlled_refinement

## Active Axis Rejected Candidates

- controlled_refinement: controlled_refinement

## Accepted Axes

- None
