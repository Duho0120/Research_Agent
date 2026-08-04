# trial_004 Demo Experiment Plan

- status: ready
- plan_type: delta_patch
- source_trial_id: trial_003
- title: Align submission via sample template merge, ordered columns, and float precision
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

## Change Details

- Adjust predict_step.main to merge predictions with sample_submission for id alignment, enforce exact column order ['id','x','y','z'], cast to float32, and save CSV with float_format='%.6f' and no index.

## Code Change Targets

- predict_step.py

## Implementation Notes

- None

## Success Criteria

- outputs/submission.csv has exactly columns id,x,y,z in order with all test ids
- No LB schema errors; score loads successfully
- Local R-Hit@1cm >= 0.591

## Failure Decision

- If schema still misaligned or rows mismatch, revert this patch and next try writer in src/baseline.py to enforce order/format.

## Expected Outputs

- outputs/submission.csv

## Issues

- None
