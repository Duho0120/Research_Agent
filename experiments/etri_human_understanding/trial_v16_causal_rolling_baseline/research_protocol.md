# trial_v16_causal_rolling_baseline Research Protocol

## Current State

```json
{
  "objective": "minimize",
  "platform": "dacon",
  "active_trial": "trial_v16_causal_rolling_baseline",
  "best_trial": {
    "trial_id": "trial_v16_causal_rolling_baseline",
    "cv_score": 0.584915,
    "lb_score": null
  },
  "consecutive_failures": 1,
  "validation_suspected": true
}
```

## Evidence

```json
{
  "cv_score": 0.584915,
  "lb_score": null,
  "best_cv_before": 0.584915,
  "best_lb_before": null,
  "cv_improved": false,
  "diagnosis_issues": [
    "CV did not improve against the current best trial."
  ],
  "task_type": "multi_output_binary_classification",
  "train_rows": 450,
  "subjects": 10,
  "target_columns": [
    "Q1",
    "Q2",
    "Q3",
    "S1",
    "S2",
    "S3",
    "S4"
  ]
}
```

## Risk

- level: medium
- local_best_public_unknown
- validation_suspected
- small_data_or_subject_count
- external_platform_manual_submission

## Candidate Actions

### safe
- Record or request leaderboard/holdout evidence for the local best.

### main
- Prepare a conservative next trial anchored to the trusted public baseline.

### aggressive
- Postpone model-family changes until public evidence exists.

## Recommended Next Trial

- trial_id: trial_v17_target_chain_stacking
- strategy: safe_submission_or_holdout_confirmation
- reason: Selected because risk flags are present: local_best_public_unknown, validation_suspected, small_data_or_subject_count, external_platform_manual_submission.

## Do Not Change

- Do not mix multiple primary improvement axes in one trial.
- Do not change model family before resolving public evidence.
- Do not replace the trusted public baseline with local-only evidence.
- Do not trust high-capacity changes without strong validation evidence.

## Need User Check

- Record or request leaderboard evidence before promoting the local best.
- Confirm platform submission limits and how leaderboard evidence will be recorded.
- Confirm target semantics before changing target dependencies or classifier chains.

## Execution Plan

- Use `trial_v17_target_chain_stacking` as the next trial id.
- Write or update pipeline_improvement_plan before code changes.
- Create patch plan and validate it before coding handoff.
- Run validation commands before training or job creation.
- Record manual or platform leaderboard result before aggressive follow-up.
