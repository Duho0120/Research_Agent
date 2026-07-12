# trial_v16_causal_rolling_baseline Research Protocol

## Current State

```json
{
  "objective": "minimize",
  "active_trial": "trial_v16_causal_rolling_baseline",
  "best_trial": {
    "trial_id": "trial_v16_causal_rolling_baseline",
    "cv_score": 0.584915,
    "lb_score": null
  },
  "consecutive_failures": 1
}
```

## Evidence

```json
{
  "score": 0.584915,
  "best_score_before": 0.584915,
  "improved": false,
  "task_type": "multi_output_binary_classification",
  "diagnosis_issues": [
    "CV did not improve against the current best trial."
  ]
}
```

## Issues

- CV did not improve against the current best trial.

## Candidate Actions

- Review validation assumptions before changing the pipeline.

## Recommended Action

- trial_id: trial_v17_target_chain_stacking
- strategy: validation_review
- reason: Selected `validation_review` from the current diagnosis: CV did not improve against the current best trial.

## Constraints

- Change only one primary improvement axis in the next trial.
- Do not bypass code validation or protected-file rules.
- Do not make a large model change until validation is reviewed.

## User Questions

- No immediate user question required.

## Execution Plan

- Use `trial_v17_target_chain_stacking` as the next trial id.
- Create the pipeline improvement plan.
- Validate the patch plan before code writing.
- Run validation commands before training.
- Evaluate and store the result before planning another trial.

## Optional Competition Evidence

```json
{
  "leaderboard_score": null,
  "best_leaderboard_score": null
}
```
