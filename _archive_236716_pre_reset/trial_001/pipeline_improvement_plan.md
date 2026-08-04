# trial_001 Pipeline Improvement Plan

## Primary Axis

hyperparameter

## Secondary Axes

- training_recipe
- feature_engineering

## Protected Axes

- validation
- model_family

## Candidate Actions

- Tune one training parameter such as learning rate, regularization, batch size, or early stopping.
- Keep data split, model family, and major feature assumptions fixed.

## Success Criteria

- The selected axis improves CV or diagnostic quality without changing protected axes.

## Do Not Change

- Do not change validation in the same trial.
- Do not change model_family in the same trial.
- Do not change validation unless this is explicitly a validation review.

## Human Review

- requires_human_review: False

## Rationale

Selected `hyperparameter` as the primary pipeline improvement axis. Diagnosis issues: CV did not improve against the current best trial.

## Evidence Used

```json
{
  "cv_score": 0.591,
  "lb_score": 0.6006,
  "rank": null,
  "leaderboard_source": "metrics",
  "issues": [
    "CV did not improve against the current best trial."
  ],
  "strategy_recommendation": "continue_refinement",
  "segment_errors_present": false,
  "prediction_correlation_with_best": null
}
```
