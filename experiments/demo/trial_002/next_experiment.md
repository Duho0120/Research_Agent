# trial_002 Next Experiment

## Strategy

sota_architecture_attempt

## Rationale

Selected `sota_architecture_attempt` after 6 consecutive failure(s). Diagnosis issues: CV did not improve against the current best trial.; Recent failures suggest strategy escalation is needed. Latest submission movement: score_delta=0.04, rank_delta=30. The current method appears saturated enough to justify a model or architecture level attempt.

## Changes

- Switch model-family or architecture instead of another parameter micro-tune.
- Prepare one SOTA-inspired candidate from the allowed model space.
- Keep validation fixed unless the diagnosis explicitly asks for validation review.

## Guardrails

- Keep validation unchanged unless this is explicitly a validation review.
- Do not submit if expected gain is within known seed variance.
- Record current and submitted leaderboard score/rank if a submission is made.
- Compare the SOTA attempt against the current best before replacing the best marker.

## Submit Gate

- requires_user_review_before_submit: True
- Preserve source and next-trial artifacts before any leaderboard submission.

## Evidence Used

```json
{
  "consecutive_failures": 6,
  "strategy_hint": "strategy_escalation",
  "issues": [
    "CV did not improve against the current best trial.",
    "Recent failures suggest strategy escalation is needed."
  ],
  "latest_submission": {
    "submission_id": "demo_trial_001_demo_trial_001_baseline_v01",
    "competition": "demo",
    "trial_id": "trial_001",
    "version_name": "demo_trial_001_baseline_v01",
    "submitted_at": "2026-05-31T05:43:48.751002+00:00",
    "submission_file": "experiments/demo/trial_001/submission.csv",
    "cv_score": 0.83,
    "previous_lb_score": 0.8,
    "previous_rank": 120,
    "submitted_lb_score": 0.84,
    "submitted_rank": 90,
    "score_delta": 0.04,
    "rank_delta": 30,
    "is_best": true,
    "notes": "Manual leaderboard entry"
  }
}
```
