# trial_004 Next Experiment

## Strategy

sota_architecture_attempt

## Rationale

Selected `sota_architecture_attempt` after 8 consecutive failure(s). Diagnosis issues: CV did not improve against the current best trial.; Recent failures suggest strategy escalation is needed. The current method appears saturated enough to justify a model or architecture level attempt.

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
  "consecutive_failures": 8,
  "strategy_hint": "strategy_escalation",
  "issues": [
    "CV did not improve against the current best trial.",
    "Recent failures suggest strategy escalation is needed."
  ],
  "latest_submission": null
}
```
