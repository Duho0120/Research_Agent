# trial_002 Plan

## Hypothesis

Recent failures suggest the current direction is saturated; switch strategy.

## Changes

- Change one high-level direction while keeping validation fixed.
- Prefer feature diagnostics over parameter micro-tuning.

## Expected Effect

A small but interpretable movement in CV, with no obvious validation break.

## Risk

- A strategy shift can make attribution harder if too many things change.

## Success Criteria

- CV improves beyond known noise or remains competitive with better diversity.
- No metric or submission format issues are observed.
- If submitted, LB movement is directionally consistent with CV.

## Config

```yaml
model:
  type: lightgbm
  params:
    learning_rate: 0.03
    num_leaves: 64
    max_depth: 8
features:
  use_frequency_encoding: false
  use_target_encoding: true
  use_interactions: true
  use_missing_indicators: true
cv:
  n_splits: 5
  seed: 42
```
