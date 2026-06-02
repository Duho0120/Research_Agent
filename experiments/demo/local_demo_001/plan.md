# local_demo_001 Plan

## Hypothesis

Continue the current focus: build reliable baseline.

## Changes

- Make one controlled config change based on recent results.
- Keep validation unchanged for comparability.

## Expected Effect

A small but interpretable movement in CV, with no obvious validation break.

## Risk

- Small changes may be within seed variance.

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
  use_frequency_encoding: true
  use_target_encoding: false
  use_interactions: false
  use_missing_indicators: true
cv:
  n_splits: 5
  seed: 42
```
