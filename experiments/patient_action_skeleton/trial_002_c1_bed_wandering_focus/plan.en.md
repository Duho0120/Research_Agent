# trial_002_c1_bed_wandering_focus Plan

## Hypothesis

The current baseline is not failing uniformly. Errors concentrate in C1 and in Bed Exit/Wandering transition windows, so the next trial should target view-sensitive transition ambiguity rather than increasing the whole model size.

## Proposed Change

- Keep the Transformer architecture and validation split unchanged.
- Add a view-aware or scenario-aware diagnostic feature path for C1-sensitive ROI geometry.
- Increase attention to Bed Exit/Wandering boundary windows without changing Fall handling.
- Track Bed Exit recall, Wandering precision, C1 error rate, and Fall recall together.

## Why This Follows From Baseline

- Baseline macro F1 is already about 0.83 and Fall F1 is about 0.95.
- The biggest actionable issue is Bed Exit/Wandering confusion.
- C1 error rate is 0.1686, much higher than C3 at 0.0772.
- Scenario 00620_H_D_SY accounts for 23/60 errors, suggesting localized edge cases.

## Success Criteria

- Bed Exit F1 improves above 0.7792.
- Macro F1 stays at or above 0.8326.
- Fall recall stays at or above 0.96.
- C1 error rate decreases below 0.15.
- Bed Exit/Wandering C1 confusion count decreases below 13.

## Risk

- Over-tuning to C1 or a few named scenarios may reduce generalization.
- Lowering Bed Exit threshold improved recall but reduced macro F1 in the baseline, so threshold-only tuning is not enough.

