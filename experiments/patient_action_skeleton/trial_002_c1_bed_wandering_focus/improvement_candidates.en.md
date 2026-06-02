# Improvement Candidates

## Primary Candidate: View-Aware Bed Exit/Wandering Separation

Problem:

- C1 has the highest validation error rate: 43/255, error rate 0.1686.
- Bed Exit/Wandering confusion is the core semantic error.
- C1 alone has 13 Bed Exit/Wandering confusion windows.

Suggested change:

- Keep the Transformer and validation split fixed.
- Add or emphasize view-aware ROI geometry features for C1-sensitive cases.
- Track C1 error rate as a first-class metric, not only global accuracy.

Why first:

- It targets the observed bottleneck without increasing model complexity broadly.
- Fall performance is already strong, so the trial should avoid disturbing Fall recall.

## Secondary Candidate: Boundary Window Sampling

Problem:

- Bed Exit and Wandering differ most around transition frames.
- Several confusion cases occur in adjacent windows with close frame ranges.

Suggested change:

- Add controlled oversampling for boundary windows where labels switch between Bed Exit and Wandering.
- Keep augmentation mild to avoid memorizing named scenarios.

## Tertiary Candidate: Selection Metric Reweighting

Problem:

- Accuracy is high because Fall dominates validation support.
- The actual research target cares more about Bed Exit/Wandering separation and Fall safety.

Suggested change:

- Select checkpoints using a combined score:
  `0.45 * BedExit_F1 + 0.35 * MacroF1 + 0.20 * FallRecall`
- Keep raw accuracy as a reporting metric only.

## Do Not Try First

- Increasing model depth or d_model before fixing the error pattern.
- Threshold-only Bed Exit tuning, because the saved threshold sweep reduced macro F1.
- Changing validation split together with model changes.

