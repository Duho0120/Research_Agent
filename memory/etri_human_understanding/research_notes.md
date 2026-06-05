# Research Notes: ETRI Human Understanding

This competition is DACON-style, not Kaggle. Use this Research Agent for memory, planning, diagnostics, validation gates, and trial tracking. Do not use Kaggle CLI submission or leaderboard polling.

## Problem

- Task type: multi-output binary classification.
- Multi-output binary classification over seven targets: `Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4`.
- Input unit is a subject-day row keyed by `subject_id`, `sleep_date`, and `lifelog_date`.
- Train has 450 rows from 10 subjects; test/submission has 250 rows.
- Additional sensor/event parquet files exist in `ch2025_data_items`.
- Metric is treated as mean binary log loss across the seven targets.

## Current Best Evidence

Public baseline:

- `V11 Anchored Three-Model Stacking`
- Trial id in this repo: `trial_v11_public_baseline`
- Subject-hole: `0.591306`
- Tail: `0.600613`
- Public LB: about `0.5984`
- Verdict: trusted public baseline.

Recent local best:

- `V16 Causal Rolling Baseline Model`
- Trial id in this repo: `trial_v16_causal_rolling_baseline`
- Subject-hole: `0.584915`
- Tail: `0.596018`
- Public LB: unknown
- Verdict: local best, but submit or record safe variant first.

Important caution:

- `V15 Subject Temporal Deviation` improved local CV but worsened Public LB to `0.5994936146`.
- Subject personalization and full-period subject baselines can overfit public/private time or subject structure.
- Prefer causal rolling baselines and conservative target-wise blending.

## Target Priors

```text
Q1 0.4956
Q2 0.5622
Q3 0.6000
S1 0.6822
S2 0.6511
S3 0.6622
S4 0.5600
```

## Next Experiment Candidates

1. Confirm or record the V16 safe Public LB.
2. V17 Target Chain Stacking:
   - Use OOF probabilities only.
   - Candidate order: `S1 -> S2 -> S4 -> S3 -> Q2 -> Q1 -> Q3`.
3. V18 Adversarial Validation + Covariate Shift Weighting:
   - Identify shifted segments/features.
   - Reflect weighting only conservatively in calibration or blending.
4. V19 Cross-fitted Platt/Logit Calibration:
   - Calibrate V11/V16 OOF predictions target by target.
   - Apply only when Subject-hole and Tail both improve.
5. V20 Mixed-effects or GPBoost prototype:
   - Prototype only for Q2/Q3/S1 first.

## Human Review Questions

- Confirm exact DACON submission limit and whether Public/Private split is time-based.
- Confirm exact target definitions from the PDF before interpreting Q/S target dependencies.
- Confirm whether V16 safe has already been submitted and what its Public LB was.
