# ETRI Human Understanding

This is a DACON-style external competition, not a Kaggle competition. The Research Agent should track planning, validation, trial memory, and submission metadata, but it must not use Kaggle CLI submission or leaderboard polling for this competition.

Task:

- Multi-output binary classification.
- One row represents a subject-day unit keyed by `subject_id`, `sleep_date`, and `lifelog_date`.
- Predict seven probability targets: `Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4`.
- The practical metric is treated as the mean binary log loss over the seven targets.

Data scale:

- Train rows: 450.
- Test/submission rows: 250.
- Subjects: 10.
- Additional unlabeled sensor/event parquet files are available under the original `ch2025_data_items` folder.

Current status:

- Trusted public baseline: `trial_v11_public_baseline`, Public LB about `0.5984`.
- Local best candidate: `trial_v16_causal_rolling_baseline`, Subject-hole `0.584915`, Tail `0.596018`, Public LB unknown.
- Recent warning: `trial_v15_subject_temporal_deviation` improved local CV but worsened Public LB to `0.5994936146`, so subject personalization is risky.

Recommended next work:

1. Confirm or record the Public LB for the V16 safe submission.
2. Plan `V17 Target Chain Stacking` using OOF probabilities only.
3. Run adversarial validation before introducing train-test shift weighting.
4. Try conservative target-wise Platt/logit calibration only when both Subject-hole and Tail validation improve.
