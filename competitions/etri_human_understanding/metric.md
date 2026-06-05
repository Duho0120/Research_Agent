# Metric

The competition is handled as a multi-output binary probability prediction task.

Working metric:

```text
mean binary log loss over Q1,Q2,Q3,S1,S2,S3,S4
```

Objective:

```text
minimize
```

Operational rules:

- Submit probabilities, not hard labels.
- Clip final probabilities conservatively, usually to `0.02..0.98`.
- Track both local validation score and DACON Public LB result.
- Do not promote a trial based only on local CV if Public LB worsens.
