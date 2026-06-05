# ETRI Data Notes

Original project root:

```text
C:\Users\ASUS\Desktop\[ZB]mentoring_project\ETRI_Human_Understand
```

Known files from handoff:

- `ch2026_metrics_train.csv`: labeled train file.
- `ch2026_submission_sample.csv`: sample submission with 250 rows.
- `ch2025_data_items/`: parquet sensor/event logs.
- `ch2026_metrics_description.pdf`: metric and target description document.

Key columns:

- `subject_id`
- `sleep_date`
- `lifelog_date`

Targets:

- `Q1`
- `Q2`
- `Q3`
- `S1`
- `S2`
- `S3`
- `S4`

Submission columns:

```text
subject_id,sleep_date,lifelog_date,Q1,Q2,Q3,S1,S2,S3,S4
```

Sensor aggregation notes:

- Event-level parquet data is aggregated by subject and sleep-date or time window.
- Known windows include `day`, `evening`, `prebed`, `late`, `deep`, `wake`, and `other`.
- Useful summaries include counts, mean, std, min, max, sum, entropy, RSSI, GPS range, and usage time.
- Rolling features have used 3, 7, 14, and causal longer-window baselines.

Validation warning:

- The train set has only 450 rows and 10 subjects, so subject-specific shortcuts are a major risk.
- Random KFold is not acceptable as the main validation.
- Always compare Subject-hole CV and Tail CV.
