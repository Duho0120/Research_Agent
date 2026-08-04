# Context Pack - workspace_code_writing

- competition: 236716
- trial_id: trial_001
- query: next experiment current project code competition data card data profile target columns feature recommendations previous trial metrics result pipeline structure decision card rejected axes recommended base workspace context snapshot validation allowed write files
- document_count: 3
- retrieval_manifest_file: `experiments/236716/trial_001/retrieval_manifest_workspace_code_writing.json`

## Retrieved Evidence

### 1. decision_card - memory/236716/latest_decision_card.md

- score: 52
- trial_id: -

```text
# Trial Decision Card: trial_001

- decision: baseline_established
- change_axis: None
- source_trial_id: None
- recommended_base_trial: trial_001
- local_score: 1.0
- local_status: baseline
- local_delta: None
- previous_local_status: baseline
- previous_local_delta: None
- active_axis: None
- axis_attempt_count: 0
- axis_attempt_limit: 3
- lb_score: None
- lb_status: missing
- lb_delta: None
- previous_lb_status: missing
- previous_lb_delta: None

## Next Guidance

Use this trial as the baseline. The next trial should change exactly one improvement axis.

## Planner Constraints

- Use `trial_001` as the base trial unless the user explicitly overrides it.
- Change exactly one primary improvement axis in the next trial.
- Keep split/model/p
```

### 2. trial_memory_card - memory/236716/latest_trial_memory_card.md

- score: 48
- trial_id: -

```text
# Trial Memory Card: trial_001

- plan_type: initial_pipeline_plan
- source_trial_id: None
- change_axis: 
- candidate_label: Baseline: Constant-Velocity Extrapolation (+80ms from t0,t-40ms)
- local_score: 1.0
- local_status: baseline
- local_delta: None
- previous_local_status: baseline
- previous_local_delta: None
- lb_score: None
- decision: baseline_established
- active_axis: None
- axis_attempt_count: 0/3
- recommended_base_trial: trial_001
- model_type: RuleBasedConstantVelocity
- model: {'estimator': 'RuleBasedConstantVelocity', 'parameters': {}, 'pretrained': False, 'pipeline_container': None}
- split: {'method': 'random_holdout_by_id', 'test_size': 0.2, 'stratify': False}

## Change Details

- None

## Kept Unchanged

- None

## Fe
```

### 3. decision_card - memory/236716/latest_decision_card.json

- score: 41
- trial_id: -

```text
{
  "schema_version": "1.0",
  "competition": "236716",
  "trial_id": "trial_001",
  "created_at": "2026-07-31T00:04:41.690183+00:00",
  "source_trial_id": null,
  "plan_type": "initial_pipeline_plan",
  "change_axis": "",
  "local_score": 1.0,
  "lb_score": null,
  "previous_local_score": null,
  "previous_lb_score": null,
  "best_local_score": null,
  "best_lb_score": null,
  "local_delta": null,
  "lb_delta": null,
  "previous_local_delta": null,
  "previous_lb_delta": null,
  "objective": "maximize",
  "local_status": "baseline",
  "lb_status": "missing",
  "previous_local_status": "baseline",
  "previous_lb_status": "missing",
  "raw_decision": "baseline_established",
  "decision": "baseline_established",
  "model_type": "RuleBasedCons
```
