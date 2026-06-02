# trial_002 Code Patch Plan

## Strategy

sota_architecture_attempt

## Target Files

- experiments/demo/trial_002/config.yaml
- scripts/demo_train.py

## Config Changes

- model.type: skeleton_transformer
- features.use_view_aware_features: True
- features.use_bed_wandering_aux_head: True
- training: {'epochs': 100, 'batch_size': 32, 'warmup_epochs': 10, 'early_stopping_patience': 25}

## Implementation Steps

- Add or select a training branch for the SOTA/model-architecture candidate.
- Keep the output contract unchanged: metrics.json and optional submission.csv.
- Run validation before any leaderboard submission.

## Validation Commands

- `python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002`
- `python -B -m unittest discover -s tests -v`

## Submit Gate

- Do not submit until training writes metrics.json.
- Before submission, record current leaderboard score/rank.
- After submission, record submitted score/rank and preserve all trial artifacts.
