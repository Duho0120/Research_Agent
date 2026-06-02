# trial_001_v07_baseline Plan

## Goal

Record the current V07 COM Lean Method 4 notebook as the baseline trial.

## Source Notebook

`C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\Notebooks\V07_COM_Lean_Method_4-Torch_XPU.ipynb`

## Current Setup

- 30-frame skeleton sequence classification
- Four classes: Normal, Bed Exit, Wandering, Fall
- Transformer encoder, input size 127, d_model 96, 2 layers, 4 heads
- LDAM main loss with DRW
- Auxiliary Bed Exit vs Wandering head
- Scenario-group validation split with no group leakage

## Baseline Observation

The model performs strongly on Fall but still confuses Bed Exit and Wandering, especially in C1 view and specific difficult scenarios.

## Success Criteria For Future Trials

- Improve Bed Exit/Wandering separation without reducing Fall recall.
- Reduce C1 view error rate.
- Reduce concentrated errors in scenario 00620_H_D_SY.
- Prefer macro F1 and Bed Exit F1 over raw accuracy only.

