# Research Notes

## Current Understanding

- Patient action skeleton is the current real-world validation case for the research agent.
- Baseline was imported from V07_COM_Lean_Method_4-Torch_XPU.ipynb.

## Promising Directions

- Focus on C1 view and Bed Exit/Wandering confusion while preserving Fall recall.
- Use human review when scenario/view/ROI ambiguity requires visual judgment.

## Failed Directions

- Threshold-only Bed Exit tuning is not enough because it can lower macro F1.

## patient_action_skeleton / trial_001_v07_baseline

- CV: 0.7926
- LB: None
- Recommendation: accept_as_candidate
- Best so far: True
- Notes: Baseline imported from V07_COM_Lean_Method_4-Torch_XPU.ipynb. Strong Fall performance, but Bed Exit/Wandering confusion remains the main improvement target.
