# Data Notes

- Window size: 30 frames
- Step size: 5 frames
- Split: scenario-group split, keeping C1/C2/C3/C4 views of the same event in the same split
- Leakage check from notebook output: 0 overlapping groups
- Validation windows: 1050
- Validation view distribution is balanced across C1/C2/C3/C4, but C1 has the highest error rate.

## Observed Validation Distribution

- Normal: 68 windows
- Bed Exit: 112 windows
- Wandering: 254 windows
- Fall: 616 windows

## Current Error Concentration

- Hardest view: C1, error rate 0.1686
- Most difficult scenario: 00620_H_D_SY, 23 errors out of 60 windows
- Main semantic confusion: Bed Exit vs Wandering

