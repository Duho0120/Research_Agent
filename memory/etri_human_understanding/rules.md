# Rules: ETRI Human Understanding

- Do not use random KFold as the main validation.
- Always evaluate Subject-hole CV and Tail CV.
- Treat Public LB as important evidence because local CV has already overestimated some personalized approaches.
- Do not submit aggressive main variants before safe variants.
- Do not use Kaggle CLI for this competition; this is DACON/external submission mode.
- Keep S2 and S3 close to V11/base unless there is strong Subject-hole, Tail, and Public evidence.
- Use OOF predictions for stacking and classifier chains.
- Do not use true labels directly as chain features.
- Avoid full-period subject baseline features unless explicitly testing transductive behavior.
- Prefer causal rolling baseline features over full-history subject deviations.
- Clip final probabilities to `0.02..0.98`.
- Track mean absolute difference versus V11 for each target before submission.
- Record manual DACON submission results with before/after score and notes.
