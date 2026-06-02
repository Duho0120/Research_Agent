# Research Rules

- One trial should test one primary hypothesis.
- Do not change validation strategy and model family in the same trial unless the Main Agent explicitly marks a validation review.
- If CV improves but LB worsens in two consecutive submissions, suspect validation mismatch.
- If three trials in a row fail to improve, shift strategy instead of making smaller tweaks.
- Do not submit when the expected gain is smaller than known seed variance.
- Treat sudden large score gains as possible leakage until checked.
