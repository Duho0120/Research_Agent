# Titanic 5-Trial CMD Runner

Deprecated: this document describes the early Titanic-only manual runner. The
current agent flow should start experiments from the CLI/app, which launches
`scripts/generic_workspace_auto_loop.py`.

Run from Command Prompt or PowerShell:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
scripts\titanic_run_5_trials_submit.cmd
```

This runs `trial_001` through `trial_005` sequentially. After each local run,
it submits that trial's `submission.csv` to Kaggle, then pauses for the public
LB score before moving to the next trial.

Local-only dry run:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
scripts\titanic_run_5_trials_local.cmd
```

Outputs are written under:

```text
demo_workspaces\titanic\manual_trials\
```

Useful files:

```text
demo_workspaces\titanic\manual_trials\summary.csv
demo_workspaces\titanic\manual_trials\trial_001\submission.csv
demo_workspaces\titanic\manual_trials\trial_002\submission.csv
demo_workspaces\titanic\manual_trials\trial_003\submission.csv
demo_workspaces\titanic\manual_trials\trial_004\submission.csv
demo_workspaces\titanic\manual_trials\trial_005\submission.csv
```

Trial axes:

| Trial | Change axis |
| --- | --- |
| trial_001 | baseline logistic regression |
| trial_002 | family-size features |
| trial_003 | title feature from passenger name |
| trial_004 | random forest with title/cabin features |
| trial_005 | gradient boosting with title/cabin features |

Important:

The script now gates progression on submission feedback. It does not silently
submit all five trials and finish; after every submit, enter the public LB score
shown by Kaggle so `summary.csv` records the leaderboard result before the next
trial starts.
