# Titanic Auto Submit Loop

Deprecated: this document describes the early Titanic-only prototype loop. The
current agent CLI, local app, web app, and AWS deployment path should use
`scripts/generic_workspace_auto_loop.py` through the main CLI/app controls.

This runner is designed so the same Python entrypoint works from local CMD and
from AWS job runners.

For non-Titanic experiments registered through the interactive CLI, use the
generic workspace runner. It follows the same runtime directory, lock, and pause
model:

```bash
python scripts/generic_workspace_auto_loop.py --competition <slug> --start-trial trial_001 --max-trials 5 --submit --kaggle-slug <slug>
```

Local CMD:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
scripts\titanic_auto_submit_loop.cmd --start trial_002 --end trial_005
```

Pause after the currently running trial is submitted and recorded:

```bat
scripts\titanic_auto_submit_loop.cmd --request-pause
```

Resume from the saved `next_trial`:

```bat
scripts\titanic_auto_submit_loop.cmd --resume
```

Show current loop state:

```bat
scripts\titanic_auto_submit_loop.cmd --status
```

AWS/ECS/Batch style:

```bash
python scripts/titanic_auto_submit_loop.py --start trial_002 --end trial_005
```

Recommended runtime settings:

```bash
export RESEARCH_AGENT_RUNTIME_DIR=/mnt/research-agent/users/<user-id>/titanic
python scripts/titanic_auto_submit_loop.py --resume
```

- Mount the runtime directory and `demo_workspaces`, `memory`, and `submissions`
  on persistent storage when a task may be replaced or restarted.
- Use one ECS/Batch task per user and competition. The runner creates an atomic
  `auto_loop.lock` so a shared runtime directory cannot run two loops at once.
- A pause request is also stored as `pause.request`, so it is not lost when the
  runner writes status at the same time.
- For a multi-host service, every task for the same user/competition must use the
  same EFS runtime directory. If that cannot be guaranteed, use a DynamoDB lease
  or Step Functions execution ID as the distributed single-run authority.
- Run the interactive CLI on a persistent shell host such as EC2 or CloudShell.
  In ECS/Batch, start `titanic_auto_submit_loop.py` as the task's main process;
  a child process does not survive after the container's main process exits.

Credential model:

- The app should not use a shared service-owner Kaggle or OpenAI key.
- Each user supplies their own credentials.
- In AWS, inject credentials into the job environment from a user-owned secret.
- Resolve the user's secret immediately before launching the task, inject it only
  into that task, and never persist or print the secret value.
- Supported Kaggle env inputs are the standard Kaggle CLI options such as
  `KAGGLE_USERNAME` + `KAGGLE_KEY`, or a CLI-authenticated runtime.

Loop behavior:

1. Run one trial locally.
2. Generate `submission.csv`.
3. Submit to Kaggle with the user's Kaggle credentials.
4. Poll `kaggle competitions submissions -c titanic`.
5. Parse the submitted public score.
6. Record the submission through the project `record_submission_result` path.
7. Update manual trial metrics and user-facing score artifacts.
8. Continue to the next trial.

Outputs:

```text
demo_workspaces\titanic\manual_trials\auto_loop_summary.csv
demo_workspaces\titanic\manual_trials\<trial_id>\metrics.json
demo_workspaces\titanic\manual_trials\<trial_id>\04_loop_decision.md
submissions\titanic\submission_log.jsonl
demo_workspaces\titanic\manual_trials\auto_loop_state.json
```

Pause/resume semantics:

- Pause is safe-stop only. It does not interrupt the active trial.
- If pause is requested during `trial_004`, the loop finishes local execution,
  submits `trial_004`, records the Kaggle public score, refreshes artifacts, and
  stops before `trial_005`.
- Resume reads `auto_loop_state.json` and starts from `next_trial`.

AWS checks before production:

1. The image contains the Kaggle CLI and the project Python dependencies.
2. The task role can read only the current user's secret and EFS access point.
3. The task has outbound HTTPS access to OpenAI and Kaggle.
4. `RESEARCH_AGENT_RUNTIME_DIR` survives task replacement.
5. Two tasks pointed at the same runtime directory demonstrate lock rejection.
6. A pause request made during a trial leaves the state at `paused` only after
   Kaggle submission and LB recording complete.
