# Autonomous Kaggle Research Agent

Starter implementation for a staged autonomous Kaggle research system.

## What is included

- Knowledge and Memory Agent: manages competition context, data profiles, research notes, trial memory, rules, review packs, and user feedback.
- Planning Agent: plans trials, chooses improvement axes, advises model-family candidates, generates baseline plans, prepares patch plans, and creates coding handoff requests.
- Training and Execution Agent: creates local/Colab jobs, runs local commands, applies scoped code-writer updates, runs validation commands, gates post-validation execution, offers a safe execution chain, and preserves run artifacts.
- Evaluation and Decision Agent: evaluates metrics, diagnoses results, validates patch/coding/submission gates, and decides when human review is needed.
- Feedback and Orchestration Agent: coordinates the loop, records decisions, updates memory, and controls bounded auto-research cycles.

The implementation keeps execution conservative. It can plan, profile data, generate a baseline, record, evaluate, create local/Colab jobs, validate configs, prepare/validate/handoff API coding requests, apply scoped code-writer updates, and preserve submission metadata. Real Kaggle API submission is still exposed as a future integration point.

Although the project began as a Kaggle research agent, competition memory can also track external competitions. External platforms such as DACON should use manual or platform-specific submission records rather than the Kaggle CLI adapter.

## Current Status

Completed:

- The project is organized as 5 top-level agents with smaller internal tools/modules for implementation detail.
- The conservative research loop can plan, validate, run locally, evaluate metrics, diagnose results, update memory, and produce next-experiment recommendations.
- Knowledge and Memory tools can inspect competitions, profile `data/<competition>/`, detect train/test/sample-submission roles, infer simple task/schema hints, and write `data_profile.md/json`.
- Planning tools can create `next_experiment.md`, choose the next improvement axis, advise task-appropriate model candidates, create a first tabular baseline plan, convert plans into `config.yaml` and `code_patch_plan.md/json`, and prepare `coding_handoff.json` / `coding_agent_request.md`.
- Training and Execution tools can apply a prepared patch plan, run a local command, then produce `metrics.json`, `evaluation.md`, `diagnosis.md`, and `code_edit_result.md/json`.
- Evaluation and Decision tools can check `code_patch_plan.json` before execution, write `patch_validation.md/json`, validate `coding_result.json` against the handoff contract, record decisions in `decision_log.jsonl`, and block unsafe or incomplete patch/code results.
- Submission tools can record before/after score and rank metadata, preserve version files, mark the best trial, and expose `submit-trial` for manual or command-hooked submission runs.
- Submission now has a safer approval gate: `prepare-submission` writes `submit_manifest.md/json` without touching leaderboard logs or best-trial markers.
- Submission best tracking now compares against the stored historical best submission, handles first submissions with no previous leaderboard score, and updates `state.yaml` from confirmed leaderboard results.
- The Kaggle CLI adapter can now build safe submit/leaderboard argument lists and return structured check/submit/leaderboard command results.
- The Kaggle CLI adapter can parse CSV/table leaderboard output and poll until a target team appears or the polling window times out.
- `submit-trial` can call the Kaggle CLI adapter directly, check CLI/auth status, submit, poll the leaderboard, and only record best markers after a leaderboard score/rank is found.
- `inspect-competition` accepts a Kaggle competition URL or slug, checks CLI/auth access, lists available data files, and writes `competition_inspection.md/json`.
- `start-competition` combines inspection, project initialization, onboarding notes, data profiling, and the first `trial_001` plan.
- `run-auto-loop` can run a bounded safe research loop across multiple trials with a no-improvement stop policy and submission disabled by default.
- `run-graph-cycle` exposes the same conservative one-trial flow through a LangGraph `StateGraph` orchestration layer while keeping the existing Python tools as graph nodes.
- Policy files under `configs/policies/` control token use, execution decisions, and human-review decisions.
- Research Operating Protocol now provides a small domain-neutral flow: current state, evidence, issues, candidate actions, constraints, user questions, and execution plan are written before code-oriented follow-up. Competition-specific checks are opt-in.
- Execution Profile provides a competition-neutral contract for an external project root, Python runtime, test/train/predict commands, expected artifacts, and allowed/forbidden write scope. Validation does not execute the project.
- `prepare-workspace` can initialize an isolated workspace from any local project path or research topic, create a bounded file inventory, draft an Execution Profile from conventional entrypoints, and surface uncertain fields as review questions without executing external code.
- `prepare-workspace --create-workspace` can scaffold a local research workspace under `demo_workspaces/<competition>` with `data/`, `src/`, `tests/`, `outputs/`, and lightweight `test/train/predict` scripts; competition data remains a manual user-provided input under `data/`.
- `run-workspace-pipeline` connects a ready Execution Profile to an explicitly approved local `test -> train -> predict` run, stops on the first failure, and records command logs plus expected-artifact checks without embedding competition-specific logic.
- `collect-workspace-metrics` copies a completed run's JSON metrics into the trial contract, accepts either `cv_score` or an explicit dotted `metrics_contract.source_key`, and requests review instead of guessing ambiguous numeric fields.
- `process-workspace-result` evaluates and diagnoses collected metrics, records trial memory idempotently, defers nonurgent Human Review until two comparable trials exist, and immediately escalates leakage, label ambiguity, safety false negatives, or missing required definitions.
- `plan-next-workspace-trial` connects processed workspace results to the next experiment plan, registers user review requests, blocks urgent review states, and allows nonurgent review states to continue with explicit caution metadata.
- `prepare-workspace-handoff` turns a workspace next-experiment plan into a scoped coding-agent request from the Execution Profile write scope without editing the external project.
- Workspace coding handoff now writes `workspace_context_snapshot.md`, combining previous-trial evidence and bounded current project code so later trials can modify the existing pipeline instead of starting from a blank request.
- `run-workspace-code-writer` applies mock/API code-writer file updates under the external `project_root` only when every changed path is project-root-relative, allowed by `allowed_write_paths`, and outside forbidden metric/submission artifacts.
- `run-workspace-after-coding` requires an accepted workspace coding result, then re-enters the existing workspace pipeline, metrics collection, and result-cycle flow; without `--run-now` it records only the planned execution.
- `demo-one-cycle` provides the two-week demo path for F-01/F-02/F-03/F-04/F-06 only: rule-based context loading, LLM/mock experiment planning, LLM/mock code writing, local execution, and file-based result recording without F-05 self-improvement, Human Review, submission, approval UI, or SQLite.
- Demo observability is CMD-only for now: `demo-one-cycle --show-progress` prints live stage messages, while `watch-demo-cycle` reads `agent_status.json` and `agent_events.jsonl` to show whether the demo agent is running, blocked, planned, or completed.
- Trial artifact organization now writes a user-facing `user_view/` folder with Korean summary, plan, pipeline-structure, code-pipeline, result files, and copied changed code; it also mirrors that compact view into `runs/<competition>/<trial>/` for SFTP/file-browser users while debug/API payloads move into `debug/` and machine-facing JSON records move into `internal/`.
- Each organized trial now writes `internal/pipeline_structure.json` plus `02_pipeline_structure.ko.md`, so the executable `.py` pipeline remains the source of truth while users and later coding handoffs can read a notebook-like stage map.
- `demo-guide` provides a demo-only interactive CMD flow that lists registered experiments, guides new workspace creation, asks the user to place manual data files, and then starts the one-cycle demo path when OpenAI API execution is explicitly confirmed.
- Demo guide execution uses the OpenAI Responses API with `OPENAI_API_KEY` and overrides the one-cycle provider/model to `openai` / `gpt-5.5`; low-cost policy calls remain on OpenAI `gpt-5.6-luna`.
- Demo guide can start from a competition URL. It attempts a lightweight page read and infers simple defaults such as Kaggle slug/platform; when the page is unreadable or dynamic, it now asks for only one optional `source_summary` instead of several separate problem/data/submission/evaluation/rule prompts. That summary is saved to `source_materials.md/json`, `overview.md`, `data_notes.md`, and `metric.md`, then included in the one-cycle planning context.
- The policy gate records local/Colab/wait/ask_user execution decisions and LLM call/skip decisions in `decision_log.jsonl`.
- Human Review now has a closed feedback loop: `request-review` creates a review pack, `record-feedback` updates that pack, and `plan-next` can use recent user feedback.
- Failed local runs now write `local_failure.json/md`, and the execution policy gate uses that structured artifact before falling back to raw log pattern matching.
- The Pipeline Improvement Planner can choose the next improvement axis such as validation, preprocessing, augmentation, loss/metric alignment, hyperparameter tuning, model family, pretraining strategy, post-processing, or human review.
- The Model Candidate Advisor can read `data_profile.json`, metrics, state, and `pipeline_improvement_plan.json` to write `model_candidates.md/json` without treating model change as the default next action.
- The Pipeline Patch Planner uses `pipeline_improvement_plan.json` to translate selected axes into config changes, target files, approval gates, and implementation steps.
- Patch validation now checks target files, generated config validity, required validation commands, protected-axis violations, user-approval gates, and forbidden submission-artifact edits before patch execution. `apply-patch` uses this validator result as the single patch safety gate.
- Coding handoff now standardizes what is sent to a future Codex/API coding worker: objective, target files, required config changes, implementation steps, guardrails, and validation commands.
- Coding handoff now has a versioned request contract with context files, allowed write files, declared new files, forbidden paths, execution constraints, and a required `coding_result` output schema.
- Coding result validation now checks future coding-worker output before downstream execution, including required fields, status values, changed-file scope, declared new files, and forbidden paths. `write-code-dry-run` can create a blocked placeholder result without calling an external API.
- Code Writer Adapter can now call an injected/mock client or the OpenAI Responses API behind an explicit `--allow-api` flag, accept JSON `file_updates`, apply only allowed paths, and immediately run `validate-coding-result`.
- Validation Command Runner can execute the handoff validation commands after an accepted coding result, write `validation_run.md/json`, and record the command outcome in `decision_log.jsonl`.
- Post-Validation Executor can continue from `validation_run.status == passed` into the existing execution policy, creating a local/Colab job or running locally when explicitly requested.
- Safe Execution Chain can run code writing, coding result validation, validation commands, and post-validation execution in one guarded path that stops at the first failed gate.
- `cycle` and `run-graph-cycle` can optionally call the Safe Execution Chain with `--run-safe-chain`, so the main trial cycle can move from coding handoff to validated execution without bypassing any gate.
- ETRI Human Understanding has been onboarded as a DACON/external competition under `etri_human_understanding`, with state, data notes, research notes, rules, prior trial memory, and V11/V15/V16 trial artifacts.
- Data onboarding now uses local files when available, or falls back to the Kaggle inspection file listing when data has not been downloaded yet.
- Baseline generation currently targets tabular CSV competitions with a detected target column and writes a local sanity-check `submission.csv` and `metrics.json` when run.
- Responsibility boundaries were tightened: Pipeline Patch Planner plans only, Patch Validator is the execution safety gate, Coding Handoff reuses existing validation, and Baseline Generator consumes the saved data profile snapshot.

Available CLI flow:

```text
start-competition / init
  -> plan / cycle
  -> run-graph-cycle
  -> profile data
  -> generate baseline pipeline
  -> validate config
  -> decide execution backend
  -> run local / create local job / ask user / create Colab job / wait
  -> evaluate metrics
  -> diagnose
  -> research protocol
  -> plan pipeline improvement
  -> advise model candidates when model choice matters
  -> decide human review
  -> prepare review pack / record feedback
  -> remember
  -> plan next experiment
  -> prepare patch plan
  -> validate patch plan
  -> prepare coding handoff
  -> run code writer / dry-run code writer
  -> validate coding result
  -> run validation commands
  -> run after validation
  -> run safe execution chain
  -> optionally run safe execution chain inside cycle / run-graph-cycle
  -> apply patch plan
  -> prepare submission manifest
  -> submit after approval and leaderboard evidence

two-week demo path:
  research_agent.cli demo-guide
    -> list registered experiments
    -> accept a competition URL and try lightweight context inspection
    -> ask for pasted source notes when URL content is unavailable
    -> guide new experiment metadata input
    -> create demo workspace and wait for manual data placement
    -> start the one-cycle demo run with OpenAI API after explicit confirmation
  demo-one-cycle
    -> load context from Execution Profile and file memory
    -> create one experiment plan through mock/API LLM
    -> create scoped coding handoff and apply mock/API code updates
    -> optionally run local test/train/predict with --run-now
    -> collect metrics and write demo result record
    -> organize trial artifacts into README / debug / internal
  watch-demo-cycle
    -> read agent_status.json and agent_events.jsonl
    -> print current stage/progress/recent events in CMD
```

Still pending:

- Real-world Kaggle smoke testing with an approved competition and submission file.
- A platform-neutral submission adapter for DACON/external competitions; ETRI currently uses manual submission tracking.
- Data download, schema analysis, and baseline training code generation from a competition inspection.
- Automatic submission policy beyond `never` / `prepare_only`.
- Real-world validation of the code-writer API path with a live key and a deliberately small approved patch.
- Full LangGraph auto-loop replacement; the current graph command covers the one-trial cycle first.
- SQLite remains postponed; current demo/state artifacts are intentionally file-based (`json`, `jsonl`, `md`) so they are easy to inspect and can be migrated later if needed.

Latest verified baseline:

```powershell
python -B -m unittest discover -s tests -v
```

Expected result: `215 tests`, `OK`.

## Agent Architecture

```text
Knowledge and Memory Agent
  -> competition inspection
  -> data profiling
  -> research notes / rules / trial memory
  -> review packs and user feedback

Planning Agent
  -> trial planning
  -> pipeline improvement planning
  -> model candidate advising
  -> baseline generation planning
  -> patch planning
  -> coding handoff preparation

Training and Execution Agent
  -> local / Colab job creation
  -> local run execution
  -> scoped code-writer file updates
  -> validation command execution
  -> post-validation execution gate
  -> safe execution chain
  -> patch application

Evaluation and Decision Agent
  -> metric evaluation
  -> diagnosis
  -> execution / LLM / human-review policy gates
  -> patch validation
  -> coding result validation
  -> submission approval and result recording

Feedback and Orchestration Agent
  -> cycle
  -> run-auto-loop
  -> decision log coordination
  -> memory update and next-trial feedback loop
```

Implementation modules are intentionally smaller than the top-level agents. Files such as `pipeline_planner.py`, `model_advisor.py`, `patch_validator.py`, `coding_handoff.py`, `code_writer_adapter.py`, `coding_result_validator.py`, and `baseline_generator.py` are internal tools used by the 5-agent architecture, not separate top-level agents.

## Policy Design

Cost-efficient autonomous research is governed by policy documents before it is encoded into rule-based gates:

- `docs/policies/execution_decision_policy.ko.md`: local / Colab / ask_user / wait_for_metrics 판단 기준
- `docs/policies/human_review_policy.ko.md`: 어떤 진단 패턴에서 사람에게 물어볼지
- `docs/policies/review_pack_schema.ko.md`: 사람이 볼 자료, 질문, 답변 저장 형식
- `docs/policies/coding_request_schema.ko.md`: Codex/API 코딩 작업자의 입력, 쓰기 범위, 결과 계약

- `docs/policies/research_operating_protocol.ko.md`: minimal common research flow and optional competition policy extensions
- `docs/policies/execution_profile_schema.ko.md`: external project execution contract and write-scope validation

Machine-readable policy files live under `configs/policies/`:

- `token_policy.yaml`
- `execution_policy.yaml`
- `human_review_policy.yaml`
- `pipeline_improvement_policy.yaml`
- `research_operating_policy.yaml`
- `model_policy.yaml`

The first rule-based policy gate is implemented in `kaggle_research_agent/agents/policy_gate.py`, and review pack creation is implemented in `kaggle_research_agent/agents/review_pack.py`.

Useful policy commands:

```powershell
python -B -m kaggle_research_agent.cli decide-llm --competition demo --trial trial_001 --reason human_review_needed
python -B -m kaggle_research_agent.cli request-review --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli research-protocol --competition demo --trial trial_001 --next-trial trial_002
python -B -m kaggle_research_agent.cli prepare-workspace --competition my_workspace --source-path "C:\path\to\project" --topic "Research objective"
python -B -m kaggle_research_agent.cli prepare-workspace --competition titanic_demo --topic "Titanic survival prediction" --platform kaggle --metric accuracy --objective maximize --create-workspace --target-column Survived --id-column PassengerId --required-data-file train.csv --required-data-file test.csv --required-data-file gender_submission.csv
python -B -m kaggle_research_agent.cli validate-execution-profile --competition demo
python -B -m kaggle_research_agent.cli run-workspace-pipeline --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli run-workspace-pipeline --competition demo --trial trial_001 --run-now
python -B -m kaggle_research_agent.cli collect-workspace-metrics --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli process-workspace-result --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli plan-next-workspace-trial --competition demo --source-trial trial_001 --next-trial trial_002
python -B -m kaggle_research_agent.cli prepare-workspace-handoff --competition demo --trial trial_002
python -B -m kaggle_research_agent.cli run-workspace-code-writer --competition demo --trial trial_002 --mock-response-file mock_response.json
python -B -m kaggle_research_agent.cli run-workspace-after-coding --competition demo --trial trial_002 --run-now
python -B -m kaggle_research_agent.cli demo-one-cycle --competition demo --trial trial_001 --mock-plan-file mock_plan_response.json --mock-response-file mock_code_response.json --run-now --show-progress
python -B -m kaggle_research_agent.cli demo-guide
python -B -m kaggle_research_agent.cli watch-demo-cycle --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli watch-demo-cycle --competition demo --trial trial_001 --follow
python -B -m kaggle_research_agent.cli organize-trial-artifacts --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli plan-improvement --competition demo --trial trial_001
python -B -m kaggle_research_agent.cli advise-models --competition demo --trial trial_001
```

Korean CMD/PowerShell demo output:

```bat
cd /d C:\Users\ASUS\Desktop\Research_Agent
set OPENAI_API_KEY=your_api_key_here
python -m research_agent.cli demo-guide
```

Legacy package path still works:

```powershell
chcp 65001
$env:PYTHONUTF8="1"
python -m kaggle_research_agent.cli demo-guide
```

For non-demo commands, use the neutral module path as well:

```bat
python -m research_agent.cli watch-demo-cycle --competition titanic --trial trial_001
```

If Korean text still looks broken, use Windows Terminal or PowerShell 7 and choose a Unicode-capable font such as D2Coding, Consolas, or Cascadia Mono.

`decide-llm` records token-policy decisions in `memory/<competition>/decision_log.jsonl`. `request-review` now prepares the standard `review_pack/` when the human-review policy asks for one.
`plan-improvement` writes `pipeline_improvement_plan.md/json` and keeps model changes as only one branch among validation, data, augmentation, loss, training, post-processing, and review decisions.
`advise-models` writes `model_candidates.md/json` and recommends model families, training strategy, and guardrails from the current data profile and trial evidence.

Human feedback closes the review loop:

```powershell
python -B -m kaggle_research_agent.cli record-feedback --competition demo --trial trial_001 --topic validation --question "Is this split appropriate?" --feedback "Use group split before large model changes." --decision change_validation --follow-up-action "Plan a validation review trial"
```

This writes `user_feedback.jsonl`, updates `review_pack/manifest.json` to `feedback_recorded`, writes `review_pack/human_feedback.md/json`, and logs a `human_feedback` decision with `user_input_used=true`.

## Quick start

```powershell
python -m kaggle_research_agent.cli init --competition playground
python -m kaggle_research_agent.cli plan --competition playground
python -m kaggle_research_agent.cli create-job --competition playground --trial trial_001
```

Or run the conservative one-trial cycle:

```powershell
python -m kaggle_research_agent.cli cycle --competition playground --trial trial_001
```

Or run the same one-trial cycle through the LangGraph orchestration layer:

```powershell
python -m kaggle_research_agent.cli run-graph-cycle --competition playground --trial trial_001
```

By default, jobs are local jobs. Use Colab only when you explicitly need it:

```powershell
python -m kaggle_research_agent.cli create-job --competition playground --trial trial_001 --backend colab
```

If the trial can run locally, execute it directly:

```powershell
python -m kaggle_research_agent.cli run-local --competition playground --trial trial_001 --run-command "python train.py --config experiments/playground/trial_001/config.yaml --output experiments/playground/trial_001"
```

Or plan, validate, run locally, evaluate, and remember in one cycle:

```powershell
python -m kaggle_research_agent.cli cycle --competition playground --trial trial_001 --run-now --run-command "python train.py --config experiments/playground/trial_001/config.yaml --output experiments/playground/trial_001"
```

After training in Colab, place `metrics.json` and optional `submission.csv` under:

```text
experiments/playground/trial_001/
```

Then evaluate and update memory:

```powershell
python -m kaggle_research_agent.cli evaluate --competition playground --trial trial_001
python -m kaggle_research_agent.cli remember --competition playground --trial trial_001
```

If you need a custom Colab command:

```powershell
python -m kaggle_research_agent.cli create-job --competition playground --trial trial_001 --run-command "python train.py --config experiments/playground/trial_001/config.yaml"
```

## Research loop core dry-run

Inspect a Kaggle competition link or slug before starting work:

```powershell
python -B -m kaggle_research_agent.cli inspect-competition --competition https://www.kaggle.com/competitions/titanic
```

This writes:

```text
competitions/<competition_slug>/competition_inspection.md
competitions/<competition_slug>/competition_inspection.json
```

Start a competition workspace from a Kaggle link or slug:

```powershell
python -B -m kaggle_research_agent.cli start-competition --competition https://www.kaggle.com/competitions/titanic --metric accuracy --objective maximize
```

This runs inspection, initializes the competition workspace, writes onboarding notes, and creates the first `trial_001` plan/config.

Profile local competition data after downloading files into `data/<competition>/`:

```powershell
python -B -m kaggle_research_agent.cli profile-data --competition titanic
```

This writes:

```text
competitions/<competition>/data_profile.md
competitions/<competition>/data_profile.json
```

Generate the first local baseline pipeline from the data profile:

```powershell
python -B -m kaggle_research_agent.cli generate-baseline --competition titanic --trial trial_001
```

Then run the generated command from `baseline_pipeline.json` or `baseline_plan.md` to create `metrics.json` and `submission.csv`.

Diagnose a completed trial:

```powershell
python -B -m kaggle_research_agent.cli diagnose --competition demo --trial trial_001
```

Run the conservative cycle on a completed trial to evaluate, diagnose, record a decision log, and remember:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job
```

Create a user review request when the diagnosis needs human input:

```powershell
python -B -m kaggle_research_agent.cli request-review --competition demo --trial trial_001
```

Record user feedback:

```powershell
python -B -m kaggle_research_agent.cli record-feedback --competition demo --trial trial_001 --topic validation --question "Is this split appropriate?" --feedback "Use group split before large model changes." --decision change_validation --follow-up-action "Plan a validation review trial"
```

Record a manual submission result without calling Kaggle:

```powershell
python -B -m kaggle_research_agent.cli record-submission --competition demo --trial trial_001 --version-name demo_trial_001_baseline_v01 --submission-file experiments/demo/trial_001/submission.csv --cv-score 0.83 --previous-lb-score 0.80 --previous-rank 120 --submitted-lb-score 0.84 --submitted-rank 90 --objective maximize --notes "Manual leaderboard entry"
```

Prepare a submission manifest before any real leaderboard action. This is the safe approval gate and does not update `submission_log.jsonl`, `VERSION.md`, or `BEST_MARKER.md`.

```powershell
python -B -m kaggle_research_agent.cli prepare-submission --competition demo --trial trial_001 --version-name demo_trial_001_v01 --submission-file experiments/demo/trial_001/submission.csv --objective maximize --notes "Ready for approval"
```

Run the submission workflow for a trial. Without `--submit-command`, this records a submission run and preserves the before/after leaderboard evidence; with command hooks, it can also execute external Kaggle CLI/API wrappers later.

```powershell
python -B -m kaggle_research_agent.cli submit-trial --competition demo --trial trial_001 --version-name demo_trial_001_v01 --submission-file experiments/demo/trial_001/submission.csv --before-score 0.80 --before-rank 120 --after-score 0.84 --after-rank 90 --objective maximize --notes "Manual leaderboard entry"
```

Run the submission workflow with the Kaggle CLI adapter after approving the manifest:

```powershell
python -B -m kaggle_research_agent.cli submit-trial --competition demo --trial trial_001 --version-name demo_trial_001_v01 --submission-file experiments/demo/trial_001/submission.csv --kaggle-competition-slug demo-competition --kaggle-team-name "my team" --kaggle-message "demo_trial_001_v01" --poll-leaderboard --poll-attempts 5 --poll-interval-seconds 30 --objective maximize --notes "Kaggle CLI submission"
```

The Kaggle CLI adapter lives under `kaggle_research_agent/integrations/kaggle_cli.py`. It exposes structured helpers for:

```text
check_cli_available
check_cli_auth
submit_competition
fetch_leaderboard
parse_leaderboard
poll_leaderboard
```

Research Planner low-level command: plan the next experiment from the latest diagnosis and submission evidence.

```powershell
python -B -m kaggle_research_agent.cli plan-next --competition demo --source-trial trial_001 --next-trial trial_002
```

Research Planner low-level command: prepare an executable code/config patch plan for that next experiment.

```powershell
python -B -m kaggle_research_agent.cli prepare-patch --competition demo --source-trial trial_001 --next-trial trial_002
```

Validate a prepared patch plan before applying it:

```powershell
python -B -m kaggle_research_agent.cli validate-patch --competition demo --trial trial_002
```

If the patch plan intentionally changes a protected or expensive area that requires approval:

```powershell
python -B -m kaggle_research_agent.cli validate-patch --competition demo --trial trial_002 --user-approved
```

Prepare a coding-agent handoff request from a validated patch plan:

```powershell
python -B -m kaggle_research_agent.cli prepare-handoff --competition demo --trial trial_002
```

Create a dry-run coding result without calling an external coding API:

```powershell
python -B -m kaggle_research_agent.cli write-code-dry-run --competition demo --trial trial_002
```

Run the Code Writer Adapter with a local mock response file:

```powershell
python -B -m kaggle_research_agent.cli run-code-writer --competition demo --trial trial_002 --mock-response-file mock_response.json
```

Run the Code Writer Adapter and immediately execute accepted validation commands:

```powershell
python -B -m kaggle_research_agent.cli run-code-writer --competition demo --trial trial_002 --mock-response-file mock_response.json --run-validation-commands
```

Run the Code Writer Adapter with the OpenAI Responses API only when you explicitly approve an API call:

```powershell
python -B -m kaggle_research_agent.cli run-code-writer --competition demo --trial trial_002 --model gpt-5 --allow-api
```

Validate a coding-agent result before downstream execution:

```powershell
python -B -m kaggle_research_agent.cli validate-coding-result --competition demo --trial trial_002
```

Execute validation commands after an accepted coding result:

```powershell
python -B -m kaggle_research_agent.cli run-validation-commands --competition demo --trial trial_002
```

Continue from passed validation into the execution policy:

```powershell
python -B -m kaggle_research_agent.cli run-after-validation --competition demo --trial trial_002 --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

Run locally immediately only when requested:

```powershell
python -B -m kaggle_research_agent.cli run-after-validation --competition demo --trial trial_002 --run-now --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

Run the guarded end-to-end code-to-execution chain:

```powershell
python -B -m kaggle_research_agent.cli run-safe-execution-chain --competition demo --trial trial_002 --mock-response-file mock_response.json --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

Run that same guarded chain from the normal cycle only when explicitly requested:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_002 --run-safe-chain --mock-response-file mock_response.json --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

The LangGraph one-trial cycle supports the same guarded branch:

```powershell
python -B -m kaggle_research_agent.cli run-graph-cycle --competition demo --trial trial_002 --run-safe-chain --mock-response-file mock_response.json --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

Or let the cycle create the next-experiment recommendation immediately after diagnosis and memory update:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002
```

To create the next-experiment recommendation and patch plan in the same cycle:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002 --prepare-next-patch
```

Run a bounded safe auto loop. By default this does not submit to Kaggle.

```powershell
python -B -m kaggle_research_agent.cli run-auto-loop --competition demo --start-trial trial_001 --max-trials 3 --submit-policy never --stop-no-improvement 3
```

Experiment Runner low-level command: apply a prepared patch plan and optionally run the next trial.

```powershell
python -B -m kaggle_research_agent.cli apply-patch --competition demo --trial trial_002 --run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

Or let the cycle prepare and apply the next patch in one pass:

```powershell
python -B -m kaggle_research_agent.cli cycle --competition demo --trial trial_001 --no-job --next-trial trial_002 --prepare-next-patch --apply-next-patch --next-run-command "python scripts/demo_train.py --config experiments/demo/trial_002/config.yaml --output experiments/demo/trial_002"
```

## Main workflow

```text
init competition
  -> plan trial
  -> validate config
  -> profile data when data is available
  -> generate first baseline pipeline
  -> decide execution backend
       -> run local
       -> create local job
       -> ask user
       -> create Colab job
       -> wait for metrics
  -> collect metrics/results
  -> evaluate
  -> diagnose
  -> plan pipeline improvement axis
  -> decide human review
       -> prepare review pack
       -> request user feedback
       -> ingest feedback
  -> update memory and decision log
  -> plan next trial
  -> prepare pipeline patch plan
  -> validate patch plan
  -> prepare coding handoff
  -> write code / dry-run code writer
  -> validate coding result
  -> apply patch plan
  -> prepare submission manifest
  -> submit only after approval/evidence
```

## Folder map

```text
kaggle_research_agent/  Python package and CLI
kaggle_research_agent/agents/  Internal tools for planning, execution, evaluation, memory, and decisions
kaggle_research_agent/integrations/  External service adapters such as Kaggle CLI
competitions/           Competition profiles and state
data/<competition>/      Local downloaded competition data
experiments/            Trial artifacts
memory/<competition>/   Research notes, rules, trial index
colab/                  Worker notebook and script
jobs/<competition>/     Local/Colab job queue files
configs/<competition>/  Allowed config/search-space templates
```

Competition-specific state is isolated by default:

```text
competitions/<competition>/state.yaml
experiments/<competition>/<trial_id>/
memory/<competition>/
jobs/<competition>/
configs/<competition>/
```
