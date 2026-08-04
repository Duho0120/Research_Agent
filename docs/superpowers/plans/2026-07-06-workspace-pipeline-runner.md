# Workspace Pipeline Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect validated Execution Profiles to a generic, explicitly approved local pipeline runner.

**Architecture:** Add one focused runner module that owns command rendering, ordered subprocess execution, result persistence, artifact inspection, and decision logging. Keep CLI parsing thin and reuse execution-profile validation and local-failure classification.

**Tech Stack:** Python standard library, unittest, existing simple YAML, store, paths, policy gate, and decision log modules.

---

### Task 1: Runner Contract

**Files:**
- Create: `tests/test_workspace_runner.py`
- Create: `research_agent/workspace_runner.py`

- [x] Write a failing dry-run test proving no external command executes without `run_now=True`.
- [x] Run `python -B -m unittest tests.test_workspace_runner.WorkspaceRunnerTest.test_dry_run_records_plan_without_executing -v` and confirm the runner import or behavior fails.
- [x] Implement profile validation, command rendering, planned result persistence, and decision logging.
- [x] Run the focused test and confirm it passes.

### Task 2: Ordered Execution And Failure Stop

**Files:**
- Modify: `tests/test_workspace_runner.py`
- Modify: `research_agent/workspace_runner.py`

- [x] Add failing tests for successful `test -> train -> predict` execution and stopping after the first failed command.
- [x] Run `python -B -m unittest tests.test_workspace_runner -v` and confirm the new assertions fail for missing execution behavior.
- [x] Implement sequential subprocess execution, per-command logs, exit-code capture, and local failure classification.
- [x] Run the focused test module and confirm it passes.

### Task 3: Artifact And Validation Gates

**Files:**
- Modify: `tests/test_workspace_runner.py`
- Modify: `research_agent/workspace_runner.py`

- [x] Add failing tests for invalid-profile blocking and missing-artifact reporting.
- [x] Run the focused test module and confirm the new assertions fail.
- [x] Implement artifact inspection and final `blocked`, `incomplete_artifacts`, or `completed` status selection.
- [x] Run the focused test module and confirm it passes.

### Task 4: CLI And Documentation

**Files:**
- Modify: `tests/test_workspace_runner.py`
- Modify: `research_agent/cli.py`
- Create: `docs/workspace_execution.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [x] Add a failing CLI test for `run-workspace-pipeline --competition demo --trial trial_001 --run-now`.
- [x] Run the focused CLI test and confirm argparse rejects the command.
- [x] Add the CLI parser and handler, then document command usage, statuses, logs, and phase boundary.
- [x] Run the focused test module and confirm it passes.

### Task 5: Full Verification

**Files:**
- Verify all changed files.

- [x] Run `python -B -m compileall -q research_agent tests` and require exit code 0.
- [x] Run `python -B -m unittest discover -s tests -v` and require zero failures.
- [x] Run `git diff --check` and require no whitespace errors.
- [x] Review `git diff --stat` and `git status --short` without reverting unrelated user changes.
