# Workspace After-Coding Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enter the workspace execution, metrics, and result-cycle flow after a workspace code change has been accepted.

**Architecture:** Add a small after-coding runner that requires `workspace_coding_result_validation.status == accepted`, then delegates to existing `run_workspace_pipeline`, `collect_workspace_metrics`, and `process_workspace_result`. Without `--run-now`, it only records the planned execution and does not run external commands.

**Tech Stack:** Python standard library, existing workspace runner/collector/result-cycle modules, existing CLI, `unittest`.

---

### Task 1: Failing Tests

**Files:**
- Create: `tests/test_workspace_after_coding.py`

- [ ] Test blocked when workspace coding result validation is missing or not accepted.
- [ ] Test dry-run mode records a planned workspace run without executing commands.
- [ ] Test `run_now=True` executes pipeline, collects metrics, and processes result.
- [ ] Test CLI command.

### Task 2: After-Coding Runner Module

**Files:**
- Create: `research_agent/workspace_after_coding.py`

- [ ] Implement `run_workspace_after_coding(competition, trial_id, run_now=False)`.
- [ ] Read `workspace_coding_result_validation.json`.
- [ ] Block unless status is `accepted`.
- [ ] Delegate execution to `run_workspace_pipeline`.
- [ ] If pipeline completes, call `collect_workspace_metrics`.
- [ ] If metrics are collected, call `process_workspace_result`.
- [ ] Write `workspace_after_coding_cycle.json/md` and log a decision.

### Task 3: CLI

**Files:**
- Modify: `research_agent/cli.py`

- [ ] Add `run-workspace-after-coding --competition --trial [--run-now]`.
- [ ] Return success for dry-run planned and completed/processed states; return nonzero for blocked/failure states.

### Task 4: Documentation

**Files:**
- Create: `docs/workspace_after_coding_cycle.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [ ] Document the post-code 1-cycle re-entry.
- [ ] Update expected test count after verification.

### Task 5: Verification

**Commands:**
- `python -B -m unittest tests.test_workspace_after_coding -v`
- `python -B -m unittest discover -s tests -v`
- `python -B -m compileall -q research_agent tests`
- `git diff --check`
