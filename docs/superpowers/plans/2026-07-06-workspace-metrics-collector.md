# Workspace Metrics Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize a completed external workspace run's JSON metrics artifact into the trial metrics contract without guessing metric fields.

**Architecture:** Add a focused collector module between workspace execution and trial evaluation. Extend Execution Profile validation with one optional mapping field, expose collection through a thin CLI command, and preserve all source data while writing structured collection evidence.

**Tech Stack:** Python standard library, unittest, existing Execution Profile, paths, state store, and decision log modules.

---

### Task 1: Canonical Metrics Collection

**Files:**
- Create: `tests/test_workspace_metrics_collector.py`
- Create: `research_agent/workspace_metrics_collector.py`

- [x] Write a test that supplies a completed workspace run and source JSON containing numeric `cv_score`, then expects a trial `metrics.json` with preserved source fields.
- [x] Run `python -B -m unittest tests.test_workspace_metrics_collector.WorkspaceMetricsCollectorTest.test_collects_existing_cv_score_without_changing_source -v` and confirm the missing module fails.
- [x] Implement completed-run validation, source artifact loading, canonical field construction, result persistence, and decision logging.
- [x] Run the focused test and confirm it passes.

### Task 2: Explicit Metric Mapping

**Files:**
- Modify: `tests/test_workspace_metrics_collector.py`
- Modify: `research_agent/workspace_metrics_collector.py`
- Modify: `tests/test_execution_profile.py`
- Modify: `research_agent/execution_profile.py`

- [x] Add tests for nested `metrics_contract.source_key` resolution and invalid mapping validation.
- [x] Run both focused test modules and confirm the mapping assertions fail.
- [x] Implement dot-path lookup, finite numeric validation, and optional profile-contract validation.
- [x] Run both focused test modules and confirm they pass.

### Task 3: Blocked And Review Outcomes

**Files:**
- Modify: `tests/test_workspace_metrics_collector.py`
- Modify: `research_agent/workspace_metrics_collector.py`

- [x] Add tests for missing mapping, invalid JSON, and non-completed workspace runs.
- [x] Run the focused collector tests and confirm the new status assertions fail.
- [x] Implement deterministic `needs_review` and `blocked` results without writing trial `metrics.json`.
- [x] Run the focused collector tests and confirm they pass.

### Task 4: CLI And Documentation

**Files:**
- Modify: `tests/test_workspace_metrics_collector.py`
- Modify: `research_agent/cli.py`
- Modify: `configs/execution_profile.example.yaml`
- Create: `docs/workspace_metrics_collection.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [x] Add a CLI test for `collect-workspace-metrics --competition demo --trial trial_001`.
- [x] Run the CLI test and confirm argparse rejects the command.
- [x] Add the CLI parser and handler, update the example profile, and document the mapping and phase boundary.
- [x] Run all focused 4차 tests and confirm they pass.

### Task 5: Full Verification

**Files:**
- Verify all changed files without reverting unrelated worktree changes.

- [x] Run `python -B -m compileall -q research_agent tests` and require exit code 0.
- [x] Run `python -B -m unittest discover -s tests -v` and require zero failures.
- [x] Run `git diff --check` and require no whitespace errors.
- [x] Review `git status --short` and keep all prior uncommitted work intact.
