# Workspace Next Experiment Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workspace-specific gate that turns a processed trial result into the next experiment plan only when it is safe to continue.

**Architecture:** The new gate reads `workspace_result_cycle.json`, registers a human review request when a review pack exists, and classifies continuation as `can_continue`, `continue_with_caution`, or `must_wait`. It calls the existing `propose_next_experiment` only for non-blocking states and writes explicit continuation metadata beside the source and next trial artifacts.

**Tech Stack:** Python standard library, existing CLI, existing memory/research planner modules, `unittest`.

---

### Task 1: Next Gate Tests

**Files:**
- Create: `tests/test_workspace_next_gate.py`

- [ ] Write tests for: nonurgent pending review continues with caution; urgent review blocks next planning; completed result plans normally; invalid/missing result blocks.
- [ ] Run the focused test file and confirm it fails because `research_agent.workspace_next_gate` does not exist.

### Task 2: Gate Module

**Files:**
- Create: `research_agent/workspace_next_gate.py`

- [ ] Implement `plan_next_workspace_trial(competition, source_trial_id, next_trial_id)`.
- [ ] Read and validate source `workspace_result_cycle.json`.
- [ ] Call `request_user_review` when `status == awaiting_human_review` and a diagnosis is available.
- [ ] Block only urgent/blocking review states.
- [ ] For nonurgent pending review, call `propose_next_experiment` and write continuation metadata.
- [ ] Log a `workspace_next_gate` decision.

### Task 3: CLI Command

**Files:**
- Modify: `research_agent/cli.py`
- Test: `tests/test_workspace_next_gate.py`

- [ ] Add `plan-next-workspace-trial --competition --source-trial --next-trial`.
- [ ] Return nonzero only when the gate status starts with `blocked`.

### Task 4: Documentation

**Files:**
- Create: `docs/workspace_next_experiment_gate.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [ ] Document continuation modes and blocking rules.
- [ ] Update README workflow and test count.

### Task 5: Verification

**Commands:**
- `python -B -m unittest tests.test_workspace_next_gate -v`
- `python -B -m unittest discover -s tests -v`
- `python -B -m compileall -q research_agent tests`
- `git diff --check`
