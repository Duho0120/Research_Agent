# Workspace Coding Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a workspace next-experiment plan into a scoped coding-agent request without editing the external project.

**Architecture:** Add a workspace-specific handoff module that reads `next_experiment.md`, `continuation_context.json`, and the competition Execution Profile. It blocks `must_wait` states, uses `write_scope.allowed` as allowed external relative paths, uses `write_scope.forbidden` plus artifacts as forbidden paths, and writes a versioned request contract under the next trial directory.

**Tech Stack:** Python standard library, existing Execution Profile loader/validator, existing CLI, `unittest`.

---

### Task 1: Failing Tests

**Files:**
- Create: `tests/test_workspace_coding_handoff.py`

- [ ] Test ready handoff from `continue_with_caution` context.
- [ ] Test blocking when continuation mode is `must_wait`.
- [ ] Test blocking when Execution Profile validation is not ready.
- [ ] Test CLI command creates the handoff file.

### Task 2: Workspace Handoff Module

**Files:**
- Create: `kaggle_research_agent/workspace_coding_handoff.py`

- [ ] Implement `prepare_workspace_coding_handoff(competition, trial_id)`.
- [ ] Load and validate Execution Profile.
- [ ] Require `next_experiment.md` and `continuation_context.json`.
- [ ] Build `workspace_coding_handoff.json` and `workspace_coding_agent_request.md`.
- [ ] Log `workspace_coding_handoff` decisions.

### Task 3: CLI

**Files:**
- Modify: `kaggle_research_agent/cli.py`

- [ ] Add `prepare-workspace-handoff --competition --trial`.
- [ ] Return nonzero for blocked status.

### Task 4: Documentation

**Files:**
- Create: `docs/workspace_coding_handoff.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [ ] Document the 1-cycle position and output contract.
- [ ] Update expected test count after verification.

### Task 5: Verification

**Commands:**
- `python -B -m unittest tests.test_workspace_coding_handoff -v`
- `python -B -m unittest discover -s tests -v`
- `python -B -m compileall -q kaggle_research_agent tests`
- `git diff --check`
