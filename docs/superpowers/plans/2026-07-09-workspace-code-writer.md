# Workspace Code Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a workspace coding handoff through a mock/API code writer, apply only allowed external project file updates, and validate the result before any training run.

**Architecture:** Add a workspace-specific code writer module that reads `workspace_coding_handoff.json`, builds a code-writer request, parses JSON output, applies `file_updates` under `project_root`, and validates changed files against Execution Profile-derived scope. It mirrors the existing internal code-writer flow but uses project-root-relative paths and writes workspace-specific artifacts.

**Tech Stack:** Python standard library, existing token policy gate, existing OpenAI Responses/FileResponse client helpers, existing CLI, `unittest`.

---

### Task 1: Failing Tests

**Files:**
- Create: `tests/test_workspace_code_writer.py`

- [ ] Test allowed external file update is applied and validation is accepted.
- [ ] Test forbidden artifact update is blocked before write.
- [ ] Test token/API gate blocks when no mock client and no explicit API approval.
- [ ] Test CLI accepts a mock response file.

### Task 2: Workspace Code Writer Module

**Files:**
- Create: `kaggle_research_agent/workspace_code_writer.py`

- [ ] Implement `run_workspace_code_writer`.
- [ ] Implement `validate_workspace_coding_result`.
- [ ] Build request payload from workspace handoff and local context files.
- [ ] Apply only safe project-root-relative `file_updates`.
- [ ] Write `workspace_coding_result.*` and `workspace_coding_result_validation.*`.
- [ ] Log decisions and token usage.

### Task 3: CLI

**Files:**
- Modify: `kaggle_research_agent/cli.py`

- [ ] Add `run-workspace-code-writer`.
- [ ] Add `validate-workspace-coding-result`.
- [ ] Return nonzero unless validation is accepted.

### Task 4: Documentation

**Files:**
- Create: `docs/workspace_code_writer.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [ ] Document path semantics and safety gates.
- [ ] Update expected test count after verification.

### Task 5: Verification

**Commands:**
- `python -B -m unittest tests.test_workspace_code_writer -v`
- `python -B -m unittest discover -s tests -v`
- `python -B -m compileall -q kaggle_research_agent tests`
- `git diff --check`
