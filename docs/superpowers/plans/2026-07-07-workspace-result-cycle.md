# Workspace Result Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect collected workspace metrics to evaluation, diagnosis, maturity-aware Human Review, and idempotent memory updates.

**Architecture:** Extend the central Human Review policy gate with optional pipeline-readiness evidence while preserving legacy callers. Add one workspace result-cycle service that owns preconditions, deferred-review queue handling, existing analysis calls, memory idempotency, and structured outputs.

**Tech Stack:** Python standard library, unittest, existing policy gate, result analyst, review pack, memory, paths, and store modules.

---

### Task 1: Maturity-Aware Review Policy

**Files:**
- Modify: `tests/test_policy_gate.py`
- Modify: `research_agent/agents/policy_gate.py`
- Modify: `research_agent/policies.py`
- Modify: `configs/policies/human_review_policy.yaml`

- [x] Add failing tests proving a nonurgent review defers before maturity, releases after two completed trials, and leakage requests immediately.
- [x] Run the focused policy tests and confirm the new timing assertions fail.
- [x] Add optional pipeline-readiness input, trigger urgency classification, policy defaults, and `request_now/defer/no_review` output.
- [x] Run the focused policy tests and confirm they pass without changing legacy behavior.

### Task 2: Result Cycle And Deferred Queue

**Files:**
- Create: `tests/test_workspace_result_cycle.py`
- Create: `research_agent/workspace_result_cycle.py`

- [x] Add a failing test that processes a first successful trial with a nonurgent diagnosis, remembers it, and queues review instead of creating a pack.
- [x] Run the focused test and confirm the missing module fails.
- [x] Implement collection precondition, evaluation, diagnosis, readiness calculation, review timing, deferred queue, memory update, and result persistence.
- [x] Run the focused test and confirm it passes.

### Task 3: Review Release And Safety

**Files:**
- Modify: `tests/test_workspace_result_cycle.py`
- Modify: `research_agent/workspace_result_cycle.py`

- [x] Add failing tests for second-trial queue release, immediate leakage review, no-review completion, blocked collection, and duplicate-memory protection.
- [x] Run the focused tests and confirm the new assertions fail.
- [x] Implement queue merge/release, urgent review pack generation, terminal statuses, and idempotency guard.
- [x] Run the focused tests and confirm they pass.

### Task 4: CLI And Documentation

**Files:**
- Modify: `tests/test_workspace_result_cycle.py`
- Modify: `research_agent/cli.py`
- Create: `docs/workspace_result_cycle.ko.md`
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [x] Add a failing CLI test for `process-workspace-result --competition demo --trial trial_001`.
- [x] Run the CLI test and confirm argparse rejects the command.
- [x] Add the CLI parser/handler and document maturity, urgent exceptions, queue behavior, and scope boundary.
- [x] Run all focused 5차 tests and confirm they pass.

### Task 5: Full Verification

**Files:**
- Verify all changes without reverting prior uncommitted work.

- [x] Run `python -B -m compileall -q research_agent tests` and require exit code 0.
- [x] Run `python -B -m unittest discover -s tests -v` and require zero failures.
- [x] Run `git diff --check` and require no whitespace errors.
- [x] Review `git status --short` and preserve all earlier phase changes.
