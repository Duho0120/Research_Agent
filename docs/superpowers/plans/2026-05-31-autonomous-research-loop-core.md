# Autonomous Research Loop Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first testable core of the autonomous research loop: trial diagnosis, user review requests, decision logging, submission metadata tracking, best trial marking, and a dry-run CLI path.

**Architecture:** Keep the current file-based architecture and add focused modules with small responsibilities. The first implementation does not call Kaggle APIs or edit model code; it creates the artifacts and decisions those later nodes will depend on.

**Tech Stack:** Python standard library, existing `research_agent` package, existing `simple_yaml`, JSON/JSONL files, Markdown artifacts, PowerShell command verification.

---

## Scope

This plan implements the first concrete slice of the approved spec:

- `diagnose_trial`
- `decide_user_review`
- `collect_user_feedback` storage
- `decision_log.jsonl`
- submission metadata and version files without real Kaggle submission
- best trial marker files
- CLI commands for dry-run verification

This plan does not implement:

- real Kaggle API calls
- LangGraph runtime
- Code Editing Agent
- Kakao or external notification integration
- automated external SOTA search

## File Structure

- Create `research_agent/decision_logger.py`
  - Appends structured decisions to `memory/<competition>/decision_log.jsonl`.

- Create `research_agent/diagnosis_agent.py`
  - Reads metrics, state, evaluation, and recent trial memory.
  - Writes `experiments/<competition>/<trial>/diagnosis.md`.
  - Returns a structured diagnosis with improvement, CV/LB gap, risks, user questions, and escalation recommendation.

- Create `research_agent/user_review_agent.py`
  - Decides whether user review is needed from diagnosis.
  - Writes `user_review_request.md`.
  - Appends user feedback entries to `memory/<competition>/user_feedback.jsonl`.

- Create `research_agent/submission_tracker.py`
  - Creates version metadata.
  - Records simulated/manual submission results in `submissions/<competition>/submission_log.jsonl`.
  - Updates `experiments/<competition>/BEST_TRIAL.md`, `memory/<competition>/best_trial.json`, and optional `BEST_MARKER.md`.

- Modify `research_agent/paths.py`
  - Add helpers for `submissions_dir()` and `competition_submissions_dir()`.

- Modify `research_agent/cli.py`
  - Add `diagnose`, `request-review`, `record-feedback`, and `record-submission` commands.

- Modify `research_agent/main_agent.py`
  - After evaluation/remembering, call diagnosis and decision logging in the conservative cycle when metrics exist.

- Create `tests/test_diagnosis_agent.py`
- Create `tests/test_user_review_agent.py`
- Create `tests/test_submission_tracker.py`
- Create `tests/test_cli_loop_core.py`

If no test runner is installed, run tests with `python -B -m unittest discover -s tests -v`.

---

### Task 1: Path Helpers And Decision Logger

**Files:**
- Modify: `research_agent/paths.py`
- Create: `research_agent/decision_logger.py`
- Test: `tests/test_decision_logger.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_decision_logger.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.decision_logger import log_decision


class DecisionLoggerTest(unittest.TestCase):
    def test_log_decision_appends_competition_scoped_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                row = log_decision(
                    "demo",
                    "trial_010",
                    decision_type="diagnosis",
                    decision="request_user_review",
                    reason="CV improved but LB worsened.",
                    evidence={"cv_score": 0.82, "lb_score": 0.79},
                    user_input_used=False,
                    next_action="write_user_review_request",
                )

                path = root / "memory" / "demo" / "decision_log.jsonl"
                self.assertTrue(path.exists())
                saved = json.loads(path.read_text(encoding="utf-8").strip())
                self.assertEqual(saved["competition"], "demo")
                self.assertEqual(saved["trial_id"], "trial_010")
                self.assertEqual(saved["decision_type"], "diagnosis")
                self.assertEqual(saved["decision"], "request_user_review")
                self.assertEqual(saved["reason"], "CV improved but LB worsened.")
                self.assertEqual(saved["evidence"]["cv_score"], 0.82)
                self.assertFalse(saved["user_input_used"])
                self.assertEqual(saved["next_action"], "write_user_review_request")
                self.assertEqual(row["decision"], saved["decision"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_decision_logger -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research_agent.decision_logger'`.

- [ ] **Step 3: Add submission path helpers**

Modify `research_agent/paths.py` by adding:

```python
def submissions_dir() -> Path:
    return project_root() / "submissions"


def competition_submissions_dir(competition: str) -> Path:
    return submissions_dir() / competition
```

- [ ] **Step 4: Implement the decision logger**

Create `research_agent/decision_logger.py`:

```python
from __future__ import annotations

import json
from typing import Any

from .paths import competition_memory_dir
from .store import now_iso


def log_decision(
    competition: str,
    trial_id: str | None,
    decision_type: str,
    decision: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    user_input_used: bool = False,
    next_action: str = "",
) -> dict[str, Any]:
    row = {
        "time": now_iso(),
        "competition": competition,
        "trial_id": trial_id,
        "decision_type": decision_type,
        "decision": decision,
        "reason": reason,
        "evidence": evidence or {},
        "user_input_used": user_input_used,
        "next_action": next_action,
    }
    out_dir = competition_memory_dir(competition)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decision_log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```powershell
python -B -m unittest tests.test_decision_logger -v
```

Expected: PASS.

- [ ] **Step 6: Check repository status**

Run:

```powershell
git status --short
```

Expected in this workspace today: `fatal: not a git repository...`. If the project has been initialized as a git repository by then, commit:

```powershell
git add research_agent/paths.py research_agent/decision_logger.py tests/test_decision_logger.py
git commit -m "feat: add decision logging"
```

---

### Task 2: Diagnosis Agent

**Files:**
- Create: `research_agent/diagnosis_agent.py`
- Test: `tests/test_diagnosis_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnosis_agent.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.diagnosis_agent import diagnose_trial


class DiagnosisAgentTest(unittest.TestCase):
    def test_diagnosis_writes_markdown_and_flags_user_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.82,
                        "lb_score": 0.78,
                        "objective": "maximize",
                        "prediction_correlation_with_best": 0.997,
                        "segment_errors": {"fold_3": 0.31},
                        "notes": "CV up but LB down.",
                    }
                ),
                encoding="utf-8",
            )
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 2\n  best_trial:\n    trial_id: old\n    cv_score: 0.8\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = diagnose_trial("demo", "trial_001")

            self.assertEqual(result["trial_id"], "trial_001")
            self.assertFalse(result["cv_improved"])
            self.assertTrue(result["needs_user_review"])
            self.assertIn("CV/LB", " ".join(result["issues"]))
            self.assertIn("사용자", " ".join(result["user_questions"]))
            diagnosis_path = trial / "diagnosis.md"
            self.assertTrue(diagnosis_path.exists())
            text = diagnosis_path.read_text(encoding="utf-8")
            self.assertIn("# trial_001 Diagnosis", text)
            self.assertIn("User Review", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_diagnosis_agent -v
```

Expected: FAIL with missing `diagnosis_agent`.

- [ ] **Step 3: Implement diagnosis agent**

Create `research_agent/diagnosis_agent.py`:

```python
from __future__ import annotations

import json
from typing import Any

from .paths import trial_dir
from .store import load_recent_trials, load_state, write_text


def diagnose_trial(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = load_state(competition)
    recent = load_recent_trials(competition)
    objective = metrics.get("objective") or state.get("competition", {}).get("objective", "maximize")
    best = state.get("current_state", {}).get("best_trial")
    best_score = best.get("cv_score") if isinstance(best, dict) else None
    cv_score = metrics.get("cv_score")
    lb_score = metrics.get("lb_score")
    corr = metrics.get("prediction_correlation_with_best")
    consecutive_failures = int(state.get("current_state", {}).get("consecutive_failures", 0))

    cv_improved = _improved(cv_score, best_score, objective) if best_score is not None else True
    issues: list[str] = []
    user_questions: list[str] = []
    improvement_candidates: list[str] = []

    if not cv_improved:
        issues.append("CV did not improve against the current best trial.")
        improvement_candidates.append("Review whether the current method is saturated before another small tweak.")
    if cv_score is not None and lb_score is not None and _direction_conflict(cv_score, lb_score, best_score, objective):
        issues.append("CV/LB movement may be inconsistent.")
        user_questions.append("사용자에게 validation split 또는 제출 전략 변경이 필요한지 의견을 요청합니다.")
    if corr is not None and corr >= 0.995:
        issues.append("Predictions are highly correlated with the current best trial.")
        improvement_candidates.append("Prefer diversity or model-family changes over another similar submission.")
    if metrics.get("leakage_warning"):
        issues.append("Leakage warning is present in metrics.")
        user_questions.append("사용자에게 leakage 가능성이 있는 feature나 data split을 확인받습니다.")
    if metrics.get("segment_errors"):
        issues.append("Errors are concentrated in one or more segments/groups/folds/features/patterns.")
        user_questions.append("사용자에게 집중 오류 구간의 도메인적 의미를 확인받습니다.")
    if consecutive_failures >= 3:
        issues.append("Recent failures suggest strategy escalation is needed.")
        improvement_candidates.append("Prepare model-family, architecture, ensemble, or SOTA exploration candidates.")

    if not improvement_candidates:
        improvement_candidates.append("Continue controlled refinement while keeping validation stable.")

    needs_user_review = bool(user_questions) or consecutive_failures >= 3
    escalation = "strategy_escalation" if consecutive_failures >= 3 else "continue_refinement"

    result = {
        "competition": competition,
        "trial_id": trial_id,
        "objective": objective,
        "cv_score": cv_score,
        "lb_score": lb_score,
        "best_cv_before": best_score,
        "cv_improved": cv_improved,
        "issues": issues,
        "improvement_candidates": improvement_candidates,
        "user_questions": user_questions,
        "needs_user_review": needs_user_review,
        "strategy_recommendation": escalation,
        "recent_trial_count": len(recent),
    }
    write_text(out_dir / "diagnosis.md", render_diagnosis(result))
    return result


def _improved(score: float | None, best: float | None, objective: str) -> bool:
    if score is None or best is None:
        return False
    return score < best if objective == "minimize" else score > best


def _direction_conflict(score: float, lb_score: float, best_score: float | None, objective: str) -> bool:
    if best_score is None:
        return False
    if objective == "minimize":
        return score < best_score and lb_score >= best_score
    return score > best_score and lb_score <= best_score


def render_diagnosis(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Diagnosis",
        "",
        "## Scores",
        "",
        f"- objective: {result['objective']}",
        f"- cv_score: {result['cv_score']}",
        f"- lb_score: {result['lb_score']}",
        f"- best_cv_before: {result['best_cv_before']}",
        f"- cv_improved: {result['cv_improved']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["No major issues detected."])
    lines.extend(["", "## Improvement Candidates", ""])
    lines.extend(f"- {item}" for item in result["improvement_candidates"])
    lines.extend(["", "## User Review", "", f"- needs_user_review: {result['needs_user_review']}"])
    lines.extend(f"- {item}" for item in result["user_questions"] or ["No user question required."])
    lines.extend(["", "## Strategy Recommendation", "", result["strategy_recommendation"], ""])
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -B -m unittest tests.test_diagnosis_agent -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add research_agent/diagnosis_agent.py tests/test_diagnosis_agent.py
git commit -m "feat: add trial diagnosis"
```

---

### Task 3: User Review Requests And Feedback Memory

**Files:**
- Create: `research_agent/user_review_agent.py`
- Test: `tests/test_user_review_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_user_review_agent.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.user_review_agent import record_user_feedback, request_user_review


class UserReviewAgentTest(unittest.TestCase):
    def test_request_user_review_writes_request_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_001").mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_001",
                "issues": ["CV/LB movement may be inconsistent."],
                "user_questions": ["사용자에게 validation split 의견을 요청합니다."],
                "improvement_candidates": ["Try a model-family change."],
            }

            with patch("research_agent.paths.project_root", return_value=root):
                path = request_user_review("demo", "trial_001", diagnosis)

            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("# trial_001 User Review Request", text)
            self.assertIn("CV/LB", text)

    def test_record_user_feedback_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                row = record_user_feedback(
                    "demo",
                    "trial_001",
                    topic="validation",
                    question="Is the split appropriate?",
                    user_feedback="Group split looks safer.",
                    decision="change_validation",
                    follow_up_action="plan validation review trial",
                )

            path = root / "memory" / "demo" / "user_feedback.jsonl"
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["topic"], "validation")
            self.assertEqual(saved["decision"], "change_validation")
            self.assertEqual(row["follow_up_action"], "plan validation review trial")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_user_review_agent -v
```

Expected: FAIL with missing `user_review_agent`.

- [ ] **Step 3: Implement user review agent**

Create `research_agent/user_review_agent.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import competition_memory_dir, trial_dir
from .store import now_iso, write_text


def request_user_review(competition: str, trial_id: str, diagnosis: dict[str, Any]) -> Path:
    out_dir = trial_dir(competition, trial_id)
    path = out_dir / "user_review_request.md"
    write_text(path, render_user_review_request(diagnosis))
    return path


def render_user_review_request(diagnosis: dict[str, Any]) -> str:
    trial_id = diagnosis["trial_id"]
    lines = [
        f"# {trial_id} User Review Request",
        "",
        "## Why Review Is Needed",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis.get("issues", []) or ["No specific issue listed."])
    lines.extend(["", "## Questions", ""])
    lines.extend(f"- {item}" for item in diagnosis.get("user_questions", []) or ["No explicit question listed."])
    lines.extend(["", "## Candidate Next Actions", ""])
    lines.extend(f"- {item}" for item in diagnosis.get("improvement_candidates", []) or ["Continue controlled refinement."])
    lines.extend(["", "## User Response", "", "Write the decision, cautions, and ideas here before recording feedback.", ""])
    return "\n".join(lines)


def record_user_feedback(
    competition: str,
    trial_id: str,
    topic: str,
    question: str,
    user_feedback: str,
    decision: str,
    follow_up_action: str,
) -> dict[str, Any]:
    row = {
        "time": now_iso(),
        "competition": competition,
        "trial_id": trial_id,
        "topic": topic,
        "question": question,
        "user_feedback": user_feedback,
        "decision": decision,
        "follow_up_action": follow_up_action,
    }
    out_dir = competition_memory_dir(competition)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "user_feedback.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    response_path = trial_dir(competition, trial_id) / "user_review_response.md"
    write_text(response_path, render_user_feedback(row))
    return row


def render_user_feedback(row: dict[str, Any]) -> str:
    return f"""# {row['trial_id']} User Review Response

## Topic

{row['topic']}

## Question

{row['question']}

## Feedback

{row['user_feedback']}

## Decision

{row['decision']}

## Follow-up Action

{row['follow_up_action']}
"""
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -B -m unittest tests.test_user_review_agent -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add research_agent/user_review_agent.py tests/test_user_review_agent.py
git commit -m "feat: add user review memory"
```

---

### Task 4: Submission Tracker And Best Trial Markers

**Files:**
- Create: `research_agent/submission_tracker.py`
- Test: `tests/test_submission_tracker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_submission_tracker.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.submission_tracker import record_submission_result


class SubmissionTrackerTest(unittest.TestCase):
    def test_record_submission_result_logs_and_marks_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0.9\n", encoding="utf-8")

            with patch("research_agent.paths.project_root", return_value=root):
                row = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    cv_score=0.82,
                    previous_lb_score=0.80,
                    previous_rank=120,
                    submitted_lb_score=0.84,
                    submitted_rank=90,
                    objective="maximize",
                    notes="Manual dry-run result.",
                )

            self.assertTrue(row["is_best"])
            log_path = root / "submissions" / "demo" / "submission_log.jsonl"
            self.assertTrue(log_path.exists())
            saved = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["score_delta"], 0.04)
            self.assertEqual(saved["rank_delta"], 30)
            self.assertTrue((root / "experiments" / "demo" / "BEST_TRIAL.md").exists())
            self.assertTrue((root / "memory" / "demo" / "best_trial.json").exists())
            self.assertTrue((trial / "BEST_MARKER.md").exists())
            self.assertTrue((trial / "VERSION.md").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_submission_tracker -v
```

Expected: FAIL with missing `submission_tracker`.

- [ ] **Step 3: Implement submission tracker**

Create `research_agent/submission_tracker.py`:

```python
from __future__ import annotations

import json
from typing import Any

from .paths import competition_memory_dir, competition_submissions_dir, experiments_dir, trial_dir
from .store import now_iso, write_text


def record_submission_result(
    competition: str,
    trial_id: str,
    version_name: str,
    submission_file: str,
    cv_score: float | None,
    previous_lb_score: float | None,
    previous_rank: int | None,
    submitted_lb_score: float | None,
    submitted_rank: int | None,
    objective: str = "maximize",
    notes: str = "",
) -> dict[str, Any]:
    score_delta = _score_delta(previous_lb_score, submitted_lb_score)
    rank_delta = _rank_delta(previous_rank, submitted_rank)
    is_best = _is_best_submission(previous_lb_score, submitted_lb_score, objective)
    row = {
        "submission_id": f"{competition}_{trial_id}_{version_name}",
        "competition": competition,
        "trial_id": trial_id,
        "version_name": version_name,
        "submitted_at": now_iso(),
        "submission_file": submission_file,
        "cv_score": cv_score,
        "previous_lb_score": previous_lb_score,
        "previous_rank": previous_rank,
        "submitted_lb_score": submitted_lb_score,
        "submitted_rank": submitted_rank,
        "score_delta": score_delta,
        "rank_delta": rank_delta,
        "is_best": is_best,
        "notes": notes,
    }
    _append_submission_log(competition, row)
    _write_trial_submission_files(competition, trial_id, row)
    if is_best:
        _write_best_files(competition, trial_id, row)
    return row


def _append_submission_log(competition: str, row: dict[str, Any]) -> None:
    out_dir = competition_submissions_dir(competition)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission_log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_trial_submission_files(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "submission_result.md", render_submission_result(row))
    write_text(out_dir / "VERSION.md", render_version(row))


def _write_best_files(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    best_md = experiments_dir() / competition / "BEST_TRIAL.md"
    write_text(best_md, render_best_trial(row))
    memory = competition_memory_dir(competition)
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "best_trial.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(trial_dir(competition, trial_id) / "BEST_MARKER.md", render_best_trial(row))


def _score_delta(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return round(current - previous, 10)


def _rank_delta(previous: int | None, current: int | None) -> int | None:
    if previous is None or current is None:
        return None
    return previous - current


def _is_best_submission(previous: float | None, current: float | None, objective: str) -> bool:
    if current is None:
        return False
    if previous is None:
        return True
    return current < previous if objective == "minimize" else current > previous


def render_submission_result(row: dict[str, Any]) -> str:
    return f"""# Submission Result: {row['version_name']}

- trial_id: {row['trial_id']}
- cv_score: {row['cv_score']}
- previous_lb_score: {row['previous_lb_score']}
- submitted_lb_score: {row['submitted_lb_score']}
- score_delta: {row['score_delta']}
- previous_rank: {row['previous_rank']}
- submitted_rank: {row['submitted_rank']}
- rank_delta: {row['rank_delta']}
- is_best: {row['is_best']}

## Notes

{row['notes']}
"""


def render_version(row: dict[str, Any]) -> str:
    return f"""# Version

- version_name: {row['version_name']}
- submission_id: {row['submission_id']}
- submission_file: {row['submission_file']}
- submitted_at: {row['submitted_at']}
"""


def render_best_trial(row: dict[str, Any]) -> str:
    return f"""# Best Trial

- competition: {row['competition']}
- trial_id: {row['trial_id']}
- version_name: {row['version_name']}
- cv_score: {row['cv_score']}
- lb_score: {row['submitted_lb_score']}
- rank: {row['submitted_rank']}
- submission_file: {row['submission_file']}
- submitted_at: {row['submitted_at']}
"""
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -B -m unittest tests.test_submission_tracker -v
```

Expected: PASS.

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add research_agent/submission_tracker.py tests/test_submission_tracker.py
git commit -m "feat: track submissions and best trial"
```

---

### Task 5: CLI Commands

**Files:**
- Modify: `research_agent/cli.py`
- Test: `tests/test_cli_loop_core.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli_loop_core.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.cli import main


class CliLoopCoreTest(unittest.TestCase):
    def test_diagnose_command_creates_diagnosis_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.7, "objective": "maximize"}), encoding="utf-8")
            (root / "competitions" / "demo").mkdir(parents=True)
            (root / "competitions" / "demo" / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            with patch("research_agent.paths.project_root", return_value=root):
                code = main(["diagnose", "--competition", "demo", "--trial", "trial_001"])
            self.assertEqual(code, 0)
            self.assertTrue((trial / "diagnosis.md").exists())

    def test_record_submission_command_creates_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_001").mkdir(parents=True)
            with patch("research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "record-submission",
                        "--competition",
                        "demo",
                        "--trial",
                        "trial_001",
                        "--version-name",
                        "demo_trial_001_baseline_v01",
                        "--submission-file",
                        "experiments/demo/trial_001/submission.csv",
                        "--cv-score",
                        "0.7",
                        "--previous-lb-score",
                        "0.6",
                        "--previous-rank",
                        "200",
                        "--submitted-lb-score",
                        "0.72",
                        "--submitted-rank",
                        "150",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_cli_loop_core -v
```

Expected: FAIL because CLI commands do not exist.

- [ ] **Step 3: Add imports**

Modify top of `research_agent/cli.py`:

```python
from .diagnosis_agent import diagnose_trial
from .submission_tracker import record_submission_result
from .user_review_agent import record_user_feedback, request_user_review
```

- [ ] **Step 4: Add parser commands**

Inside `main()`, after existing parser setup:

```python
    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--competition", required=True)
    p_diag.add_argument("--trial", required=True)

    p_review = sub.add_parser("request-review")
    p_review.add_argument("--competition", required=True)
    p_review.add_argument("--trial", required=True)

    p_feedback = sub.add_parser("record-feedback")
    p_feedback.add_argument("--competition", required=True)
    p_feedback.add_argument("--trial", required=True)
    p_feedback.add_argument("--topic", required=True)
    p_feedback.add_argument("--question", required=True)
    p_feedback.add_argument("--feedback", required=True)
    p_feedback.add_argument("--decision", required=True)
    p_feedback.add_argument("--follow-up-action", required=True)

    p_submit = sub.add_parser("record-submission")
    p_submit.add_argument("--competition", required=True)
    p_submit.add_argument("--trial", required=True)
    p_submit.add_argument("--version-name", required=True)
    p_submit.add_argument("--submission-file", required=True)
    p_submit.add_argument("--cv-score", type=float, default=None)
    p_submit.add_argument("--previous-lb-score", type=float, default=None)
    p_submit.add_argument("--previous-rank", type=int, default=None)
    p_submit.add_argument("--submitted-lb-score", type=float, default=None)
    p_submit.add_argument("--submitted-rank", type=int, default=None)
    p_submit.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    p_submit.add_argument("--notes", default="")
```

- [ ] **Step 5: Add command handlers**

Inside `main()`, before final `return 1`:

```python
    if args.command == "diagnose":
        diagnosis = diagnose_trial(args.competition, args.trial)
        print(f"Diagnosis: needs_user_review={diagnosis['needs_user_review']}")
        return 0

    if args.command == "request-review":
        diagnosis = diagnose_trial(args.competition, args.trial)
        path = request_user_review(args.competition, args.trial, diagnosis)
        print(f"Review request: {path.as_posix()}")
        return 0

    if args.command == "record-feedback":
        row = record_user_feedback(
            args.competition,
            args.trial,
            topic=args.topic,
            question=args.question,
            user_feedback=args.feedback,
            decision=args.decision,
            follow_up_action=args.follow_up_action,
        )
        print(f"Recorded feedback: {row['decision']}")
        return 0

    if args.command == "record-submission":
        row = record_submission_result(
            competition=args.competition,
            trial_id=args.trial,
            version_name=args.version_name,
            submission_file=args.submission_file,
            cv_score=args.cv_score,
            previous_lb_score=args.previous_lb_score,
            previous_rank=args.previous_rank,
            submitted_lb_score=args.submitted_lb_score,
            submitted_rank=args.submitted_rank,
            objective=args.objective,
            notes=args.notes,
        )
        print(f"Recorded submission: {row['version_name']} best={row['is_best']}")
        return 0
```

- [ ] **Step 6: Run CLI tests**

Run:

```powershell
python -B -m unittest tests.test_cli_loop_core -v
```

Expected: PASS.

- [ ] **Step 7: Run full loop-core tests**

Run:

```powershell
python -B -m unittest discover -s tests -v
```

Expected: PASS for all new tests.

- [ ] **Step 8: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add research_agent/cli.py tests/test_cli_loop_core.py
git commit -m "feat: expose research loop core CLI"
```

---

### Task 6: Integrate Diagnosis Into Conservative Cycle

**Files:**
- Modify: `research_agent/main_agent.py`
- Test: `tests/test_main_agent_diagnosis.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_main_agent_diagnosis.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.main_agent import run_cycle


class MainAgentDiagnosisTest(unittest.TestCase):
    def test_cycle_with_metrics_writes_diagnosis_and_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n  best_trial:\n    trial_id: old\n    cv_score: 0.7\nstrategy:\n  current_focus: baseline\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                "model:\n  type:\n    - lightgbm\nfeatures:\n  use_missing_indicators:\n    - True\ncv:\n  n_splits:\n    - 5\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                "model:\n  type: lightgbm\nfeatures:\n  use_missing_indicators: True\ncv:\n  n_splits: 5\n",
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.72, "objective": "maximize"}), encoding="utf-8")

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_cycle("demo", "trial_001", create_job_request=False)

            self.assertIn("diagnosed", result["steps"])
            self.assertTrue((trial / "diagnosis.md").exists())
            self.assertTrue((root / "memory" / "demo" / "decision_log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -B -m unittest tests.test_main_agent_diagnosis -v
```

Expected: FAIL because `run_cycle` does not call diagnosis/logging yet.

- [ ] **Step 3: Add imports to main_agent.py**

Modify `research_agent/main_agent.py`:

```python
from .decision_logger import log_decision
from .diagnosis_agent import diagnose_trial
```

- [ ] **Step 4: Add diagnosis after evaluation/memory**

In the `metrics_path.exists()` branch, after `remember_trial(...)`:

```python
        diagnosis = diagnose_trial(competition, trial_id)
        log_decision(
            competition,
            trial_id,
            decision_type="diagnosis",
            decision="request_user_review" if diagnosis["needs_user_review"] else "continue",
            reason="Diagnosis completed after evaluation.",
            evidence={
                "cv_improved": diagnosis["cv_improved"],
                "issues": diagnosis["issues"],
                "strategy_recommendation": diagnosis["strategy_recommendation"],
            },
            user_input_used=False,
            next_action="request-review" if diagnosis["needs_user_review"] else "plan-next-trial",
        )
        result["steps"].append("diagnosed")
        result["diagnosis"] = diagnosis
```

Make the same addition in the `run_now` branch after evaluation and memory update.

- [ ] **Step 5: Run integration test**

Run:

```powershell
python -B -m unittest tests.test_main_agent_diagnosis -v
```

Expected: PASS.

- [ ] **Step 6: Run all tests**

Run:

```powershell
python -B -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 7: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add research_agent/main_agent.py tests/test_main_agent_diagnosis.py
git commit -m "feat: diagnose completed cycles"
```

---

### Task 7: Documentation And Manual Dry-Run

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_CHANGELOG.ko.md`

- [ ] **Step 1: Add README commands**

Add a section to `README.md` after evaluation/remember usage:

```markdown
## Research loop core dry-run

Diagnose a completed trial:

```powershell
python -B -m research_agent.cli diagnose --competition demo --trial trial_001
```

Create a user review request when the diagnosis needs human input:

```powershell
python -B -m research_agent.cli request-review --competition demo --trial trial_001
```

Record user feedback:

```powershell
python -B -m research_agent.cli record-feedback --competition demo --trial trial_001 --topic validation --question "Is this split appropriate?" --feedback "Use group split before large model changes." --decision change_validation --follow-up-action "Plan a validation review trial"
```

Record a manual submission result without calling Kaggle:

```powershell
python -B -m research_agent.cli record-submission --competition demo --trial trial_001 --version-name demo_trial_001_baseline_v01 --submission-file experiments/demo/trial_001/submission.csv --cv-score 0.83 --previous-lb-score 0.80 --previous-rank 120 --submitted-lb-score 0.84 --submitted-rank 90 --objective maximize --notes "Manual leaderboard entry"
```
```

- [ ] **Step 2: Update changelog**

Append to `PROJECT_CHANGELOG.ko.md`:

```markdown
## 2026-05-31 KST

### 자율 연구 루프 core 구현 계획 수립

요약:

- `diagnose_trial`, `User Review`, decision log, submission tracking, best trial 표시를 첫 구현 단위로 확정했다.
- 실제 Kaggle API와 Code Editing Agent는 다음 구현 단계로 분리했다.

주요 계획 파일:

- `docs/superpowers/specs/2026-05-30-autonomous-research-loop-with-user-review-design.md`
- `docs/superpowers/plans/2026-05-31-autonomous-research-loop-core.md`
```

- [ ] **Step 3: Run regression commands**

Run:

```powershell
python -B -m research_agent.cli validate-config --competition demo --trial trial_001
python -B -m research_agent.cli evaluate --competition demo --trial trial_001
python -B -m research_agent.cli remember --competition demo --trial trial_001
python -B -m research_agent.cli diagnose --competition demo --trial trial_001
```

Expected:

- `validate-config`: `Config is valid`
- `evaluate`: prints a recommendation
- `remember`: prints remembered trial info
- `diagnose`: prints `Diagnosis: needs_user_review=...`

- [ ] **Step 4: Verify generated artifacts**

Run:

```powershell
Test-Path experiments\demo\trial_001\diagnosis.md
Test-Path memory\demo\decision_log.jsonl
```

Expected:

```text
True
True
```

- [ ] **Step 5: Commit if git is available**

Run:

```powershell
git status --short
```

Expected in current workspace: not a git repository. If git is initialized, commit:

```powershell
git add README.md PROJECT_CHANGELOG.ko.md
git commit -m "docs: document research loop core"
```

---

## Plan Self-Review

Spec coverage:

- `diagnose_trial`: Task 2 and Task 6.
- User Review/User Input Loop: Task 3 and Task 5.
- `decision_log.jsonl`: Task 1 and Task 6.
- Submission tracking: Task 4 and Task 5.
- Best trial marking: Task 4.
- Version name: Task 4 and Task 5.
- Strategy escalation: Task 2 diagnosis fields and Task 6 decision logging; deeper SOTA search remains a later plan as specified.
- Current architecture compatibility: Tasks use existing file-based structure and CLI.
- Real Kaggle API: excluded from this first plan, as approved in the spec.
- Code Editing Agent: excluded from this first plan, as approved in the spec.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps are intentionally left.
- Commit steps account for the current non-git workspace and provide exact commit commands if git is initialized.

Type consistency:

- `diagnose_trial(competition: str, trial_id: str) -> dict[str, Any]`
- `request_user_review(competition: str, trial_id: str, diagnosis: dict[str, Any]) -> Path`
- `record_user_feedback(...) -> dict[str, Any]`
- `log_decision(...) -> dict[str, Any]`
- `record_submission_result(...) -> dict[str, Any]`

