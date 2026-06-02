from __future__ import annotations


import json
from typing import Any

from ..paths import competition_memory_dir, memory_dir, trial_dir
from ..store import append_trial_index, load_state, now_iso, read_text, save_state, write_text


def remember_trial(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    metrics_path = out_dir / "metrics.json"
    evaluation_path = out_dir / "evaluation.md"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation = read_text(evaluation_path)
    state = load_state(competition)
    objective = state.get("competition", {}).get("objective", metrics.get("objective", "maximize"))

    current_best = state.get("current_state", {}).get("best_trial")
    is_best = _is_best(metrics.get("cv_score"), current_best, objective)
    recommendation = _extract_recommendation(evaluation)

    row = {
        "time": now_iso(),
        "competition": competition,
        "trial_id": trial_id,
        "cv_score": metrics.get("cv_score"),
        "lb_score": metrics.get("lb_score"),
        "objective": objective,
        "recommendation": recommendation,
        "is_best": is_best,
        "notes": metrics.get("notes", ""),
    }
    append_trial_index(row)

    if is_best:
        state["current_state"]["best_trial"] = {
            "trial_id": trial_id,
            "cv_score": metrics.get("cv_score"),
            "lb_score": metrics.get("lb_score"),
        }
        state["current_state"]["consecutive_failures"] = 0
    else:
        state["current_state"]["consecutive_failures"] = int(state["current_state"].get("consecutive_failures", 0)) + 1
    state["current_state"]["active_trial"] = trial_id
    save_state(competition, state)

    _append_research_notes(competition, trial_id, row)
    return row


def _is_best(cv_score: float | None, current_best: Any, objective: str) -> bool:
    if cv_score is None:
        return False
    if not isinstance(current_best, dict) or current_best.get("cv_score") is None:
        return True
    best = current_best["cv_score"]
    return cv_score < best if objective == "minimize" else cv_score > best


def _extract_recommendation(evaluation: str) -> str:
    for line in evaluation.splitlines():
        stripped = line.strip()
        if stripped in {"accept_as_candidate", "record_but_do_not_promote", "investigate_leakage", "consider_only_if_score_gain_is_large"}:
            return stripped
    return "unknown"


def _append_research_notes(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    path = competition_memory_dir(competition) / "research_notes.md"
    previous = read_text(path) if path.exists() else read_text(memory_dir() / "research_notes.md")
    addition = f"""

## {competition} / {trial_id}

- CV: {row['cv_score']}
- LB: {row['lb_score']}
- Recommendation: {row['recommendation']}
- Best so far: {row['is_best']}
- Notes: {row['notes']}
"""
    write_text(path, previous.rstrip() + addition + "\n")


import json
from typing import Any

from ..paths import competition_memory_dir
from ..store import now_iso


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
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


import json
from pathlib import Path
from typing import Any

from ..paths import competition_memory_dir, trial_dir
from ..store import now_iso, write_text


def request_user_review(competition: str, trial_id: str, diagnosis: dict[str, Any]) -> Path:
    path = trial_dir(competition, trial_id) / "user_review_request.md"
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
    lines.extend(
        f"- {item}" for item in diagnosis.get("improvement_candidates", []) or ["No candidate action listed."]
    )
    lines.extend(["", "## User Response", "", "- feedback:", "- decision:", "- follow_up_action:", ""])
    return "\n".join(lines)


def record_user_feedback(
    competition: str,
    trial_id: str,
    *,
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
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_text(trial_dir(competition, trial_id) / "user_review_response.md", render_user_feedback(row))
    _write_review_pack_feedback(competition, trial_id, row)
    log_decision(
        competition,
        trial_id,
        decision_type="human_feedback",
        decision=decision,
        reason="User feedback was recorded and should inform the next research action.",
        evidence={
            "topic": topic,
            "question": question,
            "follow_up_action": follow_up_action,
        },
        user_input_used=True,
        next_action=follow_up_action,
    )
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


def _write_review_pack_feedback(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    pack_dir = trial_dir(competition, trial_id) / "review_pack"
    if not pack_dir.exists():
        return
    feedback = {
        "competition": competition,
        "trial_id": trial_id,
        "reviewed_at": row["time"],
        "overall_decision": row["decision"],
        "answers": [
            {
                "question_id": "Q1",
                "decision": row["decision"],
                "question": row["question"],
                "notes": row["user_feedback"],
            }
        ],
        "follow_up_action": row["follow_up_action"],
        "approved_actions": [row["decision"]],
        "rejected_actions": [],
    }
    write_text(pack_dir / "human_feedback.md", render_review_pack_feedback(row))
    write_text(pack_dir / "human_feedback.json", json.dumps(feedback, ensure_ascii=False, indent=2) + "\n")
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "feedback_recorded"
        manifest["feedback_file"] = "human_feedback.json"
        write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def render_review_pack_feedback(row: dict[str, Any]) -> str:
    return f"""# Human Feedback

## Overall Decision

{row['decision']}

## Answers

### Q1

{row['user_feedback']}

## Follow-up Action

{row['follow_up_action']}
"""


