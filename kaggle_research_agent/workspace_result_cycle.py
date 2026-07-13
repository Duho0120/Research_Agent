from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.memory import log_decision, remember_trial
from .agents.policy_gate import decide_human_review
from .agents.result_analyst import diagnose_trial, evaluate_trial
from .agents.review_pack import prepare_review_pack
from .paths import competition_memory_dir, experiment_dir, trial_dir
from .store import load_trial_index, write_text
from .trial_artifacts import read_trial_json


def process_workspace_result(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    collection = read_trial_json(out_dir, "metrics_collection.json")
    if not collection:
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "steps": ["metrics_collection_invalid"],
                "issues": ["invalid_or_missing_metrics_collection"],
                "next_action": "collect-workspace-metrics",
            },
        )
    if collection.get("status") != "collected":
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "steps": ["metrics_collection_not_ready"],
                "issues": ["metrics_collection_not_collected"],
                "next_action": "collect-workspace-metrics",
            },
        )
    if not (out_dir / "metrics.json").is_file():
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "steps": ["trial_metrics_missing"],
                "issues": ["missing_trial_metrics"],
                "next_action": "collect-workspace-metrics",
            },
        )

    existing = next(
        (
            row
            for row in load_trial_index(competition)
            if row.get("competition") == competition and row.get("trial_id") == trial_id
        ),
        None,
    )
    if existing:
        return _finish(
            out_dir,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "already_processed",
                "steps": ["duplicate_memory_blocked"],
                "issues": [],
                "memory": existing,
                "next_action": "plan-next-experiment",
            },
        )

    evaluation = evaluate_trial(competition, trial_id)
    diagnosis = diagnose_trial(competition, trial_id)
    readiness = {
        "execution_profile_ready": collection.get("profile_status") == "ready",
        "workspace_run_completed": collection.get("workspace_run_status") == "completed",
        "metrics_collected": collection.get("status") == "collected",
        "completed_trial_count": len(load_trial_index(competition)) + 1,
        "pending_user_review": _has_pending_user_review(competition),
    }
    human_review = decide_human_review(
        competition,
        trial_id,
        diagnosis,
        pipeline_readiness=readiness,
    )
    steps = ["evaluated", "diagnosed"]
    if human_review["timing"] == "defer":
        _queue_deferred_review(competition, trial_id, human_review, diagnosis)
        steps.append("review_deferred")

    memory = remember_trial(competition, trial_id)
    steps.append("remembered")
    review_pack = None
    if human_review["timing"] == "request_now":
        deferred_items = _load_deferred_items(competition)
        review_diagnosis = _merge_deferred_context(diagnosis, deferred_items)
        review_pack = prepare_review_pack(competition, trial_id, review_diagnosis)
        _clear_deferred_review(competition)
        steps.append("review_pack_prepared")

    if human_review["timing"] == "request_now":
        status = "awaiting_human_review"
        next_action = "request-user-review"
    elif human_review["timing"] == "defer":
        status = "completed_review_deferred"
        next_action = "plan-next-experiment"
    else:
        status = "completed"
        next_action = "plan-next-experiment"
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": status,
        "steps": steps,
        "issues": [],
        "pipeline_readiness": readiness,
        "evaluation": evaluation,
        "diagnosis": diagnosis,
        "human_review": human_review,
        "review_pack": review_pack,
        "memory": memory,
        "next_action": next_action,
    }
    return _finish(out_dir, result)


def _finish(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    _write_result(out_dir, result)
    log_decision(
        result["competition"],
        result["trial_id"],
        decision_type="workspace_result_cycle",
        decision=result["status"],
        reason="Workspace result-cycle preconditions and analysis steps were evaluated.",
        evidence={
            "steps": result.get("steps", []),
            "review_timing": result.get("human_review", {}).get("timing"),
            "is_best": result.get("memory", {}).get("is_best"),
            "issues": result.get("issues", []),
        },
        next_action=result["next_action"],
    )
    return result


def _queue_deferred_review(
    competition: str,
    trial_id: str,
    human_review: dict[str, Any],
    diagnosis: dict[str, Any],
) -> None:
    path = competition_memory_dir(competition) / "deferred_review_queue.json"
    queue = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": []}
    triggers = _specific_triggers(human_review.get("triggers", []))
    for trigger in triggers:
        existing = next((item for item in queue["items"] if item["trigger"] == trigger), None)
        if existing:
            existing["last_trial"] = trial_id
            existing["occurrences"] += 1
            existing["issues"] = _unique([*existing["issues"], *diagnosis.get("issues", [])])
            existing["questions"] = _unique([*existing["questions"], *diagnosis.get("user_questions", [])])
        else:
            queue["items"].append(
                {
                    "trigger": trigger,
                    "first_trial": trial_id,
                    "last_trial": trial_id,
                    "occurrences": 1,
                    "issues": diagnosis.get("issues", []),
                    "questions": diagnosis.get("user_questions", []),
                }
            )
    write_text(path, json.dumps(queue, ensure_ascii=False, indent=2) + "\n")


def _load_deferred_items(competition: str) -> list[dict[str, Any]]:
    path = competition_memory_dir(competition) / "deferred_review_queue.json"
    if not path.exists():
        return []
    queue = json.loads(path.read_text(encoding="utf-8"))
    return queue.get("items", []) if isinstance(queue, dict) else []


def _merge_deferred_context(
    diagnosis: dict[str, Any],
    deferred_items: list[dict[str, Any]],
) -> dict[str, Any]:
    merged = dict(diagnosis)
    deferred_issues = [issue for item in deferred_items for issue in item.get("issues", [])]
    deferred_questions = [question for item in deferred_items for question in item.get("questions", [])]
    merged["issues"] = _unique([*diagnosis.get("issues", []), *deferred_issues])
    merged["user_questions"] = _unique([*diagnosis.get("user_questions", []), *deferred_questions])
    merged["deferred_review_items"] = deferred_items
    return merged


def _clear_deferred_review(competition: str) -> None:
    path = competition_memory_dir(competition) / "deferred_review_queue.json"
    write_text(path, json.dumps({"items": []}, ensure_ascii=False, indent=2) + "\n")


def _has_pending_user_review(competition: str) -> bool:
    root = experiment_dir(competition)
    if not root.exists():
        return False
    for manifest_path in root.glob("*/review_pack/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(manifest, dict) and manifest.get("status") == "pending_user_feedback":
            return True
    return False


def _specific_triggers(triggers: list[str]) -> list[str]:
    specific = [trigger for trigger in triggers if trigger != "diagnosis_requested_review"]
    return specific or triggers


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _write_result(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "workspace_result_cycle.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {result['competition']} / {result['trial_id']} Workspace Result Cycle",
        "",
        f"- status: {result['status']}",
        f"- review_timing: {result.get('human_review', {}).get('timing')}",
        f"- remembered: {'remembered' in result['steps']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Steps",
        "",
    ]
    lines.extend(f"- {step}" for step in result["steps"])
    lines.append("")
    write_text(out_dir / "workspace_result_cycle.md", "\n".join(lines))
