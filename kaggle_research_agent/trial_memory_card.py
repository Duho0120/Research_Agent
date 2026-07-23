from __future__ import annotations

import json
import re
from typing import Any

from .execution_facts import load_executed_trial_facts, resolve_trial_plan
from .paths import competition_memory_dir, trial_dir
from .store import now_iso, write_text


MAX_LIST_ITEMS = 12
MAX_TEXT_CHARS = 180


def write_trial_memory_card(
    competition: str,
    trial_id: str,
    *,
    plan: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    decision_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    facts = load_executed_trial_facts(competition, trial_id)
    plan = plan or resolve_trial_plan(competition, trial_id)
    metrics = metrics or _load_trial_json(out_dir, "metrics.json")
    decision_card = decision_card or _load_trial_json(out_dir, "decision_card.json")
    card = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "created_at": now_iso(),
        "plan_type": plan.get("plan_type"),
        "source_trial_id": plan.get("source_trial_id") or facts.get("source_trial_id") or decision_card.get("source_trial_id"),
        "plan_title": plan.get("plan_title"),
        "change_axis": plan.get("primary_change_axis") or facts.get("primary_change_axis") or decision_card.get("change_axis"),
        "candidate_label": decision_card.get("candidate_label"),
        "change_details": _first_items(plan.get("change_details"), limit=4),
        "kept_unchanged": _first_items(plan.get("keep_unchanged"), limit=5),
        "local_score": metrics.get("cv_score") or metrics.get("validation_accuracy") or decision_card.get("local_score"),
        "lb_score": decision_card.get("lb_score"),
        "local_status": decision_card.get("local_status"),
        "previous_local_status": decision_card.get("previous_local_status"),
        "local_delta": decision_card.get("local_delta"),
        "previous_local_delta": decision_card.get("previous_local_delta"),
        "decision": decision_card.get("decision"),
        "active_axis": decision_card.get("active_axis"),
        "axis_attempt_count": decision_card.get("axis_attempt_count"),
        "axis_attempt_limit": decision_card.get("axis_attempt_limit"),
        "recommended_base_trial": decision_card.get("recommended_base_trial"),
        "rejected_axes": _first_items(decision_card.get("rejected_axes", []), limit=MAX_LIST_ITEMS),
        "rejected_candidates": _first_items(decision_card.get("rejected_candidates", []), limit=MAX_LIST_ITEMS),
        "rejected_candidates_by_axis": _compact_mapping(decision_card.get("rejected_candidates_by_axis", {}), limit=5),
        "active_axis_rejected_candidates": _first_items(decision_card.get("active_axis_rejected_candidates", []), limit=5),
        "feature_columns": _first_items(metrics.get("feature_columns", []), limit=30),
        "numeric_features": _first_items((metrics.get("preprocessing") or {}).get("numeric_features", []), limit=20),
        "categorical_features": _first_items((metrics.get("preprocessing") or {}).get("categorical_features", []), limit=20),
        "model_type": (facts.get("model") or {}).get("estimator") or metrics.get("model_type") or metrics.get("model"),
        "model": facts.get("model") or {},
        "split": facts.get("validation") or metrics.get("split", {}),
        "consistency_issues": facts.get("consistency_issues") or [],
        "next_guidance": decision_card.get("next_guidance"),
    }
    _write_files(competition, trial_id, card)
    return card


def render_trial_memory_card(card: dict[str, Any]) -> str:
    lines = [
        f"# Trial Memory Card: {card.get('trial_id')}",
        "",
        f"- plan_type: {card.get('plan_type')}",
        f"- source_trial_id: {card.get('source_trial_id')}",
        f"- change_axis: {card.get('change_axis')}",
        f"- candidate_label: {card.get('candidate_label')}",
        f"- local_score: {card.get('local_score')}",
        f"- local_status: {card.get('local_status')}",
        f"- local_delta: {card.get('local_delta')}",
        f"- previous_local_status: {card.get('previous_local_status')}",
        f"- previous_local_delta: {card.get('previous_local_delta')}",
        f"- lb_score: {card.get('lb_score')}",
        f"- decision: {card.get('decision')}",
        f"- active_axis: {card.get('active_axis')}",
        f"- axis_attempt_count: {card.get('axis_attempt_count')}/{card.get('axis_attempt_limit')}",
        f"- recommended_base_trial: {card.get('recommended_base_trial')}",
        f"- model_type: {card.get('model_type')}",
        f"- model: {card.get('model')}",
        f"- split: {card.get('split')}",
        "",
        "## Change Details",
        "",
    ]
    lines.extend(f"- {item}" for item in card.get("change_details", []) or ["None"])
    lines.extend(["", "## Kept Unchanged", ""])
    lines.extend(f"- {item}" for item in card.get("kept_unchanged", []) or ["None"])
    lines.extend(["", "## Features", ""])
    lines.append(f"- feature_columns: {card.get('feature_columns', [])}")
    lines.append(f"- numeric_features: {card.get('numeric_features', [])}")
    lines.append(f"- categorical_features: {card.get('categorical_features', [])}")
    lines.extend(["", "## Rejected Axes", ""])
    lines.extend(f"- {item}" for item in card.get("rejected_axes", []) or ["None"])
    lines.extend(["", "## Rejected Candidates", ""])
    lines.extend(f"- {item}" for item in card.get("rejected_candidates", []) or ["None"])
    lines.extend(["", "## Active Axis Rejected Candidates", ""])
    lines.extend(f"- {item}" for item in card.get("active_axis_rejected_candidates", []) or ["None"])
    lines.extend(["", "## Next Guidance", "", str(card.get("next_guidance") or ""), ""])
    return "\n".join(lines)


def _write_files(competition: str, trial_id: str, card: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "trial_memory_card.json", json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "trial_memory_card.md", render_trial_memory_card(card))
    memory = competition_memory_dir(competition)
    memory.mkdir(parents=True, exist_ok=True)
    cards = [row for row in _load_memory_cards(competition) if row.get("trial_id") != trial_id]
    cards.append(card)
    with (memory / "trial_memory_cards.jsonl").open("w", encoding="utf-8") as file:
        for row in cards:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_text(memory / "latest_trial_memory_card.json", json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    write_text(memory / "latest_trial_memory_card.md", render_trial_memory_card(card))


def _load_memory_cards(competition: str) -> list[dict[str, Any]]:
    path = competition_memory_dir(competition) / "trial_memory_cards.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_trial_json(out_dir, name: str) -> dict[str, Any]:
    for path in [out_dir / name, out_dir / "internal" / name]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _first_items(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [_compact_text(item) for item in items if _compact_text(item)][:limit]


def _compact_mapping(value: Any, *, limit: int) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, items in value.items():
        compact_key = _compact_text(key)
        if compact_key:
            result[compact_key] = _first_items(items, limit=limit)
    return result


def _compact_text(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x20-\x7E가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" |")
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 3].rstrip() + "..."
