from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .paths import competition_memory_dir, competition_submissions_dir, trial_dir
from .store import now_iso, write_text


EPSILON = 1e-12
MAX_AXIS_NO_IMPROVE_ATTEMPTS = 3
MAX_REJECTED_CANDIDATES = 12
MAX_REJECTED_CANDIDATES_PER_AXIS = 5
MAX_CANDIDATE_LABEL_CHARS = 180


def write_trial_decision_card(
    competition: str,
    trial_id: str,
    *,
    plan: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    submission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact, rule-based F-05 decision card for the next planner."""

    out_dir = trial_dir(competition, trial_id)
    plan = plan or _load_trial_json(out_dir, "demo_experiment_plan.json")
    metrics = metrics or _load_trial_json(out_dir, "metrics.json")
    submission = submission or _load_trial_json(out_dir, "submission_run.json")

    previous_cards = _cards_before_trial(_load_decision_cards(competition, exclude_trial_id=trial_id), trial_id)
    previous_card = previous_cards[-1] if previous_cards else None
    objective = str(metrics.get("objective") or plan.get("objective") or "maximize").lower()
    if objective not in {"maximize", "minimize"}:
        objective = "maximize"
    best_card = _best_card(previous_cards, objective=objective)
    current_axis = str(plan.get("primary_change_axis") or "").strip()
    source_trial_id = plan.get("source_trial_id") or (previous_card or {}).get("trial_id")

    local_score = _score(metrics.get("cv_score"), metrics.get("validation_accuracy"), metrics.get("local_score"))
    lb_score = _score(
        submission.get("submitted_lb_score"),
        (submission.get("recorded_submission") or {}).get("submitted_lb_score")
        if isinstance(submission.get("recorded_submission"), dict)
        else None,
        _submission_log_score(competition, trial_id),
    )
    previous_local = _score((previous_card or {}).get("local_score"))
    previous_lb = _score((previous_card or {}).get("lb_score"))
    best_local = _score((best_card or {}).get("local_score"), previous_local)
    best_lb = _score((best_card or {}).get("lb_score"), previous_lb)

    previous_local_delta = _delta(previous_local, local_score)
    previous_lb_delta = _delta(previous_lb, lb_score)
    best_local_delta = _delta(best_local, local_score)
    best_lb_delta = _delta(best_lb, lb_score)
    previous_local_status = _status(previous_local, local_score, objective)
    previous_lb_status = _status(previous_lb, lb_score, objective)
    best_local_status = _status(best_local, local_score, objective)
    best_lb_status = _status(best_lb, lb_score, objective)
    raw_decision = _decision(
        has_previous=previous_card is not None,
        local_status=previous_local_status,
        lb_status=previous_lb_status,
        lb_score=lb_score,
    )
    unresolved_axis_cards = _unresolved_axis_cards(previous_cards, current_axis)
    axis_attempt_count = len(unresolved_axis_cards) + (1 if current_axis else 0)
    rejected_candidates_by_axis = _previous_rejected_candidates_by_axis(previous_cards)
    active_rejected_candidates_by_axis = (
        _previous_rejected_candidates_by_axis(unresolved_axis_cards) if current_axis else {}
    )
    rejected_candidates = _flatten_rejected_candidates(rejected_candidates_by_axis)
    candidate_label = _candidate_label(plan)
    decision = _decision(
        has_previous=previous_card is not None,
        local_status=best_local_status,
        lb_status=best_lb_status,
        lb_score=lb_score,
        current_axis=current_axis,
        axis_attempt_count=axis_attempt_count,
    )
    rejected_axes = _unique(_previous_rejected_axes(previous_cards) + ([current_axis] if _rejects_axis(decision) and current_axis else []))
    if (_continues_axis(decision) or _rejects_axis(decision)) and candidate_label:
        rejected_candidates_by_axis = _add_rejected_candidate(
            rejected_candidates_by_axis,
            current_axis or "unknown_axis",
            candidate_label,
        )
        active_rejected_candidates_by_axis = _add_rejected_candidate(
            active_rejected_candidates_by_axis,
            current_axis or "unknown_axis",
            candidate_label,
        )
        rejected_candidates = _flatten_rejected_candidates(rejected_candidates_by_axis)
    accepted_axes = _unique(_previous_accepted_axes(previous_cards) + ([current_axis] if decision in {"accept", "provisional_accept"} and current_axis else []))
    recommended_base_trial = _recommended_base_trial(
        decision=decision,
        current_trial_id=trial_id,
        source_trial_id=str(source_trial_id) if source_trial_id else None,
        best_card=best_card,
        previous_card=previous_card,
    )
    card = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "created_at": now_iso(),
        "source_trial_id": source_trial_id,
        "plan_type": plan.get("plan_type"),
        "change_axis": current_axis,
        "local_score": local_score,
        "lb_score": lb_score,
        "previous_local_score": previous_local,
        "previous_lb_score": previous_lb,
        "best_local_score": best_local,
        "best_lb_score": best_lb,
        "local_delta": best_local_delta,
        "lb_delta": best_lb_delta,
        "previous_local_delta": previous_local_delta,
        "previous_lb_delta": previous_lb_delta,
        "objective": objective,
        "local_status": best_local_status,
        "lb_status": best_lb_status,
        "previous_local_status": previous_local_status,
        "previous_lb_status": previous_lb_status,
        "raw_decision": raw_decision,
        "decision": decision,
        "active_axis": current_axis if _continues_axis(decision) else None,
        "axis_attempt_count": axis_attempt_count,
        "axis_attempt_limit": MAX_AXIS_NO_IMPROVE_ATTEMPTS,
        "candidate_label": candidate_label,
        "rejected_candidates": rejected_candidates,
        "rejected_candidates_by_axis": rejected_candidates_by_axis,
        "active_axis_rejected_candidates": active_rejected_candidates_by_axis.get(current_axis, []) if current_axis else [],
        "recommended_base_trial": recommended_base_trial,
        "rejected_axes": rejected_axes,
        "accepted_axes": accepted_axes,
        "next_guidance": _next_guidance(decision, current_axis=current_axis, recommended_base_trial=recommended_base_trial),
        "planner_constraints": _planner_constraints(decision, current_axis=current_axis, recommended_base_trial=recommended_base_trial),
    }
    _write_card_files(competition, trial_id, card)
    log_decision(
        competition,
        trial_id,
        decision_type="trial_decision_card",
        decision=decision,
        reason="Rule-based F-05 trial evaluation card was written for the next planning step.",
        evidence={
            "change_axis": current_axis,
            "local_score": local_score,
            "lb_score": lb_score,
            "local_status": best_local_status,
            "lb_status": best_lb_status,
            "previous_local_status": previous_local_status,
            "previous_lb_status": previous_lb_status,
            "recommended_base_trial": recommended_base_trial,
            "rejected_axes": rejected_axes,
            "active_axis": card.get("active_axis"),
            "axis_attempt_count": axis_attempt_count,
            "rejected_candidates": rejected_candidates,
            "active_axis_rejected_candidates": card.get("active_axis_rejected_candidates"),
        },
        next_action="use-decision-card-for-next-plan",
    )
    return card


def load_latest_decision_context(competition: str) -> dict[str, Any]:
    cards = _load_decision_cards(competition)
    latest = cards[-1] if cards else None
    objective = str((latest or {}).get("objective") or "maximize").lower()
    if objective not in {"maximize", "minimize"}:
        objective = "maximize"
    best = _best_card(cards, objective=objective)
    return {
        "latest_decision_card": latest,
        "best_decision_card": best,
        "rejected_axes": _unique(_previous_rejected_axes(cards)),
        "accepted_axes": _unique(_previous_accepted_axes(cards)),
        "active_axis": (latest or {}).get("active_axis"),
        "axis_attempt_count": (latest or {}).get("axis_attempt_count"),
        "axis_attempt_limit": (latest or {}).get("axis_attempt_limit") or MAX_AXIS_NO_IMPROVE_ATTEMPTS,
        "rejected_candidates": _unique(_previous_rejected_candidates(cards)),
        "rejected_candidates_by_axis": _previous_rejected_candidates_by_axis(cards),
        "active_axis_rejected_candidates": (latest or {}).get("active_axis_rejected_candidates", []),
        "recommended_base_trial": (latest or {}).get("recommended_base_trial") or (best or {}).get("trial_id"),
        "next_guidance": (latest or {}).get("next_guidance"),
        "planner_constraints": (latest or {}).get("planner_constraints", []),
        "decision_card_count": len(cards),
    }


def render_decision_card(card: dict[str, Any]) -> str:
    lines = [
        f"# Trial Decision Card: {card.get('trial_id')}",
        "",
        f"- decision: {card.get('decision')}",
        f"- change_axis: {card.get('change_axis') or 'None'}",
        f"- source_trial_id: {card.get('source_trial_id')}",
        f"- recommended_base_trial: {card.get('recommended_base_trial')}",
        f"- local_score: {card.get('local_score')}",
        f"- local_status: {card.get('local_status')}",
        f"- local_delta: {card.get('local_delta')}",
        f"- previous_local_status: {card.get('previous_local_status')}",
        f"- previous_local_delta: {card.get('previous_local_delta')}",
        f"- active_axis: {card.get('active_axis') or 'None'}",
        f"- axis_attempt_count: {card.get('axis_attempt_count')}",
        f"- axis_attempt_limit: {card.get('axis_attempt_limit')}",
        f"- lb_score: {card.get('lb_score')}",
        f"- lb_status: {card.get('lb_status')}",
        f"- lb_delta: {card.get('lb_delta')}",
        f"- previous_lb_status: {card.get('previous_lb_status')}",
        f"- previous_lb_delta: {card.get('previous_lb_delta')}",
        "",
        "## Next Guidance",
        "",
        str(card.get("next_guidance") or ""),
        "",
        "## Planner Constraints",
        "",
    ]
    lines.extend(f"- {item}" for item in card.get("planner_constraints", []) or ["None"])
    lines.extend(["", "## Rejected Axes", ""])
    lines.extend(f"- {item}" for item in card.get("rejected_axes", []) or ["None"])
    lines.extend(["", "## Rejected Candidates", ""])
    lines.extend(f"- {item}" for item in card.get("rejected_candidates", []) or ["None"])
    lines.extend(["", "## Active Axis Rejected Candidates", ""])
    lines.extend(f"- {item}" for item in card.get("active_axis_rejected_candidates", []) or ["None"])
    lines.extend(["", "## Accepted Axes", ""])
    lines.extend(f"- {item}" for item in card.get("accepted_axes", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _write_card_files(competition: str, trial_id: str, card: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "decision_card.json", json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "decision_card.md", render_decision_card(card))
    memory = competition_memory_dir(competition)
    memory.mkdir(parents=True, exist_ok=True)
    cards = [row for row in _load_decision_cards(competition) if row.get("trial_id") != trial_id]
    cards.append(card)
    cards = _sort_decision_cards(cards)
    with (memory / "decision_cards.jsonl").open("w", encoding="utf-8") as file:
        for row in cards:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_text(memory / "latest_decision_card.json", json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    write_text(memory / "latest_decision_card.md", render_decision_card(card))


def _load_decision_cards(competition: str, *, exclude_trial_id: str | None = None) -> list[dict[str, Any]]:
    path = competition_memory_dir(competition) / "decision_cards.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("trial_id") != exclude_trial_id:
            rows.append(row)
    return rows


def _cards_before_trial(cards: list[dict[str, Any]], trial_id: str) -> list[dict[str, Any]]:
    current_number = _trial_number(trial_id)
    if current_number is None:
        return cards
    return [
        row
        for row in cards
        if _trial_number(str(row.get("trial_id") or "")) is None
        or (_trial_number(str(row.get("trial_id") or "")) or 0) < current_number
    ]


def _sort_decision_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cards, key=lambda row: _trial_sort_key(str(row.get("trial_id") or "")))


def _trial_sort_key(trial_id: str) -> tuple[int, int | str]:
    number = _trial_number(trial_id)
    if number is None:
        return (1, trial_id)
    return (0, number)


def _trial_number(trial_id: str) -> int | None:
    prefix = "trial_"
    if not trial_id.startswith(prefix):
        return None
    suffix = trial_id[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _load_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    for path in [out_dir / name, out_dir / "internal" / name]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _submission_log_score(competition: str, trial_id: str) -> float | None:
    path = competition_submissions_dir(competition) / "submission_log.jsonl"
    if not path.exists():
        return None
    score = None
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("trial_id") == trial_id:
            score = _score(row.get("submitted_lb_score"))
    return score


def _score(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _delta(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return round(current - previous, 12)


def _status(previous: float | None, current: float | None, objective: str) -> str:
    if current is None:
        return "missing"
    if previous is None:
        return "baseline"
    diff = current - previous
    if abs(diff) <= EPSILON:
        return "flat"
    improved = diff > 0 if objective == "maximize" else diff < 0
    return "improved" if improved else "regressed"


def _decision(
    *,
    has_previous: bool,
    local_status: str,
    lb_status: str,
    lb_score: float | None,
    current_axis: str = "",
    axis_attempt_count: int = 0,
) -> str:
    if not has_previous:
        return "baseline_established"
    if lb_score is not None:
        if lb_status == "improved":
            return "accept"
        if local_status == "improved" and lb_status == "regressed":
            return "reject_or_hold_cv_lb_mismatch"
        if lb_status in {"flat", "regressed"}:
            if current_axis and axis_attempt_count < MAX_AXIS_NO_IMPROVE_ATTEMPTS:
                return "continue_axis_refinement"
            return "reject_or_hold"
    if local_status == "improved":
        return "provisional_accept"
    if local_status in {"flat", "regressed"}:
        if current_axis and axis_attempt_count < MAX_AXIS_NO_IMPROVE_ATTEMPTS:
            return "continue_axis_refinement"
        return "reject_or_hold"
    return "inconclusive"


def _rejects_axis(decision: str) -> bool:
    return decision in {"reject_or_hold", "reject_or_hold_cv_lb_mismatch"}


def _continues_axis(decision: str) -> bool:
    return decision == "continue_axis_refinement"


def _recommended_base_trial(
    *,
    decision: str,
    current_trial_id: str,
    source_trial_id: str | None,
    best_card: dict[str, Any] | None,
    previous_card: dict[str, Any] | None,
) -> str:
    if decision in {"accept", "provisional_accept", "baseline_established"}:
        return current_trial_id
    if decision == "continue_axis_refinement":
        if best_card and best_card.get("trial_id"):
            return str(best_card["trial_id"])
        if previous_card and previous_card.get("recommended_base_trial"):
            return str(previous_card["recommended_base_trial"])
        if source_trial_id:
            return source_trial_id
    if best_card and best_card.get("trial_id"):
        return str(best_card["trial_id"])
    if source_trial_id:
        return source_trial_id
    if previous_card and previous_card.get("trial_id"):
        return str(previous_card["trial_id"])
    return current_trial_id


def _best_card(cards: list[dict[str, Any]], *, objective: str = "maximize") -> dict[str, Any] | None:
    accepted = [row for row in cards if row.get("decision") in {"accept", "provisional_accept", "baseline_established"}]
    candidates = accepted or cards
    if not candidates:
        return None
    with_lb = [row for row in candidates if row.get("lb_score") is not None]
    pool = with_lb or [row for row in candidates if row.get("local_score") is not None]
    if not pool:
        return candidates[-1]
    best = pool[0]
    best_score = _score(best.get("lb_score"), best.get("local_score"))
    for row in pool[1:]:
        score = _score(row.get("lb_score"), row.get("local_score"))
        if score is None:
            continue
        if best_score is None or _is_better_or_tied(score, best_score, objective):
            best = row
            best_score = score
    return best


def _is_better_or_tied(current: float, best: float, objective: str) -> bool:
    diff = current - best
    if abs(diff) <= EPSILON:
        return True
    return diff < 0 if objective == "minimize" else diff > 0


def _next_guidance(decision: str, *, current_axis: str, recommended_base_trial: str) -> str:
    if decision == "baseline_established":
        return "Use this trial as the baseline. The next trial should change exactly one improvement axis."
    if decision == "accept":
        return "Use this trial as the new base because leaderboard score improved."
    if decision == "provisional_accept":
        return "Use this trial provisionally because local validation improved, but confirm with submission before stacking more changes."
    if decision == "continue_axis_refinement":
        return (
            f"`{current_axis}` has not improved yet, but the axis is not rejected. Start from best/base trial "
            f"`{recommended_base_trial}` and apply another candidate or parameter variant within the same axis."
        )
    if decision == "reject_or_hold_cv_lb_mismatch":
        return (
            f"Do not stack more changes on `{current_axis}`. Start from `{recommended_base_trial}` and inspect validation reliability "
            "or choose a different single axis."
        )
    if decision == "reject_or_hold":
        return f"Treat `{current_axis or 'the last change'}` as exhausted for now. Start from `{recommended_base_trial}` and try a different single axis."
    return f"Decision is inconclusive. Start from `{recommended_base_trial}` and avoid broad rewrites."


def _planner_constraints(decision: str, *, current_axis: str, recommended_base_trial: str) -> list[str]:
    constraints = [
        f"Use `{recommended_base_trial}` as the base trial unless the user explicitly overrides it.",
        "Change exactly one primary improvement axis in the next trial.",
        "Keep split/model/preprocessing fixed unless selected as the primary axis.",
    ]
    if _rejects_axis(decision) and current_axis:
        constraints.append(f"Do not keep stacking on rejected axis `{current_axis}`.")
        constraints.append("If the rejected axis was implemented in the last trial, roll back to the recommended base before applying a new change.")
    if decision == "continue_axis_refinement" and current_axis:
        constraints.append(f"Keep the primary improvement axis as `{current_axis}` for the next trial.")
        constraints.append(
            f"Use `{recommended_base_trial}` as the code/pipeline base even if the active axis was attempted on another trial."
        )
        constraints.append("Try a different candidate or parameter variant inside the same axis; do not switch to a new axis yet.")
        constraints.append("Do not preserve the failed candidate unless the new variant explicitly requires it.")
    if decision == "reject_or_hold_cv_lb_mismatch":
        constraints.append("CV improved but leaderboard regressed; consider validation reliability before trusting local gains.")
    return constraints


def _previous_rejected_axes(cards: list[dict[str, Any]]) -> list[str]:
    axes: list[str] = []
    for row in cards:
        axes.extend(str(item) for item in row.get("rejected_axes", []) if item)
    return axes


def _previous_rejected_candidates(cards: list[dict[str, Any]]) -> list[str]:
    return _flatten_rejected_candidates(_previous_rejected_candidates_by_axis(cards))


def _previous_rejected_candidates_by_axis(cards: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in cards:
        by_axis = row.get("rejected_candidates_by_axis")
        if isinstance(by_axis, dict):
            for axis, values in by_axis.items():
                axis_key = _clean_axis(axis) or "unknown_axis"
                for value in values if isinstance(values, list) else [values]:
                    grouped = _add_rejected_candidate(grouped, axis_key, str(value))
            continue
        for value in row.get("rejected_candidates", []) or []:
            axis_key = _candidate_axis_from_label(str(value)) or _clean_axis(row.get("change_axis")) or "unknown_axis"
            grouped = _add_rejected_candidate(grouped, axis_key, str(value))
    return grouped


def _add_rejected_candidate(grouped: dict[str, list[str]], axis: str, candidate: str) -> dict[str, list[str]]:
    label = _compact_text(candidate, max_chars=MAX_CANDIDATE_LABEL_CHARS)
    if not label:
        return grouped
    axis_key = _clean_axis(axis) or "unknown_axis"
    values = grouped.get(axis_key, [])
    values = _unique_keep_last(values + [label])[-MAX_REJECTED_CANDIDATES_PER_AXIS:]
    grouped[axis_key] = values
    return grouped


def _flatten_rejected_candidates(grouped: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    for axis in sorted(grouped):
        flattened.extend(grouped[axis])
    return _unique(flattened)[-MAX_REJECTED_CANDIDATES:]


def _candidate_axis_from_label(label: str) -> str:
    prefix = label.split(":", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_ -]{3,80}", prefix):
        return _clean_axis(prefix)
    return ""


def _previous_accepted_axes(cards: list[dict[str, Any]]) -> list[str]:
    axes: list[str] = []
    for row in cards:
        axes.extend(str(item) for item in row.get("accepted_axes", []) if item)
    return axes


def _unresolved_axis_cards(cards: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    if not axis:
        return []
    unresolved: list[dict[str, Any]] = []
    for row in cards:
        if row.get("change_axis") != axis:
            continue
        if row.get("decision") in {"accept", "provisional_accept"}:
            unresolved = []
            continue
        unresolved.append(row)
    return unresolved


def _candidate_label(plan: dict[str, Any]) -> str:
    axis = _clean_axis(plan.get("primary_change_axis"))
    title = _compact_text(plan.get("plan_title"), max_chars=80)
    details = [_compact_text(item, max_chars=90) for item in plan.get("change_details", []) if _compact_text(item, max_chars=90)]
    signals = _candidate_signals(details)
    if not signals and title:
        signals = [title]
    if not axis:
        return _compact_text(" | ".join(signals), max_chars=MAX_CANDIDATE_LABEL_CHARS)
    suffix = " | ".join(signals[:4]) if signals else "candidate"
    return _compact_text(f"{axis}: {suffix}", max_chars=MAX_CANDIDATE_LABEL_CHARS)


def _candidate_signals(details: list[str]) -> list[str]:
    signals: list[str] = []
    assigned_values: set[str] = set()
    for detail in details:
        for key in ["Candidate", "Model", "Penalty", "Solver", "C", "l1_ratio", "max_iter", "Feature", "With"]:
            value = _labeled_value(detail, key)
            if value:
                signals.append(f"{key.lower()}={value}")
                assigned_values.add(value)
        assignments = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_.-]+)", detail)
        for key, value in assignments:
            signals.append(f"{key}={value}")
            assigned_values.add(value)
        class_tokens = re.findall(r"\b[A-Z][A-Za-z0-9_]{4,}\b", detail)
        signals.extend(
            token
            for token in class_tokens[:2]
            if token not in {"Candidate", "Feature", "Model", "Penalty", "Solver", "Replace"}
            and token not in assigned_values
        )
    return _unique(signals)


def _labeled_value(text: str, key: str) -> str:
    match = re.search(rf"\b{re.escape(key)}\s*:\s*([^|,;]+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return _compact_text(match.group(1), max_chars=50)


def _clean_axis(value: Any) -> str:
    return _compact_text(value, max_chars=80)


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common mojibake/private-use fragments from compact planner memory.
    text = re.sub(r"[^\x20-\x7E가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" |")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _unique_keep_last(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in reversed(values):
        value = str(value).strip()
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return list(reversed(result))
