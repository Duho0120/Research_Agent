from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .execution_facts import load_executed_trial_facts, resolve_trial_plan
from .paths import competition_memory_dir, competition_submissions_dir, trial_dir
from .store import now_iso, write_text


EPSILON = 1e-12
MAX_AXIS_NO_IMPROVE_ATTEMPTS = 3
# Relative worsening beyond this ratio is treated as catastrophic: the axis is
# rejected immediately instead of getting the usual MAX_AXIS_NO_IMPROVE_ATTEMPTS
# refinement attempts. Real tuning noise in this loop is well under 5%; the
# failures this guards against were 120%+ (a bad estimator swap) and 220%+ (a
# feature set that dropped the dominant predictor), each of which previously
# burned two more full trials on refining an obviously broken idea.
CATASTROPHIC_REGRESSION_RATIO = 0.25
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
    facts = load_executed_trial_facts(competition, trial_id)
    plan = plan or resolve_trial_plan(competition, trial_id)
    metrics = metrics or _load_trial_json(out_dir, "metrics.json")
    submission = submission or _load_trial_json(out_dir, "submission_run.json")

    previous_cards = _cards_before_trial(_load_decision_cards(competition, exclude_trial_id=trial_id), trial_id)
    previous_card = previous_cards[-1] if previous_cards else None
    objective = str(metrics.get("objective") or plan.get("objective") or "maximize").lower()
    if objective not in {"maximize", "minimize"}:
        objective = "maximize"
    best_card = _best_card(previous_cards, objective=objective)
    current_axis = str(plan.get("primary_change_axis") or facts.get("primary_change_axis") or "").strip()
    source_trial_id = plan.get("source_trial_id") or facts.get("source_trial_id") or (previous_card or {}).get("trial_id")

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
    catastrophic_regression = _is_catastrophic_regression(
        best_local, local_score, objective
    ) or _is_catastrophic_regression(best_lb, lb_score, objective)
    decision = _decision(
        has_previous=previous_card is not None,
        local_status=best_local_status,
        lb_status=best_lb_status,
        lb_score=lb_score,
        current_axis=current_axis,
        axis_attempt_count=axis_attempt_count,
        catastrophic_regression=catastrophic_regression,
    )
    model_type = str(metrics.get("model_type") or "").strip() or None
    # The trial's code was built on the previous card's recommended base (the
    # workspace is restored to that snapshot before patching), so that trial
    # -- not the nominal source trial -- is the right comparator for both
    # estimator-family changes and bit-identical-score (no-op) detection.
    code_base_trial = str((previous_card or {}).get("recommended_base_trial") or source_trial_id or "") or None
    source_card = _card_for_trial(previous_cards, code_base_trial)
    base_model_type = str((source_card or {}).get("model_type") or "").strip() or None
    # An axis that swaps the estimator family is a model_family change no
    # matter what the plan named it ("model_hyperparameters:Ridge" on an HGBR
    # base is a family swap, not tuning). Free-text axis names previously let
    # such changes bypass a rejected model_family axis entirely.
    estimator_family_changed = bool(model_type and base_model_type and model_type != base_model_type)
    rejected_axes = _unique(
        _previous_rejected_axes(previous_cards)
        + ([current_axis] if _rejects_axis(decision) and current_axis else [])
        + (["model_family"] if _rejects_axis(decision) and estimator_family_changed else [])
    )
    # A score bit-identical to the source trial across a claimed code change
    # is effectively impossible unless nothing actually changed; flag it so
    # the planner fixes the implementation instead of blaming the idea.
    no_change_suspected = bool(
        current_axis
        and local_score is not None
        and (source_card or {}).get("local_score") is not None
        and abs(local_score - float(source_card["local_score"])) <= EPSILON
        and previous_lb_status in {"flat", "baseline", "missing"}
    )
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
    base_trial_axis = _base_trial_change_axis(previous_cards, recommended_base_trial)
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
        "model_type": model_type,
        "catastrophic_regression": catastrophic_regression,
        "estimator_family_changed": estimator_family_changed,
        "no_change_suspected": no_change_suspected,
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
        "planner_constraints": _planner_constraints(
            decision,
            current_axis=current_axis,
            recommended_base_trial=recommended_base_trial,
            base_trial_axis=base_trial_axis,
            catastrophic_regression=catastrophic_regression,
            estimator_family_changed=estimator_family_changed,
            no_change_suspected=no_change_suspected,
        ),
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


def candidate_label_for_plan(plan: dict[str, Any]) -> str:
    """Compute the same dedup label used for rejected_candidates, ahead of running the trial."""
    return _candidate_label(plan)


def find_duplicate_candidate(decision_context: dict[str, Any], plan: dict[str, Any]) -> str | None:
    """Return the matching previously-rejected candidate label if `plan` repeats one for its axis, else None."""
    axis = _clean_axis(plan.get("primary_change_axis"))
    if not axis:
        return None
    label = candidate_label_for_plan(plan)
    if not label:
        return None
    rejected = (decision_context.get("rejected_candidates_by_axis") or {}).get(axis, [])
    normalized = label.strip().casefold()
    for candidate in rejected:
        if str(candidate).strip().casefold() == normalized:
            return str(candidate)
    return None


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
    catastrophic_regression: bool = False,
) -> str:
    if not has_previous:
        return "baseline_established"
    if lb_score is not None:
        if lb_status == "improved":
            return "accept"
        if local_status == "improved" and lb_status == "regressed":
            return "reject_or_hold_cv_lb_mismatch"
        if lb_status in {"flat", "regressed"}:
            if catastrophic_regression:
                return "reject_or_hold"
            if current_axis and axis_attempt_count < MAX_AXIS_NO_IMPROVE_ATTEMPTS:
                return "continue_axis_refinement"
            return "reject_or_hold"
    if local_status == "improved":
        return "provisional_accept"
    if local_status in {"flat", "regressed"}:
        if catastrophic_regression:
            return "reject_or_hold"
        if current_axis and axis_attempt_count < MAX_AXIS_NO_IMPROVE_ATTEMPTS:
            return "continue_axis_refinement"
        return "reject_or_hold"
    return "inconclusive"


def _is_catastrophic_regression(best: float | None, current: float | None, objective: str) -> bool:
    """A regression so large it cannot plausibly be tuning noise.

    Status alone (improved/flat/regressed) treats a 0.1% dip and a 3x blowup
    identically, which let obviously broken axes claim their full refinement
    attempts. Relative worsening beyond CATASTROPHIC_REGRESSION_RATIO means
    the idea itself is wrong, not the tuning.
    """
    if best is None or current is None or abs(best) <= EPSILON:
        return False
    worsening = (current - best) if objective == "minimize" else (best - current)
    return worsening / abs(best) > CATASTROPHIC_REGRESSION_RATIO


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


def _planner_constraints(
    decision: str,
    *,
    current_axis: str,
    recommended_base_trial: str,
    base_trial_axis: str | None = None,
    catastrophic_regression: bool = False,
    estimator_family_changed: bool = False,
    no_change_suspected: bool = False,
) -> list[str]:
    constraints = [
        f"Use `{recommended_base_trial}` as the base trial unless the user explicitly overrides it.",
        "Change exactly one primary improvement axis in the next trial.",
        "Keep split/model/preprocessing fixed unless selected as the primary axis.",
    ]
    if catastrophic_regression and current_axis:
        constraints.append(
            f"FACT: `{current_axis}` regressed the score by more than {int(CATASTROPHIC_REGRESSION_RATIO * 100)}% "
            "-- far beyond tuning noise. The idea itself is wrong for this data; do not retry it with different "
            "parameters, and do not retry it under a different axis name."
        )
    if estimator_family_changed and _rejects_axis(decision):
        constraints.append(
            "FACT: this trial swapped the estimator family, which counts as the `model_family` axis regardless of "
            "the axis name used. `model_family` is now a rejected axis: do not propose another estimator swap "
            "(under any axis name) unless the user explicitly asks for one."
        )
    if no_change_suspected:
        constraints.append(
            f"WARNING: this trial's local score is bit-identical to base trial `{recommended_base_trial}`'s, which "
            "is effectively impossible if the planned change actually ran. The implementation most likely did not "
            "apply the change to the measured path. Before judging the axis, make the next plan verify the change "
            "lands in the executed training/validation code."
        )
    if _rejects_axis(decision) and current_axis:
        constraints.append(f"Do not keep stacking on rejected axis `{current_axis}`.")
        constraints.append("If the rejected axis was implemented in the last trial, roll back to the recommended base before applying a new change.")
    if decision == "continue_axis_refinement" and current_axis:
        constraints.append(f"Keep the primary improvement axis as `{current_axis}` for the next trial.")
        constraints.append(
            f"Use `{recommended_base_trial}` as the code/pipeline base even if the active axis was attempted on another trial."
        )
        # This is a computed fact, not a hint the planner has to infer: only
        # trials whose own change_axis matches current_axis actually contain
        # that axis's change in their code, since only accepted trials
        # become the new base and get carried forward as-is.
        if base_trial_axis and base_trial_axis != current_axis:
            constraints.append(
                f"FACT: `{recommended_base_trial}` was itself planned under axis `{base_trial_axis}`, not "
                f"`{current_axis}`. Its code does NOT contain the `{current_axis}` change yet -- apply that change "
                "in full as part of this trial (e.g. the actual model-family swap, not just a parameter tweak that "
                "assumes it already exists)."
            )
        elif base_trial_axis == current_axis:
            constraints.append(
                f"FACT: `{recommended_base_trial}` was itself planned under axis `{current_axis}`, so its code "
                "already reflects this axis -- you are refining a variant within it, not introducing it from scratch."
            )
        constraints.append("Try a different candidate or parameter variant inside the same axis; do not switch to a new axis yet.")
        constraints.append("Do not preserve the failed candidate unless the new variant explicitly requires it.")
    if decision == "reject_or_hold_cv_lb_mismatch":
        constraints.append("CV improved but leaderboard regressed; consider validation reliability before trusting local gains.")
    return constraints


def _card_for_trial(cards: list[dict[str, Any]], trial_id: str | None) -> dict[str, Any] | None:
    if not trial_id:
        return None
    for card in cards:
        if str(card.get("trial_id") or "") == trial_id:
            return card
    return None


def _base_trial_change_axis(cards: list[dict[str, Any]], recommended_base_trial: str | None) -> str | None:
    card = _card_for_trial(cards, recommended_base_trial)
    axis = str((card or {}).get("change_axis") or "").strip()
    return axis or None


# Axis names are free text written by the planner LLM, so the same conceptual
# axis shows up under several spellings ("model_params" vs
# "model_hyperparameters", stray spaces around the colon). Attempt counting
# and rejected-axis membership must compare the normalized form, or a
# re-spelled axis silently restarts its attempt budget.
_AXIS_PREFIX_SYNONYMS = {
    "model_params": "model_hyperparameters",
    "model_parameters": "model_hyperparameters",
    "model_hyperparameter": "model_hyperparameters",
    "hyperparameter": "model_hyperparameters",
    "hyperparameters": "model_hyperparameters",
}


def _normalize_axis(axis: Any) -> str:
    text = re.sub(r"\s+", "", str(axis or "").strip().casefold())
    if not text:
        return ""
    prefix, separator, suffix = text.partition(":")
    prefix = _AXIS_PREFIX_SYNONYMS.get(prefix, prefix)
    return f"{prefix}:{suffix}" if separator else prefix


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
    normalized = _normalize_axis(axis)
    unresolved: list[dict[str, Any]] = []
    for row in cards:
        if _normalize_axis(row.get("change_axis")) != normalized:
            continue
        if row.get("decision") in {"accept", "provisional_accept"}:
            unresolved = []
            continue
        unresolved.append(row)
    return unresolved


def _candidate_label(plan: dict[str, Any]) -> str:
    axis = _clean_axis(plan.get("primary_change_axis"))
    title = _compact_text(plan.get("plan_title"), max_chars=80)
    # Extract signals from the full detail text and only shorten afterwards:
    # pre-truncating before extraction used to cut class names mid-word, so
    # rejected-candidate labels stored fragments like "TransformedTarg" and
    # "HistGradientBoostin" that the planner could not match against anything.
    details = [_compact_text(item, max_chars=400) for item in plan.get("change_details", []) if _compact_text(item, max_chars=400)]
    signals = [_compact_text(item, max_chars=90) for item in _candidate_signals(details)]
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
