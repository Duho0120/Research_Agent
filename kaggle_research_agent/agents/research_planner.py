from __future__ import annotations


from pathlib import Path
from typing import Any

from .. import simple_yaml
from ..defaults import DEFAULT_CONFIG
from ..paths import competition_configs_dir, competition_memory_dir, configs_dir, memory_dir, trial_dir
from ..store import load_recent_trials, load_state, read_text, write_text
from .research_protocol import build_research_protocol


def propose_plan(competition: str, trial_id: str) -> dict[str, Any]:
    state = load_state(competition)
    recent = load_recent_trials(competition)
    notes = _read_competition_text(competition, "research_notes.md")
    rules = _read_competition_text(competition, "rules.md")
    allowed = _load_allowed_space(competition)

    focus = state.get("strategy", {}).get("current_focus", "build reliable baseline")
    best = state.get("current_state", {}).get("best_trial")
    failures = state.get("current_state", {}).get("consecutive_failures", 0)

    if not recent:
        hypothesis = "Establish a reliable baseline and verify the metric pipeline."
        changes = ["Run the baseline config without extra feature changes."]
        risk = ["The baseline may be weak, but it gives a trustworthy reference point."]
        config = DEFAULT_CONFIG
    elif failures >= 3:
        hypothesis = "Recent failures suggest the current direction is saturated; switch strategy."
        changes = ["Change one high-level direction while keeping validation fixed.", "Prefer feature diagnostics over parameter micro-tuning."]
        risk = ["A strategy shift can make attribution harder if too many things change."]
        config = _next_config_from_recent(recent, DEFAULT_CONFIG)
        config["features"]["use_interactions"] = True
    else:
        hypothesis = f"Continue the current focus: {focus}."
        changes = ["Make one controlled config change based on recent results.", "Keep validation unchanged for comparability."]
        risk = ["Small changes may be within seed variance."]
        config = _next_config_from_recent(recent, DEFAULT_CONFIG)

    plan = {
        "trial_id": trial_id,
        "hypothesis": hypothesis,
        "changes": changes,
        "expected_effect": "A small but interpretable movement in CV, with no obvious validation break.",
        "risk": risk,
        "success_criteria": [
            "CV improves beyond known noise or remains competitive with better diversity.",
            "No metric or submission format issues are observed.",
            "If submitted, LB movement is directionally consistent with CV.",
        ],
        "context_used": {
            "best_trial": best,
            "recent_trial_count": len(recent),
            "rules_seen": bool(rules.strip()),
            "notes_seen": bool(notes.strip()),
            "allowed_space_seen": bool(allowed),
        },
        "config": config,
    }

    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / "plan.md", render_plan(plan))
    simple_yaml.dump(config, out_dir / "config.yaml")
    return plan


def _next_config_from_recent(recent: list[dict[str, Any]], default: dict[str, Any]) -> dict[str, Any]:
    config = {
        "model": {
            "type": default["model"]["type"],
            "params": dict(default["model"]["params"]),
        },
        "features": dict(default["features"]),
        "cv": dict(default["cv"]),
    }
    if len(recent) % 3 == 1:
        config["features"]["use_frequency_encoding"] = True
    elif len(recent) % 3 == 2:
        config["features"]["use_target_encoding"] = True
    else:
        config["model"]["params"]["learning_rate"] = 0.02
    return config


def _read_competition_text(competition: str, filename: str) -> str:
    competition_path = competition_memory_dir(competition) / filename
    if competition_path.exists():
        return read_text(competition_path)
    return read_text(memory_dir() / filename)


def _load_allowed_space(competition: str) -> dict[str, Any]:
    competition_path = competition_configs_dir(competition) / "allowed_space.yaml"
    if competition_path.exists():
        return simple_yaml.load(competition_path, default={})
    return simple_yaml.load(configs_dir() / "allowed_space.yaml", default={})


def render_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan['trial_id']} Plan",
        "",
        "## Hypothesis",
        "",
        plan["hypothesis"],
        "",
        "## Changes",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["changes"])
    lines.extend(["", "## Expected Effect", "", plan["expected_effect"], "", "## Risk", ""])
    lines.extend(f"- {item}" for item in plan["risk"])
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(f"- {item}" for item in plan["success_criteria"])
    lines.extend(["", "## Config", "", "```yaml", simple_yaml.to_yaml(plan["config"]).rstrip(), "```", ""])
    return "\n".join(lines)


import json
from pathlib import Path
from typing import Any

from ..paths import competition_memory_dir, competition_submissions_dir, trial_dir
from ..store import load_state, write_text
from .pipeline_planner import plan_pipeline_improvement


def propose_next_experiment(competition: str, source_trial_id: str, next_trial_id: str) -> dict[str, Any]:
    """Create a next-experiment recommendation from diagnosis and submission evidence."""

    state = load_state(competition)
    failures = int(state.get("current_state", {}).get("consecutive_failures", 0))
    diagnosis = _latest_diagnosis_decision(competition, source_trial_id)
    submission = _latest_submission(competition, source_trial_id)
    user_feedback = _latest_user_feedback(competition, source_trial_id)
    pipeline_plan = _latest_or_create_pipeline_plan(competition, source_trial_id)
    research_protocol = _latest_or_create_research_protocol(competition, source_trial_id, next_trial_id)
    strategy_hint = diagnosis.get("evidence", {}).get("strategy_recommendation")
    issues = diagnosis.get("evidence", {}).get("issues", [])

    strategy = _choose_strategy(strategy_hint, failures, submission, user_feedback, pipeline_plan)
    if research_protocol and strategy == "controlled_refinement":
        strategy = research_protocol.get("recommended_next_trial", {}).get("strategy") or strategy
    plan = {
        "competition": competition,
        "source_trial_id": source_trial_id,
        "next_trial_id": next_trial_id,
        "strategy": strategy,
        "rationale": _rationale(strategy, failures, issues, submission, user_feedback),
        "changes": _changes_for_strategy(strategy),
        "guardrails": _guardrails_for_strategy(strategy),
        "requires_user_review_before_submit": strategy != "controlled_refinement",
        "evidence_used": {
            "consecutive_failures": failures,
            "strategy_hint": strategy_hint,
            "issues": issues,
            "latest_submission": submission,
            "latest_user_feedback": user_feedback,
            "pipeline_improvement": pipeline_plan,
            "research_protocol": _compact_protocol(research_protocol),
        },
    }
    out_dir = trial_dir(competition, next_trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / "next_experiment.md", render_next_experiment(plan))
    return plan


def render_next_experiment(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan['next_trial_id']} Next Experiment",
        "",
        "## Strategy",
        "",
        plan["strategy"],
        "",
        "## Rationale",
        "",
        plan["rationale"],
        "",
        "## Changes",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["changes"])
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in plan["guardrails"])
    protocol = plan["evidence_used"].get("research_protocol", {})
    if protocol:
        lines.extend(["", "## Research Protocol", ""])
        lines.append(f"- risk_level: {protocol.get('risk', {}).get('level')}")
        lines.append(f"- protocol_strategy: {protocol.get('recommended_next_trial', {}).get('strategy')}")
        lines.extend(["", "### Do Not Change", ""])
        lines.extend(f"- {item}" for item in protocol.get("do_not_change", []))
        lines.extend(["", "### Need User Check", ""])
        lines.extend(f"- {item}" for item in protocol.get("need_user_check", []) or ["No immediate user check required."])
    lines.extend(
        [
            "",
            "## Submit Gate",
            "",
            f"- requires_user_review_before_submit: {plan['requires_user_review_before_submit']}",
            "- Preserve source and next-trial artifacts before any leaderboard submission.",
            "",
            "## Evidence Used",
            "",
            "```json",
            json.dumps(plan["evidence_used"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _choose_strategy(
    strategy_hint: str | None,
    failures: int,
    submission: dict[str, Any] | None,
    user_feedback: dict[str, Any] | None = None,
    pipeline_plan: dict[str, Any] | None = None,
) -> str:
    if user_feedback:
        decision = user_feedback.get("decision")
        if decision == "change_validation":
            return "validation_review"
        if decision == "change_model_family":
            return "model_family_change"
        if decision == "prepare_sota_research":
            return "sota_architecture_attempt"

    if pipeline_plan:
        axis = pipeline_plan.get("primary_axis")
        if axis == "validation":
            return "validation_review"
        if axis in {"model_architecture", "pretraining_strategy"}:
            return "sota_architecture_attempt"
        if axis == "model_family":
            return "sota_architecture_attempt" if failures >= 3 else "model_family_change"

    lb_worsened = False
    if submission:
        score_delta = submission.get("score_delta")
        rank_delta = submission.get("rank_delta")
        lb_worsened = (score_delta is not None and score_delta < 0) or (rank_delta is not None and rank_delta < 0)

    if strategy_hint == "strategy_escalation" or failures >= 3:
        return "sota_architecture_attempt" if lb_worsened or failures >= 3 else "model_family_change"
    if lb_worsened:
        return "validation_review"
    return "controlled_refinement"


def _changes_for_strategy(strategy: str) -> list[str]:
    if strategy == "sota_architecture_attempt":
        return [
            "Switch model-family or architecture instead of another parameter micro-tune.",
            "Prepare one SOTA-inspired candidate from the allowed model space.",
            "Keep validation fixed unless the diagnosis explicitly asks for validation review.",
        ]
    if strategy == "model_family_change":
        return [
            "Change the model family while preserving the feature and validation assumptions.",
            "Prefer a diverse candidate whose predictions should differ from the current best.",
        ]
    if strategy == "validation_review":
        return [
            "Audit validation split assumptions before another submission.",
            "Run a small comparison trial focused on CV/LB alignment.",
        ]
    return [
        "Make one controlled improvement based on the latest diagnosis.",
        "Keep model family and validation fixed for attribution.",
    ]


def _guardrails_for_strategy(strategy: str) -> list[str]:
    guardrails = [
        "Keep validation unchanged unless this is explicitly a validation review.",
        "Do not submit if expected gain is within known seed variance.",
        "Record current and submitted leaderboard score/rank if a submission is made.",
    ]
    if strategy == "sota_architecture_attempt":
        guardrails.append("Compare the SOTA attempt against the current best before replacing the best marker.")
    return guardrails


def _rationale(
    strategy: str,
    failures: int,
    issues: list[str],
    submission: dict[str, Any] | None,
    user_feedback: dict[str, Any] | None = None,
) -> str:
    parts = [f"Selected `{strategy}` after {failures} consecutive failure(s)."]
    if issues:
        parts.append("Diagnosis issues: " + "; ".join(issues))
    if submission:
        parts.append(
            "Latest submission movement: "
            f"score_delta={submission.get('score_delta')}, rank_delta={submission.get('rank_delta')}."
        )
    if user_feedback:
        parts.append(
            "User feedback: "
            f"decision={user_feedback.get('decision')}, follow_up_action={user_feedback.get('follow_up_action')}."
        )
    if strategy == "sota_architecture_attempt":
        parts.append("The current method appears saturated enough to justify a model or architecture level attempt.")
    return " ".join(parts)


def _latest_diagnosis_decision(competition: str, trial_id: str) -> dict[str, Any]:
    path = competition_memory_dir(competition) / "decision_log.jsonl"
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    for row in reversed(rows):
        if row.get("trial_id") == trial_id and row.get("decision_type") == "diagnosis":
            return row
    return {}


def _latest_submission(competition: str, trial_id: str) -> dict[str, Any] | None:
    path = competition_submissions_dir(competition) / "submission_log.jsonl"
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    for row in reversed(rows):
        if row.get("trial_id") == trial_id:
            return row
    return None


def _latest_user_feedback(competition: str, trial_id: str) -> dict[str, Any] | None:
    path = competition_memory_dir(competition) / "user_feedback.jsonl"
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    for row in reversed(rows):
        if row.get("trial_id") == trial_id:
            return row
    return None


def _latest_or_create_pipeline_plan(competition: str, trial_id: str) -> dict[str, Any] | None:
    path = trial_dir(competition, trial_id) / "pipeline_improvement_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not (trial_dir(competition, trial_id) / "metrics.json").exists():
        return None
    return plan_pipeline_improvement(competition, trial_id)


def _latest_or_create_research_protocol(
    competition: str,
    source_trial_id: str,
    next_trial_id: str,
) -> dict[str, Any] | None:
    source_dir = trial_dir(competition, source_trial_id)
    if not (source_dir / "metrics.json").exists():
        return None
    path = source_dir / "research_protocol.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_research_protocol(competition, source_trial_id, next_trial_id)


def _compact_protocol(protocol: dict[str, Any] | None) -> dict[str, Any]:
    if not protocol:
        return {}
    return {
        "risk": protocol.get("risk"),
        "recommended_next_trial": protocol.get("recommended_next_trial"),
        "need_user_check": protocol.get("need_user_check"),
        "do_not_change": protocol.get("do_not_change"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


