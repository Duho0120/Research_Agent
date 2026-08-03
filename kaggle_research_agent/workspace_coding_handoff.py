from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .code_snapshot import load_trial_code_snapshot
from .execution_profile import load_execution_profile, validate_execution_profile
from .execution_plan_snapshot import capture_pending_execution_plan
from .paths import competition_dir, trial_dir
from .policies import load_policy
from .rag_policy import evaluate_rag_policy
from .retrieval.context_pack import build_context_pack
from .store import read_text, write_text


def prepare_workspace_coding_handoff(
    competition: str,
    trial_id: str,
    *,
    expanded_snapshot: bool = False,
    retry_reason: str | None = None,
    runtime_failure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    next_experiment_path = out_dir / "next_experiment.md"
    continuation_path = out_dir / "continuation_context.json"
    blocking_issues: list[str] = []

    next_experiment = read_text(next_experiment_path, default="").strip()
    if not next_experiment:
        blocking_issues.append("missing_next_experiment")
    continuation = _load_json_object(continuation_path)
    if continuation is None:
        blocking_issues.append("missing_or_invalid_continuation_context")
        continuation = {}
    if continuation.get("continuation_mode") == "must_wait":
        blocking_issues.append("continuation_requires_user_feedback")
    coding_instruction = _coding_instruction_text(out_dir, continuation, next_experiment)

    validation = validate_execution_profile(competition)
    profile: dict[str, Any] = {}
    if validation["status"] != "ready":
        blocking_issues.append("execution_profile_not_ready")
    else:
        profile = load_execution_profile(competition)

    status = "ready" if not blocking_issues else "blocked"
    handoff = _build_handoff(
        competition,
        trial_id,
        status=status,
        blocking_issues=blocking_issues,
        profile=profile,
        profile_validation=validation,
        continuation=continuation,
        expanded_snapshot=expanded_snapshot,
        retry_reason=retry_reason,
        runtime_failure_context=runtime_failure_context,
    )
    # _build_handoff may discover further issues (e.g. the plan's own Find:
    # hints not matching the base trial's actual code) that were not knowable
    # until it resolved the base trial's code snapshot -- blocking_issues is
    # shared by reference, so re-derive status from it rather than the
    # earlier, possibly stale local value.
    status = handoff.get("status", status)
    write_text(out_dir / "workspace_coding_handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    if status == "ready":
        request_text = render_workspace_coding_request(handoff, coding_instruction)
        write_text(out_dir / "workspace_coding_agent_request.md", request_text)
        capture_pending_execution_plan(
            competition,
            trial_id,
            request_text=request_text,
            request_id=handoff.get("request_id"),
        )
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_coding_handoff",
        decision=status,
        reason="Workspace coding handoff prepared from next-experiment and Execution Profile scope.",
        evidence={
            "continuation_mode": handoff.get("continuation_mode"),
            "pending_human_review": handoff.get("pending_human_review"),
            "allowed_write_paths": handoff.get("allowed_write_paths", []),
            "blocking_issues": blocking_issues,
        },
        next_action=handoff["next_action"],
    )
    return handoff


def _build_handoff(
    competition: str,
    trial_id: str,
    *,
    status: str,
    blocking_issues: list[str],
    profile: dict[str, Any],
    profile_validation: dict[str, Any],
    continuation: dict[str, Any],
    expanded_snapshot: bool = False,
    retry_reason: str | None = None,
    runtime_failure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    artifacts = profile.get("artifacts", {}) if profile else {}
    scope = profile.get("write_scope", {}) if profile else {}
    allowed = list(scope.get("allowed", [])) if isinstance(scope, dict) else []
    forbidden = list(scope.get("forbidden", [])) if isinstance(scope, dict) else []
    forbidden = _unique([*forbidden, *artifacts.get("metrics", []), *artifacts.get("submission", [])])
    validation_commands = []
    predict_commands = []
    commands = profile.get("commands", {}) if profile else {}
    if isinstance(commands, dict):
        validation_commands = list(commands.get("test", []))
        predict_commands = list(commands.get("predict", []))
    edit_mode = _edit_mode_for_continuation(continuation)
    source_trial_id = str(continuation.get("source_trial_id") or "") or None
    code_base_trial_id = str(continuation.get("recommended_base_trial") or "") or source_trial_id
    base_code_files = _source_trial_code_files(competition, code_base_trial_id)
    base_code_source = _source_trial_code_source_label(competition, code_base_trial_id) if base_code_files else None
    is_runtime_repair = bool(runtime_failure_context)
    restore_base_before_patch = bool(code_base_trial_id and base_code_files and not is_runtime_repair)
    delta_plan = _load_json_object(out_dir / "delta_plan.json") or {}
    if status == "ready" and not is_runtime_repair:
        # Check the plan's own Find: hints against the base trial's actual
        # code before any code-writing LLM call is spent generating a patch
        # from them. A plan can assume a change from an earlier, unaccepted
        # attempt at the same axis is already present in the base trial's
        # code -- it is not, since only accepted trials become the new base.
        blocking_issues.extend(_plan_find_target_issues(delta_plan, base_code_files))
        if blocking_issues:
            status = "blocked"
    next_action = "send-to-workspace-coding-agent" if status == "ready" else "resolve-workspace-handoff-blockers"
    metrics_paths = artifacts.get("metrics", []) if isinstance(artifacts, dict) else []
    metrics_path = metrics_paths[0] if metrics_paths else "outputs/metrics.json"
    context_files = _context_files(competition, trial_id)
    patch_budget = _patch_budget_for_delta(delta_plan)
    artifact_policy = load_policy("artifact_policy")
    data_card_summary = _load_data_card_summary(competition)
    retrieval_context: dict[str, Any] = {}
    if status == "ready" and profile:
        context_files.extend(
            _write_workspace_context_snapshot(
                competition,
                trial_id,
                profile,
                continuation,
                expanded_snapshot=expanded_snapshot,
                base_will_be_restored=restore_base_before_patch,
            )
        )
        rag_policy = _workspace_coding_rag_policy(continuation)
        if rag_policy["use_rag"]:
            retrieval_context = build_context_pack(
                competition,
                trial_id,
                task="workspace_code_writing",
                query=(
                    "next experiment current project code competition data card data profile target columns feature recommendations "
                    "previous trial metrics result pipeline structure "
                    "decision card rejected axes recommended base workspace context snapshot validation allowed write files"
                ),
            )
            retrieval_context["policy"] = rag_policy
            context_files.extend(
                [
                    retrieval_context["context_pack_md_file"],
                    retrieval_context["retrieval_manifest_file"],
                ]
            )
        else:
            retrieval_context = {
                "task": "workspace_code_writing",
                "document_count": 0,
                "documents": [],
                "skipped": True,
                "skip_reason": rag_policy["reason"],
                "policy": rag_policy,
            }
    return {
        "schema_version": "1.0",
        "request_id": f"{competition}:{trial_id}:workspace-coding",
        "competition": competition,
        "trial_id": trial_id,
        "handoff_type": "workspace_coding_agent_request",
        "status": status,
        "objective": "Implement the next workspace experiment within the Execution Profile write scope.",
        "project_root": profile.get("project_root") if profile else None,
        "platform": profile.get("platform") if profile else None,
        "continuation_mode": continuation.get("continuation_mode"),
        "source_trial_id": source_trial_id,
        "code_base_trial_id": code_base_trial_id,
        "recommended_base_trial": continuation.get("recommended_base_trial"),
        "pending_human_review": bool(continuation.get("pending_human_review")),
        "review_source_trial": continuation.get("review_source_trial"),
        "context_files": context_files,
        "snapshot_mode": (
            "expanded_runtime_repair"
            if is_runtime_repair
            else ("expanded_after_code_writer_blocked" if expanded_snapshot else "standard")
        ),
        "retry_reason": retry_reason,
        "runtime_failure_context": runtime_failure_context or {},
        "retrieval_context": _compact_retrieval_context(retrieval_context),
        "data_card_summary": data_card_summary,
        "edit_policy": {
            "mode": edit_mode,
            "prefer_patch_updates": edit_mode == "patch_only",
            "allow_full_file_updates": edit_mode == "full_file_allowed",
            "restore_base_before_patch": restore_base_before_patch,
            "base_code_source": base_code_source if restore_base_before_patch else None,
            "patch_budget": patch_budget,
            "patch_schema": {
                "path": "project-root-relative file path",
                "find": "exact existing text to replace",
                "replace": "replacement text",
                "reason": "short reason for the localized change",
            },
        },
        "allowed_write_paths": allowed,
        "forbidden_paths": forbidden,
        "validation_commands": validation_commands,
        "predict_commands": predict_commands,
        "execution_constraints": {
            "do_not_run_training": True,
            "do_not_submit": True,
            "do_not_edit_data_or_outputs": True,
            "do_not_write_outside_allowed_paths": True,
            "use_project_root_as_cwd": True,
            "base_trial_code_is_authoritative": restore_base_before_patch,
        },
        "metrics_output_contract": {
            "path": metrics_path,
            "score_key": "cv_score",
            "required_keys": ["cv_score", "metric", "objective"],
            "notes": [
                "Training code must write a finite numeric cv_score to the metrics artifact.",
                "metric should match the competition metric name when known.",
                "objective must be maximize or minimize.",
                "Additional diagnostic keys such as validation_accuracy are allowed, but cv_score is the canonical score.",
                "Every metrics and pipeline-summary value must be JSON serializable. Convert numpy scalars, callables, estimators, paths, and other objects to primitive values or stable strings before json.dumps.",
            ],
        },
        "artifact_policy": artifact_policy,
        "required_output": {
            "json_file": f"experiments/{competition}/{trial_id}/workspace_coding_result.json",
            "markdown_file": f"experiments/{competition}/{trial_id}/workspace_coding_result.md",
            "required_fields": ["status", "summary", "changed_files", "validation_results", "blocking_issues"],
            "status_values": ["completed", "blocked", "failed"],
            "next_action": "validate-workspace-code-change",
        },
        "profile_validation_status": profile_validation["status"],
        "profile_validation_issues": profile_validation.get("issues", []),
        "blocking_issues": _unique(blocking_issues),
        "next_action": next_action,
    }


def _edit_mode_for_continuation(continuation: dict[str, Any]) -> str:
    return "patch_only" if continuation.get("source_trial_id") else "full_file_allowed"


def _patch_budget_for_delta(delta_plan: dict[str, Any]) -> int:
    stages = delta_plan.get("affected_stages") if isinstance(delta_plan, dict) else []
    stage_count = len(stages) if isinstance(stages, list) else 0
    if stage_count <= 2:
        return 2
    if stage_count <= 4:
        return 4
    return 6


def _coding_instruction_text(out_dir: Path, continuation: dict[str, Any], fallback: str) -> str:
    if _is_active_axis_refinement(continuation):
        delta_md = read_text(out_dir / "delta_plan.md", default="").strip()
        if delta_md:
            return delta_md
        delta_json = read_text(out_dir / "delta_plan.json", default="").strip()
        if delta_json:
            return "# Delta Patch Plan\n\n```json\n" + delta_json + "\n```"
    return fallback


def _should_use_workspace_coding_rag(continuation: dict[str, Any]) -> bool:
    return bool(_workspace_coding_rag_policy(continuation)["use_rag"])


def _workspace_coding_rag_policy(continuation: dict[str, Any]) -> dict[str, Any]:
    return evaluate_rag_policy(
        continuation,
        task="workspace_code_writing",
        is_first_trial=not bool(continuation.get("source_trial_id")),
    )


def _is_active_axis_refinement(continuation: dict[str, Any]) -> bool:
    decision_context = continuation.get("decision_context") if isinstance(continuation.get("decision_context"), dict) else {}
    active_axis = str(decision_context.get("active_axis") or "").strip()
    try:
        attempt_count = int(decision_context.get("axis_attempt_count") or 0)
    except (TypeError, ValueError):
        attempt_count = 0
    try:
        attempt_limit = int(decision_context.get("axis_attempt_limit") or 3)
    except (TypeError, ValueError):
        attempt_limit = 3
    return bool(active_axis and attempt_count < attempt_limit)


def render_workspace_coding_request(handoff: dict[str, Any], next_experiment: str) -> str:
    lines = [
        f"# {handoff['trial_id']} Workspace Coding Agent Request",
        "",
        "## Objective",
        "",
        handoff["objective"],
        "",
        f"- competition: {handoff['competition']}",
        f"- trial_id: {handoff['trial_id']}",
        f"- request_id: {handoff['request_id']}",
        f"- project_root: {handoff['project_root']}",
        f"- continuation_mode: {handoff['continuation_mode']}",
        f"- source_trial_id: {handoff.get('source_trial_id')}",
        f"- code_base_trial_id: {handoff.get('code_base_trial_id')}",
        f"- pending_human_review: {handoff['pending_human_review']}",
        f"- edit_mode: {handoff.get('edit_policy', {}).get('mode')}",
        "",
        "## Input Context Files",
        "",
    ]
    lines.extend(f"- {item}" for item in handoff["context_files"] or ["None"])
    runtime_failure = handoff.get("runtime_failure_context")
    if isinstance(runtime_failure, dict) and runtime_failure:
        lines.extend(
            [
                "",
                "## Runtime Repair Context",
                "",
                "- The previous code-writing attempt was applied, but workspace execution failed.",
                "- Repair the current workspace code only. Preserve the planned experiment change and do not restore the base trial.",
                "```json",
                json.dumps(runtime_failure, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    retrieval_context = handoff.get("retrieval_context", {})
    data_card_summary = handoff.get("data_card_summary", {})
    if retrieval_context:
        lines.extend(
            [
                "",
                "## RAG Context Pack",
                "",
                f"- task: {retrieval_context.get('task')}",
                f"- documents: {retrieval_context.get('document_count')}",
                f"- skipped: {retrieval_context.get('skipped')}",
                f"- skip_reason: {retrieval_context.get('skip_reason')}",
                f"- context_pack: `{retrieval_context.get('context_pack_md_file')}`",
                f"- manifest: `{retrieval_context.get('retrieval_manifest_file')}`",
            ]
        )
    if data_card_summary:
        lines.extend(
            [
                "",
                "## Data Card Summary",
                "",
                "```json",
                json.dumps(data_card_summary, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.extend(["", "## Allowed External Write Paths", ""])
    lines.extend(f"- {item}" for item in handoff["allowed_write_paths"] or ["None"])
    edit_policy = handoff.get("edit_policy", {})
    lines.extend(
        [
            "",
            "## Edit Policy",
            "",
            f"- mode: {edit_policy.get('mode')}",
            f"- prefer_patch_updates: {edit_policy.get('prefer_patch_updates')}",
            f"- allow_full_file_updates: {edit_policy.get('allow_full_file_updates')}",
            f"- restore_base_before_patch: {edit_policy.get('restore_base_before_patch')}",
            f"- base_code_source: {edit_policy.get('base_code_source')}",
            "- When base_code_source is present, treat that source trial code as the authoritative starting point.",
            "- For patch mode, return `patch_updates` with exact `find` text and replacement text.",
            "- Do not return whole-file `file_updates` in patch mode unless the policy explicitly allows it.",
        ]
    )
    lines.extend(["", "## Forbidden External Paths", ""])
    lines.extend(f"- {item}" for item in handoff["forbidden_paths"] or ["None"])
    lines.extend(
        [
            "",
            "## Execution Constraints",
            "",
            "- Do not run training.",
            "- Do not submit to any competition platform.",
            "- Do not edit data, metrics, submission, or output artifacts.",
            "- Do not write outside the allowed external write paths.",
            "- If a base trial code snapshot is declared, do not preserve rejected changes from later failed trials.",
            "- Never fabricate, synthesize, or hardcode placeholder train/test data as a fallback when an expected "
            "file (e.g. data/train.csv, data/test.csv) is missing. A trial that raises a clear error is always "
            "correct over one that silently substitutes made-up data to produce a plausible-looking metric or "
            "submission -- a fabricated result is worse than a visible failure because it hides the real problem.",
            "- The actual data file/folder layout for this competition is listed under 'Data Card Summary' below "
            "(and in the competition's data_notes.md, if provided) -- read code against those real paths, not "
            "against a conventional train.csv/test.csv name you assume exists. If the data is split across many "
            "per-sample files or separate feature/label files, write code that loads and joins them accordingly.",
            "- If this trial changes the prediction/inference algorithm (e.g. in predict_step.py), the local "
            "validation/scoring logic (e.g. in test_step.py) must be updated to use the exact same prediction "
            "logic -- ideally by having both call one shared function (e.g. in src/) rather than each keeping its "
            "own separate copy. A local score computed from validation code that still runs the OLD algorithm "
            "does not reflect what the submission actually contains, silently making the local CV score "
            "meaningless -- this must never happen even when the change is described as small or low-risk.",
            "",
            "## Validation Commands",
            "",
        ]
    )
    lines.extend(f"```powershell\n{command}\n```" for command in handoff["validation_commands"] or ["No validation command declared."])
    metrics_contract = handoff["metrics_output_contract"]
    lines.extend(
        [
            "",
            "## Metrics Output Contract",
            "",
            f"- path: {metrics_contract['path']}",
            f"- score_key: {metrics_contract['score_key']}",
            "- required_keys:",
        ]
    )
    lines.extend(f"  - {field}" for field in metrics_contract["required_keys"])
    lines.extend(["- notes:"])
    lines.extend(f"  - {note}" for note in metrics_contract["notes"])
    artifact_policy = handoff.get("artifact_policy", {})
    if artifact_policy:
        lines.extend(
            [
                "",
                "## Artifact Policy",
                "",
                "- Metrics, submission, code snapshot, and pipeline summary are the primary trial memory.",
                "- Do not persist trained model/checkpoint artifacts by default.",
                "- Persist a model only when the policy allows it and record the reason in your summary or metrics metadata.",
                "```json",
                json.dumps(artifact_policy, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    required = handoff["required_output"]
    lines.extend(
        [
            "",
            "## Required Result Contract",
            "",
            f"- json_file: {required['json_file']}",
            f"- markdown_file: {required['markdown_file']}",
            f"- status_values: {', '.join(required['status_values'])}",
            f"- next_action: {required['next_action']}",
            "- required_fields:",
        ]
    )
    lines.extend(f"  - {field}" for field in required["required_fields"])
    if next_experiment:
        lines.extend(["", "## Next Experiment", "", next_experiment])
    lines.append("")
    return "\n".join(lines)


def _context_files(competition: str, trial_id: str) -> list[str]:
    out_dir = trial_dir(competition, trial_id)
    continuation = _load_json_object(out_dir / "continuation_context.json") or {}
    if continuation.get("source_trial_id") and (out_dir / "delta_plan.json").exists():
        candidates = [
            "delta_plan.json",
            "delta_plan.md",
        ]
        return [f"experiments/{competition}/{trial_id}/{name}" for name in candidates if (out_dir / name).exists()]
    candidates = [
        "next_experiment.md",
        "continuation_context.json",
        "continuation_context.md",
    ]
    return [f"experiments/{competition}/{trial_id}/{name}" for name in candidates if (out_dir / name).exists()]


def _write_workspace_context_snapshot(
    competition: str,
    trial_id: str,
    profile: dict[str, Any],
    continuation: dict[str, Any],
    *,
    expanded_snapshot: bool = False,
    base_will_be_restored: bool = False,
) -> list[str]:
    out_dir = trial_dir(competition, trial_id)
    source_trial_id = continuation.get("recommended_base_trial") or continuation.get("source_trial_id")
    is_delta_refinement = bool(source_trial_id and (out_dir / "delta_plan.json").exists())
    delta_plan = _load_json_object(out_dir / "delta_plan.json") if is_delta_refinement else {}
    snapshot_limits = _workspace_snapshot_limits(
        bool(source_trial_id),
        is_delta_refinement,
        expanded_snapshot=expanded_snapshot,
    )
    content = render_workspace_context_snapshot(
        competition,
        trial_id,
        profile=profile,
        source_trial_id=str(source_trial_id) if source_trial_id else None,
        delta_plan=delta_plan if isinstance(delta_plan, dict) else {},
        compact_for_delta=is_delta_refinement,
        # When the base snapshot is restored over the workspace before the
        # patch is applied, the current workspace body is guaranteed to be
        # discarded -- showing it as patch context only invites find strings
        # that can never match.
        include_current_code_body=expanded_snapshot and not base_will_be_restored,
        base_will_be_restored=base_will_be_restored,
        **snapshot_limits,
    )
    path = out_dir / "workspace_context_snapshot.md"
    write_text(path, content)
    return [f"experiments/{competition}/{trial_id}/workspace_context_snapshot.md"]


def _workspace_snapshot_limits(
    has_source_trial: bool,
    is_delta_refinement: bool,
    *,
    expanded_snapshot: bool = False,
) -> dict[str, int]:
    if expanded_snapshot:
        return {"max_files": 10, "max_chars_per_file": 16000, "max_total_chars": 52000}
    if not has_source_trial:
        return {"max_files": 5, "max_chars_per_file": 900, "max_total_chars": 2600}
    if is_delta_refinement:
        return {"max_files": 2, "max_chars_per_file": 8500, "max_total_chars": 9500}
    return {"max_files": 3, "max_chars_per_file": 13000, "max_total_chars": 15500}


def render_workspace_context_snapshot(
    competition: str,
    trial_id: str,
    *,
    profile: dict[str, Any],
    source_trial_id: str | None,
    delta_plan: dict[str, Any] | None = None,
    compact_for_delta: bool = False,
    include_current_code_body: bool = False,
    base_will_be_restored: bool = False,
    max_files: int = 16,
    max_chars_per_file: int = 3000,
    max_total_chars: int = 12000,
) -> str:
    lines = [
        f"# {trial_id} Workspace Context Snapshot",
        "",
        "This file gives the coding agent the base trial code, current workspace code, and previous trial evidence.",
        "When a recommended base trial code snapshot is present, treat it as the authoritative starting point.",
        "Use later failed trials only as negative evidence; do not preserve their rejected code changes.",
        "",
    ]
    if source_trial_id:
        lines.extend(
            [
                "## Recommended Base Trial Code Snapshot",
                "",
                f"- source_trial_id: {source_trial_id}",
                "- Use this section as the primary code reference for continuation patches.",
                "",
            ]
        )
        source_sections = _source_trial_code_sections(
            competition,
            source_trial_id,
            delta_plan=delta_plan if isinstance(delta_plan, dict) else {},
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
            max_total_chars=max_total_chars,
        )
        lines.extend(source_sections or ["- No saved source trial code snapshot was found."])
        if source_sections and not include_current_code_body:
            lines.extend(["", "## Current Workspace Code Inventory", ""])
            lines.extend(_current_code_inventory(profile, base_will_be_restored=base_will_be_restored))
        else:
            lines.extend(
                [
                    "",
                    "## Current Workspace Code",
                    "",
                    "- Used as exact patch context because no saved base-trial code snapshot was available or an expanded retry was requested.",
                    "",
                ]
            )
            current_sections = _current_code_sections(
                profile,
                max_files=max_files,
                max_chars_per_file=max_chars_per_file,
                max_total_chars=max_total_chars,
                delta_plan=delta_plan,
            )
            lines.extend(current_sections or ["- No readable current workspace code files were found."])
    else:
        lines.extend(["## Current Project Code", ""])
        code_sections = _current_code_sections(
            profile,
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
            max_total_chars=max_total_chars,
            delta_plan=delta_plan,
        )
        lines.extend(code_sections or ["- No readable code files were found in the allowed write scope."])
    if compact_for_delta:
        lines.extend(
            [
                "",
                "## Previous Trial Evidence",
                "",
                "- Omitted for compact delta patch mode. Use delta_plan and decision card outputs for trial strategy.",
            ]
        )
    else:
        lines.extend(["", "## Previous Trial Evidence", ""])
        lines.extend(_previous_trial_evidence(competition, source_trial_id))
    lines.append("")
    return "\n".join(lines)


def _previous_trial_evidence(competition: str, source_trial_id: str | None) -> list[str]:
    if not source_trial_id:
        return ["- No source trial was declared."]
    source_dir = trial_dir(competition, source_trial_id)
    lines = [f"- source_trial_id: {source_trial_id}"]
    for name in [
        "decision_card.md",
        "internal/decision_card.json",
        "metrics.json",
    ]:
        path = source_dir / name
        text = read_text(path, default="").strip()
        if not text:
            continue
        limit = 1200 if name.endswith(".md") else 1600
        lines.extend([f"", f"### {name}", "", _fenced_content(name, text[:limit])])
    if len(lines) == 1:
        lines.append("- No previous trial evidence files were found.")
    return lines


def _source_trial_code_sections(
    competition: str,
    source_trial_id: str | None,
    *,
    delta_plan: dict[str, Any] | None = None,
    max_files: int,
    max_chars_per_file: int,
    max_total_chars: int,
) -> list[str]:
    files = _source_trial_code_files(
        competition,
        source_trial_id,
        delta_plan=delta_plan,
    )
    sections: list[str] = []
    total = 0
    for relative, text in files[:max_files]:
        per_file_limit = _context_file_limit(relative, max_chars_per_file)
        snippet = _source_code_snippet(relative, text, per_file_limit, delta_plan=delta_plan)
        if total + len(snippet) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            snippet = snippet[:remaining]
        total += len(snippet)
        sections.extend([f"### {relative}", "", _fenced_content(relative, snippet), ""])
    return sections


def _source_trial_code_files(
    competition: str,
    source_trial_id: str | None,
    *,
    delta_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    if not source_trial_id:
        return []
    source_dir = trial_dir(competition, source_trial_id)
    snapshot = load_trial_code_snapshot(source_dir)
    if snapshot:
        return sorted(
            [(relative, text) for relative, text in snapshot if relative.endswith(".py")],
            key=lambda item: _delta_code_file_priority(Path(item[0]), delta_plan, text=item[1]),
        )
    code_root = source_dir / "user_view" / "code"
    files: list[tuple[str, str]] = []
    if code_root.is_dir():
        unsorted_files = []
        for path in code_root.rglob("*.py"):
            relative = path.relative_to(code_root).as_posix()
            text = read_text(path, default="")
            if text:
                unsorted_files.append((relative, text))
        files = sorted(
            unsorted_files,
            key=lambda item: _delta_code_file_priority(Path(item[0]), delta_plan, text=item[1]),
        )
    if files:
        return files
    result = _load_json_object(source_dir / "workspace_coding_result.json") or _load_json_object(
        source_dir / "internal" / "workspace_coding_result.json"
    )
    updates = result.get("file_updates") if isinstance(result, dict) else []
    if not isinstance(updates, list):
        return []
    for update in updates:
        if not isinstance(update, dict):
            continue
        relative = _safe_relative_code_path(update.get("path"))
        content = update.get("content")
        if relative and isinstance(content, str):
            files.append((relative, content))
    return sorted(
        files,
        key=lambda item: _delta_code_file_priority(Path(item[0]), delta_plan, text=item[1]),
    )


def _source_trial_code_source_label(competition: str, source_trial_id: str | None) -> str | None:
    if not source_trial_id:
        return None
    source_dir = trial_dir(competition, source_trial_id)
    if load_trial_code_snapshot(source_dir):
        return f"experiments/{competition}/{source_trial_id}/internal/code_snapshot"
    if (source_dir / "user_view" / "code").is_dir():
        return f"experiments/{competition}/{source_trial_id}/user_view/code"
    result_path = source_dir / "workspace_coding_result.json"
    if result_path.exists():
        return f"experiments/{competition}/{source_trial_id}/workspace_coding_result.json"
    internal_result_path = source_dir / "internal" / "workspace_coding_result.json"
    if internal_result_path.exists():
        return f"experiments/{competition}/{source_trial_id}/internal/workspace_coding_result.json"
    return None


def _current_code_sections(
    profile: dict[str, Any],
    *,
    max_files: int,
    max_chars_per_file: int,
    max_total_chars: int,
    delta_plan: dict[str, Any] | None = None,
) -> list[str]:
    project_root = Path(str(profile.get("project_root", "")))
    if not project_root.is_dir():
        return []
    files = _allowed_readable_files(
        project_root,
        profile.get("write_scope", {}).get("allowed", []),
        max_files=max_files,
        delta_plan=delta_plan,
    )
    sections: list[str] = []
    total = 0
    for path in files:
        relative = path.relative_to(project_root).as_posix()
        text = read_text(path, default="")
        if not text:
            continue
        per_file_limit = _context_file_limit(relative, max_chars_per_file)
        snippet = text[:per_file_limit]
        if total + len(snippet) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            snippet = snippet[:remaining]
        total += len(snippet)
        sections.extend([f"### {relative}", "", _fenced_content(relative, snippet), ""])
    return sections


def _current_code_inventory(profile: dict[str, Any], *, base_will_be_restored: bool = False) -> list[str]:
    project_root = Path(str(profile.get("project_root", "")))
    if not project_root.is_dir():
        return ["- Current workspace root is not readable."]
    files = _allowed_readable_files(project_root, profile.get("write_scope", {}).get("allowed", []), max_files=20)
    if not files:
        return ["- No readable code files were found in the allowed write scope."]
    lines = [
        "- Current workspace may contain rejected later-trial changes.",
        "- The recommended base trial snapshot above is authoritative for patch find/replace text.",
    ]
    if base_will_be_restored:
        lines.append(
            "- These files WILL BE OVERWRITTEN with the base trial snapshot before your patch is applied. "
            "Never copy find text from them; local variable names and helper structure may differ from the base."
        )
    lines.extend(f"- {path.relative_to(project_root).as_posix()}" for path in files)
    return lines


def _context_file_limit(relative_path: str, default_limit: int) -> int:
    normalized = relative_path.replace("\\", "/")
    if Path(normalized).name in {"baseline.py", "pipeline.py", "model.py"}:
        return max(default_limit, 14000)
    return default_limit


def _source_code_snippet(
    relative_path: str,
    text: str,
    max_chars: int,
    *,
    delta_plan: dict[str, Any] | None = None,
) -> str:
    normalized = relative_path.replace("\\", "/")
    if normalized.endswith(".py") and len(text) > max_chars:
        compact = _compact_python_code_for_patch(text, max_chars, delta_plan=delta_plan)
        if compact:
            return compact
    if normalized.endswith("src/baseline.py") and len(text) > max_chars:
        return _compact_baseline_code_for_patch(text, max_chars, delta_plan=delta_plan)
    return text[:max_chars]


def _compact_python_code_for_patch(
    text: str,
    max_chars: int,
    *,
    delta_plan: dict[str, Any] | None = None,
) -> str:
    if not isinstance(delta_plan, dict):
        return ""
    required = {
        str(item).strip()
        for item in delta_plan.get("required_code_symbols", [])
        if str(item).strip()
    }
    if not required:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.splitlines()
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    selected = set(required)
    for node in top_level:
        if node.name not in required:
            continue
        selected.update(
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        )
    first_definition = min((node.lineno for node in top_level), default=len(lines) + 1) - 1
    output = [
        "# Compacted Python context selected from delta_plan.required_code_symbols.",
        *lines[:first_definition],
    ]
    for node in top_level:
        if node.name not in selected:
            continue
        start = max(node.lineno - 1, 0)
        end = getattr(node, "end_lineno", node.lineno)
        output.extend(["", f"# --- {node.name} ---", *lines[start:end]])
    compact = "\n".join(output).strip() + "\n"
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "\n# ... symbol context truncated at prompt budget ...\n"


def _compact_baseline_code_for_patch(text: str, max_chars: int, *, delta_plan: dict[str, Any] | None = None) -> str:
    wanted = _wanted_baseline_blocks_for_patch(delta_plan)
    lines = text.splitlines()
    blocks: list[tuple[str, int, int]] = []
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        if line.startswith("def "):
            name = line[4:].split("(", 1)[0].strip()
            starts.append((name, index))
        elif line.startswith("class "):
            name = line[6:].split("(", 1)[0].split(":", 1)[0].strip()
            starts.append((name, index))
    for pos, (name, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        blocks.append((name, start, end))

    first_def = starts[0][1] if starts else min(len(lines), 80)
    selected_lines = [
        "# Compacted baseline.py for continuation patching.",
        "# Contains imports/constants and only the code blocks needed for safe delta find/replace patches.",
        "",
        *lines[:first_def],
    ]
    for name, start, end in blocks:
        is_class = lines[start].startswith("class ")
        if name in wanted or (not delta_plan and is_class):
            selected_lines.extend(["", f"# --- {name} ---", *lines[start:end]])
    compact = "\n".join(selected_lines).strip() + "\n"
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "\n# ... compacted baseline.py truncated at prompt budget ...\n"


def _wanted_baseline_blocks_for_patch(delta_plan: dict[str, Any] | None = None) -> set[str]:
    default = {
        "FeatureBuilder",
        "build_features",
        "build_feature_matrix",
        "build_preprocessor",
        "build_pipeline",
        "load_data",
        "make_submission_frame",
        "pipeline_summary",
        "run_experiment",
        "run_prediction",
        "load_config",
        "check_required_data",
        "train",
        "predict",
        "_make_features",
        "_build_pipeline",
        "_single_holdout_split",
    }
    if not isinstance(delta_plan, dict) or not delta_plan:
        return default
    text = json.dumps(delta_plan, ensure_ascii=False).lower()
    wanted = {"build_pipeline", "pipeline_summary", "_build_pipeline"}
    wanted.update(
        token
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,80}\b", text)
        if "_" in token
    )
    if any(keyword in text for keyword in ["feature", "numeric", "categorical", "representation"]):
        wanted.update({"FeatureBuilder", "build_features", "build_feature_matrix", "_make_features"})
    if any(keyword in text for keyword in ["imput", "missing", "preprocess", "scaler", "encoder"]):
        wanted.update({"build_preprocessor", "build_pipeline", "_build_pipeline"})
    if any(keyword in text for keyword in ["model", "classifier", "regressor", "regularization", "solver"]):
        wanted.update({"build_pipeline", "_build_pipeline", "pipeline_summary"})
    if any(keyword in text for keyword in ["split", "validation", "cv", "holdout", "metric", "score"]):
        wanted.update({"run_experiment", "_single_holdout_split", "load_data", "pipeline_summary"})
    if any(keyword in text for keyword in ["submission", "predict", "output", "inference"]):
        wanted.update({"run_prediction", "make_submission_frame", "load_data", "pipeline_summary"})
    return wanted


def _allowed_readable_files(
    project_root: Path,
    allowed_paths: list[str],
    *,
    max_files: int,
    delta_plan: dict[str, Any] | None = None,
) -> list[Path]:
    collected: list[Path] = []
    for item in allowed_paths:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = item.replace("\\", "/").strip("/")
        if not normalized or ".." in Path(normalized).parts:
            continue
        path = project_root / normalized
        if path.is_file() and _is_text_code_file(path):
            collected.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and _is_text_code_file(child):
                    collected.append(child)
                if len(collected) >= max_files:
                    return _sorted_by_code_priority(_unique_paths(collected), delta_plan)[:max_files]
    return _sorted_by_code_priority(_unique_paths(collected), delta_plan)[:max_files]


def _sorted_by_code_priority(paths: list[Path], delta_plan: dict[str, Any] | None) -> list[Path]:
    if not delta_plan:
        return sorted(paths, key=_code_file_priority)
    return sorted(
        paths,
        key=lambda path: _delta_code_file_priority(path, delta_plan, text=read_text(path, default="")),
    )


def _safe_relative_code_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    normalized = value.replace("\\", "/").strip("/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        return ""
    if path.suffix.lower() != ".py":
        return ""
    return path.as_posix()


def _code_file_priority(path: Path) -> tuple[int, str]:
    normalized = path.as_posix()
    if path.name in {"baseline.py", "pipeline.py", "model.py"} and "/src/" in f"/{normalized}":
        return (0, normalized)
    if path.name in {"train_step.py", "predict_step.py", "test_step.py"}:
        return (1, normalized)
    return (2, normalized)


def _delta_code_file_priority(
    path: Path,
    delta_plan: dict[str, Any] | None,
    *,
    text: str | None = None,
) -> tuple[int, str]:
    normalized = path.as_posix().lstrip("./")
    targets = _delta_target_code_paths(delta_plan)
    for index, target in enumerate(targets):
        if normalized == target or normalized.endswith(f"/{target}"):
            return (-100 + index, normalized)
    # No file is explicitly named anywhere in the plan (not even inside
    # change_details/required_code_symbols). Rather than falling straight
    # through to the generic alphabetical tie-break -- which can silently
    # exclude the one file that actually needs to change once the snapshot's
    # max_files budget is tight -- check whether this file's code actually
    # contains a distinctive identifier (class/library name, or a named
    # function) the plan talks about, and if so treat it like an explicit
    # target.
    if text and _references_plan_identifier(text, delta_plan):
        return (-50, normalized)
    return _code_file_priority(path)


_CAMEL_CASE_IDENTIFIER = re.compile(r"\b[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*\b")


def _plan_identifiers(delta_plan: dict[str, Any] | None) -> set[str]:
    if not isinstance(delta_plan, dict):
        return set()
    identifiers: set[str] = set()
    for symbol in delta_plan.get("required_code_symbols") or []:
        if isinstance(symbol, str) and symbol.strip():
            identifiers.add(symbol.strip())
    for field in ("primary_change_axis", "plan_title", "rationale", "change_details", "keep_unchanged"):
        value = delta_plan.get(field)
        texts = value if isinstance(value, list) else [value] if isinstance(value, str) else []
        for item in texts:
            if isinstance(item, str):
                identifiers.update(_CAMEL_CASE_IDENTIFIER.findall(item))
    return identifiers


def _references_plan_identifier(text: str, delta_plan: dict[str, Any] | None) -> bool:
    for identifier in _plan_identifiers(delta_plan):
        if re.search(rf"\bdef\s+{re.escape(identifier)}\s*\(", text) or identifier in text:
            return True
    return False


def _delta_target_code_paths(delta_plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(delta_plan, dict):
        return []
    targets: list[str] = []
    # code_change_targets is the intended field, but the planner sometimes
    # names the file inside change_details/required_code_symbols instead
    # (e.g. "Code Change Targets: train_step.py: replace ...") without also
    # copying it into code_change_targets itself. Scan those fields too so a
    # misplaced-but-present filename still counts as an explicit target.
    for field in ("code_change_targets", "change_details", "required_code_symbols"):
        for item in delta_plan.get(field, []) or []:
            if not isinstance(item, str):
                continue
            for match in re.findall(r"[A-Za-z0-9_./\\-]+\.py", item):
                normalized = match.replace("\\", "/").strip("./")
                if normalized and normalized not in targets:
                    targets.append(normalized)
    return targets


def _plan_find_target_issues(
    delta_plan: dict[str, Any],
    base_code_files: list[tuple[str, str]],
) -> list[str]:
    """Check the plan's own "File:"/"Find:" hints against the base trial's
    actual code before any code-writing LLM call is spent turning them into
    a patch. A plan can assume a change from an earlier, unaccepted attempt
    at the same axis is already present in the base trial's code -- it is
    not, since only accepted trials become the new base -- and this catches
    that mismatch deterministically, without an LLM call, before it wastes
    one on a patch that can never apply.
    """
    targets = delta_plan.get("code_change_targets")
    if not isinstance(targets, list) or not base_code_files:
        return []
    base_by_path = dict(base_code_files)
    issues: list[str] = []
    current_file: str | None = None
    for item in targets:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        lowered = stripped.lower()
        if lowered.startswith("file:"):
            current_file = stripped.split(":", 1)[1].strip()
            continue
        if not lowered.startswith("find:") or not current_file:
            continue
        find_text = stripped.split(":", 1)[1].strip()
        if not find_text:
            continue
        matched_text = None
        for relative, text in base_by_path.items():
            if relative == current_file or relative.endswith(f"/{current_file}"):
                matched_text = text
                break
        if matched_text is not None and find_text not in matched_text:
            issues.append(f"plan_find_target_missing_in_base_code:{current_file}")
    return issues


def _is_text_code_file(path: Path) -> bool:
    return path.suffix.lower() == ".py"


def _fenced_content(name: str, text: str) -> str:
    suffix = Path(name).suffix.lower()
    language = {
        ".py": "python",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
    }.get(suffix, "")
    return f"```{language}\n{text.rstrip()}\n```"


def _unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_data_card_summary(competition: str) -> dict[str, Any]:
    card = _load_json_object(competition_dir(competition) / "competition_data_card.json")
    if not card:
        card = _load_json_object(competition_dir(competition) / "data_profile.json")
    if not card:
        return {}
    recommendation = card.get("baseline_recommendation", {}) if isinstance(card.get("baseline_recommendation"), dict) else {}
    return {
        "task_type": card.get("task_type"),
        "target_column": card.get("target_column"),
        "id_column": card.get("id_column"),
        "submission_prediction_column": card.get("submission_prediction_column"),
        "train_file": card.get("train_file"),
        "test_file": card.get("test_file"),
        "sample_submission_file": card.get("sample_submission_file"),
        # Datasets that ship one CSV per sample under data/<split>/ have no
        # train_file/test_file at all; without these keys the code writer only
        # saw nulls and fell back to assuming a conventional flat train.csv.
        "dataset_layout": card.get("dataset_layout"),
        "train_dir": card.get("train_dir"),
        "test_dir": card.get("test_dir"),
        "directory_datasets": [
            {
                "name": group.get("name"),
                "role": group.get("role"),
                "file_count": group.get("file_count"),
                "filename_pattern": group.get("filename_pattern"),
                "example_files": group.get("example_files", []),
                "per_file_columns": group.get("per_file_columns", []),
                "sample_id_source": group.get("sample_id_source"),
                "id_matched_files": group.get("id_matched_files", []),
                "notes": group.get("notes", []),
            }
            for group in (card.get("directory_datasets") or [])
            if isinstance(group, dict)
        ],
        "include_features_first": recommendation.get("include_features_first", []),
        "defer_features_first": recommendation.get("defer_features_first", []),
        "exclude_columns": recommendation.get("exclude_columns", []),
        "preferred_model_families": recommendation.get("preferred_model_families", []),
        "avoid_first_trial": recommendation.get("avoid_first_trial", []),
    }


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _compact_retrieval_context(context: dict[str, Any]) -> dict[str, Any]:
    if not context:
        return {}
    return {
        "task": context.get("task"),
        "query": context.get("query"),
        "document_count": context.get("document_count"),
        "skipped": bool(context.get("skipped")),
        "skip_reason": context.get("skip_reason"),
        "policy": context.get("policy"),
        "context_pack_file": context.get("context_pack_file"),
        "context_pack_md_file": context.get("context_pack_md_file"),
        "retrieval_manifest_file": context.get("retrieval_manifest_file"),
        "documents": [
            {
                "source_path": doc.get("source_path"),
                "source_kind": doc.get("source_kind"),
                "trial_id": doc.get("trial_id"),
                "score": doc.get("score"),
            }
            for doc in context.get("documents", [])
        ],
    }
