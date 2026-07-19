from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .code_snapshot import load_trial_code_snapshot
from .execution_profile import load_execution_profile, validate_execution_profile
from .paths import competition_dir, trial_dir
from .policies import load_policy
from .retrieval.context_pack import build_context_pack
from .store import read_text, write_text


def prepare_workspace_coding_handoff(competition: str, trial_id: str) -> dict[str, Any]:
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
    )
    write_text(out_dir / "workspace_coding_handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    if status == "ready":
        write_text(out_dir / "workspace_coding_agent_request.md", render_workspace_coding_request(handoff, next_experiment))
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
) -> dict[str, Any]:
    artifacts = profile.get("artifacts", {}) if profile else {}
    scope = profile.get("write_scope", {}) if profile else {}
    allowed = list(scope.get("allowed", [])) if isinstance(scope, dict) else []
    forbidden = list(scope.get("forbidden", [])) if isinstance(scope, dict) else []
    forbidden = _unique([*forbidden, *artifacts.get("metrics", []), *artifacts.get("submission", [])])
    validation_commands = []
    commands = profile.get("commands", {}) if profile else {}
    if isinstance(commands, dict):
        validation_commands = list(commands.get("test", []))
    edit_mode = _edit_mode_for_continuation(continuation)
    source_trial_id = str(continuation.get("source_trial_id") or "") or None
    base_code_files = _source_trial_code_files(competition, source_trial_id)
    restore_base_before_patch = bool(source_trial_id and base_code_files)
    next_action = "send-to-workspace-coding-agent" if status == "ready" else "resolve-workspace-handoff-blockers"
    metrics_paths = artifacts.get("metrics", []) if isinstance(artifacts, dict) else []
    metrics_path = metrics_paths[0] if metrics_paths else "outputs/metrics.json"
    context_files = _context_files(competition, trial_id)
    artifact_policy = load_policy("artifact_policy")
    data_card_summary = _load_data_card_summary(competition)
    retrieval_context: dict[str, Any] = {}
    if status == "ready" and profile:
        context_files.extend(_write_workspace_context_snapshot(competition, trial_id, profile, continuation))
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
        context_files.extend(
            [
                retrieval_context["context_pack_md_file"],
                retrieval_context["retrieval_manifest_file"],
            ]
        )
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
        "code_base_trial_id": source_trial_id,
        "pending_human_review": bool(continuation.get("pending_human_review")),
        "review_source_trial": continuation.get("review_source_trial"),
        "context_files": context_files,
        "retrieval_context": _compact_retrieval_context(retrieval_context),
        "data_card_summary": data_card_summary,
        "edit_policy": {
            "mode": edit_mode,
            "prefer_patch_updates": edit_mode == "patch_only",
            "allow_full_file_updates": edit_mode == "full_file_allowed",
            "restore_base_before_patch": restore_base_before_patch,
            "base_code_source": f"experiments/{competition}/{source_trial_id}/internal/code_snapshot"
            if restore_base_before_patch
            else None,
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
) -> list[str]:
    out_dir = trial_dir(competition, trial_id)
    source_trial_id = continuation.get("source_trial_id")
    snapshot_limits = (
        {"max_files": 8, "max_chars_per_file": 1200, "max_total_chars": 3500}
        if not source_trial_id
        else {"max_files": 5, "max_chars_per_file": 2400, "max_total_chars": 11000}
    )
    content = render_workspace_context_snapshot(
        competition,
        trial_id,
        profile=profile,
        source_trial_id=str(source_trial_id) if source_trial_id else None,
        **snapshot_limits,
    )
    path = out_dir / "workspace_context_snapshot.md"
    write_text(path, content)
    return [f"experiments/{competition}/{trial_id}/workspace_context_snapshot.md"]


def render_workspace_context_snapshot(
    competition: str,
    trial_id: str,
    *,
    profile: dict[str, Any],
    source_trial_id: str | None,
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
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
            max_total_chars=max_total_chars,
        )
        lines.extend(source_sections or ["- No saved source trial code snapshot was found."])
        lines.extend(
            [
                "",
                "## Current Workspace Code",
                "",
                "- This may contain rejected changes from later failed trials. Use it only to understand current filesystem drift.",
                "",
            ]
        )
    else:
        lines.extend(["## Current Project Code", ""])
    code_sections = _current_code_sections(
        profile,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        max_total_chars=max_total_chars,
    )
    lines.extend(code_sections or ["- No readable code files were found in the allowed write scope."])
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
    max_files: int,
    max_chars_per_file: int,
    max_total_chars: int,
) -> list[str]:
    files = _source_trial_code_files(competition, source_trial_id)
    sections: list[str] = []
    total = 0
    for relative, text in files[:max_files]:
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


def _source_trial_code_files(competition: str, source_trial_id: str | None) -> list[tuple[str, str]]:
    if not source_trial_id:
        return []
    source_dir = trial_dir(competition, source_trial_id)
    snapshot = load_trial_code_snapshot(source_dir)
    if snapshot:
        return sorted(
            [(relative, text) for relative, text in snapshot if relative.endswith(".py")],
            key=lambda item: _code_file_priority(Path(item[0])),
        )
    code_root = source_dir / "user_view" / "code"
    files: list[tuple[str, str]] = []
    if code_root.is_dir():
        for path in sorted(code_root.rglob("*.py"), key=lambda item: _code_file_priority(item)):
            relative = path.relative_to(code_root).as_posix()
            text = read_text(path, default="")
            if text:
                files.append((relative, text))
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
    return sorted(files, key=lambda item: _code_file_priority(Path(item[0])))


def _current_code_sections(
    profile: dict[str, Any],
    *,
    max_files: int,
    max_chars_per_file: int,
    max_total_chars: int,
) -> list[str]:
    project_root = Path(str(profile.get("project_root", "")))
    if not project_root.is_dir():
        return []
    files = _allowed_readable_files(project_root, profile.get("write_scope", {}).get("allowed", []), max_files=max_files)
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


def _context_file_limit(relative_path: str, default_limit: int) -> int:
    normalized = relative_path.replace("\\", "/")
    if normalized.endswith("src/baseline.py"):
        return 9500
    return default_limit


def _allowed_readable_files(project_root: Path, allowed_paths: list[str], *, max_files: int) -> list[Path]:
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
                    return _unique_paths(collected)
    return sorted(_unique_paths(collected), key=_code_file_priority)[:max_files]


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
    if normalized.endswith("/src/baseline.py") or normalized.endswith("src/baseline.py"):
        return (0, normalized)
    if path.name in {"train_step.py", "predict_step.py", "test_step.py"}:
        return (1, normalized)
    return (2, normalized)


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
