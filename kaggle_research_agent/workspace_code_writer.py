from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from . import paths
from .agents.code_writer_adapter import CodeWriterClient, create_llm_client, provider_log_name
from .agents.memory import log_decision, log_token_usage
from .agents.policy_gate import should_call_llm
from .code_snapshot import load_trial_code_snapshot, save_trial_code_snapshot
from .paths import trial_dir
from .store import read_text, write_text
from .trial_artifacts import trial_artifact_path


def run_workspace_code_writer(
    competition: str,
    trial_id: str,
    *,
    client: CodeWriterClient | None = None,
    model: str = "gpt-5",
    provider: str = "openai",
    allow_api: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff = _load_handoff(out_dir)
    if handoff.get("status") != "ready":
        return _write_blocked_result(competition, trial_id, handoff, ["workspace_coding_handoff_not_ready"])

    token_decision = should_call_llm(
        "code_writing",
        competition=competition,
        trial_id=trial_id,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
    )
    if token_decision["decision"] != "call_llm":
        return _write_blocked_result(competition, trial_id, handoff, ["token_policy_blocked"], token_decision)

    if client is None:
        if not allow_api:
            return _write_blocked_result(competition, trial_id, handoff, ["api_call_not_enabled"], token_decision)
        client = create_llm_client(provider)

    payload = build_workspace_code_writer_payload(handoff, model=model)
    write_text(out_dir / "workspace_coding_api_request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    try:
        raw_response = client.create_response(payload)
    except RuntimeError as error:
        return _write_blocked_result(competition, trial_id, handoff, [f"api_error:{error}"], token_decision)
    write_text(out_dir / "workspace_coding_api_response.json", json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n")
    token_usage = _record_response_token_usage(
        competition,
        trial_id,
        raw_response,
        provider=provider_log_name(provider),
        model=model,
    )

    coding_result = _normalize_coding_result(_extract_coding_result(raw_response), competition, trial_id, handoff)
    blocking_issues = list(coding_result.get("blocking_issues", []))
    blocking_issues.extend(_apply_code_updates(coding_result, handoff))
    if blocking_issues:
        coding_result["status"] = "blocked"
        coding_result["blocking_issues"] = _unique(blocking_issues)
    elif coding_result.get("status") == "completed":
        coding_result["code_snapshot"] = _save_post_write_code_snapshot(out_dir, coding_result, handoff)
    _write_coding_result(out_dir, coding_result)
    validation = validate_workspace_coding_result(competition, trial_id)
    validation["blocking_issues"] = coding_result.get("blocking_issues", [])
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_code_writer_api",
        decision=validation["status"],
        reason="Workspace code writer produced a coding result and ran scope validation.",
        evidence={
            "model": model,
            "token_decision": token_decision,
            "token_usage": token_usage,
            "changed_files": coding_result.get("changed_files", []),
            "validation_issues": validation.get("issues", []),
        },
        next_action=validation["next_action"],
    )
    return validation


def validate_workspace_coding_result(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff = _load_handoff(out_dir)
    result_path = trial_artifact_path(out_dir, "workspace_coding_result.json")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing workspace coding result: {result_path}")
    coding_result = json.loads(result_path.read_text(encoding="utf-8"))
    if "validation_results" in coding_result:
        coding_result["validation_results"] = _normalize_validation_results(coding_result.get("validation_results"))
    issues = _validate_schema(coding_result, handoff)
    issues.extend(_validate_changed_files(coding_result, handoff))
    issues.extend(_validate_file_updates(coding_result, handoff))
    issues.extend(_validate_patch_updates(coding_result, handoff))
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "accepted" if not issues and coding_result.get("status") == "completed" else "blocked",
        "issues": _unique(issues),
        "coding_result_status": coding_result.get("status"),
        "changed_files": coding_result.get("changed_files", []),
        "allowed_write_paths": handoff.get("allowed_write_paths", []),
        "forbidden_paths": handoff.get("forbidden_paths", []),
        "next_action": "run-workspace-validation-commands"
        if not issues and coding_result.get("status") == "completed"
        else "revise-workspace-code-result",
    }
    write_text(out_dir / "workspace_coding_result_validation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_coding_result_validation.md", render_workspace_coding_result_validation(result))
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_coding_result_validation",
        decision=result["status"],
        reason="Workspace coding result was checked against Execution Profile-derived scope.",
        evidence={
            "issues": result["issues"],
            "coding_result_status": result["coding_result_status"],
            "changed_files": result["changed_files"],
        },
        next_action=result["next_action"],
    )
    return result


def build_workspace_code_writer_payload(handoff: dict[str, Any], *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You are a careful code writer. Return only JSON that follows the requested schema.",
            },
            {"role": "user", "content": _build_prompt(handoff)},
        ],
    }


def render_workspace_coding_result(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Workspace Coding Result",
        "",
        f"- status: {result['status']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Summary",
        "",
        result.get("summary", ""),
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("changed_files", []) or ["None"])
    lines.extend(["", "## Blocking Issues", ""])
    lines.extend(f"- {item}" for item in result.get("blocking_issues", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def render_workspace_coding_result_validation(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Workspace Coding Result Validation",
        "",
        f"- status: {result['status']}",
        f"- coding_result_status: {result.get('coding_result_status')}",
        f"- next_action: {result['next_action']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- {item}" for item in result.get("changed_files", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _build_prompt(handoff: dict[str, Any]) -> str:
    if _is_delta_patch_handoff(handoff):
        return _build_delta_patch_prompt(handoff)
    return "\n".join(
        [
            "Implement the next workspace experiment within the declared project write scope.",
            "",
            "Return one JSON object with:",
            "- status: completed | blocked | failed",
            "- summary",
            "- changed_files: project-root-relative paths only",
            "- patch_updates: list of {path, find, replace, reason}; use for localized edits whenever possible",
            "- file_updates: list of {path, content}; only use when edit_policy allows full-file updates",
            "- validation_results: list of objects; use [] if validation was not run in the response environment",
            "- blocking_issues",
            "",
            "Important execution model:",
            "You are not expected to access the filesystem or run tools directly.",
            "Use only the provided handoff, context files, retrieved evidence, and workspace snapshot excerpts.",
            "Return code changes in patch_updates when edit_policy.mode is patch_only; this local agent will apply those updates and run validation.",
            "In patch_only mode, do not return whole-file file_updates. Use exact find text from the provided code snapshot.",
            "If edit_policy.base_code_source is present, use the Recommended Base Trial Code Snapshot as the authoritative code for patch_updates.find.",
            "For patch_updates.find, copy the complete existing line or block from the authoritative code snapshot, including all existing arguments and whitespace.",
            "Do not invent a shorter find pattern from the plan; if the exact current code is not visible, return status=blocked.",
            "If edit_policy.mode is full_file_allowed, file_updates are allowed for new baselines or missing files.",
            "Do not block merely because this API response environment has no file read/write tools.",
            "Block only when the provided context is insufficient to produce a safe allowed-path update.",
            "",
            "Output budget for continuation/patch-only coding:",
            "- Return at most 2 patch_updates. Prefer 1 patch_update when the change is local.",
            "- Each patch_update.reason must be one short sentence.",
            "- summary must be at most 2 short sentences and must name only the actual code change.",
            "- validation_results must be [] because the local agent runs validation after applying your response.",
            "- Do not include unchanged code, long explanations, markdown, duplicate fallback file_updates, or analysis prose.",
            "- If the requested change needs more than 2 localized patches, return status=blocked with blocking_issues ['too_many_patch_updates_for_budget'].",
            "",
            "Use the RAG context pack and retrieval manifest when present. Prefer concrete retrieved evidence over generic changes.",
            "Your summary should state the actual pipeline changes you implemented, not only a generic performance goal.",
            "Keep generated code compact and production-like: avoid long comments, unused helper layers, notebook-style narration, and optional files that are not required by the Execution Profile.",
            "Modify the smallest necessary file set. For a first-cycle baseline, prefer one primary pipeline module plus thin train/predict/test entry points.",
            "For first-trial tabular supervised baselines, use the competition data card/data profile when available. Prefer a stable supervised baseline over a generic schema-discovery classifier.",
            "For mixed numeric/categorical tabular classification, avoid Gaussian Naive Bayes as the first baseline unless the plan explicitly justifies it with data evidence.",
            "Do not one-hot encode raw high-cardinality free-text or identifier-like columns in the first baseline merely because they exist; defer them or engineer safe derived features.",
            "If concrete target/id/feature recommendations are present, implement those concrete columns instead of broad automatic feature inclusion.",
            "If the next experiment is a continuation_delta_plan, modify the existing pipeline incrementally.",
            "For continuation trials, implement the declared primary_change_axis and keep the keep_unchanged items fixed unless the plan explicitly says otherwise.",
            "If continuation_context.decision_context lists rejected_axes or planner_constraints, do not preserve rejected changes from the latest trial; start from the recommended base evidence and apply only the new primary axis.",
            "When edit_policy.restore_base_before_patch is true, the local agent will restore the saved base trial code before applying your patch.",
            "Do not rewrite unrelated pipeline stages just to make a fresh baseline.",
            "",
            f"Objective: {handoff.get('objective')}",
            f"Project root: {handoff.get('project_root')}",
            f"Allowed external write paths: {handoff.get('allowed_write_paths', [])}",
            f"Forbidden external paths: {handoff.get('forbidden_paths', [])}",
            "Edit policy:",
            json.dumps(handoff.get("edit_policy", {}), ensure_ascii=False, indent=2),
            f"Validation commands: {handoff.get('validation_commands', [])}",
            "Artifact policy:",
            json.dumps(_compact_artifact_policy_for_prompt(handoff.get("artifact_policy", {})), ensure_ascii=False, indent=2),
            "Do not persist trained model/checkpoint artifacts by default. If you persist one, record the allowed policy reason.",
            "",
            "Retrieval context metadata:",
            json.dumps(handoff.get("retrieval_context", {}), ensure_ascii=False, indent=2),
            "",
            "Data card summary:",
            json.dumps(handoff.get("data_card_summary", {}), ensure_ascii=False, indent=2),
            "",
            "Context files:",
            _read_context(handoff.get("context_files", []), handoff=handoff),
        ]
    )


def _build_delta_patch_prompt(handoff: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Implement one small delta patch for the next ML trial.",
            "Return one JSON object only.",
            "",
            "Required JSON fields:",
            "- status: completed | blocked | failed",
            "- summary",
            "- changed_files",
            "- patch_updates: list of {path, find, replace, reason}",
            "- file_updates: []",
            "- validation_results: []",
            "- blocking_issues",
            "",
            "Patch rules:",
            "- Use patch_updates only; do not return whole-file updates.",
            "- Copy exact find text from the authoritative code snapshot.",
            "- Change only the delta_plan candidate.",
            "- Do not repeat rejected candidates listed in delta_plan.",
            "- Keep model, split, preprocessing, and unrelated features unchanged unless delta_plan explicitly targets them.",
            "- Return at most 2 patch_updates.",
            "- If exact target code is not visible, return blocked with a precise blocking issue.",
            "",
            f"Competition: {handoff.get('competition')}",
            f"Trial: {handoff.get('trial_id')}",
            f"Base/source trial: {handoff.get('code_base_trial_id') or handoff.get('source_trial_id')}",
            f"Allowed write paths: {handoff.get('allowed_write_paths', [])}",
            f"Forbidden paths: {handoff.get('forbidden_paths', [])}",
            "Edit policy:",
            json.dumps(handoff.get("edit_policy", {}), ensure_ascii=False, indent=2),
            "",
            "Context files:",
            _read_context(handoff.get("context_files", []), handoff=handoff),
        ]
    )


def _read_context(files: list[str], *, handoff: dict[str, Any] | None = None) -> str:
    root = paths.project_root()
    chunks = []
    for item in files:
        path = root / item
        if path.exists() and path.is_file():
            if _skip_context_file_body(item):
                chunks.append(f"## {item}\nBody omitted from prompt; use the metadata above and saved file path if needed.")
                continue
            limit = _context_file_prompt_limit(item, handoff=handoff)
            text = read_text(path)
            if PurePosixPath(item.replace("\\", "/")).name == "next_experiment.md":
                text = _compact_next_experiment_markdown(text)
            chunks.append(f"## {item}\n{text[:limit]}")
    return "\n\n".join(chunks) if chunks else "No context file contents available."


def _skip_context_file_body(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name
    return name.startswith("context_pack_") or name.startswith("retrieval_manifest_")


def _context_file_prompt_limit(path: str, *, handoff: dict[str, Any] | None = None) -> int:
    name = PurePosixPath(path.replace("\\", "/")).name
    if name == "delta_plan.json":
        return 2500
    if name == "delta_plan.md":
        return 1800
    if name == "next_experiment.md":
        return 3000
    if name == "workspace_context_snapshot.md":
        if _is_delta_patch_handoff(handoff):
            return 9500
        if _is_patch_only_context(handoff):
            return 16500
        return 5200
    if name.startswith("continuation_context"):
        return 1200
    return 2500


def _is_patch_only_context(handoff: dict[str, Any] | None) -> bool:
    if not isinstance(handoff, dict):
        return False
    policy = handoff.get("edit_policy") if isinstance(handoff.get("edit_policy"), dict) else {}
    return policy.get("mode") == "patch_only" and bool(policy.get("base_code_source"))


def _is_delta_patch_handoff(handoff: dict[str, Any] | None) -> bool:
    if not isinstance(handoff, dict):
        return False
    return any(PurePosixPath(str(item).replace("\\", "/")).name == "delta_plan.json" for item in handoff.get("context_files", []))


def _compact_artifact_policy_for_prompt(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        return {}
    save_model = policy.get("save_model") if isinstance(policy.get("save_model"), dict) else {}
    return {
        "save_submission": bool(policy.get("save_submission", True)),
        "save_metrics": bool(policy.get("save_metrics", True)),
        "save_code_snapshot": bool(policy.get("save_code_snapshot", True)),
        "save_pipeline_summary": bool(policy.get("save_pipeline_summary", True)),
        "save_model": {
            "default": bool(save_model.get("default", False)),
            "allowed_when": list(save_model.get("allowed_when", []))[:5],
        },
    }


def _compact_next_experiment_markdown(text: str) -> str:
    allowed_sections = {
        "objective",
        "rationale",
        "pipeline blueprint",
        "primary change axis",
        "keep unchanged",
        "change details",
        "code change targets",
        "implementation notes",
        "success criteria",
        "expected outputs",
        "failure decision",
    }
    lines = text.splitlines()
    kept: list[str] = []
    current_section: str | None = None
    before_first_section = True
    for line in lines:
        if line.startswith("## "):
            before_first_section = False
            current_section = line[3:].strip().lower()
            if current_section in allowed_sections:
                kept.extend(["", line])
            continue
        if before_first_section:
            kept.append(line)
        elif current_section in allowed_sections:
            kept.append(line)
    return "\n".join(kept).strip() + "\n"


def _load_handoff(out_dir: Path) -> dict[str, Any]:
    handoff_path = trial_artifact_path(out_dir, "workspace_coding_handoff.json")
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing workspace coding handoff: {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if not isinstance(handoff, dict):
        raise ValueError(f"Workspace coding handoff must be a JSON object: {handoff_path}")
    return handoff


def _extract_coding_result(raw_response: dict[str, Any]) -> dict[str, Any]:
    text = raw_response.get("output_text") or _extract_output_text(raw_response)
    if not text:
        return {
            "status": "blocked",
            "summary": "The workspace code writer response did not include output text.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["missing_output_text"],
        }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "summary": "The workspace code writer response was not valid JSON.",
            "changed_files": [],
            "validation_results": [],
            "blocking_issues": ["invalid_json_output"],
        }
    return parsed if isinstance(parsed, dict) else {
        "status": "blocked",
        "summary": "The workspace code writer response JSON was not an object.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": ["invalid_json_output"],
    }


def _extract_output_text(raw_response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in raw_response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part)


def _normalize_coding_result(
    result: dict[str, Any],
    competition: str,
    trial_id: str,
    handoff: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(result)
    normalized.setdefault("schema_version", "1.0")
    normalized.setdefault("request_id", handoff.get("request_id"))
    normalized.setdefault("competition", competition)
    normalized.setdefault("trial_id", trial_id)
    normalized.setdefault("status", "blocked")
    normalized.setdefault("summary", "")
    normalized.setdefault("changed_files", [])
    normalized.setdefault("validation_results", [])
    normalized.setdefault("blocking_issues", [])
    normalized.setdefault("next_action", "validate-workspace-code-change")
    normalized["validation_results"] = _normalize_validation_results(normalized.get("validation_results"))
    return normalized


def _normalize_validation_results(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        commands = value.get("commands")
        if isinstance(commands, list):
            return commands
        return [value]
    return value


def _apply_code_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if coding_result.get("patch_updates"):
        issues.extend(_restore_base_code_snapshot(handoff))
        if issues:
            return issues
    issues.extend(_apply_patch_updates(coding_result, handoff))
    issues.extend(_apply_file_updates(coding_result, handoff))
    return issues


def _apply_patch_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("patch_updates", [])
    if not updates:
        return []
    if not isinstance(updates, list):
        return ["invalid_type:patch_updates"]
    if _is_patch_only(handoff) and len(updates) > 2:
        return [f"too_many_patch_updates_for_budget:{len(updates)}"]
    issues: list[str] = []
    project_root = Path(str(handoff.get("project_root", "")))
    for update in updates:
        if not isinstance(update, dict):
            issues.append("invalid_patch_update")
            continue
        path = update.get("path")
        find = update.get("find")
        replace = update.get("replace")
        if not isinstance(path, str) or not isinstance(find, str) or not isinstance(replace, str):
            issues.append("invalid_patch_update")
            continue
        if not find:
            issues.append(f"empty_patch_find:{path}")
            continue
        path_issues = _path_scope_issues(path, handoff, update_label="patch_update")
        if path_issues:
            issues.extend(path_issues)
            continue
        target = project_root / path.replace("\\", "/")
        if not target.exists() or not target.is_file():
            issues.append(f"patch_target_missing:{path}")
            continue
        original = read_text(target, default="")
        count = original.count(find)
        if count == 0:
            issues.append(f"patch_find_not_found:{path}")
            continue
        if count > 1:
            issues.append(f"patch_find_not_unique:{path}")
            continue
        write_text(target, original.replace(find, replace, 1))
    return issues


def _apply_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    if updates and not _allow_full_file_updates(handoff):
        if coding_result.get("patch_updates"):
            return []
        return [
            f"full_file_update_not_allowed_in_patch_only:{update.get('path') if isinstance(update, dict) else 'unknown'}"
            for update in updates
        ]
    issues: list[str] = []
    project_root = Path(str(handoff.get("project_root", "")))
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        content = update.get("content") if isinstance(update, dict) else None
        if not isinstance(path, str) or not isinstance(content, str):
            issues.append("invalid_file_update")
            continue
        path_issues = _path_scope_issues(path, handoff, update_label="file_update")
        if path_issues:
            issues.extend(path_issues)
            continue
        target = project_root / path.replace("\\", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    return issues


def _restore_base_code_snapshot(handoff: dict[str, Any]) -> list[str]:
    policy = handoff.get("edit_policy") if isinstance(handoff.get("edit_policy"), dict) else {}
    if not policy.get("restore_base_before_patch"):
        return []
    competition = str(handoff.get("competition") or "")
    source_trial_id = str(handoff.get("code_base_trial_id") or handoff.get("source_trial_id") or "")
    if not competition or not source_trial_id:
        return ["missing_base_code_trial"]
    files = _base_trial_code_files(competition, source_trial_id)
    if not files:
        return [f"missing_base_code_snapshot:{source_trial_id}"]
    issues: list[str] = []
    project_root = Path(str(handoff.get("project_root", "")))
    for relative, content in files:
        path_issues = _path_scope_issues(relative, handoff, update_label="base_restore")
        if path_issues:
            issues.extend(path_issues)
            continue
        target = project_root / relative.replace("\\", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text(target, content)
    return issues


def _base_trial_code_files(competition: str, source_trial_id: str) -> list[tuple[str, str]]:
    source_dir = trial_dir(competition, source_trial_id)
    snapshot = load_trial_code_snapshot(source_dir)
    if snapshot:
        return [(relative, text) for relative, text in snapshot if relative.endswith(".py")]
    code_root = source_dir / "user_view" / "code"
    files: list[tuple[str, str]] = []
    if code_root.is_dir():
        for path in sorted(code_root.rglob("*.py")):
            relative = path.relative_to(code_root).as_posix()
            text = read_text(path, default="")
            if text:
                files.append((relative, text))
    if files:
        return files
    for path in [source_dir / "workspace_coding_result.json", source_dir / "internal" / "workspace_coding_result.json"]:
        try:
            result = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        updates = result.get("file_updates") if isinstance(result, dict) else []
        if not isinstance(updates, list):
            continue
        for update in updates:
            if not isinstance(update, dict):
                continue
            relative = _safe_relative_python_path(update.get("path"))
            content = update.get("content")
            if relative and isinstance(content, str):
                files.append((relative, content))
        if files:
            return files
    return []


def _save_post_write_code_snapshot(
    out_dir: Path, coding_result: dict[str, Any], handoff: dict[str, Any]
) -> dict[str, Any]:
    project_root = Path(str(handoff.get("project_root", "")))
    if not project_root.is_dir():
        return {"status": "skipped", "reason": "project_root_missing"}
    extra_files = ["workspace_config.json", "src/baseline.py", "train_step.py", "predict_step.py", "test_step.py"]
    return save_trial_code_snapshot(
        out_dir,
        project_root,
        list(coding_result.get("changed_files", [])),
        allowed_paths=list(handoff.get("allowed_write_paths", [])),
        extra_files=extra_files,
    )


def _safe_relative_python_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".py":
        return ""
    return path.as_posix()


def _validate_schema(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    required = handoff.get("required_output", {}).get("required_fields", [])
    status_values = set(handoff.get("required_output", {}).get("status_values", ["completed", "blocked", "failed"]))
    issues = [f"missing_required_field:{field}" for field in required if field not in coding_result]
    status = coding_result.get("status")
    if status not in status_values:
        issues.append(f"invalid_status:{status}")
    if "changed_files" in coding_result and not isinstance(coding_result["changed_files"], list):
        issues.append("invalid_type:changed_files")
    if "validation_results" in coding_result and not isinstance(coding_result["validation_results"], list):
        issues.append("invalid_type:validation_results")
    if "blocking_issues" in coding_result and not isinstance(coding_result["blocking_issues"], list):
        issues.append("invalid_type:blocking_issues")
    if "patch_updates" in coding_result and not isinstance(coding_result["patch_updates"], list):
        issues.append("invalid_type:patch_updates")
    if status in {"blocked", "failed"} and not coding_result.get("blocking_issues"):
        issues.append("missing_blocking_issue_for_incomplete_result")
    return issues


def _validate_changed_files(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    changed_files = coding_result.get("changed_files", [])
    if not isinstance(changed_files, list):
        return []
    issues: list[str] = []
    for item in changed_files:
        if not isinstance(item, str):
            issues.append("invalid_changed_file")
            continue
        issues.extend(_path_scope_issues(item, handoff, update_label="changed_file"))
    return issues


def _validate_file_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("file_updates", [])
    if not updates:
        return []
    if not isinstance(updates, list):
        return ["invalid_type:file_updates"]
    if not _allow_full_file_updates(handoff):
        if coding_result.get("patch_updates"):
            return []
        return [
            f"full_file_update_not_allowed_in_patch_only:{update.get('path') if isinstance(update, dict) else 'unknown'}"
            for update in updates
        ]
    issues: list[str] = []
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        if not isinstance(path, str):
            issues.append("invalid_file_update")
            continue
        issues.extend(_path_scope_issues(path, handoff, update_label="file_update"))
    return issues


def _validate_patch_updates(coding_result: dict[str, Any], handoff: dict[str, Any]) -> list[str]:
    updates = coding_result.get("patch_updates", [])
    if not updates:
        return []
    if not isinstance(updates, list):
        return ["invalid_type:patch_updates"]
    issues: list[str] = []
    if _is_patch_only(handoff) and len(updates) > 2:
        issues.append(f"too_many_patch_updates_for_budget:{len(updates)}")
    for update in updates:
        path = update.get("path") if isinstance(update, dict) else None
        if not isinstance(path, str):
            issues.append("invalid_patch_update")
            continue
        issues.extend(_path_scope_issues(path, handoff, update_label="patch_update"))
    return issues


def _allow_full_file_updates(handoff: dict[str, Any]) -> bool:
    policy = handoff.get("edit_policy") if isinstance(handoff.get("edit_policy"), dict) else {}
    if policy:
        return bool(policy.get("allow_full_file_updates"))
    return True


def _is_patch_only(handoff: dict[str, Any]) -> bool:
    policy = handoff.get("edit_policy") if isinstance(handoff.get("edit_policy"), dict) else {}
    return policy.get("mode") == "patch_only"


def _path_scope_issues(path: str, handoff: dict[str, Any], *, update_label: str) -> list[str]:
    issues: list[str] = []
    if not _is_safe_relative_path(path):
        issues.append(f"{update_label}_path_not_safe:{path}")
        return issues
    if not _is_allowed_path(path, handoff.get("allowed_write_paths", [])):
        issues.append(f"{update_label}_not_allowed:{path}")
    if _matches_forbidden_path(path, handoff.get("forbidden_paths", [])):
        issues.append(f"forbidden_path_touched:{path}")
    return issues


def _is_safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized.rstrip("/"))
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts and not _looks_like_windows_absolute(normalized)


def _looks_like_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] == "/"


def _is_allowed_path(path: str, allowed_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for allowed in allowed_paths:
        marker = str(allowed).replace("\\", "/").rstrip("/")
        if str(allowed).replace("\\", "/").endswith("/"):
            if normalized.startswith(marker + "/"):
                return True
        elif normalized == marker:
            return True
    return False


def _matches_forbidden_path(path: str, forbidden_paths: list[str]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for forbidden in forbidden_paths:
        marker_raw = str(forbidden).replace("\\", "/")
        marker = marker_raw.rstrip("/")
        if marker_raw.endswith("/") and normalized.startswith(marker + "/"):
            return True
        if normalized == marker:
            return True
    return False


def _write_blocked_result(
    competition: str,
    trial_id: str,
    handoff: dict[str, Any],
    issues: list[str],
    token_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    result = {
        "schema_version": "1.0",
        "request_id": handoff.get("request_id"),
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked",
        "summary": "Workspace code writer did not execute a code change.",
        "changed_files": [],
        "validation_results": [],
        "blocking_issues": issues,
        "token_decision": token_decision or {},
        "next_action": "revise-workspace-code-writer-request",
    }
    _write_coding_result(out_dir, result)
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_code_writer_api",
        decision="blocked",
        reason="Workspace code writer blocked before applying file updates.",
        evidence={"blocking_issues": issues, "token_decision": token_decision or {}},
        next_action="revise-workspace-code-writer-request",
    )
    return result


def _write_coding_result(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "workspace_coding_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "workspace_coding_result.md", render_workspace_coding_result(result))


def _record_response_token_usage(
    competition: str,
    trial_id: str,
    raw_response: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return None
    return log_token_usage(
        competition,
        trial_id,
        provider=provider,
        model=model,
        call_type="workspace_code_writing",
        usage=usage,
        request_id=raw_response.get("id"),
    )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
