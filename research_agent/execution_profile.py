from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from . import simple_yaml
from .paths import competition_dir
from .store import write_text


def load_execution_profile(competition: str) -> dict[str, Any]:
    path = competition_dir(competition) / "execution_profile.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing execution profile: {path}")
    profile = simple_yaml.load(path, default={})
    if not isinstance(profile, dict):
        raise ValueError(f"Execution profile must be a mapping: {path}")
    return profile


def validate_execution_profile(competition: str) -> dict[str, Any]:
    profile_path = competition_dir(competition) / "execution_profile.yaml"
    issues: list[str] = []
    checks: dict[str, Any] = {}

    try:
        profile = load_execution_profile(competition)
    except (FileNotFoundError, ValueError) as error:
        result = _result(competition, "blocked", [str(error)], checks, profile_path)
        _write_validation(competition, result)
        return result

    _validate_required_fields(profile, competition, issues)
    _validate_runtime_paths(profile, issues, checks)
    _validate_commands(profile, issues, checks)
    _validate_artifacts_and_scope(profile, issues, checks)
    _validate_metrics_contract(profile, issues, checks)

    result = _result(
        competition,
        "ready" if not issues else "blocked",
        _unique(issues),
        checks,
        profile_path,
    )
    _write_validation(competition, result)
    return result


def render_execution_profile_validation(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['competition']} Execution Profile Validation",
        "",
        f"- status: {result['status']}",
        f"- profile_path: {result['profile_path']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Checks", "", "```json", json.dumps(result["checks"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def _validate_required_fields(profile: dict[str, Any], competition: str, issues: list[str]) -> None:
    for field in ["schema_version", "competition", "platform", "project_root", "python", "commands", "artifacts", "write_scope"]:
        if field not in profile:
            issues.append(f"missing_field:{field}")
    if profile.get("competition") not in {None, competition}:
        issues.append("competition_mismatch")


def _validate_runtime_paths(profile: dict[str, Any], issues: list[str], checks: dict[str, Any]) -> None:
    project_value = profile.get("project_root")
    python_value = profile.get("python")
    project_path = Path(str(project_value)) if project_value else None
    python_path = Path(str(python_value)) if python_value else None

    project_exists = bool(project_path and project_path.is_absolute() and project_path.is_dir())
    python_exists = bool(python_path and python_path.is_absolute() and python_path.is_file())
    checks["project_root_exists"] = project_exists
    checks["python_exists"] = python_exists
    if project_value and project_path and not project_path.is_absolute():
        issues.append("project_root_must_be_absolute")
    elif project_value and not project_exists:
        issues.append("project_root_not_found")
    if python_value and python_path and not python_path.is_absolute():
        issues.append("python_must_be_absolute")
    elif python_value and not python_exists:
        issues.append("python_not_found")


def _validate_commands(profile: dict[str, Any], issues: list[str], checks: dict[str, Any]) -> None:
    commands = profile.get("commands", {})
    if not isinstance(commands, dict):
        issues.append("commands_must_be_mapping")
        return
    for name in ["test", "train"]:
        values = commands.get(name)
        if not _non_empty_string_list(values):
            issues.append(f"missing_command:{name}")
    predict = commands.get("predict")
    if predict is not None and not _non_empty_string_list(predict):
        issues.append("invalid_command:predict")
    checks["command_groups"] = sorted(commands)


def _validate_artifacts_and_scope(profile: dict[str, Any], issues: list[str], checks: dict[str, Any]) -> None:
    artifacts = profile.get("artifacts", {})
    scope = profile.get("write_scope", {})
    if not isinstance(artifacts, dict):
        issues.append("artifacts_must_be_mapping")
        return
    if not isinstance(scope, dict):
        issues.append("write_scope_must_be_mapping")
        return

    metric_paths = artifacts.get("metrics")
    submission_paths = artifacts.get("submission")
    allowed = scope.get("allowed")
    forbidden = scope.get("forbidden")
    if not _non_empty_string_list(metric_paths):
        issues.append("missing_artifact:metrics")
        metric_paths = []
    if not _non_empty_string_list(submission_paths):
        issues.append("missing_artifact:submission")
        submission_paths = []
    if not _non_empty_string_list(allowed):
        issues.append("missing_write_scope:allowed")
        allowed = []
    if not _non_empty_string_list(forbidden):
        issues.append("missing_write_scope:forbidden")
        forbidden = []

    for label, paths in [
        ("artifact", [*metric_paths, *submission_paths]),
        ("allowed", allowed),
        ("forbidden", forbidden),
    ]:
        for item in paths:
            if not _is_safe_relative_path(item):
                issues.append(f"{label}_path_must_be_safe_relative:{item}")

    artifacts_all = [*metric_paths, *submission_paths]
    for allowed_path in allowed:
        for forbidden_path in forbidden:
            if _paths_overlap(allowed_path, forbidden_path):
                issues.append(f"allowed_path_is_forbidden:{allowed_path}")
                break
        for artifact_path in artifacts_all:
            if _paths_overlap(allowed_path, artifact_path):
                issues.append(f"allowed_path_is_artifact:{allowed_path}")
                break

    checks["artifact_count"] = len(artifacts_all)
    checks["allowed_write_count"] = len(allowed)
    checks["forbidden_write_count"] = len(forbidden)


def _validate_metrics_contract(profile: dict[str, Any], issues: list[str], checks: dict[str, Any]) -> None:
    contract = profile.get("metrics_contract")
    checks["metrics_contract_configured"] = contract is not None
    if contract is None:
        return
    if not isinstance(contract, dict):
        issues.append("metrics_contract_must_be_mapping")
        return
    source_key = contract.get("source_key")
    if not isinstance(source_key, str) or not source_key.strip() or any(not part for part in source_key.split(".")):
        issues.append("invalid_metrics_contract:source_key")


def _result(
    competition: str,
    status: str,
    issues: list[str],
    checks: dict[str, Any],
    profile_path: Path,
) -> dict[str, Any]:
    return {
        "competition": competition,
        "status": status,
        "issues": issues,
        "checks": checks,
        "profile_path": str(profile_path.as_posix()),
        "next_action": "prepare-one-cycle" if status == "ready" else "fix-execution-profile",
    }


def _write_validation(competition: str, result: dict[str, Any]) -> None:
    out_dir = competition_dir(competition)
    write_text(out_dir / "execution_profile_validation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "execution_profile_validation.md", render_execution_profile_validation(result))


def _non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)


def _is_safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized.rstrip("/"))
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts and not _looks_like_windows_absolute(normalized)


def _looks_like_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] == "/"


def _paths_overlap(left: str, right: str) -> bool:
    left_norm = left.replace("\\", "/").rstrip("/")
    right_norm = right.replace("\\", "/").rstrip("/")
    return left_norm == right_norm or left_norm.startswith(right_norm + "/") or right_norm.startswith(left_norm + "/")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
