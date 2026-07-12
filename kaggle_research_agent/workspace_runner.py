from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .agents.policy_gate import classify_local_failure
from .execution_profile import load_execution_profile, validate_execution_profile
from .paths import trial_dir
from .store import write_text


STAGE_ORDER = ("test", "train", "predict")


def run_workspace_pipeline(
    competition: str,
    trial_id: str,
    *,
    run_now: bool = False,
) -> dict[str, Any]:
    validation = validate_execution_profile(competition)
    try:
        profile = load_execution_profile(competition)
    except (FileNotFoundError, ValueError):
        profile = {}
    commands = _render_commands(profile) if profile else {}
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "planned",
        "run_now": run_now,
        "profile_status": validation["status"],
        "profile_issues": validation["issues"],
        "project_root": profile.get("project_root"),
        "commands": commands,
        "command_results": [],
        "artifacts": {},
        "next_action": "rerun-with-run-now",
    }
    if validation["status"] != "ready":
        result["status"] = "blocked"
        result["next_action"] = "fix-execution-profile"
    elif run_now:
        _execute_commands(result, profile)
        if result["status"] != "failed":
            result["artifacts"] = _inspect_artifacts(profile)
            if _all_artifacts_exist(result["artifacts"]):
                result["status"] = "completed"
                result["next_action"] = "collect-metrics"
            else:
                result["status"] = "incomplete_artifacts"
                result["next_action"] = "fix-artifact-output"
    _write_result(result)
    log_decision(
        competition,
        trial_id,
        decision_type="workspace_pipeline",
        decision=result["status"],
        reason=(
            "External workspace commands require explicit run approval."
            if result["status"] == "planned"
            else (
                "Execution Profile validation blocked external command execution."
                if result["status"] == "blocked"
                else "Execution Profile commands were processed in stage order."
            )
        ),
        evidence={
            "profile_status": validation["status"],
            "commands": commands,
            "command_results": result["command_results"],
            "artifacts": result["artifacts"],
        },
        user_input_used=run_now,
        next_action=result["next_action"],
    )
    return result


def _render_commands(profile: dict[str, Any]) -> dict[str, list[str]]:
    python_command = subprocess.list2cmdline([str(profile["python"])])
    configured = profile.get("commands", {})
    return {
        stage: [command.replace("{python}", python_command) for command in configured.get(stage, [])]
        for stage in STAGE_ORDER
        if configured.get(stage)
    }


def _execute_commands(result: dict[str, Any], profile: dict[str, Any]) -> None:
    project_root = Path(profile["project_root"])
    out_dir = trial_dir(result["competition"], result["trial_id"]) / "workspace_logs"
    for stage in STAGE_ORDER:
        for index, command in enumerate(result["commands"].get(stage, []), start=1):
            completed = subprocess.run(
                command,
                cwd=project_root,
                shell=True,
                text=True,
                capture_output=True,
            )
            log_path = out_dir / f"{stage}_{index:02d}.log"
            write_text(
                log_path,
                "\n".join(
                    [
                        f"command: {command}",
                        f"returncode: {completed.returncode}",
                        "",
                        "[stdout]",
                        completed.stdout,
                        "[stderr]",
                        completed.stderr,
                    ]
                ),
            )
            command_result = {
                "stage": stage,
                "index": index,
                "command": command,
                "returncode": completed.returncode,
                "log_path": str(log_path.as_posix()),
            }
            result["command_results"].append(command_result)
            if completed.returncode != 0:
                result["status"] = "failed"
                result["failure"] = classify_local_failure(log_path, use_artifact=False)
                result["next_action"] = "fix-workspace-command"
                return


def _inspect_artifacts(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    project_root = Path(profile["project_root"])
    return {
        kind: [
            {
                "path": relative_path,
                "absolute_path": str((project_root / relative_path).resolve()),
                "exists": (project_root / relative_path).is_file(),
                "size": (project_root / relative_path).stat().st_size
                if (project_root / relative_path).is_file()
                else None,
            }
            for relative_path in paths
        ]
        for kind, paths in profile.get("artifacts", {}).items()
    }


def _all_artifacts_exist(artifacts: dict[str, list[dict[str, Any]]]) -> bool:
    return bool(artifacts) and all(
        item["exists"]
        for items in artifacts.values()
        for item in items
    )


def _write_result(result: dict[str, Any]) -> None:
    out_dir = trial_dir(result["competition"], result["trial_id"])
    write_text(out_dir / "workspace_run.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {result['competition']} / {result['trial_id']} Workspace Run",
        "",
        f"- status: {result['status']}",
        f"- run_now: {result['run_now']}",
        f"- profile_status: {result['profile_status']}",
        f"- project_root: {result['project_root']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Commands",
        "",
    ]
    for stage, commands in result["commands"].items():
        lines.extend(f"- {stage}: `{command}`" for command in commands)
    lines.append("")
    write_text(out_dir / "workspace_run.md", "\n".join(lines))
