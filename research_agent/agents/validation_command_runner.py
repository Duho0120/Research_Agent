from __future__ import annotations

import json
import subprocess
from typing import Any

from .. import paths
from ..paths import trial_dir
from ..store import write_text
from .memory import log_decision


def run_validation_commands(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    handoff_path = out_dir / "coding_handoff.json"
    validation_path = out_dir / "coding_result_validation.json"
    if not handoff_path.exists():
        raise FileNotFoundError(f"Missing coding handoff: {handoff_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing coding result validation: {validation_path}")

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    coding_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    commands = handoff.get("validation_commands", [])
    issues: list[str] = []
    command_results: list[dict[str, Any]] = []

    if coding_validation.get("status") != "accepted":
        issues.append("coding_result_not_accepted")
    if not commands:
        issues.append("missing_validation_command")

    if issues:
        result = _result(competition, trial_id, "blocked", issues, command_results)
        _write_result(competition, trial_id, result)
        return result

    for index, command in enumerate(commands, start=1):
        command_results.append(_run_command(competition, trial_id, index, command))

    status = "passed" if all(item["returncode"] == 0 for item in command_results) else "failed"
    result = _result(competition, trial_id, status, issues, command_results)
    _write_result(competition, trial_id, result)
    return result


def render_validation_run(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Validation Commands",
        "",
        f"- status: {result['status']}",
        f"- command_count: {len(result['commands'])}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    lines.extend(["", "## Commands", ""])
    for item in result["commands"]:
        lines.extend(
            [
                f"### Command {item['index']}",
                "",
                f"- status: {item['status']}",
                f"- returncode: {item['returncode']}",
                f"- log_path: {item['log_path']}",
                "",
                "```powershell",
                item["command"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _run_command(competition: str, trial_id: str, index: int, command: str) -> dict[str, Any]:
    root = paths.project_root()
    completed = subprocess.run(
        command,
        shell=True,
        cwd=root,
        text=True,
        capture_output=True,
    )
    out_dir = trial_dir(competition, trial_id)
    log_path = out_dir / f"validation_command_{index:03d}.log"
    write_text(
        log_path,
        "COMMAND:\n"
        + command
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
    )
    return {
        "index": index,
        "command": command,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log_path": _relative_log_path(log_path),
    }


def _result(
    competition: str,
    trial_id: str,
    status: str,
    issues: list[str],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "competition": competition,
        "trial_id": trial_id,
        "status": status,
        "issues": issues,
        "commands": commands,
        "next_action": "run-trial" if status == "passed" else "revise-code-result",
    }


def _write_result(competition: str, trial_id: str, result: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "validation_run.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "validation_run.md", render_validation_run(result))
    log_decision(
        competition,
        trial_id,
        decision_type="validation_commands",
        decision=result["status"],
        reason="Validation commands were evaluated after accepted code-writer output.",
        evidence={
            "issues": result["issues"],
            "command_count": len(result["commands"]),
            "failed_commands": [item["index"] for item in result["commands"] if item["status"] == "failed"],
        },
        next_action=result["next_action"],
    )


def _relative_log_path(path) -> str:
    try:
        return str(path.relative_to(paths.project_root()).as_posix())
    except ValueError:
        return str(path.as_posix())
