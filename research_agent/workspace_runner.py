from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .agents.memory import log_decision
from .agents.policy_gate import classify_local_failure
from .execution_profile import load_execution_profile, validate_execution_profile
from .paths import trial_dir
from .store import write_text
from .workspace_code_writer import SCORING_HARNESS_FILENAME


# "score" always runs last and is unconditional once the harness exists --
# see _render_commands, which only populates a "score" command when
# scoring_harness.py is actually present on disk, so an incomplete/blocked
# harness generation cannot break the test/train/predict stages that already
# worked before this stage was added.
STAGE_ORDER = ("test", "train", "predict", "score")


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
        run_started_at = time.time()
        _execute_commands(result, profile)
        if result["status"] != "failed":
            result["artifacts"] = _inspect_artifacts(profile)
            if _all_artifacts_exist(result["artifacts"]):
                artifact_issues = _validate_artifacts(result["artifacts"]) + _validate_artifact_freshness(
                    result["artifacts"], run_started_at
                )
                if artifact_issues:
                    result["status"] = "invalid_artifacts"
                    result["artifact_issues"] = artifact_issues
                    result["next_action"] = "fix-artifact-output"
                else:
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
    configured = dict(profile.get("commands", {}))
    if "score" not in configured:
        project_root = Path(str(profile.get("project_root", "")))
        if (project_root / SCORING_HARNESS_FILENAME).is_file():
            configured["score"] = [f"{{python}} {SCORING_HARNESS_FILENAME}"]
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
                result["failed_stage"] = stage
                result["failure"] = classify_local_failure(log_path, use_artifact=False)
                # A "score" failure is a scoring-harness bug, not a model
                # problem -- surfacing a distinct next_action stops it from
                # being reported the same way as an actual train/predict
                # failure, which would send someone looking at the wrong code.
                result["next_action"] = "fix-scoring-harness" if stage == "score" else "fix-workspace-command"
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
                "mtime": (project_root / relative_path).stat().st_mtime
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


def _validate_artifacts(artifacts: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues: list[str] = []
    invalid_numeric_tokens = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    for item in artifacts.get("submission", []):
        path = Path(str(item.get("absolute_path") or ""))
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = csv.reader(handle)
                header = next(rows, None)
                if not header:
                    issues.append(f"submission_missing_header:{path.name}")
                    continue
                for row_number, row in enumerate(rows, start=2):
                    if len(row) != len(header):
                        issues.append(f"submission_column_count_mismatch:{path.name}:row_{row_number}")
                        break
                    for column, value in zip(header, row):
                        if str(value).strip().casefold() in invalid_numeric_tokens:
                            issues.append(
                                f"submission_non_finite_value:{path.name}:row_{row_number}:{column}"
                            )
                            break
                    if issues:
                        break
        except (OSError, UnicodeError, csv.Error) as error:
            issues.append(f"submission_unreadable:{path.name}:{type(error).__name__}")
    return issues


def _validate_artifact_freshness(
    artifacts: dict[str, list[dict[str, Any]]], run_started_at: float
) -> list[str]:
    """Reject an artifact that predates this run's commands.

    Real incident: predict/train/test scripts wrote their results under a
    different file name (e.g. "outputs/metrics_local.json" instead of the
    declared "outputs/metrics.json") while leaving the previously declared
    artifact untouched. Every command reported returncode 0 and the file
    existed and parsed fine -- it was just leftover from an earlier trial --
    so the pipeline kept reporting that trial's stale score as if it were a
    fresh result, run after run, with no error anywhere. A one-second buffer
    absorbs filesystem mtime rounding, not clock drift between machines.
    """
    issues: list[str] = []
    cutoff = run_started_at - 1.0
    for items in artifacts.values():
        for item in items:
            mtime = item.get("mtime")
            if item.get("exists") and isinstance(mtime, (int, float)) and mtime < cutoff:
                issues.append(f"stale_artifact_not_regenerated_this_run:{Path(str(item.get('path'))).name}")
    return issues


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
