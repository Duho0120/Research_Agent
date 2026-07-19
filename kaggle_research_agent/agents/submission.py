from __future__ import annotations


import json
from typing import Any

from .. import paths
from ..integrations import kaggle_cli
from ..paths import competition_memory_dir, competition_submissions_dir, experiments_dir, trial_dir
from ..store import load_state, now_iso, save_state, write_text
from ..trial_decision import write_trial_decision_card
from ..trial_memory_card import write_trial_memory_card


def record_submission_result(
    competition: str,
    trial_id: str,
    version_name: str,
    submission_file: str,
    cv_score: float | None = None,
    previous_lb_score: float | None = None,
    previous_rank: int | None = None,
    submitted_lb_score: float | None = None,
    submitted_rank: int | None = None,
    objective: str = "maximize",
    notes: str = "",
) -> dict[str, Any]:
    score_delta = _score_delta(previous_lb_score, submitted_lb_score)
    rank_delta = _rank_delta(previous_rank, submitted_rank)
    historical_best = _load_best_submission(competition)
    best_reference_score = (
        historical_best.get("submitted_lb_score")
        if historical_best and historical_best.get("submitted_lb_score") is not None
        else previous_lb_score
    )
    is_best = _is_best_submission(best_reference_score, submitted_lb_score, objective)
    row = {
        "submission_id": f"{competition}_{trial_id}_{version_name}",
        "competition": competition,
        "trial_id": trial_id,
        "version_name": version_name,
        "submitted_at": now_iso(),
        "submission_file": submission_file,
        "cv_score": cv_score,
        "previous_lb_score": previous_lb_score,
        "previous_rank": previous_rank,
        "submitted_lb_score": submitted_lb_score,
        "submitted_rank": submitted_rank,
        "score_delta": score_delta,
        "rank_delta": rank_delta,
        "best_reference_score": best_reference_score,
        "is_best": is_best,
        "notes": notes,
    }
    _append_submission_log(competition, row)
    _write_trial_files(competition, trial_id, row)
    decision_card = write_trial_decision_card(
        competition,
        trial_id,
        metrics={"cv_score": cv_score, "objective": objective},
        submission={"submitted_lb_score": submitted_lb_score, "submitted_rank": submitted_rank},
    )
    write_trial_memory_card(
        competition,
        trial_id,
        metrics={"cv_score": cv_score, "objective": objective},
        decision_card=decision_card,
    )
    if is_best:
        _write_best_files(competition, trial_id, row)
    return row


def render_submission_result(row: dict[str, Any]) -> str:
    return f"""# Submission Result: {row['version_name']}

- submission_id: {row['submission_id']}
- competition: {row['competition']}
- trial_id: {row['trial_id']}
- submitted_at: {row['submitted_at']}
- submission_file: {row['submission_file']}
- cv_score: {row['cv_score']}
- previous_lb_score: {row['previous_lb_score']}
- submitted_lb_score: {row['submitted_lb_score']}
- score_delta: {row['score_delta']}
- previous_rank: {row['previous_rank']}
- submitted_rank: {row['submitted_rank']}
- rank_delta: {row['rank_delta']}
- is_best: {row['is_best']}

## Notes

{row['notes']}
"""


def render_version(row: dict[str, Any]) -> str:
    return f"""# Version

- version_name: {row['version_name']}
- submission_id: {row['submission_id']}
- competition: {row['competition']}
- trial_id: {row['trial_id']}
- submission_file: {row['submission_file']}
- submitted_at: {row['submitted_at']}
"""


def render_best_trial(row: dict[str, Any]) -> str:
    return f"""# Best Trial

- competition: {row['competition']}
- trial_id: {row['trial_id']}
- version_name: {row['version_name']}
- cv_score: {row['cv_score']}
- lb_score: {row['submitted_lb_score']}
- rank: {row['submitted_rank']}
- submission_file: {row['submission_file']}
- submitted_at: {row['submitted_at']}
"""


def _append_submission_log(competition: str, row: dict[str, Any]) -> None:
    out_dir = competition_submissions_dir(competition)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission_log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_trial_files(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "submission_result.md", render_submission_result(row))
    write_text(out_dir / "VERSION.md", render_version(row))


def _write_best_files(competition: str, trial_id: str, row: dict[str, Any]) -> None:
    memory = competition_memory_dir(competition)
    best_path = memory / "best_trial.json"
    if best_path.exists():
        previous = json.loads(best_path.read_text(encoding="utf-8"))
        previous_trial_id = previous.get("trial_id")
        if previous_trial_id and previous_trial_id != trial_id:
            previous_marker = trial_dir(competition, previous_trial_id) / "BEST_MARKER.md"
            if previous_marker.exists():
                previous_marker.unlink()

    write_text(experiments_dir() / competition / "BEST_TRIAL.md", render_best_trial(row))
    memory.mkdir(parents=True, exist_ok=True)
    best_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text(trial_dir(competition, trial_id) / "BEST_MARKER.md", render_best_trial(row))
    _update_state_best_trial_from_submission(competition, row)


def _load_best_submission(competition: str) -> dict[str, Any] | None:
    path = competition_memory_dir(competition) / "best_trial.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _update_state_best_trial_from_submission(competition: str, row: dict[str, Any]) -> None:
    state = load_state(competition)
    if not isinstance(state, dict):
        return
    current_state = state.setdefault("current_state", {})
    if not isinstance(current_state, dict):
        state["current_state"] = {}
        current_state = state["current_state"]
    current_state["best_trial"] = {
        "trial_id": row["trial_id"],
        "cv_score": row.get("cv_score"),
        "lb_score": row.get("submitted_lb_score"),
        "rank": row.get("submitted_rank"),
        "version_name": row.get("version_name"),
        "source": "leaderboard_submission",
        "submitted_at": row.get("submitted_at"),
    }
    current_state["active_trial"] = row["trial_id"]
    save_state(competition, state)


def _score_delta(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return round(current - previous, 10)


def _rank_delta(previous: int | None, current: int | None) -> int | None:
    if previous is None or current is None:
        return None
    return previous - current


def _is_best_submission(previous: float | None, current: float | None, objective: str) -> bool:
    if objective not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported objective: {objective}")
    if current is None:
        return False
    if previous is None:
        return True
    return current < previous if objective == "minimize" else current > previous


from pathlib import Path


def prepare_submission(
    competition: str,
    trial_id: str,
    version_name: str,
    submission_file: str,
    objective: str = "maximize",
    notes: str = "",
) -> dict[str, Any]:
    """Create a submission manifest without touching leaderboard records."""

    out_dir = trial_dir(competition, trial_id)
    submission_path = _resolve_path(submission_file)
    metrics_path = out_dir / "metrics.json"
    checks: list[str] = []
    check_details: dict[str, str] = {}

    if not submission_path.exists():
        checks.append("Missing submission file")
        check_details["submission_file"] = submission_file
    if not metrics_path.exists():
        checks.append("Missing metrics file")
        check_details["metrics_file"] = str(metrics_path.as_posix())

    metrics = _load_metrics(out_dir) if metrics_path.exists() else {}
    manifest = {
        "competition": competition,
        "trial_id": trial_id,
        "version_name": version_name,
        "submission_file": submission_file,
        "cv_score": metrics.get("cv_score"),
        "objective": objective or metrics.get("objective", "maximize"),
        "status": "blocked" if checks else "ready",
        "requires_user_approval": True,
        "approved": False,
        "checks": checks,
        "check_details": check_details,
        "created_at": now_iso(),
        "notes": notes,
        "next_step": "submit-trial after user approval" if not checks else "fix blocked checks before submission",
    }
    _write_submit_manifest(out_dir, manifest)
    return manifest


def submit_trial(
    competition: str,
    trial_id: str,
    version_name: str,
    submission_file: str,
    before_score: float | None = None,
    before_rank: int | None = None,
    after_score: float | None = None,
    after_rank: int | None = None,
    objective: str = "maximize",
    before_command: str | None = None,
    submit_command: str | None = None,
    after_command: str | None = None,
    kaggle_competition_slug: str | None = None,
    kaggle_team_name: str | None = None,
    kaggle_message: str | None = None,
    poll_leaderboard: bool = False,
    poll_attempts: int = 5,
    poll_interval_seconds: float = 30.0,
    kaggle_runner: kaggle_cli.CommandRunner | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Submit or record a trial while preserving before/after leaderboard evidence."""

    out_dir = trial_dir(competition, trial_id)
    submission_path = _resolve_path(submission_file)
    if not submission_path.exists():
        result = {
            "competition": competition,
            "trial_id": trial_id,
            "version_name": version_name,
            "status": "blocked",
            "reason": f"Missing submission file: {submission_file}",
        }
        _write_submit_run(out_dir, result)
        return result

    before = _leaderboard_state(before_command, before_score, before_rank)
    if kaggle_competition_slug:
        submit_result = _run_kaggle_submission(
            competition=competition,
            trial_id=trial_id,
            version_name=version_name,
            submission_file=submission_file,
            competition_slug=kaggle_competition_slug,
            team_name=kaggle_team_name,
            message=kaggle_message or version_name,
            poll_leaderboard=poll_leaderboard,
            poll_attempts=poll_attempts,
            poll_interval_seconds=poll_interval_seconds,
            runner=kaggle_runner,
        )
    else:
        submit_result = _run_command(submit_command) if submit_command else {"returncode": 0, "stdout": "", "stderr": ""}

    if kaggle_competition_slug and submit_result["status"] not in {"submitted", "submitted_without_polling"}:
        result = {
            "competition": competition,
            "trial_id": trial_id,
            "version_name": version_name,
            "status": submit_result["status"],
            "reason": submit_result.get("reason"),
            "before": before,
            "submit_result": submit_result,
        }
        _write_submit_run(out_dir, result)
        return result

    if submit_result["returncode"] != 0:
        result = {
            "competition": competition,
            "trial_id": trial_id,
            "version_name": version_name,
            "status": "submit_failed",
            "reason": submit_result["stderr"] or submit_result["stdout"],
            "before": before,
            "submit_command": submit_command,
            "submit_result": submit_result,
        }
        _write_submit_run(out_dir, result)
        return result

    if submit_result.get("leaderboard_result", {}).get("status") == "found":
        leaderboard = submit_result["leaderboard_result"]
        after = {"score": leaderboard.get("score"), "rank": leaderboard.get("rank"), "poll_result": leaderboard}
    elif submit_result.get("submission_result", {}).get("public_score") is not None:
        submission_result = submit_result["submission_result"]
        after = {"score": submission_result.get("public_score"), "rank": None, "submission_result": submission_result}
    else:
        after = _leaderboard_state(after_command, after_score, after_rank)

    metrics = _load_metrics(out_dir)
    row = record_submission_result(
        competition=competition,
        trial_id=trial_id,
        version_name=version_name,
        submission_file=submission_file,
        cv_score=metrics.get("cv_score"),
        previous_lb_score=before.get("score"),
        previous_rank=before.get("rank"),
        submitted_lb_score=after.get("score"),
        submitted_rank=after.get("rank"),
        objective=objective or metrics.get("objective", "maximize"),
        notes=notes,
    )
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "version_name": version_name,
        "status": "submitted" if submit_command else "recorded",
        "submission_file": submission_file,
        "previous_lb_score": before.get("score"),
        "previous_rank": before.get("rank"),
        "submitted_lb_score": after.get("score"),
        "submitted_rank": after.get("rank"),
        "submit_command": submit_command,
        "submit_result": submit_result,
        "recorded_submission": row,
    }
    if kaggle_competition_slug:
        result["status"] = "submitted"
        result["kaggle_competition_slug"] = kaggle_competition_slug
        result["kaggle_team_name"] = kaggle_team_name
    _write_submit_run(out_dir, result)
    return result


def render_submission_run(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Submission Run",
        "",
        f"- status: {result['status']}",
        f"- version_name: {result['version_name']}",
        f"- previous_lb_score: {result.get('previous_lb_score')}",
        f"- previous_rank: {result.get('previous_rank')}",
        f"- submitted_lb_score: {result.get('submitted_lb_score')}",
        f"- submitted_rank: {result.get('submitted_rank')}",
    ]
    if result.get("reason"):
        lines.extend(["", "## Reason", "", result["reason"]])
    if result.get("submit_command"):
        lines.extend(["", "## Submit Command", "", f"```bash\n{result['submit_command']}\n```"])
    lines.append("")
    return "\n".join(lines)


def _leaderboard_state(command: str | None, score: float | None, rank: int | None) -> dict[str, Any]:
    if command:
        completed = _run_command(command)
        if completed["returncode"] != 0:
            return {"score": score, "rank": rank, "command_result": completed}
        parsed = _parse_score_rank(completed["stdout"])
        parsed["command_result"] = completed
        return parsed
    return {"score": score, "rank": rank}


def _parse_score_rank(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {"score": None, "rank": None}
    data = json.loads(stripped.splitlines()[-1])
    return {"score": data.get("score"), "rank": data.get("rank")}


def _run_kaggle_submission(
    competition: str,
    trial_id: str,
    version_name: str,
    submission_file: str,
    competition_slug: str,
    team_name: str | None,
    message: str,
    poll_leaderboard: bool,
    poll_attempts: int,
    poll_interval_seconds: float,
    runner: kaggle_cli.CommandRunner | None,
) -> dict[str, Any]:
    root = paths.project_root()
    command_runner = runner or kaggle_cli.run_args
    cli_check = kaggle_cli.check_cli_available(root, runner=command_runner)
    if not cli_check["ok"]:
        return {
            "status": "kaggle_cli_unavailable",
            "returncode": cli_check["returncode"],
            "reason": cli_check["stderr"] or cli_check["stdout"],
            "cli_check": cli_check,
        }
    auth_check = kaggle_cli.check_cli_auth(root, runner=command_runner)
    if not auth_check["ok"]:
        return {
            "status": "kaggle_auth_failed",
            "returncode": auth_check["returncode"],
            "reason": auth_check["stderr"] or auth_check["stdout"],
            "cli_check": cli_check,
            "auth_check": auth_check,
        }
    submit = kaggle_cli.submit_competition(
        competition_slug=competition_slug,
        submission_file=submission_file,
        message=message,
        cwd=root,
        runner=command_runner,
    )
    if not submit["ok"]:
        return {
            "status": "submit_failed",
            "returncode": submit["returncode"],
            "reason": submit["stderr"] or submit["stdout"],
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submit": submit,
        }
    if not poll_leaderboard:
        submissions = kaggle_cli.fetch_submissions(
            competition_slug=competition_slug,
            cwd=root,
            page_size=5,
            runner=command_runner,
        )
        latest_submission = (
            kaggle_cli.latest_completed_submission(
                submissions.get("stdout", ""),
                description=message,
                file_name=Path(submission_file).name,
            )
            if submissions["ok"]
            else None
        )
        return {
            "status": "submitted_without_polling",
            "returncode": 0,
            "stdout": submit["stdout"],
            "stderr": submit["stderr"],
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submit": submit,
            "submissions": submissions,
            "submission_result": latest_submission or {},
        }
    if not team_name:
        return {
            "status": "leaderboard_polling_blocked",
            "returncode": 0,
            "reason": "Missing Kaggle team name for leaderboard polling.",
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submit": submit,
        }
    leaderboard = kaggle_cli.poll_leaderboard(
        competition_slug=competition_slug,
        team_name=team_name,
        cwd=root,
        attempts=poll_attempts,
        sleep_seconds=poll_interval_seconds,
        runner=command_runner,
    )
    if leaderboard["status"] != "found":
        return {
            "status": "leaderboard_timeout",
            "returncode": 0,
            "reason": "Leaderboard result was not available before polling timed out.",
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submit": submit,
            "leaderboard_result": leaderboard,
        }
    return {
        "status": "submitted",
        "returncode": 0,
        "stdout": submit["stdout"],
        "stderr": submit["stderr"],
        "cli_check": cli_check,
        "auth_check": auth_check,
        "submit": submit,
        "leaderboard_result": leaderboard,
        "competition": competition,
        "trial_id": trial_id,
        "version_name": version_name,
    }


def _run_command(command: str | None) -> dict[str, Any]:
    if not command:
        return {"returncode": 0, "stdout": "", "stderr": ""}
    return kaggle_cli.run_command(command, cwd=paths.project_root())


def _resolve_path(path: str) -> Path:
    file_path = Path(path)
    if file_path.is_absolute():
        return file_path
    return paths.project_root() / file_path


def _load_metrics(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_submit_run(out_dir: Path, result: dict[str, Any]) -> None:
    write_text(out_dir / "submission_run.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "submission_run.md", render_submission_run(result))


def _write_submit_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_text(out_dir / "submit_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "submit_manifest.md", render_submit_manifest(manifest))


def render_submit_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        f"# {manifest['trial_id']} Submit Manifest",
        "",
        f"- status: {manifest['status']}",
        f"- version_name: {manifest['version_name']}",
        f"- submission_file: {manifest['submission_file']}",
        f"- cv_score: {manifest['cv_score']}",
        f"- objective: {manifest['objective']}",
        f"- requires_user_approval: {manifest['requires_user_approval']}",
        f"- approved: {manifest['approved']}",
        f"- next_step: {manifest['next_step']}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["checks"] or ["ready"])
    lines.extend(["", "## Notes", "", manifest.get("notes", ""), ""])
    return "\n".join(lines)


