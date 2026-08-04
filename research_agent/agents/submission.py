from __future__ import annotations


import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import os

from .. import paths
from ..integrations import dacon_api, kaggle_cli
from ..paths import competition_memory_dir, competition_submissions_dir, experiments_dir, trial_dir
from ..store import load_state, now_iso, save_state, write_text
from ..trial_artifacts import prepare_trial_execution_facts
from ..trial_decision import write_trial_decision_card
from ..trial_memory_card import write_trial_memory_card
from ..user_insight_policy import evaluate_user_insight_submission


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
    _merge_submission_evidence_into_metrics(competition, trial_id, row)
    prepare_trial_execution_facts(competition, trial_id)
    evaluate_user_insight_submission(
        competition,
        trial_id,
        submitted_lb_score,
        objective=objective,
    )
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


def _merge_submission_evidence_into_metrics(
    competition: str,
    trial_id: str,
    row: dict[str, Any],
) -> None:
    metrics_path = trial_dir(competition, trial_id) / "metrics.json"
    if not metrics_path.exists():
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(metrics, dict):
        return
    metrics.update(
        {
            "lb_score": row.get("submitted_lb_score"),
            "submitted_lb_score": row.get("submitted_lb_score"),
            "submitted_rank": row.get("submitted_rank"),
            "submission_file": row.get("submission_file"),
            "submission_status": "submitted",
            "is_best_submission": bool(row.get("is_best")),
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    dacon_competition_id: str | None = None,
    dacon_team_name: str | None = None,
    dacon_message: str | None = None,
    dacon_submit_fn: dacon_api.SubmitFn | None = None,
    dacon_fetch_submissions_fn: Callable[..., dict[str, Any]] | None = None,
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
    elif dacon_competition_id:
        submit_result = _run_dacon_submission(
            # Use the resolved absolute path: dacon_api has no subprocess cwd
            # to resolve a project-root-relative path against (Kaggle's path
            # only works because kaggle_cli's runner sets cwd=project_root).
            submission_file=str(submission_path),
            competition_id=dacon_competition_id,
            team_name=dacon_team_name or "",
            message=dacon_message or version_name,
            submit_fn=dacon_submit_fn,
            fetch_submissions_fn=dacon_fetch_submissions_fn,
            display_file_name=f"{version_name}{submission_path.suffix}",
        )
    else:
        submit_result = _run_command(submit_command) if submit_command else {"returncode": 0, "stdout": "", "stderr": ""}

    platform_competition_ref = kaggle_competition_slug or dacon_competition_id
    if platform_competition_ref and submit_result["status"] not in {"submitted", "submitted_without_polling"}:
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
    elif dacon_competition_id:
        result["status"] = "submitted"
        result["dacon_competition_id"] = dacon_competition_id
        result["dacon_team_name"] = dacon_team_name
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
    file_name = Path(submission_file).name
    existing = kaggle_cli.fetch_submissions(
        competition_slug=competition_slug,
        cwd=root,
        page_size=20,
        runner=command_runner,
    )
    existing_matches = (
        kaggle_cli.matching_submissions(
            existing.get("stdout", ""),
            description=message,
            file_name=file_name,
        )
        if existing.get("ok")
        else []
    )
    if existing_matches:
        lookup = kaggle_cli.poll_submission(
            competition_slug,
            description=message,
            file_name=file_name,
            cwd=root,
            attempts=poll_attempts,
            sleep_seconds=poll_interval_seconds,
            runner=command_runner,
        )
        if lookup["status"] == "found":
            return {
                "status": "submitted_without_polling",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "cli_check": cli_check,
                "auth_check": auth_check,
                "submissions": lookup.get("command_result"),
                "submission_result": lookup["submission"],
                "reused_existing_submission": True,
            }
        return {
            "status": "submission_pending",
            "returncode": 0,
            "reason": "A matching Kaggle submission already exists and is still being processed.",
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submissions": lookup.get("command_result"),
            "submission_result": lookup.get("submission") or existing_matches[0],
            "reused_existing_submission": True,
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
        lookup = kaggle_cli.poll_submission(
            competition_slug,
            description=message,
            file_name=file_name,
            cwd=root,
            attempts=poll_attempts,
            sleep_seconds=poll_interval_seconds,
            runner=command_runner,
        )
        latest_submission = lookup.get("submission") or {}
        if lookup["status"] != "found":
            return {
                "status": "submission_pending",
                "returncode": 0,
                "reason": "Kaggle accepted the submission, but its public score is still pending.",
                "stdout": submit["stdout"],
                "stderr": submit["stderr"],
                "cli_check": cli_check,
                "auth_check": auth_check,
                "submit": submit,
                "submissions": lookup.get("command_result"),
                "submission_result": latest_submission,
                "reused_existing_submission": False,
            }
        return {
            "status": "submitted_without_polling",
            "returncode": 0,
            "stdout": submit["stdout"],
            "stderr": submit["stderr"],
            "cli_check": cli_check,
            "auth_check": auth_check,
            "submit": submit,
            "submissions": lookup.get("command_result"),
            "submission_result": latest_submission,
            "reused_existing_submission": False,
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


def _run_dacon_submission(
    *,
    submission_file: str,
    competition_id: str,
    team_name: str,
    message: str,
    submit_fn: dacon_api.SubmitFn | None,
    fetch_submissions_fn: Callable[..., dict[str, Any]] | None = None,
    display_file_name: str | None = None,
) -> dict[str, Any]:
    """Submit via the DACON adapter, translated into the same result shape
    _run_kaggle_submission produces (status/returncode/reason) so the shared
    downstream code in submit_trial does not need to branch by platform.

    DACON's submit endpoint returns no score, but its submission-history
    endpoint does, so the scored value is read back from there and surfaced
    as `submission_result.public_score` -- the same field the Kaggle path
    uses -- which is what lets lb_score be recorded automatically instead of
    having to be typed in from a manual leaderboard check.

    Every trial writes to the same `outputs/submission.csv` path, so without
    display_file_name every row in DACON's own submission history page would
    show the identical file name "submission.csv" with no way to tell which
    trial produced which entry short of opening each one's memo. When given,
    a same-content copy is uploaded under that name instead -- the workspace's
    canonical output path is left untouched.
    """
    upload_path = submission_file
    temp_dir: str | None = None
    if display_file_name:
        temp_dir = tempfile.mkdtemp(prefix="dacon_submit_")
        upload_path = str(Path(temp_dir) / display_file_name)
        shutil.copyfile(submission_file, upload_path)
    try:
        result = dacon_api.submit_competition(
            competition_id=competition_id,
            submission_file=upload_path,
            team_name=team_name,
            message=message,
            env=dict(os.environ),
            submit_fn=submit_fn,
        )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    if not result["ok"]:
        return {
            "status": result["status"],
            "returncode": 1,
            "reason": result.get("error"),
            "dacon_result": result,
        }
    submitted = {
        "status": "submitted",
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "dacon_result": result,
    }
    scored = _fetch_dacon_submitted_score(
        competition_id=competition_id,
        message=message,
        fetch_submissions_fn=fetch_submissions_fn,
    )
    if scored is not None:
        submitted["submission_result"] = scored
    return submitted


def _fetch_dacon_submitted_score(
    *,
    competition_id: str,
    message: str,
    fetch_submissions_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Read back the score DACON assigned to the submission just filed.

    Best effort: a missing session token, a network error, or a history that
    does not list this submission yet all return None rather than failing an
    upload that already succeeded. The submission is located by its memo,
    which this codebase sets, so a concurrent manual submission is not
    mistaken for the one recorded here.
    """
    fetch = fetch_submissions_fn or dacon_api.fetch_my_submissions
    try:
        history = fetch(competition_id, env=dict(os.environ))
    except Exception:  # noqa: BLE001 - never fail a completed upload over the score lookup
        return None
    if not history.get("ok"):
        return None
    matches = [
        row
        for row in (history.get("submissions") or [])
        if isinstance(row, dict) and str(row.get("memo") or "") == message
    ]
    if not matches:
        return None
    latest = max(matches, key=lambda row: row.get("sub_id") or 0)
    score = latest.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return {
        "public_score": score,
        "sub_id": latest.get("sub_id"),
        "submitted_at": latest.get("c_time"),
        "file_link": latest.get("file_link"),
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


