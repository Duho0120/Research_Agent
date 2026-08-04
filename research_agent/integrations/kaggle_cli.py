from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


CommandRunner = Callable[[list[str], Path], dict[str, Any]]


def run_command(command: str, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def run_args(args: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        shell=False,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def check_cli_available(cwd: Path, runner: CommandRunner = run_args) -> dict[str, Any]:
    args = ["kaggle", "--version"]
    return _structured_result("check_cli", args, runner(args, cwd))


def check_cli_auth(cwd: Path, runner: CommandRunner = run_args) -> dict[str, Any]:
    args = ["kaggle", "competitions", "list", "--page-size", "1"]
    return _structured_result("check_auth", args, runner(args, cwd))


def submit_competition(
    competition_slug: str,
    submission_file: str,
    message: str,
    cwd: Path,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    args = build_submit_args(competition_slug, submission_file, message)
    return _structured_result("submit", args, runner(args, cwd))


def fetch_leaderboard(
    competition_slug: str,
    cwd: Path,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    args = build_leaderboard_args(competition_slug)
    return _structured_result("leaderboard", args, runner(args, cwd))


def fetch_submissions(
    competition_slug: str,
    cwd: Path,
    page_size: int = 5,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    args = build_submissions_args(competition_slug, page_size=page_size)
    return _structured_result("submissions", args, runner(args, cwd))


def fetch_competition_files(
    competition_slug: str,
    cwd: Path,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    args = build_competition_files_args(competition_slug)
    return _structured_result("competition_files", args, runner(args, cwd))


def poll_leaderboard(
    competition_slug: str,
    team_name: str,
    cwd: Path,
    attempts: int = 5,
    sleep_seconds: float = 30.0,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        command_result = fetch_leaderboard(competition_slug, cwd=cwd, runner=runner)
        parsed = parse_leaderboard(command_result.get("stdout", ""), team_name=team_name)
        last_result = {**parsed, "attempts": attempt, "command_result": command_result}
        if parsed["status"] == "found":
            return last_result
        if attempt < attempts and sleep_seconds:
            time.sleep(sleep_seconds)
    return {
        "status": "timeout",
        "team_name": team_name,
        "score": None,
        "rank": None,
        "attempts": attempts,
        "command_result": last_result.get("command_result") if last_result else None,
    }


def parse_leaderboard(text: str, team_name: str) -> dict[str, Any]:
    rows = _parse_csv_leaderboard(text)
    if not rows:
        rows = _parse_table_leaderboard(text)
    ranked = _rank_rows(rows)
    needle = team_name.casefold()
    for row in ranked:
        if str(row.get("team_name", "")).casefold() == needle:
            return {
                "status": "found",
                "team_name": row["team_name"],
                "score": row["score"],
                "rank": row["rank"],
            }
    return {"status": "not_found", "team_name": team_name, "score": None, "rank": None}


def parse_submissions(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(stripped.splitlines()):
        normalized = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
        rows.append(
            {
                "ref": str(normalized.get("ref", "")).strip(),
                "file_name": str(normalized.get("filename", "")).strip(),
                "date": str(normalized.get("date", "")).strip(),
                "description": str(normalized.get("description", "")).strip(),
                "status": str(normalized.get("status", "")).strip(),
                "public_score": _to_float(normalized.get("publicscore")),
                "private_score": _to_float(normalized.get("privatescore")),
            }
        )
    return rows


def latest_completed_submission(
    text: str,
    *,
    description: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any] | None:
    rows = matching_submissions(text, description=description, file_name=file_name)
    for row in rows:
        if "complete" in str(row.get("status", "")).casefold() and row.get("public_score") is not None:
            return row
    return rows[0] if rows else None


def matching_submissions(
    text: str,
    *,
    description: str | None = None,
    file_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = parse_submissions(text)
    if description:
        rows = [row for row in rows if row.get("description") == description]
    if file_name:
        rows = [row for row in rows if row.get("file_name") == file_name]
    return rows


def poll_submission(
    competition_slug: str,
    *,
    description: str,
    file_name: str,
    cwd: Path,
    attempts: int = 5,
    sleep_seconds: float = 30.0,
    runner: CommandRunner = run_args,
) -> dict[str, Any]:
    last_submission: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    for attempt in range(1, max(1, attempts) + 1):
        command_result = fetch_submissions(
            competition_slug=competition_slug,
            cwd=cwd,
            page_size=20,
            runner=runner,
        )
        last_result = command_result
        matches = (
            matching_submissions(
                command_result.get("stdout", ""),
                description=description,
                file_name=file_name,
            )
            if command_result.get("ok")
            else []
        )
        if matches:
            last_submission = matches[0]
            if (
                "complete" in str(last_submission.get("status", "")).casefold()
                and last_submission.get("public_score") is not None
            ):
                return {
                    "status": "found",
                    "submission": last_submission,
                    "attempts": attempt,
                    "command_result": command_result,
                }
        if attempt < max(1, attempts) and sleep_seconds:
            time.sleep(sleep_seconds)
    return {
        "status": "pending" if last_submission else "not_found",
        "submission": last_submission,
        "attempts": max(1, attempts),
        "command_result": last_result,
    }


def parse_competition_files(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if "," in stripped.splitlines()[0]:
        return _parse_competition_files_csv(stripped)
    return _parse_competition_files_table(stripped)


def normalize_competition_slug(reference: str) -> str:
    value = reference.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if "competitions" in parts:
            index = parts.index("competitions")
            if index + 1 < len(parts):
                return parts[index + 1]
        if parts:
            return parts[-1]
    if "/" in value:
        return value.split("/")[-1]
    return value


def build_submit_args(competition_slug: str, submission_file: str, message: str) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition_slug,
        "-f",
        submission_file,
        "-m",
        message,
    ]


def build_submit_command(competition_slug: str, submission_file: str, message: str) -> str:
    return _command_text(build_submit_args(competition_slug, submission_file, message))


def build_leaderboard_args(competition_slug: str) -> list[str]:
    return ["kaggle", "competitions", "leaderboard", competition_slug, "--show"]


def build_submissions_args(competition_slug: str, page_size: int = 5) -> list[str]:
    return ["kaggle", "competitions", "submissions", competition_slug, "--csv", "--page-size", str(page_size)]


def build_competition_files_args(competition_slug: str) -> list[str]:
    return ["kaggle", "competitions", "files", "-c", competition_slug]


def build_leaderboard_command(competition_slug: str) -> str:
    return _command_text(build_leaderboard_args(competition_slug))


def _structured_result(action: str, args: list[str], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": action,
        "args": args,
        "command": _command_text(args),
        "returncode": result.get("returncode"),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "ok": result.get("returncode") == 0,
        "status": "ok" if result.get("returncode") == 0 else "failed",
    }


def _command_text(args: list[str]) -> str:
    return subprocess.list2cmdline(args)


def _parse_csv_leaderboard(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if "," not in stripped:
        return []
    rows = []
    for row in csv.DictReader(stripped.splitlines()):
        normalized = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
        team = normalized.get("teamname") or normalized.get("team") or normalized.get("name")
        score = _to_float(normalized.get("score"))
        if team and score is not None:
            rows.append({"team_name": team.strip(), "score": score})
    return rows


def _parse_table_leaderboard(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-", " "}:
            continue
        parts = stripped.rsplit(None, 1)
        if len(parts) != 2:
            continue
        score = _to_float(parts[1])
        if score is None:
            continue
        team = parts[0].strip()
        if team.casefold() in {"team", "teamname", "name"}:
            continue
        rows.append({"team_name": team, "score": score})
    return rows


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _parse_competition_files_csv(text: str) -> list[dict[str, Any]]:
    files = []
    for row in csv.DictReader(text.splitlines()):
        normalized = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
        name = normalized.get("name") or normalized.get("filename") or normalized.get("file")
        if not name:
            continue
        files.append(
            {
                "name": name.strip(),
                "size": str(normalized.get("size", "")).strip(),
                "creation_date": str(normalized.get("creationdate", "")).strip(),
            }
        )
    return files


def _parse_competition_files_table(text: str) -> list[dict[str, Any]]:
    files = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"-", " "}:
            continue
        lower = stripped.casefold()
        if lower.startswith("name ") or lower in {"name", "file"}:
            continue
        parts = stripped.split()
        if not parts:
            continue
        files.append({"name": parts[0], "size": parts[1] if len(parts) > 1 else "", "creation_date": ""})
    return files
