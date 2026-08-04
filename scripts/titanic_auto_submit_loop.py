"""Deprecated Titanic-only manual loop.

This helper is kept for reproducing the early prototype only. The agent CLI,
local app, web app, and deployment path must use generic_workspace_auto_loop.py.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPTS_DIR]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from research_agent.agents.submission import record_submission_result  # noqa: E402

from titanic_run_5_trials import (  # noqa: E402
    ROOT,
    RUN_DIR,
    TRIALS,
    feature_columns,
    run_trial,
    write_user_artifacts,
)


SUBMISSION_STATUS_COMPLETE = "SubmissionStatus.COMPLETE"
RUNTIME_DIR = Path(os.environ.get("RESEARCH_AGENT_RUNTIME_DIR", str(RUN_DIR))).expanduser().resolve()
STATE_PATH = RUNTIME_DIR / "auto_loop_state.json"
LOCK_PATH = RUNTIME_DIR / "auto_loop.lock"
PAUSE_REQUEST_PATH = RUNTIME_DIR / "pause.request"


def run_command(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def submit_to_kaggle(submission_file: str, message: str) -> None:
    command = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "titanic",
        "-f",
        submission_file,
        "-m",
        message,
    ]
    completed = run_command(command, timeout=180)
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Kaggle submit failed with exit code {completed.returncode}")


def fetch_submissions() -> str:
    completed = run_command(["kaggle", "competitions", "submissions", "-c", "titanic"], timeout=120)
    if completed.returncode != 0:
        print(completed.stdout.strip())
        print(completed.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Kaggle submissions query failed with exit code {completed.returncode}")
    return completed.stdout


def wait_for_submission_result(message: str, *, attempts: int, interval_seconds: float) -> dict[str, Any]:
    last_output = ""
    for attempt in range(1, attempts + 1):
        output = fetch_submissions()
        last_output = output
        result = parse_submission_table(output, message)
        if result and result.get("status") == SUBMISSION_STATUS_COMPLETE and result.get("public_score") is not None:
            return result
        print(f"Waiting for Kaggle score ({attempt}/{attempts})...")
        time.sleep(interval_seconds)
    raise RuntimeError("Could not find a completed Kaggle submission with publicScore.\n" + last_output)


def parse_submission_table(output: str, message: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        if message not in line:
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 6:
            continue
        if len(parts) >= 7:
            status = parts[-3]
            public_score = parse_float(parts[-2])
            private_score = parse_float(parts[-1])
        else:
            status = parts[-2]
            public_score = parse_float(parts[-1])
            private_score = None
        return {
            "ref": parts[0],
            "file_name": parts[1],
            "date": parts[2],
            "description": message,
            "status": status,
            "public_score": public_score,
            "private_score": private_score,
            "raw": line,
        }
    return None


def parse_float(value: str) -> float | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def update_manual_metrics(metrics: dict[str, Any], submission_result: dict[str, Any]) -> dict[str, Any]:
    updated = dict(metrics)
    updated["kaggle_submitted"] = True
    updated["kaggle_returncode"] = 0
    updated["kaggle_ref"] = submission_result.get("ref")
    updated["kaggle_status"] = submission_result.get("status")
    updated["kaggle_lb_score"] = submission_result.get("public_score")
    updated["kaggle_private_score"] = submission_result.get("private_score")
    return updated


def save_manual_metrics(metrics: dict[str, Any]) -> None:
    trial_dir = Path(metrics["submission_file"]).parent
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def write_loop_decision(
    trial_dir: Path,
    metrics: dict[str, Any],
    previous_lb: float | None,
    best_lb: float | None,
    next_trial: str | None,
) -> None:
    current_lb = metrics.get("kaggle_lb_score")
    delta_previous = None if previous_lb is None or current_lb is None else round(current_lb - previous_lb, 10)
    delta_best = None if best_lb is None or current_lb is None else round(current_lb - best_lb, 10)
    lines = [
        f"# {metrics['trial_id']} loop decision",
        "",
        f"- local_score: {metrics.get('local_score')}",
        f"- kaggle_lb_score: {current_lb}",
        f"- previous_lb_score: {previous_lb}",
        f"- delta_vs_previous: {delta_previous}",
        f"- best_lb_before: {best_lb}",
        f"- delta_vs_best_before: {delta_best}",
        f"- next_trial: {next_trial or '-'}",
        "",
    ]
    if current_lb is None:
        lines.append("Decision: stop, missing leaderboard score.")
    elif best_lb is None or current_lb >= best_lb:
        lines.append("Decision: continue from this trial as the current loop best.")
    else:
        lines.append("Decision: continue, but keep the previous best as the comparison baseline.")
    lines.append("")
    (trial_dir / "04_loop_decision.md").write_text("\n".join(lines), encoding="utf-8")


def record_project_submission(
    metrics: dict[str, Any],
    submission_result: dict[str, Any],
    previous_lb: float | None,
) -> dict[str, Any]:
    ref = submission_result.get("ref") or "unknown"
    return record_submission_result(
        competition="titanic",
        trial_id=metrics["trial_id"],
        version_name=f"{metrics['trial_id']}_kaggle_{ref}",
        submission_file=metrics["submission_file"],
        cv_score=metrics.get("local_score"),
        previous_lb_score=previous_lb,
        submitted_lb_score=submission_result.get("public_score"),
        objective="maximize",
        notes=f"Kaggle ref {ref}; recorded by titanic_auto_submit_loop.py",
    )


def write_summary(rows: list[dict[str, Any]]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "auto_loop_summary.csv"
    fieldnames = [
        "trial_id",
        "local_score",
        "kaggle_lb_score",
        "kaggle_ref",
        "change_axis",
        "feature_mode",
        "model",
        "submission_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Auto-loop summary: {path}")


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_loop_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_loop_state(**updates: Any) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    state = load_loop_state()
    state.update(updates)
    state["competition"] = "titanic"
    state["updated_at"] = now_text()
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)
    return state


def acquire_loop_lock() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = load_lock_owner()
            if owner and process_is_alive(owner):
                raise SystemExit(f"Another auto loop is already running (PID {owner}).")
            try:
                LOCK_PATH.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps({"pid": os.getpid(), "started_at": now_text()}) + "\n")
        atexit.register(release_loop_lock)
        return
    raise SystemExit("Could not acquire the auto-loop lock.")


def load_lock_owner() -> int | None:
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("pid")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def release_loop_lock() -> None:
    if load_lock_owner() != os.getpid():
        return
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def pause_is_requested() -> bool:
    return PAUSE_REQUEST_PATH.exists() or bool(load_loop_state().get("pause_requested"))


def clear_pause_request() -> None:
    try:
        PAUSE_REQUEST_PATH.unlink()
    except FileNotFoundError:
        pass


def request_pause() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PAUSE_REQUEST_PATH.write_text(now_text() + "\n", encoding="utf-8")
    state = save_loop_state(pause_requested=True)
    print(f"Pause requested. Current status={state.get('status', '-')}. The loop will stop after the current trial is submitted and recorded.")


def print_status() -> None:
    state = load_loop_state()
    if not state:
        print("No auto-loop state found.")
        return
    print(json.dumps(state, indent=2))


def next_trial_after(trial_id: str | None) -> str | None:
    trial_ids = [trial.trial_id for trial in TRIALS]
    if not trial_id:
        return trial_ids[0] if trial_ids else None
    if trial_id not in trial_ids:
        return None
    index = trial_ids.index(trial_id) + 1
    return trial_ids[index] if index < len(trial_ids) else None


def resume_start_trial() -> str:
    state = load_loop_state()
    next_trial = state.get("next_trial") or next_trial_after(state.get("last_completed_trial"))
    if not next_trial:
        raise SystemExit("No next trial to resume.")
    save_loop_state(status="resuming", pause_requested=False, next_trial=next_trial)
    return str(next_trial)


def validate_credential_mode() -> None:
    has_legacy = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    if not has_legacy and not has_token:
        print(
            "Warning: no Kaggle credential environment variables detected. "
            "The Kaggle CLI may still work if the user has a local Kaggle login.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    trial_ids = [trial.trial_id for trial in TRIALS]
    parser = argparse.ArgumentParser(description="Run Titanic trials, submit to Kaggle, record LB, then continue.")
    parser.add_argument("--start", default="trial_001", choices=trial_ids)
    parser.add_argument("--end", default="trial_005", choices=trial_ids)
    parser.add_argument("--poll-attempts", type=int, default=12)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--stop-on-submit-failure", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from auto_loop_state.json next_trial.")
    parser.add_argument("--request-pause", action="store_true", help="Ask a running loop to pause after the current trial.")
    parser.add_argument("--status", action="store_true", help="Print auto-loop state and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print_status()
        return 0
    if args.request_pause:
        request_pause()
        return 0
    acquire_loop_lock()
    state_before_start = load_loop_state()
    if state_before_start.get("status") in {"paused", "completed", "failed"}:
        clear_pause_request()
    validate_credential_mode()

    trial_ids = [trial.trial_id for trial in TRIALS]
    start_trial = resume_start_trial() if args.resume else args.start
    start_index = trial_ids.index(start_trial)
    end_index = trial_ids.index(args.end)
    if start_index > end_index:
        raise SystemExit("--start must be before or equal to --end")

    previous_lb: float | None = None
    best_lb: float | None = None
    rows: list[dict[str, Any]] = []

    selected = TRIALS[start_index : end_index + 1]
    save_loop_state(
        status="running",
        pause_requested=False,
        pid=os.getpid(),
        start_trial=selected[0].trial_id if selected else None,
        end_trial=selected[-1].trial_id if selected else None,
        next_trial=selected[0].trial_id if selected else None,
    )
    for index, spec in enumerate(selected):
        next_trial = selected[index + 1].trial_id if index + 1 < len(selected) else None
        print(f"\n=== {spec.trial_id}: {spec.change_axis} ===", flush=True)
        save_loop_state(status="running", current_trial=spec.trial_id, next_trial=spec.trial_id)
        metrics = run_trial(spec, submit=False, poll_seconds=0)
        message = f"titanic {spec.trial_id} local CV {metrics['local_score']:.6f}"
        submit_to_kaggle(metrics["submission_file"], message)
        submission_result = wait_for_submission_result(
            message,
            attempts=args.poll_attempts,
            interval_seconds=args.poll_interval_seconds,
        )
        metrics = update_manual_metrics(metrics, submission_result)
        save_manual_metrics(metrics)

        numeric, categorical = feature_columns(spec.feature_mode)
        write_user_artifacts(Path(metrics["submission_file"]).parent, spec, metrics, numeric, categorical)
        record_project_submission(metrics, submission_result, previous_lb)
        write_loop_decision(Path(metrics["submission_file"]).parent, metrics, previous_lb, best_lb, next_trial)

        current_lb = metrics.get("kaggle_lb_score")
        previous_lb = current_lb
        if current_lb is not None and (best_lb is None or current_lb > best_lb):
            best_lb = current_lb
        rows.append(metrics)
        state = save_loop_state(
            status="running",
            current_trial=None,
            last_completed_trial=spec.trial_id,
            next_trial=next_trial,
            best_lb_score=best_lb,
            previous_lb_score=previous_lb,
        )
        print(f"Recorded Kaggle LB for {spec.trial_id}: {current_lb}")
        if state.get("pause_requested") or pause_is_requested():
            save_loop_state(status="paused", current_trial=None, next_trial=next_trial)
            print(f"Paused after {spec.trial_id}. Resume will start from {next_trial or 'no remaining trial'}.")
            write_summary(rows)
            return 0

    clear_pause_request()
    save_loop_state(status="completed", current_trial=None, next_trial=None, pause_requested=False, pid=None)
    write_summary(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        save_loop_state(status="failed", current_trial=None, error=str(error), pid=None)
        print(f"Auto loop failed: {error}", file=sys.stderr)
        raise
