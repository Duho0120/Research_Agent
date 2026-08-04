"""Colab job worker.

Run this inside Colab after mounting/syncing the project folder. It scans the
`jobs` directory, executes pending jobs, and expects each job command to produce
metrics.json and optional submission.csv in the trial directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from research_agent import simple_yaml

ROOT = Path.cwd()
JOBS_DIR = ROOT / "jobs"


def load_job(path: Path) -> dict[str, Any]:
    job = simple_yaml.load(path, default={})
    if not isinstance(job, dict):
        raise ValueError(f"Job file must contain a YAML mapping: {path}")
    return job


def update_job(path: Path, job: dict[str, Any], *, status: str, error: str | None = None) -> dict[str, Any]:
    updated = dict(job)
    updated["status"] = status
    if error:
        updated["worker_error"] = error
    else:
        updated.pop("worker_error", None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(simple_yaml.to_yaml(updated), encoding="utf-8")
    temporary.replace(path)
    return updated


def run_pending_jobs(
    *,
    root: Path | None = None,
    jobs_dir: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    effective_root = root or ROOT
    effective_jobs_dir = jobs_dir or effective_root / "jobs"
    for job_path in sorted(effective_jobs_dir.rglob("*.yaml")):
        job = load_job(job_path)
        if str(job.get("status") or "").strip().lower() != "pending":
            continue
        command = str(job.get("command") or "").strip()
        if not command:
            update_job(job_path, job, status="failed", error="missing_command")
            print(f"Skipped {job.get('job_id', job_path.stem)}: missing command")
            continue
        print(f"Running {job.get('job_id', job_path.stem)}: {command}")
        running_job = update_job(job_path, job, status="running")
        try:
            result = runner(command, shell=True, cwd=effective_root)
            final_status = "done" if result.returncode == 0 else "failed"
            error = None if result.returncode == 0 else f"return_code_{result.returncode}"
        except Exception as exc:  # Keep the queue readable even when process creation fails.
            final_status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        update_job(job_path, running_job, status=final_status, error=error)
        print(f"Finished with status={final_status}")


if __name__ == "__main__":
    run_pending_jobs()
