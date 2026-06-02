"""Colab worker skeleton.

Run this inside Colab after mounting/syncing the project folder. It scans the
`jobs` directory, executes pending jobs, and expects each job command to produce
metrics.json and optional submission.csv in the trial directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path.cwd()
JOBS_DIR = ROOT / "jobs"


def parse_simple_job(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
    return data


def run_pending_jobs() -> None:
    for job_path in sorted(JOBS_DIR.rglob("*.yaml")):
        text = job_path.read_text(encoding="utf-8")
        if "status: pending" not in text:
            continue
        job = parse_simple_job(job_path)
        command = job["command"]
        print(f"Running {job.get('job_id', job_path.stem)}: {command}")
        job_path.write_text(text.replace("status: pending", "status: running"), encoding="utf-8")
        result = subprocess.run(command, shell=True, cwd=ROOT)
        updated = job_path.read_text(encoding="utf-8")
        final_status = "done" if result.returncode == 0 else "failed"
        job_path.write_text(updated.replace("status: running", f"status: {final_status}"), encoding="utf-8")
        print(f"Finished with status={final_status}")


if __name__ == "__main__":
    run_pending_jobs()
