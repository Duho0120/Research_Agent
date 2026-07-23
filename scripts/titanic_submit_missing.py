"""Deprecated Titanic-only submission repair helper.

This helper is kept for archived prototype cleanup only. New agent experiments
should submit and record scores through generic_workspace_auto_loop.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPTS_DIR]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from titanic_auto_submit_loop import (  # noqa: E402
    record_project_submission,
    save_manual_metrics,
    submit_to_kaggle,
    update_manual_metrics,
    wait_for_submission_result,
    write_loop_decision,
)
from titanic_run_5_trials import RUN_DIR, TRIALS, feature_columns, write_user_artifacts  # noqa: E402


def load_metrics(trial_id: str) -> dict[str, Any]:
    path = RUN_DIR / trial_id / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics for {trial_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def submitted_lb_before(trial_id: str) -> float | None:
    previous: float | None = None
    for spec in TRIALS:
        if spec.trial_id == trial_id:
            return previous
        try:
            metrics = load_metrics(spec.trial_id)
        except FileNotFoundError:
            continue
        lb = metrics.get("kaggle_lb_score")
        if metrics.get("kaggle_submitted") and lb is not None:
            previous = float(lb)
    return previous


def best_lb_before(trial_id: str) -> float | None:
    best: float | None = None
    for spec in TRIALS:
        if spec.trial_id == trial_id:
            return best
        try:
            metrics = load_metrics(spec.trial_id)
        except FileNotFoundError:
            continue
        lb = metrics.get("kaggle_lb_score")
        if metrics.get("kaggle_submitted") and lb is not None:
            value = float(lb)
            best = value if best is None else max(best, value)
    return best


def submit_missing_trial(trial_id: str, *, poll_attempts: int, poll_interval_seconds: float) -> dict[str, Any]:
    spec_by_id = {spec.trial_id: spec for spec in TRIALS}
    if trial_id not in spec_by_id:
        raise ValueError(f"Unknown Titanic trial: {trial_id}")

    metrics = load_metrics(trial_id)
    if metrics.get("kaggle_submitted") and metrics.get("kaggle_lb_score") is not None:
        print(f"{trial_id}: already submitted, LB={metrics.get('kaggle_lb_score')}")
        return metrics

    submission_file = metrics.get("submission_file")
    if not submission_file or not Path(submission_file).exists():
        raise FileNotFoundError(f"{trial_id}: submission file is missing: {submission_file}")

    message = f"titanic {trial_id} local CV {float(metrics['local_score']):.6f}"
    print(f"{trial_id}: submitting {submission_file}")
    submit_to_kaggle(str(submission_file), message)
    submission_result = wait_for_submission_result(
        message,
        attempts=poll_attempts,
        interval_seconds=poll_interval_seconds,
    )

    previous_lb = submitted_lb_before(trial_id)
    best_lb = best_lb_before(trial_id)
    updated = update_manual_metrics(metrics, submission_result)
    save_manual_metrics(updated)

    spec = spec_by_id[trial_id]
    numeric, categorical = feature_columns(spec.feature_mode)
    trial_dir = Path(updated["submission_file"]).parent
    write_user_artifacts(trial_dir, spec, updated, numeric, categorical)
    record_project_submission(updated, submission_result, previous_lb)

    next_trial = next_trial_after(trial_id)
    write_loop_decision(trial_dir, updated, previous_lb, best_lb, next_trial)
    print(f"{trial_id}: recorded Kaggle LB={updated.get('kaggle_lb_score')} ref={updated.get('kaggle_ref')}")
    return updated


def next_trial_after(trial_id: str) -> str | None:
    trial_ids = [spec.trial_id for spec in TRIALS]
    index = trial_ids.index(trial_id)
    return trial_ids[index + 1] if index + 1 < len(trial_ids) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit existing unsubmitted Titanic manual trial files.")
    parser.add_argument("--trials", nargs="*", default=None, help="Specific trial ids. Defaults to all unsubmitted trials.")
    parser.add_argument("--poll-attempts", type=int, default=12)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial_ids = args.trials or [spec.trial_id for spec in TRIALS]
    submitted = []
    for trial_id in trial_ids:
        metrics = load_metrics(trial_id)
        if metrics.get("kaggle_submitted") and metrics.get("kaggle_lb_score") is not None:
            print(f"{trial_id}: skip, already submitted.")
            continue
        submitted.append(
            submit_missing_trial(
                trial_id,
                poll_attempts=args.poll_attempts,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        )
    if not submitted:
        print("No missing Titanic submissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
