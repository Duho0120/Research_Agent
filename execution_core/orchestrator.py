"""One execution path for a trial.

Every caller -- a fresh trial, a retry, a resumed run -- goes through
``run_trial``. That is deliberate. The previous constant-predictor check lived
inside the coding function, so resuming an already-completed trial skipped it
entirely and a predictor returning the same answer for every sample was
recorded as a result twice.

The order below is also deliberate: the artifacts are written last, after the
observations pass. A trial whose predictions cannot respond to anything
produces no metrics file, so nothing downstream has a score to record.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import metrics as metrics_module
from .contract import (
    ContractViolation,
    DEFAULT_HOLDOUT_RATIO,
    DEFAULT_SEED,
    LOADER_MODULE,
    MODEL_MODULE,
)
from .runner_script import build_runner_source
from .submission_writer import write_submission
from .verification import read_template_ids, verify_test_ids


CONSTANT_PREDICTION_ISSUE = "predict_ignores_input:same_output_for_every_sample"


def run_trial(
    project_root: Path | str,
    python: str,
    *,
    data_dir: Path | str,
    metric: str,
    outputs_dir: Path | str | None = None,
    submission_template: Path | str | None = None,
    loader_module: str = LOADER_MODULE,
    model_module: str = MODEL_MODULE,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_SEED,
    allow_constant_predictions: bool = False,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Fit, predict, score and write this trial's artifacts.

    `allow_constant_predictions` is the caller's decision, not the core's: a
    first trial is legitimately a constant submission-format baseline, while a
    trial meant to build on a previous one is not.
    """
    root = Path(project_root)
    outputs = Path(outputs_dir) if outputs_dir else root / "outputs"
    result: dict[str, Any] = {
        "status": "blocked",
        "metric": metric,
        "cv_score": None,
        "issues": [],
        "project_root": str(root),
    }

    missing = [module for module in (loader_module, model_module) if not (root / f"{module}.py").is_file()]
    if missing:
        result["issues"] = [f"missing_module:{name}" for name in missing]
        result["next_action"] = "generate-missing-module"
        return result
    if metric not in metrics_module.METRICS:
        result["issues"] = [f"unknown_metric:{metric}"]
        result["next_action"] = "register-competition-metric"
        return result

    child = _run_child(
        root,
        python,
        data_dir=data_dir,
        loader_module=loader_module,
        model_module=model_module,
        holdout_ratio=holdout_ratio,
        seed=seed,
        timeout=timeout,
    )
    result["child"] = {key: value for key, value in child.items() if key not in _BULK_KEYS}
    if not child.get("ok"):
        result["status"] = "failed"
        result["failed_stage"] = child.get("stage")
        result["error"] = child.get("error")
        result["issues"] = [f"execution_failed:{child.get('stage') or 'unknown'}"]
        result["next_action"] = "fix-model-code"
        return result

    try:
        target_keys = _target_keys(child)
    except ContractViolation as violation:
        result["status"] = "failed"
        result["failed_stage"] = "loader_declarations"
        result["error"] = str(violation)
        result["issues"] = ["loader_declarations_inconsistent"]
        result["next_action"] = "fix-data-loader"
        return result

    # The loader is the one place that knows this competition's layout, so its
    # output is checked against an anchor the competition itself supplies. A
    # loader reading the wrong files otherwise produces a perfectly plausible
    # score.
    template_check = _verify_against_template(child, submission_template)
    result["template_verification"] = template_check
    if template_check.get("issues"):
        result["status"] = "failed"
        result["failed_stage"] = "loader_anchor"
        result["issues"] = template_check["issues"]
        result["next_action"] = "fix-data-loader"
        return result

    distinct = _distinct_predictions(child["holdout_predictions"])
    result["distinct_holdout_predictions"] = distinct
    result["holdout_count"] = child["n_holdout"]
    result["n_train"] = child["n_train"]
    result["n_test"] = child["n_test"]
    result["fit_returned"] = child.get("fit_returned")

    # Observed, not inferred. A constant predictor is syntactically legal under
    # any contract, so no signature can rule it out -- only running it can.
    if distinct < 2 and child["n_holdout"] >= 2 and not allow_constant_predictions:
        result["status"] = "blocked_constant_predictor"
        result["issues"] = [CONSTANT_PREDICTION_ISSUE]
        result["next_action"] = "make-predict-use-its-input"
        return result

    try:
        score = metrics_module.compute(metric, child["holdout_predictions"], child["truths"], target_keys)
    except (ValueError, KeyError) as error:
        result["status"] = "failed"
        result["failed_stage"] = "score"
        result["error"] = f"{type(error).__name__}: {error}"
        result["issues"] = ["scoring_failed"]
        result["next_action"] = "fix-model-output-shape"
        return result

    try:
        submission = write_submission(
            outputs / "submission.csv",
            columns=child["submission_columns"],
            id_column=child["id_column"],
            ids=child["test_ids"],
            predictions=child["test_predictions"],
        )
    except ContractViolation as violation:
        result["status"] = "failed"
        result["failed_stage"] = "submission"
        result["error"] = str(violation)
        result["issues"] = ["submission_write_failed"]
        result["next_action"] = "fix-model-output-shape"
        return result

    result["cv_score"] = score
    result["objective"] = metrics_module.objective_of(metric)
    result["submission"] = submission
    result["metrics_path"] = str(_write_metrics(outputs, result, child))
    result["status"] = "completed"
    result["next_action"] = "collect-metrics"
    return result


_BULK_KEYS = {"truths", "holdout_predictions", "test_predictions", "test_ids"}


def _run_child(
    root: Path,
    python: str,
    *,
    data_dir: Path | str,
    loader_module: str,
    model_module: str,
    holdout_ratio: float,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    core_path = str(Path(__file__).resolve().parent.parent)
    with tempfile.TemporaryDirectory() as workdir:
        out_path = Path(workdir) / "result.json"
        script_path = Path(workdir) / "execution_core_runner.py"
        script_path.write_text(
            build_runner_source(
                core_path=core_path,
                out_path=str(out_path),
                data_dir=str(data_dir),
                loader_module=loader_module,
                model_module=model_module,
                holdout_ratio=holdout_ratio,
                seed=seed,
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [python, str(script_path)],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "timeout", "error": f"exceeded {timeout}s"}
        if not out_path.is_file():
            # The child died before it could report -- a segfault, an OOM kill,
            # or an interpreter-level failure. Its stderr is all we have.
            return {
                "ok": False,
                "stage": "child_process",
                "error": (completed.stderr or completed.stdout or "child produced no result").strip()[-4000:],
                "returncode": completed.returncode,
            }
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return {"ok": False, "stage": "child_result", "error": f"unreadable result: {error}"}
    payload["stdout_tail"] = (completed.stdout or "")[-2000:]
    return payload


def _target_keys(child: dict[str, Any]) -> list[str]:
    """The keys predict() must produce, cross-checked against both declarations.

    Requiring label_keys() and the submission's non-id columns to agree is a
    real constraint on the loader, and a deliberate one: it collapses "what the
    model is scored on" and "what gets submitted" into one set of names, so a
    model can never be scored on one thing and submitted for another.
    """
    label_keys = list(child["label_keys"])
    submission_targets = [c for c in child["submission_columns"] if c != child["id_column"]]
    if set(label_keys) != set(submission_targets):
        raise ContractViolation(
            f"label_keys() {sorted(label_keys)} does not match the submission's non-id columns "
            f"{sorted(submission_targets)}; they must name the same values."
        )
    return submission_targets


def _verify_against_template(
    child: dict[str, Any], submission_template: Path | str | None
) -> dict[str, Any]:
    """Check the loader's test ids against the competition's own template.

    A caller with no template declares that by passing None; the skip is
    recorded rather than assumed, so an unverified run is visible in the
    result instead of looking identical to a verified one.
    """
    if submission_template is None:
        return {"status": "skipped", "reason": "no_template_declared", "issues": []}
    path = Path(submission_template)
    if not path.is_file():
        return {
            "status": "failed",
            "reason": "template_not_found",
            "template": str(path),
            "issues": [f"submission_template_not_found:{path.name}"],
        }
    try:
        template_ids = read_template_ids(path, child["id_column"])
    except (ValueError, OSError, csv.Error) as error:
        return {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "template": str(path),
            "issues": ["submission_template_unreadable"],
        }
    issues = verify_test_ids(child["test_ids"], template_ids)
    return {
        "status": "failed" if issues else "verified",
        "template": str(path),
        "template_rows": len(template_ids),
        "loaded_rows": len(child["test_ids"]),
        "issues": issues,
    }


def _distinct_predictions(predictions: list[Any]) -> int:
    return len({json.dumps(prediction, sort_keys=True, default=str) for prediction in predictions})


def _write_metrics(outputs: Path, result: dict[str, Any], child: dict[str, Any]) -> Path:
    """Written by the framework, from the score the framework computed.

    Same path and same keys as the old agent-written artifact, so the metrics
    collector, the state DB and the dashboard keep working untouched.
    """
    path = outputs / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cv_score": result["cv_score"],
                "metric": result["metric"],
                "objective": result["objective"],
                "holdout_count": child["n_holdout"],
                "n_train": child["n_train"],
                "n_test": child["n_test"],
                "distinct_holdout_predictions": result["distinct_holdout_predictions"],
                "generated_by": "execution_core",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
