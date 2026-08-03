from __future__ import annotations

import json
from typing import Any

from .data_onboarding import profile_competition_data
from .paths import competition_dir, trial_dir
from .store import load_state, write_text


def generate_baseline_pipeline(competition: str, trial_id: str = "trial_001") -> dict[str, Any]:
    profile = _load_or_create_data_profile(competition)
    competition_state = load_state(competition).get("competition", {})
    metric = str(competition_state.get("metric") or "unknown")
    objective = str(competition_state.get("objective") or "maximize")
    out_dir = trial_dir(competition, trial_id)
    target = profile["target_candidates"][0] if profile.get("target_candidates") else None
    blocking_issues = []
    if profile["status"] != "ready":
        blocking_issues.append("data_not_ready")
    if profile["task_type"] != "tabular":
        blocking_issues.append(f"unsupported_task_type:{profile['task_type']}")
    if not target:
        blocking_issues.append("missing_target_column")

    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked" if blocking_issues else "ready",
        "task_type": profile["task_type"],
        "target_column": target,
        "id_columns": profile.get("id_candidates", []),
        "metric": metric,
        "objective": objective,
        "blocking_issues": blocking_issues,
        "run_command": "",
        "outputs": [],
    }
    if blocking_issues:
        write_text(out_dir / "baseline_pipeline.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        write_text(out_dir / "baseline_plan.md", render_baseline_plan(result))
        return result

    script_path = out_dir / "baseline_train.py"
    write_text(script_path, _tabular_baseline_script())
    result["run_command"] = (
        f"python experiments/{competition}/{trial_id}/baseline_train.py "
        f"--competition {competition} --trial {trial_id} --target {target} "
        f"--metric {metric} --objective {objective}"
    )
    result["outputs"] = [
        f"experiments/{competition}/{trial_id}/metrics.json",
        f"experiments/{competition}/{trial_id}/submission.csv",
    ]
    write_text(out_dir / "baseline_pipeline.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "baseline_plan.md", render_baseline_plan(result))
    return result


def _load_or_create_data_profile(competition: str) -> dict[str, Any]:
    profile_path = competition_dir(competition) / "data_profile.json"
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    return profile_competition_data(competition)


def render_baseline_plan(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Baseline Pipeline",
        "",
        f"- status: {result['status']}",
        f"- task_type: {result['task_type']}",
        f"- target_column: {result.get('target_column')}",
        f"- id_columns: {', '.join(result.get('id_columns', [])) if result.get('id_columns') else 'None'}",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("blocking_issues", []) or ["None"])
    lines.extend(["", "## Run Command", ""])
    if result.get("run_command"):
        lines.append(f"```powershell\n{result['run_command']}\n```")
    else:
        lines.append("- Not available until blocking issues are resolved.")
    lines.extend(["", "## Expected Outputs", ""])
    lines.extend(f"- {item}" for item in result.get("outputs", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _tabular_baseline_script() -> str:
    return r'''from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--metric", default="unknown")
    parser.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    args = parser.parse_args()

    root = Path.cwd()
    data_dir = root / "data" / args.competition
    out_dir = root / "experiments" / args.competition / args.trial
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = _sample_submission_path(data_dir)
    train_rows = _read_csv(train_path)
    test_rows = _read_csv(test_path)
    sample_rows = _read_csv(sample_path) if sample_path.exists() else []
    if not train_rows:
        raise SystemExit(f"No training rows found: {train_path}")

    target_values = [row.get(args.target, "") for row in train_rows if row.get(args.target, "") != ""]
    task_kind = _task_kind(args.metric, args.objective, target_values)
    prediction_value = (
        _mean_value(target_values)
        if task_kind == "regression"
        else _majority_value(train_rows, args.target)
    )
    id_columns = _id_columns(train_rows, test_rows, sample_rows)
    submission_columns = _submission_columns(sample_rows, id_columns, args.target)
    submission_rows = _make_submission_rows(
        test_rows,
        sample_rows,
        submission_columns,
        id_columns,
        args.target,
        prediction_value,
    )
    _write_csv(out_dir / "submission.csv", submission_columns, submission_rows)

    metrics = {
        "trial_id": args.trial,
        "cv_score": (
            _regression_score(target_values, float(prediction_value), args.metric)
            if task_kind == "regression"
            else _training_accuracy(train_rows, args.target, str(prediction_value))
        ),
        "lb_score": None,
        "metric": args.metric,
        "objective": args.objective,
        "model_type": "mean_regression_baseline" if task_kind == "regression" else "majority_class_baseline",
        "task_kind": task_kind,
        "target_column": args.target,
        "prediction_value": prediction_value,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "notes": "Generated baseline for a first local execution and submission-format sanity check.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'submission.csv'}")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _sample_submission_path(data_dir: Path) -> Path:
    for path in data_dir.glob("*.csv"):
        normalized = "".join(character for character in path.stem.casefold() if character.isalnum())
        if normalized in {"samplesubmission", "gendersubmission"}:
            return path
    return data_dir / "sample_submission.csv"


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _majority_value(rows: list[dict[str, str]], target: str) -> str:
    counts = Counter(row.get(target, "") for row in rows)
    counts.pop("", None)
    if not counts:
        raise SystemExit(f"Target column has no values: {target}")
    return counts.most_common(1)[0][0]


def _task_kind(metric: str, objective: str, values: list[str]) -> str:
    normalized = metric.strip().casefold().replace("-", "_")
    if normalized in {"rmse", "rmsle", "mae", "mse", "mean_squared_error", "mean_absolute_error"}:
        return "regression"
    if normalized in {"accuracy", "f1", "roc_auc", "auc", "log_loss", "logloss"}:
        return "classification"
    try:
        [float(value) for value in values]
    except ValueError:
        return "classification"
    return "regression" if objective == "minimize" else "classification"


def _mean_value(values: list[str]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        raise SystemExit("Regression target has no numeric values.")
    return sum(numeric) / len(numeric)


def _regression_score(values: list[str], prediction: float, metric: str) -> float:
    actual = [float(value) for value in values]
    normalized = metric.strip().casefold().replace("-", "_")
    errors = [value - prediction for value in actual]
    if normalized in {"mae", "mean_absolute_error"}:
        return sum(abs(value) for value in errors) / len(errors)
    if normalized in {"mse", "mean_squared_error"}:
        return sum(value * value for value in errors) / len(errors)
    if normalized == "rmsle":
        prediction_log = math.log1p(max(0.0, prediction))
        return math.sqrt(
            sum((math.log1p(max(0.0, value)) - prediction_log) ** 2 for value in actual) / len(actual)
        )
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def _training_accuracy(rows: list[dict[str, str]], target: str, value: str) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row.get(target) == value)
    return round(correct / len(rows), 6)


def _id_columns(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], sample_rows: list[dict[str, str]]) -> list[str]:
    groups = [rows[0].keys() for rows in [train_rows, test_rows, sample_rows] if rows]
    if not groups:
        return []
    common = set(groups[0])
    for columns in groups[1:]:
        common &= set(columns)
    return [column for column in groups[0] if column in common and "id" in column.lower()]


def _submission_columns(sample_rows: list[dict[str, str]], id_columns: list[str], target: str) -> list[str]:
    if sample_rows:
        return list(sample_rows[0].keys())
    return [*(id_columns or ["id"]), target]


def _make_submission_rows(
    test_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    columns: list[str],
    id_columns: list[str],
    target: str,
    value: str,
) -> list[dict[str, str]]:
    source_rows = sample_rows if sample_rows else test_rows
    rows = []
    for index, source in enumerate(source_rows):
        row = {}
        for column in columns:
            if column == target:
                row[column] = value
            elif column in source:
                row[column] = source[column]
            elif column in id_columns and index < len(test_rows):
                row[column] = test_rows[index].get(column, "")
            else:
                row[column] = value
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
'''
