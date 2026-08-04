from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "workspace_config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def check_required_data() -> None:
    config = load_config()
    missing = []
    for name in config.get("required_data_files", []):
        if not (ROOT / "data" / name).exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError("Missing data files in data/: " + ", ".join(missing))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_kind(config: dict, labels: list[str]) -> str:
    metric = str(config.get("metric") or "").strip().casefold().replace("-", "_")
    if metric in {"rmse", "rmsle", "mae", "mse", "mean_squared_error", "mean_absolute_error"}:
        return "regression"
    if metric in {"accuracy", "f1", "roc_auc", "auc", "log_loss", "logloss"}:
        return "classification"
    numeric = [_to_float(value) for value in labels]
    if numeric and all(value is not None for value in numeric) and config.get("objective") == "minimize":
        return "regression"
    return "classification"


def _regression_score(metric: str, actual: list[float], prediction: float) -> float:
    if not actual:
        return 0.0
    errors = [value - prediction for value in actual]
    normalized = metric.strip().casefold().replace("-", "_")
    if normalized == "mae" or normalized == "mean_absolute_error":
        return sum(abs(value) for value in errors) / len(errors)
    if normalized == "mse" or normalized == "mean_squared_error":
        return sum(value * value for value in errors) / len(errors)
    if normalized == "rmsle":
        predicted_log = math.log1p(max(0.0, prediction))
        return math.sqrt(
            sum((math.log1p(max(0.0, value)) - predicted_log) ** 2 for value in actual) / len(actual)
        )
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def train() -> dict:
    check_required_data()
    config = load_config()
    target = config.get("target_column")
    if not target:
        raise ValueError("workspace_config.json must define target_column before training.")
    train_path = _find_train_file(target)
    rows = _read_csv(train_path)
    if not rows:
        raise ValueError(f"No rows found in {train_path}")
    if target not in rows[0]:
        raise ValueError(f"Target column {target!r} not found in {train_path}")

    split_at = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split_at]
    valid_rows = rows[split_at:] or rows[:]
    labels = [row[target] for row in train_rows if row.get(target) not in {None, ""}]
    if not labels:
        raise ValueError(f"No labels found in target column {target!r}")
    task_kind = _task_kind(config, labels)
    metric = str(config.get("metric") or "unknown")
    if task_kind == "regression":
        numeric_labels = [value for value in (_to_float(item) for item in labels) if value is not None]
        if not numeric_labels:
            raise ValueError(f"Regression target {target!r} does not contain numeric labels.")
        prediction = sum(numeric_labels) / len(numeric_labels)
        valid_labels = [
            value
            for value in (_to_float(row.get(target)) for row in valid_rows)
            if value is not None
        ]
        score = _regression_score(metric, valid_labels, prediction)
        strategy = "mean_regression"
    else:
        prediction = Counter(labels).most_common(1)[0][0]
        correct = sum(1 for row in valid_rows if row.get(target) == prediction)
        score = correct / len(valid_rows) if valid_rows else 0.0
        strategy = "majority_class"

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    model = {
        "strategy": strategy,
        "task_kind": task_kind,
        "prediction": prediction,
        "target_column": target,
        "id_column": config.get("id_column"),
        "train_file": train_path.name,
    }
    metrics = {
        "cv_score": score,
        "metric": metric,
        "objective": config.get("objective", "maximize"),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "strategy": strategy,
        "model_type": "MeanRegressor" if task_kind == "regression" else "MajorityClassClassifier",
    }
    (outputs / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (outputs / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def predict() -> Path:
    config = load_config()
    outputs = ROOT / "outputs"
    model_path = outputs / "model.json"
    if not model_path.exists():
        train()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    test_path = _find_test_file(model.get("target_column"))
    rows = _read_csv(test_path)
    target = model.get("target_column") or config.get("target_column") or "prediction"
    id_column = config.get("id_column") or model.get("id_column") or _first_column(rows)
    submission_path = outputs / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[id_column, target])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({id_column: row.get(id_column, index), target: model["prediction"]})
    return submission_path


def _find_train_file(target: str) -> Path:
    preferred = ROOT / "data" / "train.csv"
    if preferred.exists():
        return preferred
    for path in sorted((ROOT / "data").glob("*.csv")):
        try:
            rows = _read_csv(path, limit=1)
        except ValueError:
            continue
        if rows and target in rows[0]:
            return path
    raise FileNotFoundError("Could not find a CSV train file containing the target column.")


def _find_test_file(target: str | None) -> Path:
    preferred = ROOT / "data" / "test.csv"
    if preferred.exists():
        return preferred
    for path in sorted((ROOT / "data").glob("*.csv")):
        if path.name.lower().startswith("train"):
            continue
        rows = _read_csv(path, limit=1)
        if rows and (not target or target not in rows[0]):
            return path
    raise FileNotFoundError("Could not find a CSV test file.")


def _read_csv(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = []
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows


def _first_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "id"
    return next(iter(rows[0].keys()), "id")
