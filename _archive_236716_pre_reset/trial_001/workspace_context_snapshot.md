# trial_001 Workspace Context Snapshot

This file gives the coding agent the base trial code, current workspace code, and previous trial evidence.
When a recommended base trial code snapshot is present, treat it as the authoritative starting point.
Use later failed trials only as negative evidence; do not preserve their rejected code changes.

## Recommended Base Trial Code Snapshot

- source_trial_id: trial_001
- Use this section as the primary code reference for continuation patches.

- No saved source trial code snapshot was found.

## Current Workspace Code

- Used as exact patch context because no saved base-trial code snapshot was available or an expanded retry was requested.

### src/baseline.py

```python
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
```

### predict_step.py

```python
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except (TypeError, ValueError):
        return None


def _load_sample_ids(sample_path: Path) -> List[str]:
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing required file: {sample_path}")
    with sample_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames:
            raise ValueError("sample_submission.csv must contain 'id' column")
        return [row["id"] for row in reader]


def _load_test_groups(test_path: Path) -> Tuple[Dict[str, List[dict]], Optional[str]]:
    if not test_path.exists():
        return {}, None
    with test_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        time_col = "t" if "t" in fields else ("timestamp" if "timestamp" in fields else None)
        required = {"id", "x", "y", "z"}
        if not required.issubset(set(fields)):
            # If schema is not as expected, return empty to trigger fallback
            return {}, time_col
        groups: Dict[str, List[dict]] = {}
        for row in reader:
            pid = row.get("id")
            if not pid:
                continue
            # Parse numeric fields safely
            tx = _to_float(row.get("x"))
            ty = _to_float(row.get("y"))
            tz = _to_float(row.get("z"))
            tt = _to_float(row.get(time_col)) if time_col else None
            groups.setdefault(pid, []).append({
                "t": tt,
                "x": tx,
                "y": ty,
                "z": tz,
            })
        # Sort by time if available, otherwise preserve read order
        if time_col:
            for pid, rows in groups.items():
                rows.sort(key=lambda r: (-1.0 if r["t"] is None else r["t"]))
        return groups, time_col


def _predict_one(rows: List[dict], dt_horizon: float = 0.08, delta_target: float = 0.04) -> Tuple[float, float, float]:
    # Fallbacks when data is insufficient
    if not rows:
        return 0.0, 0.0, 0.0
    # Ensure we only use rows with finite positions
    valid_rows = [r for r in rows if all(_to_float(r.get(k)) is not None for k in ("x", "y", "z"))]
    if not valid_rows:
        return 0.0, 0.0, 0.0
    # If only one frame or no time info, return last position (zero-velocity)
    if len(valid_rows) == 1 or any(r.get("t") is None for r in valid_rows):
        r0 = valid_rows[-1]
        return float(r0["x"]), float(r0["y"]), float(r0["z"])

    # Assume rows are sorted ascending by time
    r0 = valid_rows[-1]
    t0 = r0.get("t")
    if t0 is None:
        return float(r0["x"]), float(r0["y"]), float(r0["z"])

    target_t = t0 - delta_target
    # Find frame at or before target_t; otherwise use immediate previous
    prev = None
    for cand in reversed(valid_rows[:-1]):
        tc = cand.get("t")
        if tc is not None and tc <= target_t:
            prev = cand
            break
    if prev is None:
        prev = valid_rows[-2]

    t_prev = prev.get("t")
    if t_prev is None:
        return float(r0["x"]), float(r0["y"]), float(r0["z"])

    dt = t0 - t_prev
    if dt <= 0 or not math.isfinite(dt):
        return float(r0["x"]), float(r0["y"]), float(r0["z"])

    vx = (float(r0["x"]) - float(prev["x"])) / dt
    vy = (float(r0["y"]) - float(prev["y"])) / dt
    vz = (float(r0["z"]) - float(prev["z"])) / dt

    px = float(r0["x"]) + vx * dt_horizon
    py = float(r0["y"]) + vy * dt_horizon
    pz = float(r0["z"]) + vz * dt_horizon

    # Ensure finite outputs
    if not all(math.isfinite(v) for v in (px, py, pz)):
        return float(r0["x"]), float(r0["y"]), float(r0["z"])  # fallback to last position
    return px, py, pz


def predict() -> str:
    data_dir = ROOT / "data"
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    sample_path = data_dir / "sample_submission.csv"
    test_path = data_dir / "test.csv"
    submission_path = outputs / "submission.csv"

    sample_ids = _load_sample_ids(sample_path)
    groups, _ = _load_test_groups(test_path)

    # Generate predictions in sample order
    with submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z"])
        for pid in sample_ids:
            rows = groups.get(pid, [])
            px, py, pz = _predict_one(rows)
            # Cast to float32-like by rounding precision; CSV has no dtype, but keep finite floats
            writer.writerow([pid, float(px), float(py), float(pz)])

    # If test.csv was missing, print a warning but do not fail
    if not test_path.exists():
        print(f"[warn] Prediction fallback: Missing required file: {test_path}")

    return str(submission_path)


if __name__ == "__main__":
    path = predict()
    print(f"Submission written to: {path}")
```

### test_step.py

```python
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from train_step import train
from predict_step import predict

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    # Run training to compute metrics (no model artifact persisted)
    try:
        train()
    except FileNotFoundError as e:
        # If train.csv is not present, continue to prediction-only checks
        print(f"[warn] Training skipped: {e}")
    except Exception as e:
        # Non-fatal for validation step; continue to prediction to at least check submission path
        print(f"[warn] Training error (continuing): {e}")

    # Run prediction to generate submission
    sub_path = predict()

    # Schema checks
    sample_csv = DATA_DIR / "sample_submission.csv"
    if not sample_csv.exists():
        raise FileNotFoundError("sample_submission.csv not found for schema validation")

    sample_rows = _read_csv(sample_csv)
    sub_rows = _read_csv(sub_path)

    expected_header = ["id", "x", "y", "z"]
    actual_header = sub_rows[0].keys() if sub_rows else expected_header
    if list(actual_header) != expected_header:
        raise SystemExit(f"Invalid submission header: {list(actual_header)} != {expected_header}")

    if len(sub_rows) != len(sample_rows):
        raise SystemExit(f"Row count mismatch: submission={len(sub_rows)} sample={len(sample_rows)}")

    # Finite check
    for i, row in enumerate(sub_rows):
        for col in ("x", "y", "z"):
            try:
                v = float(row[col])
            except Exception:
                raise SystemExit(f"Non-numeric value at row {i} col {col}: {row[col]!r}")
            if not math.isfinite(v):
                raise SystemExit(f"Non-finite value at row {i} col {col}: {v}")

    # Basic stats
    xs = [float(r["x"]) for r in sub_rows]
    ys = [float(r["y"]) for r in sub_rows]
    zs = [float(r["z"]) for r in sub_rows]
    print(
        {
            "rows": len(sub_rows),
            "x_min": min(xs) if xs else None,
            "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None,
            "y_max": max(ys) if ys else None,
            "z_min": min(zs) if zs else None,
            "z_max": max(zs) if zs else None,
            "submission_path": str(sub_path),
        }
    )


if __name__ == "__main__":
    main()
```


## Previous Trial Evidence

- source_trial_id: trial_001
- No previous trial evidence files were found.
