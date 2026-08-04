# trial_002 Workspace Context Snapshot

This file gives the coding agent the base trial code, current workspace code, and previous trial evidence.
When a recommended base trial code snapshot is present, treat it as the authoritative starting point.
Use later failed trials only as negative evidence; do not preserve their rejected code changes.

## Recommended Base Trial Code Snapshot

- source_trial_id: trial_001
- Use this section as the primary code reference for continuation patches.

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
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

TIME_CANDIDATES = ["timestep_ms", "timestamp_ms", "time", "t"]
POS_COLS = ["x", "y", "z"]
HORIZON_S = 0.08


def _read_sample_submission(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if [c.strip().casefold() for c in header] != ["id", "x", "y", "z"]:
            # Still accept as long as id present; output will be id,x,y,z
            if "id" not in header:
                raise ValueError("sample_submission.csv must contain columns: id,x,y,z")
        for row in reader:
            ids.append(str(row.get("id", "")).strip())
    return ids, ["id", "x", "y", "z"]


def _build_file_map(directory: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if directory.exists():
        for p in directory.glob("*.csv"):
            mapping[p.stem] = p
    return mapping


def _detect_time_column(header: List[str]) -> Optional[str]:
    lower = {h.strip().casefold(): h for h in header}
    for c in TIME_CANDIDATES:
        if c in lower:
            return lower[c]
    return None


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _read_series(path: Path) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        tcol = _detect_time_column(header)
        if not tcol:
            raise ValueError(f"Time column not found in {path.name}; expected one of {TIME_CANDIDATES}")
        rows: List[Tuple[float, float, float, float]] = []  # (t_ms, x, y, z)
        for row in reader:
            t = _to_float(row.get(tcol))
            x = _to_float(row.get("x"))
            y = _to_float(row.get("y"))
            z = _to_float(row.get("z"))
            if None in (t, x, y, z):
                continue
            rows.append((t, x, y, z))
    rows.sort(key=lambda r: r[0])
    times = [r[0] for r in rows]
    poss = [(r[1], r[2], r[3]) for r in rows]
    return times, poss


def _predict_last_cv(times: List[float], poss: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not times or not poss:
        return (0.0, 0.0, 0.0)
    if len(times) == 1 or len(poss) == 1:
        return poss[-1]
    t0, t1 = times[-2], times[-1]
    p0, p1 = poss[-2], poss[-1]
    dt_ms = t1 - t0
    if dt_ms <= 0:
        return p1
    scale = HORIZON_S / (dt_ms / 1000.0)
    vx = (p1[0] - p0[0]) * scale
    vy = (p1[1] - p0[1]) * scale
    vz = (p1[2] - p0[2]) * scale
    pred = (p1[0] + vx, p1[1] + vy, p1[2] + vz)
    if any(not math.isfinite(v) for v in pred):
        return p1
    return pred


def main() -> None:
    test_dir = DATA_DIR / "test"
    sample_sub = DATA_DIR / "sample_submission.csv"
    if not sample_sub.exists():
        raise FileNotFoundError("data/sample_submission.csv not found")
    if not test_dir.exists():
        raise FileNotFoundError("data/test/ directory not found")

    ids, out_cols = _read_sample_submission(sample_sub)
    fmap = _build_file_map(test_dir)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "submission.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_cols)
        for sid in ids:
            path = fmap.get(sid)
            if path is None:
                # Try case variants or direct filename
                cand = test_dir / f"{sid}.csv"
                path = cand if cand.exists() else None
            if path is None:
                raise FileNotFoundError(f"Test CSV for id {sid!r} not found in data/test/")
            times, poss = _read_series(path)
            px, py, pz = _predict_last_cv(times, poss)
            # Safety: ensure finite
            if not all(math.isfinite(v) for v in (px, py, pz)):
                # Fallback to last observed or zeros
                if poss:
                    px, py, pz = poss[-1]
                else:
                    px, py, pz = 0.0, 0.0, 0.0
            writer.writerow([sid, f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])


if __name__ == "__main__":
    main()
```

### test_step.py

```python
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

TIME_CANDIDATES = ["timestep_ms", "timestamp_ms", "time", "t"]
HORIZON_S = 0.08


def _detect_time_column(header: List[str]) -> Optional[str]:
    lower = {h.strip().casefold(): h for h in header}
    for c in TIME_CANDIDATES:
        if c in lower:
            return lower[c]
    return None


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _read_series(path: Path) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        tcol = _detect_time_column(header)
        if not tcol:
            raise ValueError(f"Time column not found in {path.name}; expected one of {TIME_CANDIDATES}")
        rows: List[Tuple[float, float, float, float]] = []  # (t_ms, x, y, z)
        for row in reader:
            t = _to_float(row.get(tcol))
            x = _to_float(row.get("x"))
            y = _to_float(row.get("y"))
            z = _to_float(row.get("z"))
            if None in (t, x, y, z):
                continue
            rows.append((t, x, y, z))
    rows.sort(key=lambda r: r[0])
    times = [r[0] for r in rows]
    poss = [(r[1], r[2], r[3]) for r in rows]
    return times, poss


def _predict_from_pair(p_prev, p_now, dt_ms: float) -> Tuple[float, float, float]:
    if dt_ms <= 0:
        return p_now
    scale = HORIZON_S / (dt_ms / 1000.0)
    vx = (p_now[0] - p_prev[0]) * scale
    vy = (p_now[1] - p_prev[1]) * scale
    vz = (p_now[2] - p_prev[2]) * scale
    pred = (p_now[0] + vx, p_now[1] + vy, p_now[2] + vz)
    if any(not math.isfinite(v) for v in pred):
        return p_now
    return pred


def _euclid(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def main() -> None:
    train_dir = DATA_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError("data/train/ directory not found for validation run")

    ids = [p.stem for p in train_dir.glob("*.csv")]
    ids.sort()

    if not ids:
        raise ValueError("No training CSV files found in data/train/")

    # Random holdout by id
    random.seed(42)
    n_valid = max(1, int(0.1 * len(ids)))
    valid_ids = set(random.sample(ids, n_valid))

    hits = 0
    total = 0
    evaluated_ids = 0

    for sid in ids:
        if sid not in valid_ids:
            continue
        path = train_dir / f"{sid}.csv"
        times, poss = _read_series(path)
        if len(times) < 3:
            continue
        # Choose anchor t0 as an index where a target near t0+80ms exists
        best_idx = None
        best_gap = None
        for i in range(1, len(times) - 1):  # leave at least one future point
            t0 = times[i]
            target_time = t0 + 80.0
            # Find nearest future index j with minimal |t[j]-target_time|
            j = None
            gap = None
            for k in range(i + 1, len(times)):
                g = abs(times[k] - target_time)
                if gap is None or g < gap:
                    gap = g
                    j = k
                if times[k] >= target_time and g <= (gap or g):
                    break
            # Accept if within 20 ms tolerance
            if j is not None and gap is not None and gap <= 20.0:
                # Prefer latest feasible anc
```


## Current Workspace Code Inventory

- Current workspace may contain rejected later-trial changes.
- The recommended base trial snapshot above is authoritative for patch find/replace text.
- These files WILL BE OVERWRITTEN with the base trial snapshot before your patch is applied. Never copy find text from them; local variable names and helper structure may differ from the base.
- src/baseline.py
- predict_step.py
- test_step.py
- train_step.py
- src/__init__.py

## Previous Trial Evidence

- source_trial_id: trial_001

### decision_card.md

```markdown
# Trial Decision Card: trial_001

- decision: baseline_established
- change_axis: None
- source_trial_id: trial_001
- recommended_base_trial: trial_001
- local_score: 0.591
- local_status: baseline
- local_delta: None
- previous_local_status: baseline
- previous_local_delta: None
- active_axis: None
- axis_attempt_count: 0
- axis_attempt_limit: 3
- lb_score: 0.6006
- lb_status: baseline
- lb_delta: None
- previous_lb_status: baseline
- previous_lb_delta: None

## Next Guidance

Use this trial as the baseline. The next trial should change exactly one improvement axis.

## Planner Constraints

- Use `trial_001` as the base trial unless the user explicitly overrides it.
- Change exactly one primary improvement axis in the next trial.
- Keep split/model/preprocessing fixed unless selected as the primary axis.

## Rejected Axes

- None

## Rejected Candidates

- None

## Active Axis Rejected Candidates

- None

## Accepted Axes

- None
```

### internal/decision_card.json

```json
{
  "schema_version": "1.0",
  "competition": "236716",
  "trial_id": "trial_001",
  "created_at": "2026-07-31T00:52:15.872933+00:00",
  "source_trial_id": "trial_001",
  "plan_type": "continuation_delta_plan",
  "change_axis": "",
  "local_score": 0.591,
  "lb_score": null,
  "previous_local_score": null,
  "previous_lb_score": null,
  "best_local_score": null,
  "best_lb_score": null,
  "local_delta": null,
  "lb_delta": null,
  "previous_local_delta": null,
  "previous_lb_delta": null,
  "objective": "maximize",
  "local_status": "baseline",
  "lb_status": "missing",
  "previous_local_status": "baseline",
  "previous_lb_status": "missing",
  "raw_decision": "baseline_established",
  "decision": "baseline_established",
  "model_type": "RuleBasedConstantVelocity",
  "catastrophic_regression": false,
  "estimator_family_changed": false,
  "no_change_suspected": false,
  "active_axis": null,
  "axis_attempt_count": 0,
  "axis_attempt_limit": 3,
  "candidate_label": "",
  "rejected_candidates": [],
  "rejected_candidates_by_axis": {},
  "active_axis_rejected_candidates": [],
  "recommended_base_trial": "trial_001",
  "rejected_axes": [],
  "accepted_axes": [],
  "next_guidance": "Use this trial as the baseline. The next trial should change exactly one improvement axis.",
  "planner_constraints": [
    "Use `trial_001` as the base trial unless the user explicitly overrides it.",
    "Change exactly one primary improvement axis in the next trial.",
    "Keep split/model/preprocessing fixed unless selected as the primary axis."
  ]
}
```

### metrics.json

```json
{
  "metric": "R-Hit@1cm",
  "objective": "maximize",
  "validation_method": "random_holdout_by_id",
  "validation_size": 0.1,
  "random_seed": 42,
  "R-Hit@1cm": 0.591,
  "n_ids": 1000,
  "total_pairs": 1000,
  "model_type": "RuleBasedConstantVelocity",
  "horizon_s": 0.08,
  "final_features": [
    "x",
    "y",
    "z"
  ],
  "time_candidates": [
    "timestep_ms",
    "timestamp_ms",
    "time",
    "t"
  ],
  "competition": "236716",
  "trial": "trial_001",
  "trial_id": "trial_001",
  "base_trial": null,
  "change_axis": null,
  "cv_score": 0.591,
  "source_metrics_path": "C:\\Users\\ASUS\\Desktop\\Research_Agent\\demo_workspaces\\236716\\outputs\\metrics.json",
  "pipeline_summary": {
    "competition": "236716",
    "trial": "trial_001",
    "trial_id": "trial_001",
    "base_trial": null,
    "change_axis": null
  },
  "lb_score": 0.6006,
  "submitted_lb_score": 0.6006,
  "submitted_rank": null,
  "submission_file": "demo_workspaces/236716/outputs/submission.csv",
  "submission_status": "submitted",
  "is_best_submission": true
}
```
