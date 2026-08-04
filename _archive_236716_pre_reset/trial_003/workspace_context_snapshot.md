# trial_003 Workspace Context Snapshot

This file gives the coding agent the base trial code, current workspace code, and previous trial evidence.
When a recommended base trial code snapshot is present, treat it as the authoritative starting point.
Use later failed trials only as negative evidence; do not preserve their rejected code changes.

## Recommended Base Trial Code Snapshot

- source_trial_id: trial_001
- Use this section as the primary code reference for continuation patches.

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
                # Prefer latest feasible anchor (closer to end) to emulate test
                best_idx = (i, j)
                best_gap = gap
        if not best_idx:
            continue
        i, j = best_idx
        p_prev, p_now = poss[i - 1], poss[i]
        dt_ms = times[i] - times[i - 1]
        pred = _predict_from_pair(p_prev, p_now, dt_ms)
        true_p = poss[j]
        dist = _euclid(pred, true_p)
        total += 1
        if dist <= 0.01:
            hits += 1
        evaluated_ids += 1

    score = (hits / total) if total > 0 else 0.0

    OUTPUTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "metric": "R-Hit@1cm",
        "objective": "maximize",
        "validation_method": "random_holdout_by_id",
        "validation_size": 0.1,
        "random_seed": 42,
        "R-Hit@1cm": float(score),
        "n_ids": int(evaluated_ids),
        "total_pairs": int(total),
        "model_type": "RuleBasedConstantVelocity",
        "horizon_s": HORIZON_S,
        "final_features": ["x", "y", "z"],
        "time_candidates": TIME_CANDIDATES,
    }

    with (OUTPUTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(me
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

- Omitted for compact delta patch mode. Use delta_plan and decision card outputs for trial strategy.
