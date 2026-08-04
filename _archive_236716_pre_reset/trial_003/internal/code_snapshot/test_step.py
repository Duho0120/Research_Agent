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
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Also print a concise summary for logs
    print(f"R-Hit@1cm: {score:.6f} on {evaluated_ids} ids ({total} pairs)")


if __name__ == "__main__":
    main()
