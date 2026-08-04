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
