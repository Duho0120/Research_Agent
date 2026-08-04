from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def _detect_time_column(header: list[str]) -> Optional[str]:
    candidates = ["timestep_ms", "timestamp_ms", "time", "t"]
    lower = {h.strip().casefold(): h for h in header}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _peek_first_train_file() -> Optional[Path]:
    train_dir = DATA_DIR / "train"
    if not train_dir.exists():
        return None
    for p in sorted(train_dir.glob("*.csv")):
        return p
    return None


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [h.strip() for h in header]


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    sample_sub = DATA_DIR / "sample_submission.csv"

    n_train_files = len(list(train_dir.glob("*.csv"))) if train_dir.exists() else 0
    n_test_files = len(list(test_dir.glob("*.csv"))) if test_dir.exists() else 0

    header = []
    time_col = None
    feat_cols = []
    example = _peek_first_train_file()
    if example and example.exists():
        header = _read_header(example)
        time_col = _detect_time_column(header)
        # Feature columns are positional axes available
        for c in ["x", "y", "z"]:
            if c in header:
                feat_cols.append(c)

    summary = {
        "pipeline": "RuleBasedConstantVelocity",
        "horizon_s": 0.08,
        "uses_fit": False,
        "dataset_layout": "per_sample_files",
        "data_paths": {
            "train_dir": str(train_dir) if train_dir.exists() else None,
            "test_dir": str(test_dir) if test_dir.exists() else None,
            "sample_submission": str(sample_sub) if sample_sub.exists() else None,
        },
        "counts": {
            "train_files": n_train_files,
            "test_files": n_test_files,
        },
        "detected_columns": {
            "time_column": time_col,
            "feature_columns": feat_cols,
            "label_columns": None,
            "id_column": "filename_stem",
        },
        "validation": {
            "method": "random_holdout_by_id",
            "validation_size": 0.1,
            "random_seed": 42,
        },
        "preprocessing": [
            "group_by_id_from_filename",
            "sort_by_time_column",
            "extract_last_two_observations",
            "compute_velocity_from_dt_ms",
        ],
        "submission": {
            "columns": ["id", "x", "y", "z"],
            "order_from_sample_submission": True,
        },
    }

    with (OUTPUTS_DIR / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
