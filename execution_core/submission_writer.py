"""The submission file, written by the framework.

Agent-written submission code produced a run of failures that all looked
alike from outside: columns in the wrong order, a stale file left from an
earlier trial and reported as fresh, and a file written under a name nobody
had declared. None of those are possible when the framework holds the pen --
it knows the column order from the competition's own template and it writes
every row itself.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .contract import ContractViolation


def write_submission(
    path: Path | str,
    *,
    columns: list[str],
    id_column: str,
    ids: list[Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write `predictions` to `path` using the template's own column order."""
    if id_column not in columns:
        raise ContractViolation(
            f"id_column {id_column!r} is not among submission_columns() {columns!r}."
        )
    if len(ids) != len(predictions):
        raise ContractViolation(
            f"got {len(ids)} test ids but {len(predictions)} predictions."
        )
    target_columns = [column for column in columns if column != id_column]
    if not target_columns:
        raise ContractViolation("submission_columns() holds only the id column; nothing to predict.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row_number, (identifier, prediction) in enumerate(zip(ids, predictions), start=1):
            if not isinstance(prediction, dict):
                raise ContractViolation(
                    f"predict() returned a {type(prediction).__name__} for row {row_number}; expected a dict."
                )
            missing = [column for column in target_columns if column not in prediction]
            if missing:
                raise ContractViolation(
                    f"predict() omitted {missing} for row {row_number}; "
                    f"expected keys {target_columns}."
                )
            values = [_finite(prediction[column], column, row_number) for column in target_columns]
            writer.writerow(
                [identifier if column == id_column else values[target_columns.index(column)] for column in columns]
            )
    return {"path": str(out_path), "rows": len(ids), "columns": columns}


def _finite(value: Any, column: str, row_number: int) -> Any:
    """Reject NaN/inf here rather than after upload.

    A submission carrying NaN is accepted by the CSV writer and rejected by
    the platform, which costs one of a small daily submission budget.
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ContractViolation(f"predict() produced a non-finite value for {column!r} at row {row_number}.")
    return value
