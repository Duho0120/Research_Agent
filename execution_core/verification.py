"""The observations that no contract can replace.

Three failures survive any amount of structural design, because each one is
syntactically legal code that simply does the wrong thing. They can only be
caught by running something and looking at what came out:

1. a predictor that answers identically for every sample (in orchestrator.py,
   free of charge -- the holdout predictions are already in hand);
2. a loader that read the wrong files (here);
3. a metric implementation that does not measure what the competition
   measures (stage 2).

(2) is the one that makes varying competition layouts safe. The framework
never learns where this competition keeps its data -- it only checks the
loader's output against an anchor the competition itself provides: the
submission template says which ids must be predicted, and in what order.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_template_ids(template_path: Path | str, id_column: str) -> list[str]:
    """Read the id column from the competition's own submission template."""
    path = Path(template_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"submission template {path.name} has no header row.")
        try:
            index = header.index(id_column)
        except ValueError:
            raise ValueError(
                f"submission template {path.name} has no {id_column!r} column; found {header}."
            ) from None
        return [row[index] for row in reader if row]


def verify_test_ids(test_ids: list[Any], template_ids: list[str]) -> list[str]:
    """Compare what the loader returned against what must be submitted.

    Order matters, not just membership: a submission whose rows are a
    permutation of the template scores as noise on most platforms while
    looking entirely valid on disk.
    """
    issues: list[str] = []
    loaded = [str(identifier) for identifier in test_ids]
    expected = [str(identifier) for identifier in template_ids]
    if not loaded:
        return ["loader_returned_no_test_samples"]
    if len(loaded) != len(expected):
        issues.append(f"test_count_differs_from_template:{len(loaded)}_vs_{len(expected)}")
    if set(loaded) != set(expected):
        missing = len(set(expected) - set(loaded))
        unexpected = len(set(loaded) - set(expected))
        issues.append(f"test_ids_do_not_match_template:missing_{missing}:unexpected_{unexpected}")
    elif loaded != expected:
        issues.append("test_ids_match_template_but_in_a_different_order")
    return issues
