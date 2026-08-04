"""What the competition says it measures.

Every competition already carries a metric.md, but most hold only a name and a
direction:

    # Metric
    - name: R-Hit@1cm
    - objective: maximize

Implementing a metric from that is guesswork, and the guess that hurts is the
silent one: read "1cm" against millimetre coordinates and the threshold is off
by a hundred, while the score still reacts to predictions, still peaks on
perfect ones, and still points the right way. No amount of verification
catches it. A sentence of prose from whoever read the competition page does.

Supplying that prose is optional. A competition without it still runs -- the
resulting metric is just marked low-confidence, so a score nobody should lean
on is visible as such instead of looking like every other score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkedExample:
    """A hand-checked input/output pair, which is worth more than prose.

    Prose can be misread; an example is executable. A generated metric that
    disagrees with it is rejected outright, and a unit mistake -- the one
    failure the other checks all pass -- cannot survive it.
    """

    predictions: list[dict[str, Any]]
    truths: list[dict[str, Any]]
    target_keys: list[str]
    expected: float
    tolerance: float = 1e-6


@dataclass(frozen=True)
class MetricDeclaration:
    name: str
    objective: str
    prose: str = ""
    worked_example: WorkedExample | None = None

    @property
    def has_definition(self) -> bool:
        """Whether this says anything beyond a name and a direction."""
        return bool(self.prose.strip())

    @property
    def confidence(self) -> str:
        if self.worked_example is not None:
            return "high"
        return "medium" if self.has_definition else "low"


def parse_metric_spec(path: Path | str) -> MetricDeclaration | None:
    """Read a competition's metric.md. Returns None if there is no usable name."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    name = _field(text, "name")
    if not name:
        return None
    objective = (_field(text, "objective") or "").strip().lower()
    if objective not in {"minimize", "maximize"}:
        objective = "maximize"
    return MetricDeclaration(
        name=name,
        objective=objective,
        prose=_prose(text),
        worked_example=_worked_example(text),
    )


def normalize_metric_name(name: str) -> str:
    """Fold a competition's own spelling into a registry key.

    "R-Hit@1cm" and "r_hit_at_1cm" name the same thing; only one of them is a
    usable identifier.
    """
    folded = name.strip().lower().replace("@", "_at_").replace("%", "_pct_")
    folded = re.sub(r"[^a-z0-9]+", "_", folded)
    return folded.strip("_")


def _field(text: str, key: str) -> str:
    match = re.search(rf"^\s*[-*]\s*{re.escape(key)}\s*[:：]\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _prose(text: str) -> str:
    """Everything past the name/objective header, minus the example fence.

    Kept verbatim rather than parsed into fields: this is handed to whoever
    implements the metric, and a definition survives being read better than it
    survives being restructured.
    """
    body = re.split(r"^##\s", text, maxsplit=1, flags=re.MULTILINE)
    if len(body) < 2:
        return ""
    remainder = "## " + body[1]
    return re.sub(r"```json.*?```", "", remainder, flags=re.DOTALL).strip()


def _worked_example(text: str) -> WorkedExample | None:
    for block in re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "expected" not in payload:
            continue
        try:
            return WorkedExample(
                predictions=list(payload["predictions"]),
                truths=list(payload["truths"]),
                target_keys=[str(key) for key in payload["target_keys"]],
                expected=float(payload["expected"]),
                tolerance=float(payload.get("tolerance", 1e-6)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None
