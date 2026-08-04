"""Scoring, owned by the framework.

The agent cannot reach this module from its trial code, which is the point:
previously the agent wrote the scorer, and a scorer that returns a hardcoded
number, or one that never reads the predictions at all, both look like a
working pipeline from the outside.

Stage 1 ships only metrics that are competition-independent. Competition
metrics arrive in stage 2, registered here at onboarding time -- outside the
trial loop, so a trial can never redefine how it is judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MetricSpec:
    name: str
    objective: str  # "minimize" or "maximize"
    fn: Callable[[list[dict[str, Any]], list[dict[str, Any]], list[str]], float]


METRICS: dict[str, MetricSpec] = {}


def register(spec: MetricSpec, *, overwrite: bool = False) -> None:
    if spec.name in METRICS and not overwrite:
        raise ValueError(f"metric {spec.name!r} is already registered.")
    if spec.objective not in {"minimize", "maximize"}:
        raise ValueError(f"objective must be 'minimize' or 'maximize', got {spec.objective!r}.")
    METRICS[spec.name] = spec


def compute(
    metric: str,
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    target_keys: list[str],
) -> float:
    """Score `predictions` against `truths`. Never called from agent code."""
    spec = METRICS.get(metric)
    if spec is None:
        raise KeyError(f"unknown metric {metric!r}; registered: {sorted(METRICS)}")
    if len(predictions) != len(truths):
        raise ValueError(f"prediction/truth count mismatch: {len(predictions)} vs {len(truths)}")
    if not predictions:
        raise ValueError("cannot score an empty holdout.")
    return float(spec.fn(predictions, truths, target_keys))


def objective_of(metric: str) -> str:
    spec = METRICS.get(metric)
    if spec is None:
        raise KeyError(f"unknown metric {metric!r}; registered: {sorted(METRICS)}")
    return spec.objective


def _pairs(
    predictions: list[dict[str, Any]], truths: list[dict[str, Any]], target_keys: list[str]
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for predicted, actual in zip(predictions, truths):
        for key in target_keys:
            if key not in predicted:
                raise ValueError(f"prediction is missing target key {key!r}.")
            pairs.append((float(predicted[key]), float(actual[key])))
    return pairs


def _rmse(predictions, truths, target_keys) -> float:
    pairs = _pairs(predictions, truths, target_keys)
    return math.sqrt(sum((p - a) ** 2 for p, a in pairs) / len(pairs))


def _mae(predictions, truths, target_keys) -> float:
    pairs = _pairs(predictions, truths, target_keys)
    return sum(abs(p - a) for p, a in pairs) / len(pairs)


def _accuracy(predictions, truths, target_keys) -> float:
    hits = 0
    for predicted, actual in zip(predictions, truths):
        if all(str(predicted.get(key)) == str(actual.get(key)) for key in target_keys):
            hits += 1
    return hits / len(truths)


def _mean_euclidean_distance(predictions, truths, target_keys) -> float:
    """Mean straight-line error across however many coordinate axes there are.

    Axis-count-agnostic on purpose: target_keys decides the dimensionality, so
    this is not tied to any one competition's coordinate names.
    """
    total = 0.0
    for predicted, actual in zip(predictions, truths):
        squared = 0.0
        for key in target_keys:
            if key not in predicted:
                raise ValueError(f"prediction is missing target key {key!r}.")
            squared += (float(predicted[key]) - float(actual[key])) ** 2
        total += math.sqrt(squared)
    return total / len(truths)


register(MetricSpec("rmse", "minimize", _rmse))
register(MetricSpec("mae", "minimize", _mae))
register(MetricSpec("accuracy", "maximize", _accuracy))
register(MetricSpec("mean_euclidean_distance", "minimize", _mean_euclidean_distance))
