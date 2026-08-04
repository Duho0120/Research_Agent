"""Proving a metric measures something, and measures it the right way round.

Reacting to predictions is not enough. Suppose R-Hit@1cm -- the share of
predictions landing within a centimetre, higher is better -- gets implemented
as mean distance instead. Wreck the predictions and the score moves, so a
sensitivity check passes. But now every improvement rule is reading the number
backwards: the agent optimises for distance while believing it optimises for
hit rate, three trials of "no improvement" trigger an axis change that was
never needed, and the whole experiment walks away from the answer.

So direction is checked, not just reaction. Predictions are degraded in
stages, and the scores must order themselves the way the declared objective
says they should.
"""

from __future__ import annotations

import random
from typing import Any, Callable


MetricFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], list[str]], float]

WRECK_OFFSET = 1000.0
NUDGE_OFFSET = 0.25


def verify_metric(
    fn: MetricFn,
    *,
    objective: str,
    target_keys: list[str],
    truths: list[dict[str, Any]] | None = None,
    worked_example: Any | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run every check that does not need the competition's own scorer.

    Returns the observations alongside the issues, so a rejection can be shown
    with the numbers that caused it rather than as a bare verdict.
    """
    if objective not in {"minimize", "maximize"}:
        return {"issues": [f"metric_objective_invalid:{objective}"], "observations": {}}
    if not target_keys:
        return {"issues": ["metric_has_no_target_keys"], "observations": {}}

    truths = truths or _synthetic_truths(target_keys, seed=seed)
    perfect = [dict(truth) for truth in truths]
    nudged = _shift(truths, target_keys, NUDGE_OFFSET)
    wrecked = _shift(truths, target_keys, WRECK_OFFSET)

    scores: dict[str, Any] = {}
    for label, predictions in (("perfect", perfect), ("nudged", nudged), ("wrecked", wrecked)):
        try:
            scores[label] = float(fn(predictions, truths, target_keys))
        except Exception as error:  # the metric is unknown code; any failure is a rejection
            return {
                "issues": [f"metric_raised_on_{label}_predictions:{type(error).__name__}"],
                "observations": {"error": str(error)[:400], **scores},
            }

    issues: list[str] = []
    issues += _reacts(scores)
    issues += _points_the_right_way(scores, objective)
    issues += _is_deterministic(fn, perfect, truths, target_keys, scores["perfect"])
    example_result = _matches_worked_example(fn, worked_example)
    issues += example_result["issues"]

    return {
        "issues": issues,
        "observations": {**scores, "worked_example": example_result["observation"]},
    }


def _reacts(scores: dict[str, float]) -> list[str]:
    """A scorer that ignores its predictions returns the same number regardless.

    Real incident: a generated harness returned a hardcoded constant, and
    another computed a real-looking number from the labels alone. Both looked
    like a working pipeline from outside.
    """
    if scores["perfect"] == scores["wrecked"]:
        return ["metric_ignores_predictions:score_unchanged_when_predictions_wrecked"]
    return []


def _points_the_right_way(scores: dict[str, float], objective: str) -> list[str]:
    """Better predictions must score better, in the declared direction."""
    better = (lambda a, b: a > b) if objective == "maximize" else (lambda a, b: a < b)
    if not better(scores["perfect"], scores["wrecked"]):
        return [f"metric_direction_contradicts_objective:{objective}"]
    # Non-strict between neighbours: a discrete metric such as accuracy scores
    # a small miss and a large one identically, which is correct behaviour.
    if better(scores["wrecked"], scores["nudged"]) or better(scores["nudged"], scores["perfect"]):
        return [f"metric_is_not_monotonic_in_error:{objective}"]
    return []


def _is_deterministic(
    fn: MetricFn,
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    target_keys: list[str],
    first: float,
) -> list[str]:
    """Two trials compared against each other must have been scored the same way.

    A metric that samples or shuffles internally turns the improvement rules
    into a coin flip.
    """
    try:
        second = float(fn([dict(p) for p in predictions], truths, target_keys))
    except Exception as error:
        return [f"metric_raised_on_rerun:{type(error).__name__}"]
    return [] if second == first else ["metric_is_not_deterministic"]


def _matches_worked_example(fn: MetricFn, worked_example: Any | None) -> dict[str, Any]:
    """The only check that catches a unit mistake.

    Everything else passes when "1cm" is read against millimetre coordinates:
    the score still reacts, still peaks on perfect predictions, still points
    the right way. Only a number someone worked out by hand disagrees.
    """
    if worked_example is None:
        return {"issues": [], "observation": "not_provided"}
    try:
        actual = float(
            fn(
                list(worked_example.predictions),
                list(worked_example.truths),
                list(worked_example.target_keys),
            )
        )
    except Exception as error:
        return {
            "issues": [f"metric_raised_on_worked_example:{type(error).__name__}"],
            "observation": str(error)[:200],
        }
    if abs(actual - worked_example.expected) > worked_example.tolerance:
        return {
            "issues": [f"metric_disagrees_with_worked_example:got_{actual}_expected_{worked_example.expected}"],
            "observation": {"got": actual, "expected": worked_example.expected},
        }
    return {"issues": [], "observation": {"got": actual, "expected": worked_example.expected}}


def _synthetic_truths(target_keys: list[str], *, count: int = 12, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [{key: round(rng.uniform(1.0, 9.0), 4) for key in target_keys} for _ in range(count)]


def _shift(truths: list[dict[str, Any]], target_keys: list[str], offset: float) -> list[dict[str, Any]]:
    return [{key: float(truth[key]) + offset for key in target_keys} for truth in truths]
