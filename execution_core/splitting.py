"""The holdout split, owned by the framework.

Under the old model the agent built its own split inside a scoring harness it
also wrote. One trial split off a slice that carried no labels and scored it
anyway; the check that caught it was a text search for the word "train" in the
harness source, which the next trial satisfied without splitting anything.
Here the split is framework code and label presence is a precondition, so an
unlabeled holdout cannot be constructed in the first place.
"""

from __future__ import annotations

import random
from typing import Any

from .contract import ContractViolation, DEFAULT_HOLDOUT_RATIO, DEFAULT_SEED


def split_samples(
    samples: list[dict[str, Any]],
    label_keys: list[str],
    *,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split `samples` into (train, holdout), deterministically.

    Deterministic because two trials whose scores are compared must have been
    scored on the same holdout -- otherwise the improvement rules are reading
    noise.
    """
    if not label_keys:
        raise ContractViolation("label_keys() returned nothing; the framework cannot verify a labeled split.")
    if not 0.0 < holdout_ratio < 1.0:
        raise ContractViolation(f"holdout_ratio must be between 0 and 1, got {holdout_ratio!r}.")
    if len(samples) < 2:
        raise ContractViolation(f"need at least 2 train samples to split, got {len(samples)}.")

    missing = _first_sample_missing_labels(samples, label_keys)
    if missing is not None:
        index, absent = missing
        raise ContractViolation(
            f"train sample at index {index} is missing label key(s) {sorted(absent)}; "
            "load_samples('train') must return every key named by label_keys()."
        )

    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    # At least one sample on each side, however small the dataset or ratio.
    holdout_size = min(len(samples) - 1, max(1, round(len(samples) * holdout_ratio)))
    holdout_index = set(order[:holdout_size])

    holdout = [samples[i] for i in range(len(samples)) if i in holdout_index]
    train = [samples[i] for i in range(len(samples)) if i not in holdout_index]
    return train, holdout


def _first_sample_missing_labels(
    samples: list[dict[str, Any]], label_keys: list[str]
) -> tuple[int, set[str]] | None:
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ContractViolation(
                f"load_samples() returned a {type(sample).__name__} at index {index}; every sample must be a dict."
            )
        absent = {key for key in label_keys if key not in sample}
        if absent:
            return index, absent
    return None
