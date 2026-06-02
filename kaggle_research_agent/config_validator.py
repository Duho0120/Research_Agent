from __future__ import annotations

from typing import Any


def validate_config(config: dict[str, Any], allowed_space: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_node(config, allowed_space, path="", errors=errors)
    return errors


def _validate_node(value: Any, allowed: Any, path: str, errors: list[str]) -> None:
    if isinstance(allowed, dict) and "min" in allowed and "max" in allowed:
        if not isinstance(value, (int, float)):
            errors.append(f"{path}: expected number in range {allowed['min']}..{allowed['max']}")
            return
        if value < allowed["min"] or value > allowed["max"]:
            errors.append(f"{path}: {value} is outside {allowed['min']}..{allowed['max']}")
        return

    if isinstance(allowed, list):
        if value not in allowed:
            errors.append(f"{path}: {value!r} is not one of {allowed!r}")
        return

    if isinstance(allowed, dict):
        if not isinstance(value, dict):
            errors.append(f"{path}: expected mapping")
            return
        for key in value:
            if key not in allowed:
                errors.append(f"{path}.{key}".strip(".") + ": not allowed by search space")
        for key, child_allowed in allowed.items():
            if key in value:
                _validate_node(value[key], child_allowed, f"{path}.{key}".strip("."), errors)
        return

