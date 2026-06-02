"""Tiny YAML-like reader/writer for this project's simple config files.

It supports nested dictionaries, lists, booleans, numbers, nulls, and strings.
The writer emits a conservative YAML subset. The reader also accepts JSON.
This avoids a hard PyYAML dependency for the starter project.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def load(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return default
    if text[0] in "[{":
        return json.loads(text)
    return _parse_lines(text.splitlines())


def dump(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(to_yaml(data), encoding="utf-8")


def to_yaml(data: Any, indent: int = 0) -> str:
    lines = _emit(data, indent)
    return "\n".join(lines) + "\n"


def _emit(data: Any, indent: int) -> list[str]:
    pad = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_emit(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines = []
        for value in data:
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_emit(value, indent + 2))
            else:
                lines.append(f"{pad}- {_scalar(value)}")
        return lines
    return [f"{pad}{_scalar(data)}"]


def _scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or text.strip() != text or any(ch in text for ch in ":#[]{}"):
        return json.dumps(text, ensure_ascii=False)
    return text


def _parse_lines(lines: list[str]) -> Any:
    clean = [line.rstrip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for line_number, raw in enumerate(clean):
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if text.startswith("- "):
            item_text = text[2:].strip()
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw}")
            parent.append(_parse_scalar(item_text))
            continue

        if text == "-":
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw}")
            new_obj: dict[str, Any] = {}
            parent.append(new_obj)
            stack.append((indent, new_obj))
            continue

        if ":" not in text:
            raise ValueError(f"Unsupported YAML line: {raw}")

        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue

        child = _guess_child(clean, line_number, indent)
        parent[key] = child
        stack.append((indent, child))

    return root


def _guess_child(lines: list[str], current_index: int, indent: int) -> dict[str, Any] | list[Any]:
    for next_line in lines[current_index + 1 :]:
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_indent <= indent:
            break
        return [] if next_line.strip().startswith("-") else {}
    return {}


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith(('"', "'")):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value.strip('"').strip("'")
    return value
