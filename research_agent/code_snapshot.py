from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from .store import read_text, write_text


SNAPSHOT_DIR_NAME = "code_snapshot"
SNAPSHOT_MANIFEST_NAME = "code_snapshot_manifest.json"

TEXT_CODE_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
}


def save_trial_code_snapshot(
    out_dir: Path,
    project_root: Path,
    changed_files: list[Any],
    *,
    allowed_paths: list[Any] | None = None,
    extra_files: list[str] | None = None,
) -> dict[str, Any]:
    """Persist the post-code-writer source files for this trial.

    This snapshot is the immutable code reference for future continuation trials.
    It prevents regenerated user-facing artifacts from accidentally copying the
    current workspace after later trials have changed it.
    """

    snapshot_root = out_dir / "internal" / SNAPSHOT_DIR_NAME
    _reset_snapshot_dir(out_dir, snapshot_root)

    files: list[dict[str, Any]] = []
    for relative in _snapshot_candidates(changed_files, extra_files or []):
        if allowed_paths and not _is_allowed_path(relative, allowed_paths):
            continue
        source = project_root / Path(*PurePosixPath(relative).parts)
        if not source.is_file() or not _is_text_snapshot_file(source):
            continue
        text = read_text(source, default="")
        destination = snapshot_root / Path(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_text(destination, text)
        files.append(
            {
                "path": relative,
                "bytes": len(text.encode("utf-8")),
                "source": "project_root_after_code_writer",
            }
        )

    manifest = {
        "schema_version": "1.0",
        "snapshot_dir": f"internal/{SNAPSHOT_DIR_NAME}",
        "files": files,
    }
    write_text(out_dir / "internal" / SNAPSHOT_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_trial_code_snapshot(out_dir: Path, *, max_chars_per_file: int | None = None) -> list[tuple[str, str]]:
    manifest = _read_json(out_dir / "internal" / SNAPSHOT_MANIFEST_NAME)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    snapshot_root = out_dir / "internal" / SNAPSHOT_DIR_NAME
    result: list[tuple[str, str]] = []

    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            relative = _safe_relative_path(item.get("path"))
            if not relative:
                continue
            path = snapshot_root / Path(*PurePosixPath(relative).parts)
            if not path.is_file() or not _is_text_snapshot_file(path):
                continue
            text = read_text(path, default="")
            if max_chars_per_file is not None:
                text = text[:max_chars_per_file]
            result.append((relative, text))
        if result:
            return sorted(result, key=lambda pair: _code_file_priority(pair[0]))

    if snapshot_root.is_dir():
        for path in sorted(snapshot_root.rglob("*")):
            if not path.is_file() or not _is_text_snapshot_file(path):
                continue
            relative = path.relative_to(snapshot_root).as_posix()
            if not _safe_relative_path(relative):
                continue
            text = read_text(path, default="")
            if max_chars_per_file is not None:
                text = text[:max_chars_per_file]
            result.append((relative, text))
    return sorted(result, key=lambda pair: _code_file_priority(pair[0]))


def _snapshot_candidates(changed_files: list[Any], extra_files: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in [*changed_files, *extra_files]:
        relative = _safe_relative_path(item)
        if not relative or relative in seen:
            continue
        seen.add(relative)
        result.append(relative)
    return result


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return ""
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return ""
    return path.as_posix()


def _is_allowed_path(path: str, allowed_paths: list[Any]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for allowed in allowed_paths:
        if not isinstance(allowed, str) or not allowed.strip():
            continue
        marker_raw = allowed.replace("\\", "/").rstrip("/")
        if allowed.replace("\\", "/").endswith("/"):
            if normalized.startswith(marker_raw + "/"):
                return True
        elif normalized == marker_raw:
            return True
    return False


def _is_text_snapshot_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_CODE_SUFFIXES


def _reset_snapshot_dir(out_dir: Path, snapshot_root: Path) -> None:
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_root.exists():
        if snapshot_root.parent.resolve() != (out_dir / "internal").resolve():
            raise ValueError(f"Refusing to reset unexpected code snapshot directory: {snapshot_root}")
        shutil.rmtree(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _code_file_priority(path: str) -> tuple[int, str]:
    normalized = path.replace("\\", "/")
    if normalized.endswith("src/baseline.py"):
        return (0, normalized)
    if normalized.endswith("train_step.py") or normalized.endswith("train.py"):
        return (1, normalized)
    if normalized.endswith("predict_step.py") or normalized.endswith("predict.py"):
        return (2, normalized)
    if normalized.endswith("test_step.py") or normalized.endswith("test.py"):
        return (3, normalized)
    return (9, normalized)
