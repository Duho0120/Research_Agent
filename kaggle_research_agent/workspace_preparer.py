from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import simple_yaml
from .execution_profile import validate_execution_profile
from .paths import competition_dir
from .store import init_project, load_state, save_state, write_text


SKIP_DIRECTORIES = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}
DATA_SUFFIXES = {".csv", ".parquet", ".feather", ".arrow", ".xlsx", ".jsonl"}
MAX_INVENTORY_FILES = 500
MAX_SCAN_DEPTH = 3


def prepare_workspace(
    competition: str,
    *,
    source_path: str | None = None,
    topic: str | None = None,
    platform: str = "external",
    metric: str = "unknown",
    objective: str = "maximize",
    python_path: str | None = None,
) -> dict[str, Any]:
    if not source_path and not topic:
        raise ValueError("Either source_path or topic is required.")

    init_project(competition, metric=metric, objective=objective)
    source = Path(source_path).expanduser().resolve() if source_path else None
    runtime = Path(python_path).expanduser().resolve() if python_path else Path(sys.executable).resolve()
    source_record = {
        "competition": competition,
        "topic": topic,
        "platform": platform,
        "source_path": str(source) if source else None,
        "python": str(runtime),
        "status": "pending_inspection" if source else "needs_project_path",
    }
    _update_workspace_state(competition, platform, topic, source)
    _write_source_record(competition, source_record)

    if source is None:
        return {
            **source_record,
            "status": "needs_project_path",
            "review_questions": ["Provide the local project or data path before execution setup."],
            "steps": ["initialized", "source_recorded"],
        }
    if not source.is_dir():
        source_record["status"] = "blocked"
        _write_source_record(competition, source_record)
        return {
            **source_record,
            "status": "blocked",
            "review_questions": ["The provided source path does not exist or is not a directory."],
            "steps": ["initialized", "source_recorded", "source_invalid"],
        }

    inventory = _inspect_source(source)
    _write_inventory(competition, inventory)
    profile, review_questions = _build_profile_draft(
        competition,
        source,
        runtime,
        platform,
        inventory,
    )
    simple_yaml.dump(profile, competition_dir(competition) / "execution_profile.yaml")
    validation = validate_execution_profile(competition)
    status = "ready" if validation["status"] == "ready" and not review_questions else "needs_review"
    source_record["status"] = status
    _write_source_record(competition, source_record)
    _write_preparation_summary(competition, status, review_questions, validation, inventory)
    return {
        **source_record,
        "status": status,
        "review_questions": review_questions,
        "execution_profile_validation": validation,
        "inventory": inventory,
        "steps": ["initialized", "source_recorded", "source_inspected", "execution_profile_drafted", "profile_validated"],
    }


def _inspect_source(source: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    stack = [(source, 0)]
    while stack and len(files) < MAX_INVENTORY_FILES:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if child.name not in SKIP_DIRECTORIES and depth < MAX_SCAN_DEPTH:
                    stack.append((child, depth + 1))
                continue
            relative = child.relative_to(source).as_posix()
            files.append(
                {
                    "path": relative,
                    "name": child.name,
                    "suffix": child.suffix.lower(),
                    "category": _file_category(child),
                    "size": child.stat().st_size,
                }
            )
            if len(files) >= MAX_INVENTORY_FILES:
                break
    return {
        "source_path": str(source),
        "file_count": len(files),
        "truncated": len(files) >= MAX_INVENTORY_FILES,
        "files": files,
    }


def _build_profile_draft(
    competition: str,
    source: Path,
    runtime: Path,
    platform: str,
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    paths = {item["path"] for item in inventory["files"]}
    commands: dict[str, list[str]] = {}
    if "tests" in {part for path in paths for part in Path(path).parts} or any(
        path in paths for path in ["pytest.ini", "pyproject.toml", "tox.ini"]
    ):
        commands["test"] = ["{python} -m pytest tests -q"]
    train_entry = _first_existing(paths, ["train.py", "src/train.py", "scripts/train.py"])
    if train_entry:
        commands["train"] = [f"{{python}} {train_entry}"]
    predict_entry = _first_existing(paths, ["predict.py", "inference.py", "src/predict.py", "scripts/predict.py"])
    if predict_entry:
        commands["predict"] = [f"{{python}} {predict_entry}"]

    metrics_paths = _matching_paths(paths, {"metrics.json"})
    submission_paths = _matching_paths(paths, {"submission.csv"})
    artifacts: dict[str, list[str]] = {}
    if metrics_paths:
        artifacts["metrics"] = metrics_paths
    if submission_paths:
        artifacts["submission"] = submission_paths

    allowed = _allowed_write_paths(source, paths)
    forbidden = _forbidden_paths(source, inventory, metrics_paths, submission_paths)
    profile = {
        "schema_version": "1.0",
        "competition": competition,
        "platform": platform,
        "project_root": str(source),
        "python": str(runtime),
        "commands": commands,
        "artifacts": artifacts,
        "write_scope": {"allowed": allowed, "forbidden": forbidden},
        "submission_mode": "kaggle_cli" if platform == "kaggle" else "manual_external",
    }
    questions = []
    if "test" not in commands:
        questions.append("Confirm the test command.")
    if "train" not in commands:
        questions.append("Confirm the train command.")
    if "predict" not in commands:
        questions.append("Confirm whether a separate predict command is required.")
    if "metrics" not in artifacts:
        questions.append("Confirm the metrics artifact path.")
    if "submission" not in artifacts:
        questions.append("Confirm the submission artifact path or mark submission as not applicable.")
    if not allowed:
        questions.append("Confirm which source files the coding agent may modify.")
    return profile, questions


def _allowed_write_paths(source: Path, paths: set[str]) -> list[str]:
    allowed = []
    for directory in ["src", "tests", "scripts"]:
        if (source / directory).is_dir():
            allowed.append(f"{directory}/")
    for filename in ["train.py", "predict.py", "inference.py"]:
        if filename in paths:
            allowed.append(filename)
    return allowed


def _forbidden_paths(
    source: Path,
    inventory: dict[str, Any],
    metrics_paths: list[str],
    submission_paths: list[str],
) -> list[str]:
    forbidden = []
    for directory in ["data", "dataset", "datasets", "input"]:
        if (source / directory).is_dir():
            forbidden.append(f"{directory}/")
    for item in inventory["files"]:
        if item["suffix"] in DATA_SUFFIXES or item["category"] in {"data", "submission", "metrics"}:
            forbidden.append(item["path"])
    forbidden.extend(metrics_paths)
    forbidden.extend(submission_paths)
    return list(dict.fromkeys(forbidden)) or ["data/"]


def _matching_paths(paths: set[str], names: set[str]) -> list[str]:
    return sorted(path for path in paths if Path(path).name.casefold() in names)


def _first_existing(paths: set[str], candidates: list[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in paths), None)


def _file_category(path: Path) -> str:
    lowered = path.name.casefold()
    if lowered == "metrics.json":
        return "metrics"
    if "submission" in lowered and path.suffix.lower() == ".csv":
        return "submission"
    if path.suffix.lower() in DATA_SUFFIXES:
        return "data"
    if path.suffix.lower() == ".ipynb":
        return "notebook"
    if path.suffix.lower() == ".py":
        return "code"
    return "other"


def _update_workspace_state(competition: str, platform: str, topic: str | None, source: Path | None) -> None:
    state = load_state(competition)
    state.setdefault("competition", {})["platform"] = platform
    state["competition"]["topic"] = topic
    state["competition"]["source_path"] = str(source) if source else None
    save_state(competition, state)


def _write_source_record(competition: str, record: dict[str, Any]) -> None:
    out_dir = competition_dir(competition)
    write_text(out_dir / "workspace_source.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {competition} Workspace Source",
        "",
        f"- status: {record['status']}",
        f"- platform: {record['platform']}",
        f"- topic: {record.get('topic')}",
        f"- source_path: {record.get('source_path')}",
        f"- python: {record.get('python')}",
        "",
    ]
    write_text(out_dir / "workspace_source.md", "\n".join(lines))


def _write_inventory(competition: str, inventory: dict[str, Any]) -> None:
    out_dir = competition_dir(competition)
    write_text(out_dir / "workspace_inventory.json", json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {competition} Workspace Inventory",
        "",
        f"- source_path: {inventory['source_path']}",
        f"- file_count: {inventory['file_count']}",
        f"- truncated: {inventory['truncated']}",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- {item['path']} [{item['category']}]" for item in inventory["files"])
    lines.append("")
    write_text(out_dir / "workspace_inventory.md", "\n".join(lines))


def _write_preparation_summary(
    competition: str,
    status: str,
    questions: list[str],
    validation: dict[str, Any],
    inventory: dict[str, Any],
) -> None:
    result = {
        "competition": competition,
        "status": status,
        "review_questions": questions,
        "execution_profile_status": validation["status"],
        "execution_profile_issues": validation["issues"],
        "inventory_file_count": inventory["file_count"],
    }
    out_dir = competition_dir(competition)
    write_text(out_dir / "workspace_preparation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# {competition} Workspace Preparation", "", f"- status: {status}", "", "## Review Questions", ""]
    lines.extend(f"- {item}" for item in questions or ["None"])
    lines.extend(["", "## Profile Issues", ""])
    lines.extend(f"- {item}" for item in validation["issues"] or ["None"])
    lines.append("")
    write_text(out_dir / "workspace_preparation.md", "\n".join(lines))
