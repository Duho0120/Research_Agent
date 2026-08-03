from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import paths
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
    create_workspace: bool = False,
    workspace_root: str | None = None,
    target_column: str | None = None,
    id_column: str | None = None,
    required_data_files: list[str] | None = None,
) -> dict[str, Any]:
    if not source_path and not topic and not create_workspace:
        raise ValueError("Either source_path, topic, or create_workspace is required.")

    init_project(competition, metric=metric, objective=objective)
    runtime = Path(python_path).expanduser().resolve() if python_path else Path(sys.executable).resolve()
    data_files = _normalize_data_files(required_data_files)
    scaffold: dict[str, Any] | None = None
    if create_workspace:
        source = _workspace_scaffold_path(competition, source_path=source_path, workspace_root=workspace_root)
        scaffold = _create_workspace_scaffold(
            competition,
            source,
            platform=platform,
            metric=metric,
            objective=objective,
            topic=topic,
            target_column=target_column,
            id_column=id_column,
            required_data_files=data_files,
        )
    else:
        source = Path(source_path).expanduser().resolve() if source_path else None
    source_record = {
        "competition": competition,
        "topic": topic,
        "platform": platform,
        "source_path": str(source) if source else None,
        "python": str(runtime),
        "status": "pending_inspection" if source else "needs_project_path",
        "created_workspace": bool(scaffold),
        "required_data_files": data_files,
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
    data_check = _check_required_data_files(source, data_files)
    status = _preparation_status(validation, review_questions, data_check, scaffold)
    source_record["status"] = status
    _write_source_record(competition, source_record)
    _write_preparation_summary(competition, status, review_questions, validation, inventory, data_check, scaffold)
    return {
        **source_record,
        "status": status,
        "review_questions": review_questions,
        "execution_profile_validation": validation,
        "inventory": inventory,
        "data_check": data_check,
        "scaffold": scaffold,
        "steps": ["initialized", "source_recorded", "source_inspected", "execution_profile_drafted", "profile_validated"],
    }


def refresh_workspace_inventory(competition: str, source: Path) -> dict[str, Any]:
    """Recompute and persist workspace_inventory.json for an existing
    registered experiment.

    prepare_workspace() only builds this inventory once, at registration
    time -- if data files are uploaded afterward (e.g. via the data-upload
    endpoint), the cached inventory still shows the pre-upload, data-less
    workspace unless this is called to refresh it. Planning/coding context
    (demo_one_cycle.load_demo_context) reads the cached file, not a live
    scan, so a stale inventory here silently hides uploaded data from the
    agent.
    """
    inventory = _inspect_source(source)
    _write_inventory(competition, inventory)
    return inventory


def _workspace_scaffold_path(competition: str, *, source_path: str | None, workspace_root: str | None) -> Path:
    if source_path:
        return Path(source_path).expanduser().resolve()
    root = Path(workspace_root).expanduser().resolve() if workspace_root else paths.project_root() / "demo_workspaces"
    return (root / competition).resolve()


def _create_workspace_scaffold(
    competition: str,
    source: Path,
    *,
    platform: str,
    metric: str,
    objective: str,
    topic: str | None,
    target_column: str | None,
    id_column: str | None,
    required_data_files: list[str],
) -> dict[str, Any]:
    for directory in ["data", "src", "tests", "outputs"]:
        (source / directory).mkdir(parents=True, exist_ok=True)
    config = {
        "competition": competition,
        "platform": platform,
        "topic": topic,
        "metric": metric,
        "objective": objective,
        "target_column": target_column,
        "id_column": id_column,
        "required_data_files": required_data_files,
        "data_dir": "data",
        "outputs_dir": "outputs",
    }
    files = {
        "workspace_config.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        "src/__init__.py": "",
        "src/baseline.py": _baseline_script(),
        "test_step.py": _test_step_script(),
        "train_step.py": _train_step_script(),
        "predict_step.py": _predict_step_script(),
        "README.md": _workspace_readme(competition, required_data_files),
        "data/README.md": _data_readme(required_data_files),
        "tests/README.md": "Place optional workspace tests here.\n",
    }
    created_files = []
    preserved_files = []
    for relative, content in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            preserved_files.append(relative)
            continue
        write_text(target, content)
        created_files.append(relative)
    return {
        "source_path": str(source),
        "created_files": created_files,
        "preserved_files": preserved_files,
        "data_policy": "manual",
        "next_action": "copy-required-data-files" if required_data_files else "confirm-data-files-and-target-column",
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
    test_entry = _first_existing(paths, ["test_step.py", "tests/test_step.py"])
    if test_entry:
        commands["test"] = [f"{{python}} {test_entry}"]
    elif "tests" in {part for path in paths for part in Path(path).parts} or any(
        path in paths for path in ["pytest.ini", "pyproject.toml", "tox.ini"]
    ):
        commands["test"] = ["{python} -m pytest tests -q"]
    train_entry = _first_existing(paths, ["train_step.py", "train.py", "src/train.py", "scripts/train.py"])
    if train_entry:
        commands["train"] = [f"{{python}} {train_entry}"]
    predict_entry = _first_existing(
        paths,
        ["predict_step.py", "predict.py", "inference.py", "src/predict.py", "scripts/predict.py"],
    )
    if predict_entry:
        commands["predict"] = [f"{{python}} {predict_entry}"]

    metrics_paths = _matching_paths(paths, {"metrics.json"})
    submission_paths = _matching_paths(paths, {"submission.csv"})
    artifacts: dict[str, list[str]] = {}
    if not metrics_paths and (source / "outputs").is_dir():
        metrics_paths = ["outputs/metrics.json"]
    if not submission_paths and (source / "outputs").is_dir():
        submission_paths = ["outputs/submission.csv"]
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
    for filename in ["train_step.py", "predict_step.py", "test_step.py", "train.py", "predict.py", "inference.py", "workspace_config.json"]:
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
        f"- created_workspace: {record.get('created_workspace')}",
        f"- required_data_files: {record.get('required_data_files')}",
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
    data_check: dict[str, Any] | None = None,
    scaffold: dict[str, Any] | None = None,
) -> None:
    result = {
        "competition": competition,
        "status": status,
        "review_questions": questions,
        "execution_profile_status": validation["status"],
        "execution_profile_issues": validation["issues"],
        "inventory_file_count": inventory["file_count"],
        "data_check": data_check or {},
        "scaffold": scaffold or {},
    }
    out_dir = competition_dir(competition)
    write_text(out_dir / "workspace_preparation.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# {competition} Workspace Preparation", "", f"- status: {status}", "", "## Review Questions", ""]
    lines.extend(f"- {item}" for item in questions or ["None"])
    lines.extend(["", "## Profile Issues", ""])
    lines.extend(f"- {item}" for item in validation["issues"] or ["None"])
    if data_check:
        lines.extend(["", "## Data Check", ""])
        lines.extend(f"- missing: {item}" for item in data_check.get("missing_files", []) or ["None"])
    if scaffold:
        lines.extend(["", "## Created Workspace", ""])
        lines.extend(f"- {item}" for item in scaffold.get("created_files", []) or ["None"])
    lines.append("")
    write_text(out_dir / "workspace_preparation.md", "\n".join(lines))


def _normalize_data_files(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip().replace("\\", "/")
            if item:
                normalized.append(item)
    return list(dict.fromkeys(normalized))


def _check_required_data_files(source: Path, required_data_files: list[str]) -> dict[str, Any]:
    missing = [item for item in required_data_files if not (source / "data" / item).exists()]
    return {
        "status": "ready" if not missing else "missing_data",
        "required_files": required_data_files,
        "missing_files": missing,
        "data_dir": str((source / "data").as_posix()),
        "manual_data_required": bool(missing),
    }


def _preparation_status(
    validation: dict[str, Any],
    review_questions: list[str],
    data_check: dict[str, Any],
    scaffold: dict[str, Any] | None,
) -> str:
    if review_questions:
        return "needs_review"
    if validation["status"] != "ready":
        return "blocked"
    if scaffold and data_check.get("status") == "missing_data":
        return "needs_data"
    return "ready"


def _workspace_readme(competition: str, required_data_files: list[str]) -> str:
    lines = [
        f"# {competition} Workspace",
        "",
        "This workspace was scaffolded by the research agent.",
        "",
        "## Manual Data Step",
        "",
        "Copy the competition data files into `data/` before running the local loop.",
        "",
    ]
    lines.extend(f"- data/{item}" for item in required_data_files or ["<competition files>"])
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```powershell",
            "python test_step.py",
            "python train_step.py",
            "python predict_step.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _data_readme(required_data_files: list[str]) -> str:
    lines = ["# Data", "", "Place manually downloaded competition data files here.", ""]
    lines.extend(f"- {item}" for item in required_data_files or ["train.csv", "test.csv"])
    lines.append("")
    return "\n".join(lines)


def _test_step_script() -> str:
    return """from src.baseline import check_required_data


if __name__ == "__main__":
    check_required_data()
    print("workspace contract ok")
"""


def _train_step_script() -> str:
    return """from src.baseline import train


if __name__ == "__main__":
    train()
"""


def _predict_step_script() -> str:
    return """from src.baseline import predict


if __name__ == "__main__":
    predict()
"""


def _baseline_script() -> str:
    return r'''from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "workspace_config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def check_required_data() -> None:
    config = load_config()
    missing = []
    for name in config.get("required_data_files", []):
        if not (ROOT / "data" / name).exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError("Missing data files in data/: " + ", ".join(missing))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_kind(config: dict, labels: list[str]) -> str:
    metric = str(config.get("metric") or "").strip().casefold().replace("-", "_")
    if metric in {"rmse", "rmsle", "mae", "mse", "mean_squared_error", "mean_absolute_error"}:
        return "regression"
    if metric in {"accuracy", "f1", "roc_auc", "auc", "log_loss", "logloss"}:
        return "classification"
    numeric = [_to_float(value) for value in labels]
    if numeric and all(value is not None for value in numeric) and config.get("objective") == "minimize":
        return "regression"
    return "classification"


def _regression_score(metric: str, actual: list[float], prediction: float) -> float:
    if not actual:
        return 0.0
    errors = [value - prediction for value in actual]
    normalized = metric.strip().casefold().replace("-", "_")
    if normalized == "mae" or normalized == "mean_absolute_error":
        return sum(abs(value) for value in errors) / len(errors)
    if normalized == "mse" or normalized == "mean_squared_error":
        return sum(value * value for value in errors) / len(errors)
    if normalized == "rmsle":
        predicted_log = math.log1p(max(0.0, prediction))
        return math.sqrt(
            sum((math.log1p(max(0.0, value)) - predicted_log) ** 2 for value in actual) / len(actual)
        )
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def train() -> dict:
    check_required_data()
    config = load_config()
    target = config.get("target_column")
    if not target:
        raise ValueError("workspace_config.json must define target_column before training.")
    train_path = _find_train_file(target)
    rows = _read_csv(train_path)
    if not rows:
        raise ValueError(f"No rows found in {train_path}")
    if target not in rows[0]:
        raise ValueError(f"Target column {target!r} not found in {train_path}")

    split_at = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split_at]
    valid_rows = rows[split_at:] or rows[:]
    labels = [row[target] for row in train_rows if row.get(target) not in {None, ""}]
    if not labels:
        raise ValueError(f"No labels found in target column {target!r}")
    task_kind = _task_kind(config, labels)
    metric = str(config.get("metric") or "unknown")
    if task_kind == "regression":
        numeric_labels = [value for value in (_to_float(item) for item in labels) if value is not None]
        if not numeric_labels:
            raise ValueError(f"Regression target {target!r} does not contain numeric labels.")
        prediction = sum(numeric_labels) / len(numeric_labels)
        valid_labels = [
            value
            for value in (_to_float(row.get(target)) for row in valid_rows)
            if value is not None
        ]
        score = _regression_score(metric, valid_labels, prediction)
        strategy = "mean_regression"
    else:
        prediction = Counter(labels).most_common(1)[0][0]
        correct = sum(1 for row in valid_rows if row.get(target) == prediction)
        score = correct / len(valid_rows) if valid_rows else 0.0
        strategy = "majority_class"

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    model = {
        "strategy": strategy,
        "task_kind": task_kind,
        "prediction": prediction,
        "target_column": target,
        "id_column": config.get("id_column"),
        "train_file": train_path.name,
    }
    metrics = {
        "cv_score": score,
        "metric": metric,
        "objective": config.get("objective", "maximize"),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "strategy": strategy,
        "model_type": "MeanRegressor" if task_kind == "regression" else "MajorityClassClassifier",
    }
    (outputs / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (outputs / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def predict() -> Path:
    config = load_config()
    outputs = ROOT / "outputs"
    model_path = outputs / "model.json"
    if not model_path.exists():
        train()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    test_path = _find_test_file(model.get("target_column"))
    rows = _read_csv(test_path)
    target = model.get("target_column") or config.get("target_column") or "prediction"
    id_column = config.get("id_column") or model.get("id_column") or _first_column(rows)
    submission_path = outputs / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[id_column, target])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({id_column: row.get(id_column, index), target: model["prediction"]})
    return submission_path


def _find_train_file(target: str) -> Path:
    preferred = ROOT / "data" / "train.csv"
    if preferred.exists():
        return preferred
    for path in sorted((ROOT / "data").glob("*.csv")):
        try:
            rows = _read_csv(path, limit=1)
        except ValueError:
            continue
        if rows and target in rows[0]:
            return path
    raise FileNotFoundError("Could not find a CSV train file containing the target column.")


def _find_test_file(target: str | None) -> Path:
    preferred = ROOT / "data" / "test.csv"
    if preferred.exists():
        return preferred
    for path in sorted((ROOT / "data").glob("*.csv")):
        if path.name.lower().startswith("train"):
            continue
        rows = _read_csv(path, limit=1)
        if rows and (not target or target not in rows[0]):
            return path
    raise FileNotFoundError("Could not find a CSV test file.")


def _read_csv(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = []
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows


def _first_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "id"
    return next(iter(rows[0].keys()), "id")
'''
