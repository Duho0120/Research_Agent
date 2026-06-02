from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .paths import competition_data_dir, competition_dir
from .store import now_iso, write_text


def profile_competition_data(competition: str) -> dict[str, Any]:
    local_dir = competition_data_dir(competition)
    local_files = [path for path in local_dir.iterdir() if path.is_file()] if local_dir.exists() else []
    inspection_files = _inspection_files(competition)
    file_profiles = [_profile_file(path, local_dir) for path in sorted(local_files)]
    if not file_profiles:
        file_profiles = [_profile_inspection_file(item) for item in inspection_files]

    train = _first_role(file_profiles, "train")
    test = _first_role(file_profiles, "test")
    sample = _first_role(file_profiles, "sample_submission")
    target_candidates = _target_candidates(train, test, sample)
    id_candidates = _id_candidates(train, test, sample)
    task_type = _infer_task_type(file_profiles, train)
    status = "ready" if local_files else "needs_data_download"
    questions = _human_review_questions(status, task_type, target_candidates, sample)
    profile = {
        "competition": competition,
        "status": status,
        "profiled_at": now_iso(),
        "local_data_dir": f"data/{competition}",
        "task_type": task_type,
        "target_candidates": target_candidates,
        "id_candidates": id_candidates,
        "files": file_profiles,
        "human_review_questions": questions,
        "next_steps": _next_steps(status, task_type, target_candidates),
    }
    out_dir = competition_dir(competition)
    write_text(out_dir / "data_profile.json", json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "data_profile.md", render_data_profile(profile))
    return profile


def render_data_profile(profile: dict[str, Any]) -> str:
    lines = [
        f"# Data Profile: {profile['competition']}",
        "",
        f"- status: {profile['status']}",
        f"- task_type: {profile['task_type']}",
        f"- local_data_dir: {profile['local_data_dir']}",
        f"- target_candidates: {', '.join(profile['target_candidates']) if profile['target_candidates'] else 'None'}",
        f"- id_candidates: {', '.join(profile['id_candidates']) if profile['id_candidates'] else 'None'}",
        "",
        "## Files",
        "",
    ]
    for item in profile["files"] or []:
        columns = ", ".join(item.get("columns", [])) if item.get("columns") else "schema not loaded"
        lines.append(f"- {item['name']} [{item['role']}] rows_sampled={item.get('rows_sampled', 0)} columns={columns}")
    if not profile["files"]:
        lines.append("- No files discovered.")
    lines.extend(["", "## Human Review Questions", ""])
    lines.extend(f"- {item}" for item in profile["human_review_questions"] or ["None"])
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in profile["next_steps"])
    lines.append("")
    return "\n".join(lines)


def _inspection_files(competition: str) -> list[dict[str, Any]]:
    path = competition_dir(competition) / "competition_inspection.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("files", [])


def _profile_file(path: Path, data_root: Path) -> dict[str, Any]:
    role = _infer_role(path.name)
    suffix = path.suffix.lower()
    profile: dict[str, Any] = {
        "name": path.relative_to(data_root).as_posix(),
        "role": role,
        "format": suffix.lstrip(".") or "unknown",
        "path": f"data/{data_root.name}/{path.relative_to(data_root).as_posix()}",
        "columns": [],
        "rows_sampled": 0,
    }
    if suffix == ".csv":
        schema = _read_csv_schema(path)
        profile.update(schema)
    return profile


def _profile_inspection_file(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name", "")
    suffix = Path(name).suffix.lower()
    return {
        "name": name,
        "role": _infer_role(name),
        "format": suffix.lstrip(".") or "unknown",
        "path": "",
        "size": item.get("size", ""),
        "columns": [],
        "rows_sampled": 0,
    }


def _read_csv_schema(path: Path, max_rows: int = 20) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append(row)
    return {
        "columns": columns,
        "rows_sampled": len(rows),
        "column_types": {column: _infer_column_type([row.get(column, "") for row in rows]) for column in columns},
    }


def _infer_role(name: str) -> str:
    lowered = name.lower()
    if "sample_submission" in lowered or "sample-submission" in lowered:
        return "sample_submission"
    if Path(lowered).stem == "train" or "train" in lowered:
        return "train"
    if Path(lowered).stem == "test" or "test" in lowered:
        return "test"
    return "unknown"


def _first_role(files: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    return next((item for item in files if item.get("role") == role), None)


def _target_candidates(
    train: dict[str, Any] | None,
    test: dict[str, Any] | None,
    sample: dict[str, Any] | None,
) -> list[str]:
    if not train:
        return []
    train_cols = train.get("columns", [])
    test_cols = set(test.get("columns", []) if test else [])
    sample_cols = [col for col in sample.get("columns", [])] if sample else []
    candidates = [col for col in train_cols if col not in test_cols]
    if sample_cols:
        sample_targets = [col for col in sample_cols if col in candidates]
        if sample_targets:
            return sample_targets
    return candidates[:3]


def _id_candidates(
    train: dict[str, Any] | None,
    test: dict[str, Any] | None,
    sample: dict[str, Any] | None,
) -> list[str]:
    column_groups = [item.get("columns", []) for item in [train, test, sample] if item]
    if not column_groups:
        return []
    common = set(column_groups[0])
    for columns in column_groups[1:]:
        common &= set(columns)
    preferred = [col for col in column_groups[0] if col in common and ("id" in col.lower() or col.lower().endswith("_id"))]
    return preferred[:3]


def _infer_task_type(files: list[dict[str, Any]], train: dict[str, Any] | None) -> str:
    formats = {item.get("format") for item in files}
    names = " ".join(item.get("name", "").lower() for item in files)
    if "jpg" in formats or "jpeg" in formats or "png" in formats or "images" in names:
        return "image"
    if "csv" in formats and train and train.get("columns"):
        return "tabular"
    return "unknown"


def _infer_column_type(values: list[str]) -> str:
    non_empty = [value for value in values if value not in {"", None}]
    if not non_empty:
        return "unknown"
    try:
        for value in non_empty:
            float(value)
        return "numeric"
    except (TypeError, ValueError):
        return "categorical_or_text"


def _human_review_questions(
    status: str,
    task_type: str,
    target_candidates: list[str],
    sample: dict[str, Any] | None,
) -> list[str]:
    if status == "needs_data_download":
        return ["Download competition data into data/<competition>/ before baseline generation."]
    questions = []
    if not target_candidates:
        questions.append("Which column is the training target?")
    if task_type == "unknown":
        questions.append("What is the competition task type: tabular, image, text, time-series, or other?")
    if not sample:
        questions.append("What submission columns are required if sample_submission is missing?")
    return questions or ["Confirm that detected target and task type are correct before baseline generation."]


def _next_steps(status: str, task_type: str, target_candidates: list[str]) -> list[str]:
    if status == "needs_data_download":
        return ["Download data.", "Run profile-data again after files are available."]
    steps = ["Review data_profile.md."]
    if target_candidates and task_type != "unknown":
        steps.append("Generate a baseline pipeline for the detected task type.")
    else:
        steps.append("Resolve human review questions before baseline generation.")
    return steps
