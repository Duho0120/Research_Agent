from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import simple_yaml
from .defaults import DEFAULT_ALLOWED_SPACE, DEFAULT_CONFIG, DEFAULT_NOTES, DEFAULT_RULES, DEFAULT_STATE
from .paths import (
    competition_configs_dir,
    competition_dir,
    competition_jobs_dir,
    competition_memory_dir,
    configs_dir,
    experiment_dir,
    jobs_dir,
    memory_dir,
    trial_dir,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_project(competition: str, metric: str = "unknown", objective: str = "maximize") -> None:
    competition_path = competition_dir(competition)
    competition_path.mkdir(parents=True, exist_ok=True)
    experiment_dir(competition).mkdir(parents=True, exist_ok=True)
    memory_dir().mkdir(parents=True, exist_ok=True)
    jobs_dir().mkdir(parents=True, exist_ok=True)
    configs_dir().mkdir(parents=True, exist_ok=True)
    competition_memory_dir(competition).mkdir(parents=True, exist_ok=True)
    competition_jobs_dir(competition).mkdir(parents=True, exist_ok=True)
    competition_configs_dir(competition).mkdir(parents=True, exist_ok=True)
    Path("colab").mkdir(parents=True, exist_ok=True)

    state = DEFAULT_STATE.copy()
    state["competition"] = dict(state["competition"])
    state["competition"]["name"] = competition
    state["competition"]["metric"] = metric
    state["competition"]["objective"] = objective

    write_if_missing(competition_path / "overview.md", f"# {competition}\n\nDescribe the competition, data, and important constraints here.\n")
    write_if_missing(competition_path / "metric.md", f"# Metric\n\n- name: {metric}\n- objective: {objective}\n")
    write_if_missing(competition_path / "data_notes.md", "# Data Notes\n\nAdd column notes, leakage risks, and split assumptions here.\n")
    write_yaml_if_missing(competition_path / "state.yaml", state)
    write_yaml_if_missing(competition_configs_dir(competition) / "allowed_space.yaml", DEFAULT_ALLOWED_SPACE)
    write_if_missing(competition_memory_dir(competition) / "research_notes.md", DEFAULT_NOTES)
    write_if_missing(competition_memory_dir(competition) / "rules.md", DEFAULT_RULES)
    write_if_missing(competition_memory_dir(competition) / "trial_index.jsonl", "")

    first_trial = trial_dir(competition, "trial_001")
    first_trial.mkdir(parents=True, exist_ok=True)
    write_yaml_if_missing(first_trial / "config.yaml", DEFAULT_CONFIG)
    write_if_missing(first_trial / "plan.md", "# Trial 001 Plan\n\nInitial baseline trial.\n")
    write_if_missing(first_trial / "metrics.example.json", json.dumps(example_metrics("trial_001"), indent=2) + "\n")


def write_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_yaml_if_missing(path: Path, data: Any) -> None:
    if not path.exists():
        simple_yaml.dump(data, path)


def load_state(competition: str) -> dict[str, Any]:
    return simple_yaml.load(competition_dir(competition) / "state.yaml", default={})


def save_state(competition: str, state: dict[str, Any]) -> None:
    simple_yaml.dump(state, competition_dir(competition) / "state.yaml")


def load_trial_index(competition: str | None = None) -> list[dict[str, Any]]:
    path = _trial_index_path(competition)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_trial_index(row: dict[str, Any]) -> None:
    competition = row.get("competition")
    path = competition_memory_dir(competition) / "trial_index.jsonl" if competition else memory_dir() / "trial_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_recent_trials(competition: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = [row for row in load_trial_index(competition) if row.get("competition") == competition]
    return rows[-limit:]


def _trial_index_path(competition: str | None = None) -> Path:
    if competition:
        path = competition_memory_dir(competition) / "trial_index.jsonl"
        if path.exists():
            return path
    return memory_dir() / "trial_index.jsonl"


def read_text(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def example_metrics(trial_id: str) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "cv_score": 0.0,
        "lb_score": None,
        "objective": "maximize",
        "seed": 42,
        "prediction_correlation_with_best": None,
        "notes": "Replace this file with real metrics.json after training.",
    }
