from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_agent import simple_yaml
from research_agent.state_db import default_db_path, state_db_connection
from research_agent.state_db_sync import sync_state_db


SEED_NAMESPACES = [
    "competitions",
    "experiments",
    "runs",
    "memory",
    "demo_workspaces",
    "submissions",
]


def initialize_demo_seed(
    seed_root: Path,
    storage_dir: Path,
    *,
    competition: str = "titanic",
) -> dict[str, Any]:
    copied: list[str] = []
    skipped: list[str] = []
    storage_dir.mkdir(parents=True, exist_ok=True)
    for namespace in SEED_NAMESPACES:
        source = seed_root / namespace / competition
        target = storage_dir / namespace / competition
        if not source.is_dir():
            continue
        if target.exists() and any(target.iterdir()):
            skipped.append(namespace)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rmdir()
        shutil.copytree(source, target)
        copied.append(namespace)

    _normalize_execution_paths(storage_dir, competition)
    sync_result = sync_state_db(competition)
    runtime_dir = storage_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cli_state = runtime_dir / "cli_state.json"
    if not cli_state.exists():
        _write_json(cli_state, {"selected_competition": competition})
    loop_state = runtime_dir / "auto_loop_state.json"
    if not loop_state.exists():
        _write_json(loop_state, _initial_loop_state(competition))
    return {
        "competition": competition,
        "copied": copied,
        "skipped": skipped,
        "db_path": str(default_db_path()),
        "sync_status": sync_result.get("status"),
    }


def _normalize_execution_paths(storage_dir: Path, competition: str) -> None:
    workspace = (storage_dir / "demo_workspaces" / competition).resolve()
    profile_path = storage_dir / "competitions" / competition / "execution_profile.yaml"
    profile = simple_yaml.load(profile_path, default={})
    if isinstance(profile, dict) and profile:
        profile["project_root"] = str(workspace)
        profile["python"] = sys.executable
        simple_yaml.dump(profile, profile_path)

    source_path = storage_dir / "competitions" / competition / "workspace_source.json"
    source = _load_json(source_path)
    if source:
        source["source_path"] = str(workspace)
        source["python"] = sys.executable
        _write_json(source_path, source)


def _initial_loop_state(competition: str) -> dict[str, Any]:
    with state_db_connection(default_db_path()) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT trial_id, status
                FROM trials
                WHERE competition_id = ?
                ORDER BY trial_id
                """,
                [competition],
            )
        ]
    planned = [row for row in rows if str(row.get("status") or "").casefold() in {"planned", "ready"}]
    completed = [
        row
        for row in rows
        if str(row.get("status") or "").casefold() not in {"planned", "ready", "discovered"}
    ]
    last_completed = str(completed[-1]["trial_id"]) if completed else None
    next_trial = str(planned[-1]["trial_id"]) if planned else _next_trial_id(last_completed or "trial_000")
    return {
        "competition": competition,
        "status": "completed",
        "phase": "planned" if planned else "planning",
        "current_trial": None,
        "last_completed_trial": last_completed,
        "next_trial": next_trial,
        "pause_requested": False,
        "pid": None,
        "error": None,
    }


def _next_trial_id(trial_id: str) -> str:
    prefix, _, suffix = trial_id.rpartition("_")
    try:
        return f"{prefix}_{int(suffix) + 1:03d}"
    except ValueError:
        return "trial_001"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize empty container storage with demo records.")
    parser.add_argument("--seed-root", required=True)
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--competition", default="titanic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = initialize_demo_seed(
        Path(args.seed_root).expanduser().resolve(),
        Path(args.storage_dir).expanduser().resolve(),
        competition=args.competition,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
