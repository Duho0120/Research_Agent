from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    competition = "titanic"
    workspace = root / "demo_workspaces" / competition
    competition_dir = root / "competitions" / competition
    trial_dir = root / "experiments" / competition / "trial_001"

    required = [
        workspace / "data" / "train.csv",
        workspace / "data" / "test.csv",
        workspace / "data" / "gender_submission.csv",
        trial_dir / "mock_plan_response.json",
        trial_dir / "mock_code_response.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        print("Missing required demo files:")
        for item in missing:
            print(f"- {item}")
        return 1

    workspace_config = {
        "competition": competition,
        "required_data_files": ["train.csv", "test.csv"],
        "target_column": "Survived",
        "id_column": "PassengerId",
        "metric": "accuracy",
        "objective": "maximize",
    }
    (workspace / "workspace_config.json").write_text(
        json.dumps(workspace_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    python_path = Path(sys.executable).resolve()
    execution_profile = "\n".join(
        [
            "schema_version: 1.0",
            f"competition: {competition}",
            "platform: kaggle",
            f'project_root: "{_yaml_path(workspace)}"',
            f'python: "{_yaml_path(python_path)}"',
            "commands:",
            "  test:",
            '    - "{python} test_step.py"',
            "  train:",
            '    - "{python} train_step.py"',
            "  predict:",
            '    - "{python} predict_step.py"',
            "artifacts:",
            "  metrics:",
            "    - outputs/metrics.json",
            "  submission:",
            "    - outputs/submission.csv",
            "write_scope:",
            "  allowed:",
            "    - src/",
            "    - tests/",
            "    - train_step.py",
            "    - predict_step.py",
            "    - test_step.py",
            "    - workspace_config.json",
            "  forbidden:",
            "    - data/",
            "    - outputs/metrics.json",
            "    - outputs/submission.csv",
            "submission_mode: kaggle_cli",
            "",
        ]
    )
    competition_dir.mkdir(parents=True, exist_ok=True)
    (competition_dir / "execution_profile.yaml").write_text(execution_profile, encoding="utf-8")

    state = "\n".join(
        [
            "competition:",
            f"  name: {competition}",
            "  metric: accuracy",
            "  objective: maximize",
            "  submission_limit_per_day: 5",
            "  platform: kaggle",
            "  topic: Titanic - Machine Learning from Disaster",
            f'  source_path: "{_yaml_path(workspace)}"',
            "current_state:",
            "  active_trial: trial_001",
            "  best_trial: null",
            "  consecutive_failures: 0",
            "  submissions_today: 0",
            "  validation_suspected: false",
            "strategy:",
            "  current_focus: build reliable baseline",
            "  promising_directions:",
            "  forbidden_directions:",
            "",
        ]
    )
    (competition_dir / "state.yaml").write_text(state, encoding="utf-8")

    workspace_source = {
        "competition": competition,
        "topic": "Titanic - Machine Learning from Disaster",
        "platform": "kaggle",
        "source_path": str(workspace),
        "python": str(python_path),
        "status": "ready",
        "created_workspace": True,
        "required_data_files": ["train.csv", "test.csv"],
    }
    (competition_dir / "workspace_source.json").write_text(
        json.dumps(workspace_source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Demo profile is ready.")
    print(f"- competition: {competition}")
    print(f"- workspace: {workspace}")
    print(f"- python: {python_path}")
    return 0


def _yaml_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


if __name__ == "__main__":
    raise SystemExit(main())
