import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from research_agent.cli import main
from research_agent.operations import build_operations_status, next_trial_id
from research_agent.state_db import (
    initialize_state_db,
    upsert_competition,
    upsert_trial,
    upsert_trial_decision,
    upsert_trial_score,
)


class OperationsCliTest(unittest.TestCase):
    def test_operations_status_recommends_next_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_operations_fixture(db_path)

            status = build_operations_status(competition="demo", db_path=db_path)
            operation = status["experiment"]["operation"]

            self.assertEqual("ready_next_trial", operation["state"])
            self.assertEqual("trial_003", operation["next_trial_id"])
            self.assertEqual("feature_engineering", operation["active_axis"])
            self.assertEqual(2, operation["axis_attempt_count"])
            self.assertEqual(3, operation["axis_attempt_limit"])
            self.assertIn("feature_engineering", operation["next_action_label"])
            self.assertIn("run-next-trial", "\n".join(operation["commands"]))

    def test_next_trial_id_handles_empty_and_numbered_trials(self):
        self.assertEqual("trial_001", next_trial_id([]))
        self.assertEqual("trial_010", next_trial_id([{"trial_id": "trial_002"}, {"trial_id": "trial_009"}]))
        self.assertEqual("trial_004", next_trial_id([{"trial_id": "trial_v16"}, {"trial_id": "baseline"}, {"trial_id": "trial_003"}]))

    def test_discovered_scaffold_is_ready_first_trial_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)
            upsert_competition(
                {
                    "competition_id": "demo",
                    "platform": "kaggle",
                    "topic": "Demo Competition",
                    "metric": "rmsle",
                    "objective": "minimize",
                    "status": "has_trials",
                    "workspace_path": "demo_workspaces/demo",
                },
                db_path,
            )
            upsert_trial(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "status": "discovered",
                },
                db_path,
            )

            status = build_operations_status(competition="demo", db_path=db_path)
            operation = status["experiment"]["operation"]

            self.assertEqual("ready_first_trial", operation["state"])
            self.assertEqual("trial_001", operation["next_trial_id"])
            self.assertEqual("discovered", operation["latest_trial"]["status"])

    def test_status_command_renders_operator_friendly_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_operations_fixture(db_path)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["status", "--competition", "demo", "--db-path", str(db_path), "--no-sync"])

            text = output.getvalue()
            self.assertEqual(0, code)
            self.assertIn("실험 운영 상세: demo", text)
            self.assertIn("다음 trial: trial_003", text)
            self.assertIn("현재 개선축: feature_engineering (2/3)", text)
            self.assertIn("run-next-trial", text)

    def test_run_next_trial_dry_run_uses_computed_trial_id_without_running_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_operations_fixture(db_path)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "run-next-trial",
                        "--competition",
                        "demo",
                        "--db-path",
                        str(db_path),
                        "--no-sync",
                        "--dry-run",
                    ]
                )

            text = output.getvalue()
            self.assertEqual(0, code)
            self.assertIn("Dry run: 다음 실행 대상은 demo / trial_003", text)


def _write_operations_fixture(db_path: Path) -> None:
    initialize_state_db(db_path)
    upsert_competition(
        {
            "competition_id": "demo",
            "platform": "kaggle",
            "topic": "Demo Competition",
            "metric": "accuracy",
            "objective": "maximize",
            "status": "has_trials",
            "workspace_path": "demo_workspaces/demo",
        },
        db_path,
    )
    for trial_id, score, is_best in [("trial_001", 0.81, False), ("trial_002", 0.84, True)]:
        upsert_trial(
            {
                "competition_id": "demo",
                "trial_id": trial_id,
                "status": "completed",
                "primary_change_axis": "feature_engineering",
                "recommended_base_trial": "trial_002",
            },
            db_path,
        )
        upsert_trial_score(
            {
                "competition_id": "demo",
                "trial_id": trial_id,
                "metric": "accuracy",
                "objective": "maximize",
                "local_score": score,
                "is_best_local": is_best,
            },
            db_path,
        )
    upsert_trial_decision(
        {
            "competition_id": "demo",
            "trial_id": "trial_002",
            "decision": "continue_axis_refinement",
            "change_axis": "feature_engineering",
            "active_axis": "feature_engineering",
            "axis_attempt_count": 2,
            "axis_attempt_limit": 3,
            "recommended_base_trial": "trial_002",
        },
        db_path,
    )


if __name__ == "__main__":
    unittest.main()
