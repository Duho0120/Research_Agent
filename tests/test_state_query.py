import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from research_agent.cli import main
from research_agent.state_db import (
    initialize_state_db,
    record_token_usage,
    upsert_competition,
    upsert_submission,
    upsert_trial,
    upsert_trial_artifact,
    upsert_trial_decision,
    upsert_trial_score,
)
from research_agent.state_query import (
    get_experiment_status,
    get_trial_detail,
    list_experiment_statuses,
    render_experiment_status,
    render_experiment_statuses,
    render_trial_detail,
)


class StateQueryTest(unittest.TestCase):
    def test_state_query_builds_ui_ready_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_state_db_fixture(db_path)

            experiments = list_experiment_statuses(db_path)
            experiment = get_experiment_status("demo", db_path)
            detail = get_trial_detail("demo", "trial_001", db_path)

            self.assertEqual(1, experiments["experiment_count"])
            self.assertEqual("demo", experiments["experiments"][0]["competition_id"])
            self.assertEqual("trial_001", experiments["experiments"][0]["best_trial"]["trial_id"])
            self.assertEqual(2, experiment["trial_count"])
            self.assertEqual(300, experiment["total_tokens"])
            self.assertEqual(1, experiment["artifact_count"])
            self.assertEqual(1, experiment["submission_count"])
            self.assertEqual("trial_001", detail["summary"]["trial_id"])
            self.assertEqual(300, detail["token_total"])
            self.assertEqual(["plan_ko"], [item["artifact_type"] for item in detail["user_artifacts"]])

            self.assertIn("실험 상태 요약", render_experiment_statuses(experiments))
            self.assertIn("실험 현황: demo", render_experiment_status(experiment))
            self.assertIn("Trial 상세: demo / trial_001", render_trial_detail(detail))

    def test_cli_state_query_commands_render_text_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_state_db_fixture(db_path)

            text_output = io.StringIO()
            with redirect_stdout(text_output):
                exit_code = main(["show-trial", "--competition", "demo", "--trial", "trial_001", "--db-path", str(db_path)])

            json_output = io.StringIO()
            with redirect_stdout(json_output):
                json_exit_code = main(["show-experiment", "--competition", "demo", "--db-path", str(db_path), "--json"])

            self.assertEqual(0, exit_code)
            self.assertEqual(0, json_exit_code)
            self.assertIn("사용자가 확인할 파일", text_output.getvalue())
            self.assertIn('"competition_id": "demo"', json_output.getvalue())

    def test_kaggle_best_prefers_leaderboard_score_over_local_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_state_db_fixture(db_path)
            upsert_trial_score(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "metric": "accuracy",
                    "objective": "maximize",
                    "local_score": 0.83,
                    "lb_score": 0.76,
                    "is_best_local": True,
                    "is_best_lb": False,
                },
                db_path,
            )
            upsert_trial_score(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_002",
                    "metric": "accuracy",
                    "objective": "maximize",
                    "local_score": 0.81,
                    "lb_score": 0.78,
                    "is_best_local": False,
                    "is_best_lb": True,
                },
                db_path,
            )

            experiment = get_experiment_status("demo", db_path)

            self.assertEqual("trial_002", experiment["best_trial"]["trial_id"])


def _write_state_db_fixture(db_path: Path) -> None:
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
    for trial_id, score in [("trial_001", 0.83), ("trial_002", 0.81)]:
        upsert_trial(
            {
                "competition_id": "demo",
                "trial_id": trial_id,
                "status": "completed",
                "primary_change_axis": "preprocessing",
                "recommended_base_trial": "trial_001",
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
                "is_best_local": trial_id == "trial_001",
            },
            db_path,
        )
    upsert_trial_decision(
        {
            "competition_id": "demo",
            "trial_id": "trial_001",
            "decision": "baseline_established",
            "change_axis": "preprocessing",
            "active_axis": "preprocessing",
            "recommended_base_trial": "trial_001",
        },
        db_path,
    )
    upsert_trial_artifact(
        {
            "competition_id": "demo",
            "trial_id": "trial_001",
            "artifact_type": "plan_ko",
            "path": "runs/demo/trial_001/01_plan.ko.md",
            "is_user_facing": True,
        },
        db_path,
    )
    record_token_usage(
        {
            "source_key": "memory/demo/token_usage.jsonl:1",
            "competition_id": "demo",
            "trial_id": "trial_001",
            "provider": "openai",
            "model": "gpt-5.5",
            "call_type": "experiment_planning",
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        },
        db_path,
    )
    upsert_submission(
        {
            "competition_id": "demo",
            "trial_id": "trial_001",
            "platform": "kaggle",
            "submission_file": "runs/demo/trial_001/submission.csv",
            "status": "prepared",
            "requires_user_approval": True,
        },
        db_path,
    )


if __name__ == "__main__":
    unittest.main()

