import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.state_db import (
    create_pending_action,
    get_best_trial,
    get_trial_summary,
    initialize_state_db,
    list_competitions,
    list_pending_actions,
    list_trials,
    record_token_usage,
    resolve_pending_action,
    upsert_competition,
    upsert_submission,
    upsert_trial,
    upsert_trial_artifact,
    upsert_trial_decision,
    upsert_trial_score,
)


class StateDbTest(unittest.TestCase):
    def test_initialize_state_db_creates_minimal_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()

            self.assertEqual(root / "memory" / "research_agent.sqlite3", db_path)
            with closing(sqlite3.connect(db_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }

            self.assertIn("competitions", tables)
            self.assertIn("trials", tables)
            self.assertIn("trial_scores", tables)
            self.assertIn("trial_decisions", tables)
            self.assertIn("trial_artifacts", tables)
            self.assertIn("pending_actions", tables)

    def test_upserts_trial_state_and_queries_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)

            upsert_competition(
                {
                    "competition_id": "demo",
                    "platform": "kaggle",
                    "topic": "Titanic",
                    "metric": "accuracy",
                    "objective": "maximize",
                    "status": "running",
                    "workspace_path": "demo_workspaces/demo",
                },
                db_path,
            )
            upsert_trial(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "status": "completed",
                    "plan_type": "initial_pipeline_plan",
                    "primary_change_axis": None,
                    "recommended_base_trial": "trial_001",
                },
                db_path,
            )
            upsert_trial_score(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "metric": "accuracy",
                    "objective": "maximize",
                    "local_score": 0.83,
                    "local_status": "baseline",
                    "is_best_local": True,
                },
                db_path,
            )
            upsert_trial_decision(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "decision": "baseline_established",
                    "axis_attempt_count": 0,
                    "axis_attempt_limit": 3,
                    "recommended_base_trial": "trial_001",
                    "rejected_axes": [],
                    "rejected_candidates": [],
                    "planner_constraints": ["Change exactly one primary improvement axis."],
                },
                db_path,
            )
            artifact = upsert_trial_artifact(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "artifact_type": "plan_ko",
                    "path": "runs/demo/trial_001/01_plan.ko.md",
                    "is_user_facing": True,
                },
                db_path,
            )

            competitions = list_competitions(db_path)
            trials = list_trials("demo", db_path)
            summary = get_trial_summary("demo", "trial_001", db_path)
            best = get_best_trial("demo", db_path)

            self.assertEqual(["demo"], [row["competition_id"] for row in competitions])
            self.assertEqual(["trial_001"], [row["trial_id"] for row in trials])
            self.assertEqual("completed", summary["status"])
            self.assertEqual(0.83, summary["local_score"])
            self.assertTrue(summary["is_best_local"])
            self.assertEqual("baseline_established", summary["decision"])
            self.assertEqual("trial_001", best["trial_id"])
            self.assertTrue(artifact["is_user_facing"])

    def test_best_trial_respects_minimize_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)
            upsert_competition({"competition_id": "demo", "objective": "minimize"}, db_path)
            for trial_id, score in [("trial_001", 0.7), ("trial_002", 0.5)]:
                upsert_trial({"competition_id": "demo", "trial_id": trial_id, "status": "completed"}, db_path)
                upsert_trial_score(
                    {
                        "competition_id": "demo",
                        "trial_id": trial_id,
                        "metric": "rmse",
                        "objective": "minimize",
                        "local_score": score,
                    },
                    db_path,
                )

            self.assertEqual("trial_002", get_best_trial("demo", db_path)["trial_id"])

    def test_pending_actions_and_operational_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)
            upsert_competition({"competition_id": "demo"}, db_path)
            upsert_trial({"competition_id": "demo", "trial_id": "trial_001"}, db_path)

            usage = record_token_usage(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "call_type": "experiment_planning",
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                },
                db_path,
            )
            submission = upsert_submission(
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
            action = create_pending_action(
                {
                    "action_id": "action_001",
                    "competition_id": "demo",
                    "trial_id": "trial_001",
                    "action_type": "approval_required",
                    "priority": 10,
                    "message": "Submission approval required.",
                    "payload": {"submission_file": submission["submission_file"]},
                },
                db_path,
            )

            pending = list_pending_actions("demo", db_path)
            resolved = resolve_pending_action("action_001", db_path)

            self.assertGreaterEqual(usage["usage_id"], 1)
            self.assertTrue(submission["requires_user_approval"])
            self.assertEqual({"submission_file": "runs/demo/trial_001/submission.csv"}, action["payload"])
            self.assertEqual(["action_001"], [item["action_id"] for item in pending])
            self.assertEqual("resolved", resolved["status"])

    def test_missing_required_field_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)

            with self.assertRaises(ValueError):
                upsert_trial({"competition_id": "demo"}, db_path)


if __name__ == "__main__":
    unittest.main()
