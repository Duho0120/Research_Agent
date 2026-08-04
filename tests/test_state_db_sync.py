import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from research_agent.cli import main
from research_agent.state_db import (
    get_best_trial,
    get_trial_summary,
    initialize_state_db,
    list_competitions,
    upsert_competition,
    upsert_trial,
    upsert_trial_artifact,
)
from research_agent.state_db_sync import sync_state_db


class StateDbSyncTest(unittest.TestCase):
    def test_sync_state_db_reads_topic_from_competition_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)
            workspace_config = root / "demo_workspaces" / "demo" / "workspace_config.json"
            config = json.loads(workspace_config.read_text(encoding="utf-8"))
            config.pop("topic", None)
            _write_json(workspace_config, config)
            (root / "competitions" / "demo" / "state.yaml").write_text(
                "\n".join(
                    [
                        "competition:",
                        "  name: demo",
                        "  topic: Bike Sharing Demand",
                        "  metric: rmsle",
                        "  objective: minimize",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                sync_state_db("demo", db_path=db_path)

            self.assertEqual("Bike Sharing Demand", list_competitions(db_path)[0]["topic"])

    def test_sync_state_db_imports_file_based_trial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)

            with patch("research_agent.paths.project_root", return_value=root):
                result = sync_state_db("demo", db_path=db_path)
                second_result = sync_state_db("demo", db_path=db_path)

            self.assertEqual("completed", result["status"])
            self.assertEqual(1, result["competition_count"])
            self.assertEqual(1, result["trial_count"])
            self.assertGreaterEqual(result["artifact_count"], 3)
            self.assertEqual(1, result["token_usage_count"])
            self.assertEqual(1, result["submission_count"])

            competitions = list_competitions(db_path)
            summary = get_trial_summary("demo", "trial_001", db_path)
            best = get_best_trial("demo", db_path)

            self.assertEqual(["demo"], [row["competition_id"] for row in competitions])
            self.assertEqual("kaggle", competitions[0]["platform"])
            self.assertEqual("completed", summary["status"])
            self.assertEqual("preprocessing", summary["primary_change_axis"])
            self.assertAlmostEqual(0.83, summary["local_score"])
            self.assertAlmostEqual(0.77, summary["lb_score"])
            self.assertEqual("trial_001", best["trial_id"])

            with closing(sqlite3.connect(db_path)) as connection:
                token_rows = connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
                artifact_rows = connection.execute("SELECT COUNT(*) FROM trial_artifacts").fetchone()[0]
                submission_rows = connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                user_artifact_types = [
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT artifact_type
                        FROM trial_artifacts
                        WHERE competition_id = 'demo'
                          AND trial_id = 'trial_001'
                          AND is_user_facing = 1
                        ORDER BY artifact_type
                        """
                    ).fetchall()
                ]

            self.assertEqual(1, token_rows)
            self.assertEqual(result["artifact_count"], artifact_rows)
            self.assertEqual(1, submission_rows)
            self.assertEqual(["pipeline_structure_ko", "plan_ko"], user_artifact_types)
            self.assertEqual(second_result["artifact_count"], result["artifact_count"])
            self.assertGreater(second_result["removed_stale_rows"], 0)

    def test_sync_state_db_removes_deleted_file_based_trial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)

            initialize_state_db(db_path)
            upsert_competition({"competition_id": "demo", "status": "has_trials"}, db_path)
            upsert_trial({"competition_id": "demo", "trial_id": "trial_old", "status": "completed"}, db_path)
            upsert_trial_artifact(
                {
                    "competition_id": "demo",
                    "trial_id": "trial_old",
                    "artifact_type": "plan_ko",
                    "path": "runs/demo/trial_old/01_plan.ko.md",
                    "is_user_facing": True,
                },
                db_path,
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = sync_state_db("demo", db_path=db_path)

            self.assertEqual("completed", result["status"])
            self.assertGreaterEqual(result["removed_stale_rows"], 2)
            self.assertIsNone(get_trial_summary("demo", "trial_old", db_path))
            self.assertEqual("completed", get_trial_summary("demo", "trial_001", db_path)["status"])

    def test_sync_state_db_imports_manual_trials_as_official_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)
            _write_manual_trial_tree(root, "trial_003")

            with patch("research_agent.paths.project_root", return_value=root):
                result = sync_state_db("demo", db_path=db_path)

            summary = get_trial_summary("demo", "trial_003", db_path)
            best_local = get_best_trial("demo", db_path)
            best_lb = get_best_trial("demo", db_path, prefer_lb=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual(2, result["trial_count"])
            self.assertEqual("completed", summary["status"])
            self.assertEqual("feature_engineering_name_title", summary["primary_change_axis"])
            self.assertAlmostEqual(0.85, summary["local_score"])
            self.assertAlmostEqual(0.78, summary["lb_score"])
            self.assertTrue(summary["is_best_local"])
            self.assertTrue(summary["is_best_lb"])
            self.assertEqual("trial_003", best_local["trial_id"])
            self.assertEqual("trial_003", best_lb["trial_id"])

            with closing(sqlite3.connect(db_path)) as connection:
                user_artifacts = connection.execute(
                    """
                    SELECT artifact_type, path
                    FROM trial_artifacts
                    WHERE competition_id = 'demo'
                      AND trial_id = 'trial_003'
                      AND is_user_facing = 1
                    ORDER BY artifact_type
                    """
                ).fetchall()
                submission = connection.execute(
                    """
                    SELECT lb_score, submission_file, requires_user_approval
                    FROM submissions
                    WHERE competition_id = 'demo' AND trial_id = 'trial_003'
                    """
                ).fetchone()

            self.assertEqual(["pipeline_structure_ko", "plan_ko", "scores_ko"], [row[0] for row in user_artifacts])
            self.assertIn("manual_trials/trial_003", user_artifacts[0][1])
            self.assertAlmostEqual(0.78, submission[0])
            self.assertIn("manual_trials", submission[1])
            self.assertEqual(0, submission[2])

    def test_sync_state_db_prefers_final_graph_cycle_status_over_partial_result_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)
            _write_json(root / "experiments" / "demo" / "trial_001" / "demo_graph_cycle.json", {"status": "blocked"})

            with patch("research_agent.paths.project_root", return_value=root):
                sync_state_db("demo", db_path=db_path)

            self.assertEqual("blocked", get_trial_summary("demo", "trial_001", db_path)["status"])

    def test_sync_state_db_recovers_axis_from_pipeline_plan_and_next_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)

            trial_002 = root / "experiments" / "demo" / "trial_002"
            _write_json(trial_002 / "metrics.json", {"metric": "accuracy", "objective": "maximize", "local_score": 0.81})
            _write_json(trial_002 / "workspace_result_cycle.json", {"status": "completed"})
            _write_json(trial_002 / "decision_card.json", {"decision": "accept", "change_axis": ""})
            _write_json(trial_002 / "pipeline_improvement_plan.json", {"primary_axis": "hyperparameter"})

            trial_003 = root / "experiments" / "demo" / "trial_003"
            _write_json(trial_003 / "metrics.json", {"metric": "accuracy", "objective": "maximize", "local_score": 0.82})
            _write_json(trial_003 / "workspace_result_cycle.json", {"status": "completed"})
            _write_json(trial_003 / "decision_card.json", {"decision": "accept", "change_axis": ""})
            (trial_003 / "next_experiment.md").write_text(
                "# trial_003 Next Experiment\n\n## Strategy\n\nmodel_ensemble\n\n## Rationale\n\nUser insight.\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                sync_state_db("demo", db_path=db_path)

            self.assertEqual("hyperparameter", get_trial_summary("demo", "trial_002", db_path)["primary_change_axis"])
            self.assertEqual("model_ensemble", get_trial_summary("demo", "trial_003", db_path)["primary_change_axis"])

    def test_sync_state_db_prefers_executed_trial_facts_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)
            trial = root / "experiments" / "demo" / "trial_001"
            _write_json(
                trial / "internal" / "executed_trial_facts.json",
                {
                    "source_trial_id": "trial_000",
                    "plan_type": "continuation_delta_plan",
                    "primary_change_axis": "model_ensemble",
                    "model": {"estimator": "VotingClassifier"},
                },
            )

            with patch("research_agent.paths.project_root", return_value=root):
                sync_state_db("demo", db_path=db_path)

            summary = get_trial_summary("demo", "trial_001", db_path)
            self.assertEqual("trial_000", summary["source_trial_id"])
            self.assertEqual("model_ensemble", summary["primary_change_axis"])
            self.assertEqual("continuation_delta_plan", summary["plan_type"])

    def test_sync_state_db_records_next_plan_as_planned_trial_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            _write_demo_competition_tree(root)
            planned = root / "experiments" / "demo" / "trial_002"
            _write_json(
                planned / "next_experiment.json",
                {
                    "status": "planned",
                    "source_trial_id": "trial_001",
                    "next_trial_id": "trial_002",
                    "strategy": "feature_engineering",
                },
            )
            (planned / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\nfeature_engineering\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                sync_state_db("demo", db_path=db_path)

            summary = get_trial_summary("demo", "trial_002", db_path)

        self.assertEqual("planned", summary["status"])
        self.assertEqual("feature_engineering", summary["plan_summary"])
        self.assertEqual("feature_engineering", summary["primary_change_axis"])
        self.assertIsNone(summary["local_score"])
        self.assertIsNone(summary["lb_score"])

    def test_cli_sync_state_db_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "state.sqlite3"
            _write_demo_competition_tree(root)

            with patch("research_agent.paths.project_root", return_value=root):
                exit_code = main(["sync-state-db", "--competition", "demo", "--db-path", str(db_path)])

            self.assertEqual(0, exit_code)
            self.assertEqual("completed", get_trial_summary("demo", "trial_001", db_path)["status"])


def _write_demo_competition_tree(root: Path) -> None:
    competition_dir = root / "competitions" / "demo"
    trial_dir = root / "experiments" / "demo" / "trial_001"
    internal_dir = trial_dir / "internal"
    run_dir = root / "runs" / "demo" / "trial_001"
    memory_dir = root / "memory" / "demo"
    submissions_dir = root / "submissions" / "demo"

    for path in [competition_dir, internal_dir, run_dir / "code", memory_dir, submissions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    (competition_dir / "execution_profile.yaml").write_text(
        "\n".join(
            [
                "platform: kaggle",
                "metric: accuracy",
                "objective: maximize",
                f"project_root: {root / 'demo_workspaces' / 'demo'}",
            ]
        ),
        encoding="utf-8",
    )
    workspace_dir = root / "demo_workspaces" / "demo"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        workspace_dir / "workspace_config.json",
        {
            "platform": "kaggle",
            "topic": "Demo Competition",
            "metric": "accuracy",
            "objective": "maximize",
        },
    )
    _write_json(
        trial_dir / "metrics.json",
        {
            "metric": "accuracy",
            "objective": "maximize",
            "local_score": 0.83,
        },
    )
    _write_json(trial_dir / "workspace_result_cycle.json", {"status": "completed"})
    _write_json(
        internal_dir / "demo_experiment_plan.json",
        {
            "plan_type": "initial_pipeline_plan",
            "source_trial_id": None,
            "primary_change_axis": "preprocessing",
            "recommended_base_trial": "trial_001",
        },
    )
    _write_json(
        internal_dir / "decision_card.json",
        {
            "decision": "baseline_established",
            "change_axis": "preprocessing",
            "active_axis": "preprocessing",
            "axis_attempt_count": 1,
            "axis_attempt_limit": 3,
            "recommended_base_trial": "trial_001",
            "planner_constraints": ["Change exactly one primary improvement axis."],
        },
    )
    _write_json(internal_dir / "pipeline_structure.json", {"pipeline_steps": []})
    _write_json(internal_dir / "code_snapshot_manifest.json", {"files": ["train_step.py"]})
    _write_json(
        trial_dir / "submit_manifest.json",
        {
            "platform": "kaggle",
            "submission_file": "runs/demo/trial_001/submission.csv",
            "status": "prepared",
            "requires_user_approval": True,
        },
    )
    _write_json(
        trial_dir / "submission_run.json",
        {
            "status": "submitted",
            "submitted_lb_score": 0.77,
            "submitted_rank": 1234,
            "submission_file": "runs/demo/trial_001/submission.csv",
        },
    )
    (trial_dir / "node_events.jsonl").write_text('{"node":"recorder"}\n', encoding="utf-8")
    (run_dir / "README.ko.md").write_text("# Demo\n", encoding="utf-8")
    (run_dir / "00_summary_card.ko.md").write_text("# Summary\n", encoding="utf-8")
    (run_dir / "01_plan.ko.md").write_text("# Plan\n", encoding="utf-8")
    (run_dir / "02_pipeline_structure.ko.md").write_text("# Pipeline\n", encoding="utf-8")
    (run_dir / "06_decision.ko.md").write_text("# Decision\n", encoding="utf-8")
    (run_dir / "code" / "train_step.py").write_text("print('train')\n", encoding="utf-8")
    (memory_dir / "token_usage.jsonl").write_text(
        json.dumps(
            {
                "competition": "demo",
                "trial_id": "trial_001",
                "provider": "openai",
                "model": "gpt-5.5",
                "call_type": "experiment_planning",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (submissions_dir / "submission_log.jsonl").write_text(
        json.dumps(
            {
                "competition": "demo",
                "trial_id": "trial_001",
                "platform": "kaggle",
                "submission_file": "runs/demo/trial_001/submission.csv",
                "status": "submitted",
                "submitted_lb_score": 0.77,
                "submitted_rank": 1234,
                "requires_user_approval": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manual_trial_tree(root: Path, trial_id: str) -> None:
    manual_dir = root / "demo_workspaces" / "demo" / "manual_trials" / trial_id
    user_view = manual_dir / "user_view"
    user_view.mkdir(parents=True, exist_ok=True)
    _write_json(
        manual_dir / "metrics.json",
        {
            "trial_id": trial_id,
            "local_score": 0.85,
            "metric": "accuracy",
            "objective": "maximize",
            "change_axis": "feature_engineering_name_title",
            "submission_file": str(manual_dir / "submission.csv"),
            "kaggle_submitted": True,
            "kaggle_status": "SubmissionStatus.COMPLETE",
            "kaggle_lb_score": 0.78,
        },
    )
    (manual_dir / "submission.csv").write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
    (user_view / "01_plan.ko.md").write_text("# Plan\n", encoding="utf-8")
    (user_view / "02_pipeline_structure.ko.md").write_text("# Pipeline\n", encoding="utf-8")
    (user_view / "03_scores.ko.md").write_text("# Scores\n", encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
