import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent import simple_yaml
from research_agent.cli import main
from research_agent.state_db import get_trial_summary
from research_agent.workspace_result_cycle import process_workspace_result


class WorkspaceResultCycleTest(unittest.TestCase):
    def test_first_nonurgent_review_is_deferred_but_trial_is_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(
                root,
                "trial_001",
                {
                    "cv_score": 0.70,
                    "objective": "maximize",
                    "segment_errors": [{"segment": "group_a", "error_rate": 0.4}],
                    "notes": "first baseline",
                },
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = process_workspace_result("demo", "trial_001")

            self.assertEqual("completed_review_deferred", result["status"])
            self.assertEqual("defer", result["human_review"]["timing"])
            self.assertTrue(result["memory"]["is_best"])
            self.assertFalse((root / "experiments" / "demo" / "trial_001" / "review_pack").exists())
            queue = json.loads(
                (root / "memory" / "demo" / "deferred_review_queue.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, len(queue["items"]))
            rows = (root / "memory" / "demo" / "trial_index.jsonl").read_text().splitlines()
            self.assertEqual(1, len(rows))

    def test_repeated_nonurgent_evidence_accumulates_before_requesting_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            metrics = {
                "cv_score": 0.70,
                "objective": "maximize",
                "segment_errors": [{"segment": "group_a", "error_rate": 0.4}],
            }
            self._write_trial(root, "trial_001", metrics)
            self._write_trial(root, "trial_002", {**metrics, "cv_score": 0.72})
            self._write_trial(root, "trial_003", {**metrics, "cv_score": 0.71})

            with patch("research_agent.paths.project_root", return_value=root):
                first = process_workspace_result("demo", "trial_001")
                second = process_workspace_result("demo", "trial_002")
                third = process_workspace_result("demo", "trial_003")

            self.assertEqual("completed_review_deferred", first["status"])
            self.assertEqual("completed_review_deferred", second["status"])
            self.assertEqual("awaiting_human_review", third["status"])
            self.assertEqual("request_now", third["human_review"]["timing"])
            self.assertEqual("request-user-review", third["next_action"])
            self.assertTrue((root / "experiments" / "demo" / "trial_003" / "review_pack" / "manifest.json").exists())
            db_path = root / "memory" / "research_agent.sqlite3"
            from research_agent.state_db import list_pending_actions

            pending = list_pending_actions("demo", db_path)
            self.assertEqual(1, len(pending))
            self.assertEqual("trial_003", pending[0]["trial_id"])
            queue = json.loads(
                (root / "memory" / "demo" / "deferred_review_queue.json").read_text(encoding="utf-8")
            )
            self.assertEqual([], queue["items"])
            rows = (root / "memory" / "demo" / "trial_index.jsonl").read_text().splitlines()
            self.assertEqual(3, len(rows))

    def test_leakage_requests_review_on_first_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(
                root,
                "trial_001",
                {"cv_score": 0.70, "objective": "maximize", "leakage_warning": True},
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = process_workspace_result("demo", "trial_001")

            self.assertEqual("awaiting_human_review", result["status"])
            self.assertTrue(result["human_review"]["urgent"])
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "review_pack").exists())
            rows = (root / "memory" / "demo" / "trial_index.jsonl").read_text().splitlines()
            self.assertEqual(1, len(rows))

    def test_pending_review_prevents_repeated_nonurgent_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(
                root,
                "trial_001",
                {"cv_score": 0.70, "objective": "maximize", "leakage_warning": True},
            )
            self._write_trial(
                root,
                "trial_002",
                {
                    "cv_score": 0.72,
                    "objective": "maximize",
                    "segment_errors": [{"segment": "group_a", "error_rate": 0.4}],
                },
            )

            with patch("research_agent.paths.project_root", return_value=root):
                first = process_workspace_result("demo", "trial_001")
                second = process_workspace_result("demo", "trial_002")

            self.assertEqual("awaiting_human_review", first["status"])
            self.assertEqual("completed", second["status"])
            self.assertEqual("suppressed_pending", second["human_review"]["action"])
            self.assertTrue(second["pipeline_readiness"]["pending_user_review"])
            self.assertFalse((root / "experiments" / "demo" / "trial_002" / "review_pack").exists())

    def test_no_review_path_completes_and_remembers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(root, "trial_001", {"cv_score": 0.70, "objective": "maximize"})

            with patch("research_agent.paths.project_root", return_value=root):
                result = process_workspace_result("demo", "trial_001")

            self.assertEqual("completed", result["status"])
            self.assertEqual("no_review", result["human_review"]["timing"])
            self.assertEqual("plan-next-experiment", result["next_action"])
            self.assertTrue(result["memory"]["is_best"])
            self.assertEqual("synced", result["state_db_sync"]["status"])
            db_path = root / "memory" / "research_agent.sqlite3"
            self.assertTrue(db_path.exists())
            summary = get_trial_summary("demo", "trial_001", db_path)
            self.assertEqual("completed", summary["status"])
            self.assertEqual(0.70, summary["local_score"])

    def test_collection_must_be_collected_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(
                root,
                "trial_001",
                {"cv_score": 0.70, "objective": "maximize"},
                collection_status="needs_review",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = process_workspace_result("demo", "trial_001")

            self.assertEqual("blocked", result["status"])
            self.assertEqual("collect-workspace-metrics", result["next_action"])
            self.assertFalse((root / "experiments" / "demo" / "trial_001" / "evaluation.md").exists())
            self.assertFalse((root / "memory" / "demo" / "trial_index.jsonl").exists())

    def test_collected_status_without_metrics_file_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(root, "trial_001", {"cv_score": 0.70, "objective": "maximize"})
            (root / "experiments" / "demo" / "trial_001" / "metrics.json").unlink()

            with patch("research_agent.paths.project_root", return_value=root):
                result = process_workspace_result("demo", "trial_001")

            self.assertEqual("blocked", result["status"])
            self.assertIn("missing_trial_metrics", result["issues"])
            self.assertEqual("collect-workspace-metrics", result["next_action"])

    def test_duplicate_trial_is_not_remembered_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(root, "trial_001", {"cv_score": 0.70, "objective": "maximize"})

            with patch("research_agent.paths.project_root", return_value=root):
                process_workspace_result("demo", "trial_001")
                duplicate = process_workspace_result("demo", "trial_001")

            self.assertEqual("already_processed", duplicate["status"])
            rows = (root / "memory" / "demo" / "trial_index.jsonl").read_text().splitlines()
            self.assertEqual(1, len(rows))

    def test_process_workspace_result_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_trial(root, "trial_001", {"cv_score": 0.70, "objective": "maximize"})

            with patch("research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "process-workspace-result",
                        "--competition",
                        "demo",
                        "--trial",
                        "trial_001",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue(
                (root / "experiments" / "demo" / "trial_001" / "workspace_result_cycle.json").exists()
            )

    def _write_state(self, root: Path) -> None:
        competition = root / "competitions" / "demo"
        competition.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": "accuracy", "objective": "maximize"},
                "current_state": {
                    "active_trial": None,
                    "best_trial": None,
                    "consecutive_failures": 0,
                    "submissions_today": 0,
                    "validation_suspected": False,
                },
                "strategy": {"current_focus": "baseline", "promising_directions": [], "forbidden_directions": []},
            },
            competition / "state.yaml",
        )

    def _write_trial(self, root: Path, trial_id: str, metrics: dict, *, collection_status: str = "collected") -> None:
        trial = root / "experiments" / "demo" / trial_id
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "plan.md").write_text(f"# {trial_id} Plan\n", encoding="utf-8")
        (trial / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (trial / "metrics_collection.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": trial_id,
                    "status": collection_status,
                    "profile_status": "ready",
                    "workspace_run_status": "completed",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
