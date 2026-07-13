import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_metrics_collector import collect_workspace_metrics


class WorkspaceMetricsCollectorTest(unittest.TestCase):
    def test_collects_existing_cv_score_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"cv_score": 0.83, "accuracy": 0.91, "metric": "accuracy", "notes": "baseline"}),
                encoding="utf-8",
            )
            original = source.read_text(encoding="utf-8")
            self._write_workspace(root, project, "trial_001", status="completed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_001")

            self.assertEqual("collected", result["status"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_001" / "metrics.json").read_text())
            self.assertEqual(0.83, metrics["cv_score"])
            self.assertEqual(0.91, metrics["accuracy"])
            self.assertEqual("trial_001", metrics["trial_id"])
            self.assertEqual("maximize", metrics["objective"])
            self.assertEqual(original, source.read_text(encoding="utf-8"))
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "metrics_collection.json").exists())

    def test_collects_nested_score_from_explicit_source_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"validation": {"macro_f1": 0.77}, "epoch": 5}),
                encoding="utf-8",
            )
            self._write_workspace(
                root,
                project,
                "trial_002",
                status="completed",
                metrics_contract={"source_key": "validation.macro_f1"},
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_002")

            self.assertEqual("collected", result["status"])
            self.assertEqual("validation.macro_f1", result["score_source"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_002" / "metrics.json").read_text())
            self.assertEqual(0.77, metrics["cv_score"])
            self.assertEqual("accuracy", metrics["metric"])
            self.assertEqual(5, metrics["epoch"])

    def test_collects_validation_metric_key_from_competition_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"validation_accuracy": 0.79, "training_rows": 712}),
                encoding="utf-8",
            )
            self._write_workspace(root, project, "trial_validation_metric", status="completed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_validation_metric")

            self.assertEqual("collected", result["status"])
            self.assertEqual("validation_accuracy", result["score_source"])
            self.assertEqual(0.79, result["cv_score"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_validation_metric" / "metrics.json").read_text())
            self.assertEqual(0.79, metrics["cv_score"])
            self.assertEqual(712, metrics["training_rows"])

    def test_missing_score_mapping_requests_review_without_writing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"accuracy": 0.91}), encoding="utf-8")
            self._write_workspace(root, project, "trial_003", status="completed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_003")

            self.assertEqual("needs_review", result["status"])
            self.assertEqual("confirm-metrics-source-key", result["next_action"])
            self.assertTrue(result["review_questions"])
            self.assertFalse((root / "experiments" / "demo" / "trial_003" / "metrics.json").exists())

    def test_invalid_json_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text("not-json", encoding="utf-8")
            self._write_workspace(root, project, "trial_004", status="completed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_004")

            self.assertEqual("blocked", result["status"])
            self.assertIn("invalid_metrics_json", result["issues"])
            self.assertFalse((root / "experiments" / "demo" / "trial_004" / "metrics.json").exists())

    def test_non_completed_workspace_run_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"cv_score": 0.5}), encoding="utf-8")
            self._write_workspace(root, project, "trial_005", status="failed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_005")

            self.assertEqual("blocked", result["status"])
            self.assertIn("workspace_run_not_completed", result["issues"])
            self.assertFalse((root / "experiments" / "demo" / "trial_005" / "metrics.json").exists())

    def test_workspace_run_must_be_json_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_workspace(root, project, "trial_007", status="completed")
            workspace_run = root / "experiments" / "demo" / "trial_007" / "workspace_run.json"
            workspace_run.write_text("[]", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_007")

            self.assertEqual("blocked", result["status"])
            self.assertIn("invalid_or_missing_workspace_run", result["issues"])

    def test_collect_workspace_metrics_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"cv_score": 0.72}), encoding="utf-8")
            self._write_workspace(root, project, "trial_006", status="completed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "collect-workspace-metrics",
                        "--competition",
                        "demo",
                        "--trial",
                        "trial_006",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_006" / "metrics.json").exists())

    def _write_workspace(
        self,
        root: Path,
        project: Path,
        trial_id: str,
        *,
        status: str,
        metrics_contract: dict | None = None,
    ) -> None:
        competition = root / "competitions" / "demo"
        competition.mkdir(parents=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": "accuracy", "objective": "maximize"},
                "current_state": {},
            },
            competition / "state.yaml",
        )
        profile = {
            "schema_version": "1.0",
            "competition": "demo",
            "platform": "external",
            "project_root": str(project),
            "python": sys.executable,
            "commands": {"test": ["echo test"], "train": ["echo train"], "predict": ["echo predict"]},
            "artifacts": {
                "metrics": ["outputs/metrics.json"],
                "submission": ["outputs/submission.csv"],
            },
            "write_scope": {
                "allowed": ["src/"],
                "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
            },
            "submission_mode": "manual_external",
        }
        if metrics_contract is not None:
            profile["metrics_contract"] = metrics_contract
        simple_yaml.dump(profile, competition / "execution_profile.yaml")

        trial = root / "experiments" / "demo" / trial_id
        trial.mkdir(parents=True)
        (trial / "workspace_run.json").write_text(
            json.dumps({"competition": "demo", "trial_id": trial_id, "status": status}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
