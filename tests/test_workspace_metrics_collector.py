import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent import simple_yaml
from research_agent.cli import main
from research_agent.workspace_metrics_collector import collect_workspace_metrics


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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_002")

            self.assertEqual("collected", result["status"])
            self.assertEqual("validation.macro_f1", result["score_source"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_002" / "metrics.json").read_text())
            self.assertEqual(0.77, metrics["cv_score"])
            self.assertEqual("accuracy", metrics["metric"])
            self.assertEqual(5, metrics["epoch"])

    def test_falls_back_past_a_stale_configured_source_key(self):
        # metrics_contract.source_key is set from a previous trial's metrics
        # shape. A later trial's code writer can reasonably name its score
        # field differently, so a missing configured key must not fail
        # outright -- it should fall through to metric-name-based detection
        # instead of forcing another config edit every time a new trial uses
        # a different (but still reasonable) key.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"metric": "accuracy", "value": 0.83, "epoch": 5}),
                encoding="utf-8",
            )
            self._write_workspace(
                root,
                project,
                "trial_003",
                status="completed",
                metrics_contract={"source_key": "cv_mean_accuracy"},
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_003")

            self.assertEqual("collected", result["status"])
            self.assertEqual("value", result["score_source"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_003" / "metrics.json").read_text())
            self.assertEqual(0.83, metrics["cv_score"])

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

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_validation_metric")

            self.assertEqual("collected", result["status"])
            self.assertEqual("validation_accuracy", result["score_source"])
            self.assertEqual(0.79, result["cv_score"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_validation_metric" / "metrics.json").read_text())
            self.assertEqual(0.79, metrics["cv_score"])
            self.assertEqual(712, metrics["training_rows"])

    def test_collects_direct_metric_key_from_competition_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"accuracy": 0.91, "metric": "accuracy", "objective": "maximize"}),
                encoding="utf-8",
            )
            self._write_workspace(root, project, "trial_direct_metric", status="completed")

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_direct_metric")

            self.assertEqual("collected", result["status"])
            self.assertEqual("accuracy", result["score_source"])
            self.assertEqual(0.91, result["cv_score"])
            metrics = json.loads((root / "experiments" / "demo" / "trial_direct_metric" / "metrics.json").read_text())
            self.assertEqual(0.91, metrics["cv_score"])
            self.assertEqual(0.91, metrics["accuracy"])

    def test_collects_value_when_reported_metric_matches_competition_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"metric": "accuracy", "value": 0.88, "split": "validation"}),
                encoding="utf-8",
            )
            self._write_workspace(root, project, "trial_metric_value", status="completed")

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_metric_value")

            self.assertEqual("collected", result["status"])
            self.assertEqual("value", result["score_source"])
            self.assertEqual(0.88, result["cv_score"])

    def test_does_not_collect_value_when_reported_metric_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps({"metric": "loss", "value": 0.12}),
                encoding="utf-8",
            )
            self._write_workspace(root, project, "trial_mismatched_metric_value", status="completed")

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_mismatched_metric_value")

            self.assertEqual("needs_review", result["status"])
            self.assertIsNone(result["cv_score"])

    def test_missing_score_mapping_requests_review_without_writing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"score": 0.91}), encoding="utf-8")
            self._write_workspace(root, project, "trial_003", status="completed")

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
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

    def test_matches_metric_key_despite_case_and_punctuation_differences(self):
        # The code writer names the score key after the metric using whatever
        # spelling it picks that run. "R-Hit@1cm" in state.yaml previously
        # only matched a literal "r-hit@1cm" key, so a metrics.json written
        # as "R-Hit@1cm" was reported as missing_or_invalid_primary_score and
        # required a hand-configured metrics_contract.source_key.
        for written_key in ("R-Hit@1cm", "r_hit_at_1cm", "val_RHit1cm"):
            with self.subTest(written_key=written_key):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "agent"
                    project = Path(tmp) / "project"
                    root.mkdir()
                    project.mkdir()
                    source = project / "outputs" / "metrics.json"
                    source.parent.mkdir()
                    source.write_text(
                        json.dumps({written_key: 0.591, "n_ids": 1000, "model_type": "rule"}),
                        encoding="utf-8",
                    )
                    self._write_workspace(root, project, "trial_001", status="completed", metric="R-Hit@1cm")

                    with patch("research_agent.paths.project_root", return_value=root):
                        result = collect_workspace_metrics("demo", "trial_001")

                    self.assertEqual("collected", result["status"])
                    self.assertEqual(0.591, result["cv_score"])
                    self.assertEqual(written_key, result["score_source"])

    def test_ignores_non_numeric_keys_when_matching_metric_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            # "metric" holds the metric's *name*, not its value -- matching on
            # it would otherwise pick up a string and crash the collector.
            source.write_text(
                json.dumps({"metric": "R-Hit@1cm", "R-Hit@1cm": 0.42}),
                encoding="utf-8",
            )
            self._write_workspace(root, project, "trial_001", status="completed", metric="R-Hit@1cm")

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_001")

            self.assertEqual("collected", result["status"])
            self.assertEqual(0.42, result["cv_score"])

    def test_stale_source_key_falls_back_to_normalized_metric_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            source = project / "outputs" / "metrics.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"R-Hit@1cm": 0.7}), encoding="utf-8")
            self._write_workspace(
                root,
                project,
                "trial_001",
                status="completed",
                metric="R-Hit@1cm",
                metrics_contract={"source_key": "r_hit_at_1cm"},
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = collect_workspace_metrics("demo", "trial_001")

            self.assertEqual("collected", result["status"])
            self.assertEqual(0.7, result["cv_score"])
            self.assertEqual("R-Hit@1cm", result["score_source"])

    def _write_workspace(
        self,
        root: Path,
        project: Path,
        trial_id: str,
        *,
        status: str,
        metrics_contract: dict | None = None,
        metric: str = "accuracy",
    ) -> None:
        competition = root / "competitions" / "demo"
        competition.mkdir(parents=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": metric, "objective": "maximize"},
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
