import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.trial_artifacts import organize_trial_artifacts, trial_artifact_exists


class TrialArtifactsTest(unittest.TestCase):
    def test_organize_trial_artifacts_writes_readme_and_moves_debug_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.8, "metric": "accuracy", "objective": "maximize"}),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(
                json.dumps({"status": "collected", "score_source": "cv_score"}),
                encoding="utf-8",
            )
            (trial / "workspace_run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "command_results": [{"log_path": "experiments/demo/trial_001/workspace_logs/train.log"}],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["train.py"]}),
                encoding="utf-8",
            )
            (trial / "next_experiment.md").write_text("- title: Baseline\n", encoding="utf-8")
            (trial / "workspace_coding_api_request.json").write_text("{}", encoding="utf-8")
            (trial / "workspace_coding_api_response.json").write_text("{}", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = organize_trial_artifacts("demo", "trial_001")

            self.assertEqual("demo", result["competition"])
            self.assertEqual(0.8, result["local_score"])
            self.assertTrue((trial / "README.md").exists())
            self.assertTrue((trial / "internal" / "artifact_manifest.json").exists())
            self.assertFalse((trial / "workspace_coding_api_request.json").exists())
            self.assertFalse((trial / "workspace_coding_api_response.json").exists())
            self.assertTrue((trial / "debug" / "workspace_coding_api_request.json").exists())
            self.assertTrue((trial / "debug" / "workspace_coding_api_response.json").exists())
            self.assertFalse((trial / "workspace_run.json").exists())
            self.assertTrue((trial / "internal" / "workspace_run.json").exists())
            self.assertTrue((trial / "internal" / "pipeline_structure.json").exists())
            self.assertTrue(trial_artifact_exists(trial, "workspace_run.json"))
            self.assertTrue((trial / "user_view" / "README.ko.md").exists())
            self.assertTrue((trial / "user_view" / "01_plan.ko.md").exists())
            self.assertTrue((trial / "user_view" / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((trial / "user_view" / "03_code_pipeline.ko.md").exists())
            self.assertTrue((trial / "user_view" / "04_result.ko.md").exists())
            user_readme = (trial / "user_view" / "README.ko.md").read_text(encoding="utf-8")
            self.assertIn("사용자가 바로 확인할 만한 파일", user_readme)
            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            self.assertEqual("1.0", structure["schema_version"])
            self.assertTrue(any(stage["id"] == "data_split_cv" for stage in structure["stages"]))
            browse = root / "runs" / "demo" / "trial_001"
            self.assertTrue((browse / "README.ko.md").exists())
            self.assertTrue((browse / "01_plan.ko.md").exists())
            self.assertTrue((browse / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((browse / "03_code_pipeline.ko.md").exists())
            self.assertTrue((browse / "04_result.ko.md").exists())
            self.assertTrue((browse / "05_paths.ko.md").exists())
            self.assertTrue((root / "runs" / "demo" / "README.ko.md").exists())
            browse_paths = (browse / "05_paths.ko.md").read_text(encoding="utf-8")
            self.assertIn("실험 원본 기록", browse_paths)


if __name__ == "__main__":
    unittest.main()
