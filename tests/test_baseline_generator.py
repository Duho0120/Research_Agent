import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.baseline_generator import generate_baseline_pipeline


class BaselineGeneratorTest(unittest.TestCase):
    def _write_titanic_data(self, root: Path) -> None:
        data = root / "data" / "titanic"
        data.mkdir(parents=True)
        (data / "train.csv").write_text(
            "PassengerId,Survived,Pclass,Sex,Age\n1,0,3,male,22\n2,1,1,female,38\n3,1,2,female,26\n",
            encoding="utf-8",
        )
        (data / "test.csv").write_text(
            "PassengerId,Pclass,Sex,Age\n4,3,male,35\n5,1,female,40\n",
            encoding="utf-8",
        )
        (data / "sample_submission.csv").write_text("PassengerId,Survived\n4,0\n5,0\n", encoding="utf-8")

    def test_generate_baseline_pipeline_creates_tabular_script_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_titanic_data(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = generate_baseline_pipeline("titanic", "trial_001")

            trial = root / "experiments" / "titanic" / "trial_001"
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["task_type"], "tabular")
            self.assertEqual(result["target_column"], "Survived")
            self.assertTrue((trial / "baseline_train.py").exists())
            self.assertTrue((trial / "baseline_plan.md").exists())
            self.assertTrue((trial / "baseline_pipeline.json").exists())
            self.assertIn("baseline_train.py", result["run_command"])

    def test_generated_tabular_baseline_runs_and_writes_metrics_and_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_titanic_data(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = generate_baseline_pipeline("titanic", "trial_001")

            completed = subprocess.run(
                result["run_command"],
                shell=True,
                cwd=root,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            trial = root / "experiments" / "titanic" / "trial_001"
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["model_type"], "majority_class_baseline")
            self.assertEqual(metrics["target_column"], "Survived")
            submission = (trial / "submission.csv").read_text(encoding="utf-8")
            self.assertIn("PassengerId,Survived", submission)
            self.assertIn("4,1", submission)

    def test_generate_baseline_pipeline_blocks_without_ready_data_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            competition = root / "competitions" / "titanic"
            competition.mkdir(parents=True)
            (competition / "competition_inspection.json").write_text(
                json.dumps({"competition_slug": "titanic", "status": "ready", "files": [{"name": "train.csv"}]}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = generate_baseline_pipeline("titanic", "trial_001")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("data_not_ready", result["blocking_issues"])
            self.assertFalse((root / "experiments" / "titanic" / "trial_001" / "baseline_train.py").exists())

    def test_generate_baseline_pipeline_uses_existing_data_profile_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "titanic"
            data.mkdir(parents=True)
            (data / "train.csv").write_text(
                "id,manual_target,auto_target,x\n1,1,0,10\n2,1,0,11\n",
                encoding="utf-8",
            )
            (data / "test.csv").write_text("id,x\n3,12\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("id,auto_target\n3,0\n", encoding="utf-8")
            competition = root / "competitions" / "titanic"
            competition.mkdir(parents=True)
            (competition / "data_profile.json").write_text(
                json.dumps(
                    {
                        "competition": "titanic",
                        "status": "ready",
                        "task_type": "tabular",
                        "target_candidates": ["manual_target"],
                        "id_candidates": ["id"],
                        "files": [],
                        "human_review_questions": [],
                        "next_steps": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = generate_baseline_pipeline("titanic", "trial_001")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["target_column"], "manual_target")


if __name__ == "__main__":
    unittest.main()
