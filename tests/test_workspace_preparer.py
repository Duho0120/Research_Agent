import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_preparer import prepare_workspace


class WorkspacePreparerTest(unittest.TestCase):
    def test_prepare_workspace_detects_conventional_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            source = Path(tmp) / "source"
            root.mkdir()
            (source / "tests").mkdir(parents=True)
            (source / "outputs").mkdir()
            (source / "train.py").write_text("print('train')\n", encoding="utf-8")
            (source / "predict.py").write_text("print('predict')\n", encoding="utf-8")
            (source / "tests" / "test_train.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
            (source / "outputs" / "metrics.json").write_text("{}\n", encoding="utf-8")
            (source / "outputs" / "submission.csv").write_text("id,target\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace(
                    "demo",
                    source_path=str(source),
                    platform="external",
                    metric="score",
                    objective="maximize",
                )

            self.assertEqual("ready", result["status"])
            self.assertEqual([], result["review_questions"])
            profile = simple_yaml.load(root / "competitions" / "demo" / "execution_profile.yaml")
            self.assertEqual(str(source.resolve()), profile["project_root"])
            self.assertTrue(profile["commands"]["test"])
            self.assertTrue(profile["commands"]["train"])
            self.assertTrue(profile["commands"]["predict"])
            self.assertIn("train.py", profile["write_scope"]["allowed"])
            self.assertIn("outputs/metrics.json", profile["write_scope"]["forbidden"])

    def test_prepare_workspace_marks_unknown_entrypoints_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            source = Path(tmp) / "source"
            root.mkdir()
            source.mkdir()
            (source / "experiment.ipynb").write_text("{}\n", encoding="utf-8")
            (source / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace("demo", source_path=str(source), topic="Binary classification")

            self.assertEqual("needs_review", result["status"])
            self.assertIn("Confirm the test command.", result["review_questions"])
            self.assertIn("Confirm the train command.", result["review_questions"])
            self.assertIn("Confirm the metrics artifact path.", result["review_questions"])
            inventory = root / "competitions" / "demo" / "workspace_inventory.json"
            self.assertTrue(inventory.exists())

    def test_topic_only_creates_workspace_without_execution_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace("topic_demo", topic="Forecast energy demand")

            self.assertEqual("needs_project_path", result["status"])
            self.assertTrue((root / "competitions" / "topic_demo" / "workspace_source.json").exists())
            self.assertFalse((root / "competitions" / "topic_demo" / "execution_profile.yaml").exists())

    def test_prepare_workspace_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            source = Path(tmp) / "source"
            root.mkdir()
            source.mkdir()
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "prepare-workspace",
                        "--competition",
                        "demo",
                        "--source-path",
                        str(source),
                        "--topic",
                        "Demo research",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "competitions" / "demo" / "workspace_source.json").exists())


if __name__ == "__main__":
    unittest.main()
