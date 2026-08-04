import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent import simple_yaml
from research_agent.cli import main
from research_agent.workspace_preparer import prepare_workspace, refresh_workspace_inventory


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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
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
            with patch("research_agent.paths.project_root", return_value=root):
                result = prepare_workspace("topic_demo", topic="Forecast energy demand")

            self.assertEqual("needs_project_path", result["status"])
            self.assertTrue((root / "competitions" / "topic_demo" / "workspace_source.json").exists())
            self.assertFalse((root / "competitions" / "topic_demo" / "execution_profile.yaml").exists())

    def test_create_workspace_scaffolds_local_project_and_waits_for_manual_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            with patch("research_agent.paths.project_root", return_value=root):
                result = prepare_workspace(
                    "titanic_demo",
                    topic="Titanic survival prediction",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="Survived",
                    id_column="PassengerId",
                    required_data_files=["train.csv", "test.csv", "gender_submission.csv"],
                )

            source = root / "demo_workspaces" / "titanic_demo"
            self.assertEqual("needs_data", result["status"])
            self.assertEqual(["train.csv", "test.csv", "gender_submission.csv"], result["data_check"]["missing_files"])
            self.assertTrue((source / "data").is_dir())
            self.assertTrue((source / "src" / "baseline.py").exists())
            self.assertTrue((source / "train_step.py").exists())
            self.assertTrue((source / "predict_step.py").exists())
            profile = simple_yaml.load(root / "competitions" / "titanic_demo" / "execution_profile.yaml")
            self.assertEqual(str(source.resolve()), profile["project_root"])
            self.assertEqual(["{python} train_step.py"], profile["commands"]["train"])
            self.assertEqual(["outputs/metrics.json"], profile["artifacts"]["metrics"])
            self.assertIn("data/", profile["write_scope"]["forbidden"])

    def test_scaffolded_workspace_runs_after_user_adds_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()
            with patch("research_agent.paths.project_root", return_value=root):
                result = prepare_workspace(
                    "titanic_demo",
                    topic="Titanic survival prediction",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="Survived",
                    id_column="PassengerId",
                    required_data_files=["train.csv", "test.csv"],
                )
            source = Path(result["source_path"])
            (source / "data" / "train.csv").write_text(
                "PassengerId,Survived,Sex\n1,0,male\n2,1,female\n3,1,female\n4,0,male\n5,1,female\n",
                encoding="utf-8",
            )
            (source / "data" / "test.csv").write_text(
                "PassengerId,Sex\n6,male\n7,female\n",
                encoding="utf-8",
            )

            for command in [["test_step.py"], ["train_step.py"], ["predict_step.py"]]:
                completed = subprocess.run(
                    [sys.executable, *command],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            metrics = json.loads((source / "outputs" / "metrics.json").read_text(encoding="utf-8"))
            submission = (source / "outputs" / "submission.csv").read_text(encoding="utf-8")
            self.assertEqual("accuracy", metrics["metric"])
            self.assertIn("PassengerId,Survived", submission)

    def test_scaffolded_workspace_supports_regression_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()
            with patch("research_agent.paths.project_root", return_value=root):
                result = prepare_workspace(
                    "bike_demo",
                    topic="Bike sharing demand",
                    platform="kaggle",
                    metric="rmsle",
                    objective="minimize",
                    create_workspace=True,
                    target_column="count",
                    id_column="datetime",
                    required_data_files=["train.csv", "test.csv"],
                )
            source = Path(result["source_path"])
            (source / "data" / "train.csv").write_text(
                "datetime,count,temp\n"
                "2020-01-01 00:00:00,2,10\n"
                "2020-01-01 01:00:00,4,11\n"
                "2020-01-01 02:00:00,8,12\n"
                "2020-01-01 03:00:00,16,13\n"
                "2020-01-01 04:00:00,32,14\n",
                encoding="utf-8",
            )
            (source / "data" / "test.csv").write_text(
                "datetime,temp\n2020-01-02 00:00:00,15\n",
                encoding="utf-8",
            )

            for command in [["test_step.py"], ["train_step.py"], ["predict_step.py"]]:
                completed = subprocess.run(
                    [sys.executable, *command],
                    cwd=source,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(0, completed.returncode, completed.stderr)

            metrics = json.loads((source / "outputs" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("rmsle", metrics["metric"])
            self.assertEqual("mean_regression", metrics["strategy"])
            self.assertGreaterEqual(metrics["cv_score"], 0)

    def test_prepare_workspace_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            source = Path(tmp) / "source"
            root.mkdir()
            source.mkdir()
            with patch("research_agent.paths.project_root", return_value=root):
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

    def test_prepare_workspace_cli_can_create_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()
            with patch("research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "prepare-workspace",
                        "--competition",
                        "titanic_demo",
                        "--topic",
                        "Titanic survival prediction",
                        "--platform",
                        "kaggle",
                        "--metric",
                        "accuracy",
                        "--objective",
                        "maximize",
                        "--create-workspace",
                        "--target-column",
                        "Survived",
                        "--id-column",
                        "PassengerId",
                        "--required-data-file",
                        "train.csv",
                        "--required-data-file",
                        "test.csv",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "demo_workspaces" / "titanic_demo" / "workspace_config.json").exists())


class RefreshWorkspaceInventoryTest(unittest.TestCase):
    def test_refresh_picks_up_files_added_after_initial_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            source = Path(tmp) / "source"
            source.mkdir(parents=True)

            with patch("research_agent.paths.project_root", return_value=root):
                with patch("research_agent.workspace_preparer.paths.project_root", return_value=root):
                    prepare_workspace(
                        "demo",
                        source_path=str(source),
                        topic="Demo",
                        create_workspace=True,
                    )
                    inventory_path = root / "competitions" / "demo" / "workspace_inventory.json"
                    before = json.loads(inventory_path.read_text(encoding="utf-8"))
                    self.assertNotIn("train_labels.csv", [f["name"] for f in before["files"]])

                    # Data uploaded after registration (this is exactly what
                    # web_app.py's /api/upload-data endpoint writes into).
                    (source / "data" / "train_labels.csv").write_text("id,x\n1,0.5\n", encoding="utf-8")

                    refresh_workspace_inventory("demo", source)
                    after = json.loads(inventory_path.read_text(encoding="utf-8"))

        self.assertIn("train_labels.csv", [f["name"] for f in after["files"]])


if __name__ == "__main__":
    unittest.main()
