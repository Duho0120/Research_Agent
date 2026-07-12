import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.execution_profile import validate_execution_profile


class ExecutionProfileTest(unittest.TestCase):
    def test_valid_profile_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "external_project"
            project.mkdir()
            python_path = project / "python.exe"
            python_path.write_text("", encoding="utf-8")
            self._write_profile(root, project, python_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_execution_profile("demo")

            self.assertEqual("ready", result["status"])
            self.assertEqual([], result["issues"])
            self.assertTrue((root / "competitions" / "demo" / "execution_profile_validation.json").exists())

    def test_missing_project_and_python_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_profile(root, root / "missing_project", root / "missing_python.exe")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_execution_profile("demo")

            self.assertEqual("blocked", result["status"])
            self.assertIn("project_root_not_found", result["issues"])
            self.assertIn("python_not_found", result["issues"])

    def test_protected_artifacts_cannot_be_in_allowed_write_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "external_project"
            project.mkdir()
            python_path = project / "python.exe"
            python_path.write_text("", encoding="utf-8")
            self._write_profile(
                root,
                project,
                python_path,
                allowed=["src/train.py", "data/train.csv", "outputs/submission.csv"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_execution_profile("demo")

            self.assertEqual("blocked", result["status"])
            self.assertIn("allowed_path_is_forbidden:data/train.csv", result["issues"])
            self.assertIn("allowed_path_is_artifact:outputs/submission.csv", result["issues"])

    def test_cli_validates_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "external_project"
            project.mkdir()
            python_path = project / "python.exe"
            python_path.write_text("", encoding="utf-8")
            self._write_profile(root, project, python_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(["validate-execution-profile", "--competition", "demo"])

            self.assertEqual(0, code)

    def test_invalid_metrics_contract_source_key_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "external_project"
            project.mkdir()
            python_path = project / "python.exe"
            python_path.write_text("", encoding="utf-8")
            self._write_profile(
                root,
                project,
                python_path,
                metrics_contract={"source_key": "validation..score"},
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_execution_profile("demo")

            self.assertEqual("blocked", result["status"])
            self.assertIn("invalid_metrics_contract:source_key", result["issues"])

    def _write_profile(
        self,
        root: Path,
        project: Path,
        python_path: Path,
        *,
        allowed: list[str] | None = None,
        metrics_contract: dict | None = None,
    ) -> None:
        competition_dir = root / "competitions" / "demo"
        competition_dir.mkdir(parents=True)
        profile = {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": str(python_path),
                "commands": {
                    "test": ["{python} -m pytest tests -q"],
                    "train": ["{python} train.py"],
                    "predict": ["{python} predict.py"],
                },
                "artifacts": {
                    "metrics": ["outputs/metrics.json"],
                    "submission": ["outputs/submission.csv"],
                },
                "write_scope": {
                    "allowed": allowed or ["src/train.py", "src/features.py", "tests/"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            }
        if metrics_contract is not None:
            profile["metrics_contract"] = metrics_contract
        simple_yaml.dump(profile, competition_dir / "execution_profile.yaml")


if __name__ == "__main__":
    unittest.main()
