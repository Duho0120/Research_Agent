import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from research_agent import simple_yaml
from research_agent.cli import main
from research_agent.workspace_after_coding import run_workspace_after_coding


class WorkspaceAfterCodingTest(unittest.TestCase):
    def test_blocks_when_workspace_coding_result_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_coding_validation(root, status="blocked")

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_workspace_after_coding("demo", "trial_002", run_now=True)

            self.assertEqual("blocked", result["status"])
            self.assertIn("workspace_coding_result_not_accepted", result["issues"])
            self.assertFalse((root / "experiments" / "demo" / "trial_002" / "workspace_run.json").exists())

    def test_dry_run_records_planned_execution_without_running_external_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            self._write_coding_validation(root, status="accepted")

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_workspace_after_coding("demo", "trial_002", run_now=False)

            self.assertEqual("ready_to_run", result["status"])
            self.assertEqual("planned", result["workspace_run"]["status"])
            self.assertFalse((project / "order.txt").exists())
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "workspace_after_coding_cycle.json").exists())

    def test_run_now_executes_collects_metrics_and_processes_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            self._write_coding_validation(root, status="accepted")

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_workspace_after_coding("demo", "trial_002", run_now=True)

            trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("completed", result["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertIn(result["workspace_result_cycle"]["status"], {"completed", "completed_review_deferred"})
            self.assertTrue((trial / "metrics.json").exists())
            self.assertTrue((trial / "evaluation.md").exists())
            self.assertTrue((trial / "diagnosis.md").exists())
            self.assertTrue((root / "memory" / "demo" / "trial_index.jsonl").exists())
            self.assertEqual("test\ntrain\npredict\n", (project / "order.txt").read_text(encoding="utf-8"))

    def test_run_workspace_after_coding_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            self._write_coding_validation(root, status="accepted")

            with patch("research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["run-workspace-after-coding", "--competition", "demo", "--trial", "trial_002", "--run-now"])

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "workspace_after_coding_cycle.json").exists())

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
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
            comp / "state.yaml",
        )

    def _write_profile(self, root: Path, project: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": sys.executable,
                "commands": {
                    "test": ["{python} test_step.py"],
                    "train": ["{python} train_step.py"],
                    "predict": ["{python} predict_step.py"],
                },
                "artifacts": {
                    "metrics": ["outputs/metrics.json"],
                    "submission": ["outputs/submission.csv"],
                },
                "write_scope": {
                    "allowed": ["src/", "tests/", "train_step.py"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            },
            comp / "execution_profile.yaml",
        )

    def _write_project_scripts(self, project: Path) -> None:
        (project / "outputs").mkdir(exist_ok=True)
        (project / "test_step.py").write_text(
            "from pathlib import Path\nPath('order.txt').write_text('test\\n')\n",
            encoding="utf-8",
        )
        (project / "train_step.py").write_text(
            "from pathlib import Path\n"
            "path = Path('order.txt')\n"
            "path.write_text(path.read_text() + 'train\\n')\n"
            "Path('outputs').mkdir(exist_ok=True)\n"
            "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.77, \"objective\": \"maximize\", \"metric\": \"accuracy\"}\\n')\n",
            encoding="utf-8",
        )
        (project / "predict_step.py").write_text(
            "from pathlib import Path\n"
            "path = Path('order.txt')\n"
            "path.write_text(path.read_text() + 'predict\\n')\n"
            "Path('outputs/submission.csv').write_text('id,target\\n1,0\\n')\n",
            encoding="utf-8",
        )

    def _write_coding_validation(self, root: Path, *, status: str) -> None:
        trial = root / "experiments" / "demo" / "trial_002"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "workspace_coding_result_validation.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": status,
                    "issues": [] if status == "accepted" else ["changed_file_not_allowed:data/train.csv"],
                    "coding_result_status": "completed" if status == "accepted" else "blocked",
                    "changed_files": ["train_step.py"] if status == "accepted" else ["data/train.csv"],
                    "next_action": "run-workspace-validation-commands"
                    if status == "accepted"
                    else "revise-workspace-code-result",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
