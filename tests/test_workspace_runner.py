import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_runner import run_workspace_pipeline


class WorkspaceRunnerTest(unittest.TestCase):
    def test_dry_run_records_plan_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            marker = project / "executed.txt"
            self._write_profile(
                root,
                project,
                test_commands=[f'{{python}} -c "from pathlib import Path; Path(r\'{marker}\').write_text(\'ran\')"'],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_001", run_now=False)

            self.assertEqual("planned", result["status"])
            self.assertFalse(marker.exists())
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "workspace_run.json").exists())

    def test_run_now_executes_test_train_predict_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "test_step.py").write_text(
                "from pathlib import Path\nPath('order.txt').write_text('test\\n')\n",
                encoding="utf-8",
            )
            (project / "train_step.py").write_text(
                "from pathlib import Path\n"
                "path = Path('order.txt')\n"
                "path.write_text(path.read_text() + 'train\\n')\n"
                "Path('outputs').mkdir(exist_ok=True)\n"
                "Path('outputs/metrics.json').write_text('{}\\n')\n",
                encoding="utf-8",
            )
            (project / "predict_step.py").write_text(
                "from pathlib import Path\n"
                "path = Path('order.txt')\n"
                "path.write_text(path.read_text() + 'predict\\n')\n"
                "Path('outputs/submission.csv').write_text('id,target\\n')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                test_commands=["{python} test_step.py"],
                train_commands=["{python} train_step.py"],
                predict_commands=["{python} predict_step.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_001", run_now=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual("test\ntrain\npredict\n", (project / "order.txt").read_text())
            self.assertEqual(["test", "train", "predict"], [item["stage"] for item in result["command_results"]])
            self.assertTrue(result["artifacts"]["metrics"][0]["exists"])
            self.assertTrue(result["artifacts"]["submission"][0]["exists"])

    def test_failed_test_stops_later_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "fail.py").write_text(
                "from pathlib import Path\nPath('test_ran.txt').write_text('yes')\nraise SystemExit(3)\n",
                encoding="utf-8",
            )
            (project / "later.py").write_text(
                "from pathlib import Path\nPath('later_ran.txt').write_text('yes')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                test_commands=["{python} fail.py"],
                train_commands=["{python} later.py"],
                predict_commands=["{python} later.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_002", run_now=True)

            self.assertEqual("failed", result["status"])
            self.assertTrue((project / "test_ran.txt").exists())
            self.assertFalse((project / "later_ran.txt").exists())
            self.assertEqual(1, len(result["command_results"]))
            self.assertEqual(3, result["command_results"][0]["returncode"])

    def test_invalid_profile_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_profile(root, project)
            project.rmdir()

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_003", run_now=True)

            self.assertEqual("blocked", result["status"])
            self.assertEqual([], result["command_results"])
            self.assertIn("project_root_not_found", result["profile_issues"])

    def test_missing_profile_is_recorded_as_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_006", run_now=True)

            self.assertEqual("blocked", result["status"])
            self.assertEqual({}, result["commands"])
            self.assertIsNone(result["project_root"])
            self.assertTrue(result["profile_issues"])

    def test_missing_declared_artifacts_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_profile(root, project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_004", run_now=True)

            self.assertEqual("incomplete_artifacts", result["status"])
            self.assertEqual("fix-artifact-output", result["next_action"])
            self.assertFalse(result["artifacts"]["metrics"][0]["exists"])
            self.assertFalse(result["artifacts"]["submission"][0]["exists"])

    def test_non_finite_submission_is_rejected_before_submission_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "train.py").write_text(
                "from pathlib import Path\n"
                "Path('outputs').mkdir(exist_ok=True)\n"
                "Path('outputs/metrics.json').write_text('{}\\n')\n",
                encoding="utf-8",
            )
            (project / "predict.py").write_text(
                "from pathlib import Path\n"
                "Path('outputs/submission.csv').write_text('id,target\\n1,inf\\n')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                train_commands=["{python} train.py"],
                predict_commands=["{python} predict.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_007", run_now=True)

            self.assertEqual("invalid_artifacts", result["status"])
            self.assertEqual("fix-artifact-output", result["next_action"])
            self.assertIn("submission_non_finite_value", result["artifact_issues"][0])

    def test_stale_leftover_artifact_is_rejected_even_though_commands_succeeded(self):
        # Real incident: predict/train step wrote their real output under a
        # different file name than the declared artifact, leaving the
        # declared outputs/metrics.json and outputs/submission.csv untouched
        # from an earlier trial. Every command returned 0 and the declared
        # files existed and parsed fine, so nothing before this check caught
        # it -- the trial kept reporting that earlier trial's stale score as
        # if it were fresh, run after run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "outputs").mkdir()
            (project / "outputs" / "metrics.json").write_text('{"cv_score": 0.591}\n', encoding="utf-8")
            (project / "outputs" / "submission.csv").write_text("id,x,y,z\n1,0,0,0\n", encoding="utf-8")
            long_ago = time.time() - 3600
            os.utime(project / "outputs" / "metrics.json", (long_ago, long_ago))
            os.utime(project / "outputs" / "submission.csv", (long_ago, long_ago))
            (project / "train_step.py").write_text(
                # Writes real output under a different name, never touching
                # the declared outputs/metrics.json.
                "from pathlib import Path\nPath('outputs/metrics_local.json').write_text('{}\\n')\n",
                encoding="utf-8",
            )
            (project / "predict_step.py").write_text(
                "from pathlib import Path\nPath('outputs/submission_local.csv').write_text('id,x,y,z\\n')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                train_commands=["{python} train_step.py"],
                predict_commands=["{python} predict_step.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_008", run_now=True)

            self.assertEqual("invalid_artifacts", result["status"])
            self.assertEqual("fix-artifact-output", result["next_action"])
            self.assertTrue(
                any("stale_artifact_not_regenerated_this_run:metrics.json" in issue for issue in result["artifact_issues"])
            )
            self.assertTrue(
                any(
                    "stale_artifact_not_regenerated_this_run:submission.csv" in issue
                    for issue in result["artifact_issues"]
                )
            )

    def test_freshly_regenerated_artifact_is_not_flagged_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "train_step.py").write_text(
                "from pathlib import Path\n"
                "Path('outputs').mkdir(exist_ok=True)\n"
                "Path('outputs/metrics.json').write_text('{}\\n')\n",
                encoding="utf-8",
            )
            (project / "predict_step.py").write_text(
                "from pathlib import Path\nPath('outputs/submission.csv').write_text('id,target\\n')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                train_commands=["{python} train_step.py"],
                predict_commands=["{python} predict_step.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_pipeline("demo", "trial_009", run_now=True)

            self.assertEqual("completed", result["status"])

    def test_cli_runs_workspace_pipeline_with_explicit_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            (project / "test_step.py").write_text("print('ok')\n", encoding="utf-8")
            (project / "train_step.py").write_text(
                "from pathlib import Path\n"
                "Path('outputs').mkdir(exist_ok=True)\n"
                "Path('outputs/metrics.json').write_text('{}\\n')\n",
                encoding="utf-8",
            )
            (project / "predict_step.py").write_text(
                "from pathlib import Path\nPath('outputs/submission.csv').write_text('id,target\\n')\n",
                encoding="utf-8",
            )
            self._write_profile(
                root,
                project,
                test_commands=["{python} test_step.py"],
                train_commands=["{python} train_step.py"],
                predict_commands=["{python} predict_step.py"],
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "run-workspace-pipeline",
                        "--competition",
                        "demo",
                        "--trial",
                        "trial_005",
                        "--run-now",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_005" / "workspace_run.json").exists())

    def _write_profile(
        self,
        root: Path,
        project: Path,
        *,
        test_commands: list[str] | None = None,
        train_commands: list[str] | None = None,
        predict_commands: list[str] | None = None,
    ) -> None:
        destination = root / "competitions" / "demo"
        destination.mkdir(parents=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": sys.executable,
                "commands": {
                    "test": test_commands or ["{python} -c \"print('test')\""],
                    "train": train_commands or ["{python} -c \"print('train')\""],
                    "predict": predict_commands or ["{python} -c \"print('predict')\""],
                },
                "artifacts": {
                    "metrics": ["outputs/metrics.json"],
                    "submission": ["outputs/submission.csv"],
                },
                "write_scope": {
                    "allowed": ["src/", "tests/"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            },
            destination / "execution_profile.yaml",
        )


if __name__ == "__main__":
    unittest.main()
