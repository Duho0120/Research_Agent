import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.demo_one_cycle import run_demo_one_cycle


class DemoOneCycleTest(unittest.TestCase):
    def test_demo_one_cycle_runs_mock_plan_code_execution_and_records_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("completed", result["status"])
            self.assertEqual("ready", result["plan"]["status"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertEqual(0.88, result["record"]["local_score"])
            self.assertTrue((trial / "demo_context.md").exists())
            self.assertTrue((trial / "next_experiment.md").exists())
            self.assertTrue((trial / "workspace_coding_result.json").exists())
            self.assertTrue((trial / "workspace_run.json").exists())
            self.assertTrue((trial / "demo_cycle_record.json").exists())
            self.assertTrue((trial / "agent_status.json").exists())
            self.assertTrue((trial / "agent_events.jsonl").exists())
            status = json.loads((trial / "agent_status.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", status["status"])
            self.assertEqual("done", status["current_stage"])
            events = (trial / "agent_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "stage_started"' in line for line in events))
            self.assertTrue(any('"event": "cycle_completed"' in line for line in events))
            self.assertTrue((root / "memory" / "demo" / "demo_trial_index.jsonl").exists())
            usage_lines = (root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(usage_lines))
            usages = [json.loads(line) for line in usage_lines]
            self.assertEqual(["experiment_planning", "workspace_code_writing"], [row["call_type"] for row in usages])
            self.assertEqual({"anthropic_messages"}, {row["provider"] for row in usages})
            self.assertEqual({"claude-sonnet-5"}, {row["model"] for row in usages})
            self.assertEqual("gpt-5.6-luna", result["model_policy"]["low_cost"]["model"])
            self.assertIn("improved", (project / "train_step.py").read_text(encoding="utf-8"))

    def test_demo_one_cycle_blocks_without_plan_api_or_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001")

            self.assertEqual("blocked", result["status"])
            self.assertIn("api_call_not_enabled", result["issues"])

    def test_demo_one_cycle_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(
                        [
                            "demo-one-cycle",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--mock-plan-file",
                            str(plan_response),
                            "--mock-response-file",
                            str(code_response),
                            "--run-now",
                            "--show-progress",
                        ]
                    )

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "demo_one_cycle.json").exists())
            self.assertIn("[RUN] F-01", output.getvalue())
            self.assertIn("[OK] done", output.getvalue())

    def test_watch_demo_cycle_cli_prints_current_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["watch-demo-cycle", "--competition", "demo", "--trial", "trial_001"])

            text = output.getvalue()
            self.assertEqual(0, code)
            self.assertIn("Demo agent status: demo / trial_001", text)
            self.assertIn("status        : completed", text)
            self.assertIn("Recent events:", text)

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": "accuracy", "objective": "maximize", "platform": "external"},
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
        (root / "memory" / "demo").mkdir(parents=True, exist_ok=True)
        (root / "memory" / "demo" / "trial_index.jsonl").write_text("", encoding="utf-8")

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
        (project / "outputs").mkdir()
        (project / "src").mkdir()
        (project / "tests").mkdir()
        (project / "test_step.py").write_text("from pathlib import Path\nPath('order.txt').write_text('test\\n')\n", encoding="utf-8")
        (project / "train_step.py").write_text(
            "from pathlib import Path\n"
            "Path('outputs').mkdir(exist_ok=True)\n"
            "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.5, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n",
            encoding="utf-8",
        )
        (project / "predict_step.py").write_text(
            "from pathlib import Path\n"
            "path = Path('order.txt')\n"
            "path.write_text(path.read_text() + 'predict\\n')\n"
            "Path('outputs/submission.csv').write_text('id,target\\n1,0\\n')\n",
            encoding="utf-8",
        )

    def _write_plan_response(self, root: Path) -> Path:
        path = root / "mock_plan_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_plan",
                    "usage": {"input_tokens": 300, "output_tokens": 90, "total_tokens": 390},
                    "output_text": json.dumps(
                        {
                            "plan_title": "First local baseline",
                            "objective": "Write a small train script and verify local metrics.",
                            "rationale": "A minimal runnable baseline proves the loop before deeper optimization.",
                            "implementation_notes": ["Update train_step.py to emit metrics."],
                            "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_code_response(self, root: Path) -> Path:
        path = root / "mock_code_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_code",
                    "usage": {"input_tokens": 500, "output_tokens": 120, "total_tokens": 620},
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Wrote improved first-cycle train script.",
                            "changed_files": ["train_step.py"],
                            "file_updates": [
                                {
                                    "path": "train_step.py",
                                    "content": (
                                        "from pathlib import Path\n"
                                        "# improved demo baseline\n"
                                        "path = Path('order.txt')\n"
                                        "path.write_text(path.read_text() + 'train\\n')\n"
                                        "Path('outputs').mkdir(exist_ok=True)\n"
                                        "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.88, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n"
                                    ),
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
