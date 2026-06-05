import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.post_validation_executor import run_after_validation


class PostValidationExecutorTest(unittest.TestCase):
    def test_run_after_validation_creates_local_job_when_validation_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_validation_run(trial, "passed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_after_validation("demo", "trial_002", command="python train.py")

            self.assertEqual(result["status"], "job_created")
            self.assertEqual(result["execution_decision"], "create_local_job")
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())
            self.assertTrue((trial / "post_validation_execution.json").exists())
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            last = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["decision_type"], "post_validation_execution")
            self.assertEqual(last["decision"], "job_created")

    def test_run_after_validation_runs_local_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_validation_run(trial, "passed")
            command = (
                "python -c \"from pathlib import Path; "
                "Path(r'experiments/demo/trial_002/metrics.json').write_text('{\\\"cv_score\\\": 0.7, \\\"objective\\\": \\\"maximize\\\"}', encoding='utf-8')\""
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_after_validation("demo", "trial_002", command=command, run_now=True)

            self.assertEqual(result["status"], "executed")
            self.assertEqual(result["execution_decision"], "run_local")
            self.assertTrue((trial / "metrics.json").exists())
            self.assertTrue((trial / "local_run.log").exists())

    def test_run_after_validation_blocks_when_validation_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_validation_run(trial, "failed")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_after_validation("demo", "trial_002", command="python train.py", run_now=True)

            self.assertEqual(result["status"], "blocked")
            self.assertIn("validation_run_not_passed", result["issues"])
            self.assertFalse((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def _write_validation_run(self, trial: Path, status: str) -> None:
        (trial / "validation_run.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": status,
                    "issues": [],
                    "commands": [],
                    "next_action": "run-trial" if status == "passed" else "revise-code-result",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
