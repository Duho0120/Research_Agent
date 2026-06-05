import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.safe_execution_chain import run_safe_execution_chain


class FakeCodeWriterClient:
    def __init__(self, response: dict):
        self.response = response

    def create_response(self, payload: dict) -> dict:
        return self.response


class SafeExecutionChainTest(unittest.TestCase):
    def test_safe_execution_chain_runs_to_local_job_after_validation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_handoff(
                trial,
                [
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            client = FakeCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated config safely.",
                            "changed_files": ["experiments/demo/trial_002/config.yaml"],
                            "file_updates": [
                                {
                                    "path": "experiments/demo/trial_002/config.yaml",
                                    "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_safe_execution_chain("demo", "trial_002", client=client, command="python train.py")

            self.assertEqual(result["status"], "job_created")
            self.assertEqual(result["code_writer_status"], "accepted")
            self.assertEqual(result["validation_status"], "passed")
            self.assertEqual(result["execution_status"], "job_created")
            self.assertTrue((trial / "coding_result_validation.json").exists())
            self.assertTrue((trial / "validation_run.json").exists())
            self.assertTrue((trial / "post_validation_execution.json").exists())
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def test_safe_execution_chain_stops_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_handoff(trial, ["python -c \"raise SystemExit(2)\""])
            client = FakeCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated config safely.",
                            "changed_files": ["experiments/demo/trial_002/config.yaml"],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_safe_execution_chain("demo", "trial_002", client=client, command="python train.py")

            self.assertEqual(result["status"], "validation_failed")
            self.assertEqual(result["execution_status"], None)
            self.assertFalse((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def _write_handoff(self, trial: Path, validation_commands: list[str]) -> None:
        (trial / "coding_handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "request_id": "demo:trial_002:coding",
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": "ready",
                    "objective": "Implement the validated code patch plan without expanding scope.",
                    "context_files": ["experiments/demo/trial_002/config.yaml"],
                    "allowed_write_files": ["experiments/demo/trial_002/config.yaml"],
                    "create_files": [],
                    "forbidden_paths": [
                        "data/",
                        "submissions/",
                        "experiments/demo/trial_002/submission.csv",
                        "experiments/demo/trial_002/metrics.json",
                    ],
                    "implementation_steps": ["Add balanced sampler config."],
                    "validation_commands": validation_commands,
                    "required_output": {
                        "required_fields": [
                            "status",
                            "summary",
                            "changed_files",
                            "validation_results",
                            "blocking_issues",
                        ],
                        "status_values": ["completed", "blocked", "failed"],
                    },
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
