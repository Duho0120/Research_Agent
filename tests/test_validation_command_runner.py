import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.validation_command_runner import run_validation_commands


class ValidationCommandRunnerTest(unittest.TestCase):
    def test_run_validation_commands_executes_handoff_commands_after_accepted_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            marker = trial / "validated.txt"
            self._write_handoff(
                trial,
                [
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            self._write_coding_validation(trial, "accepted")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_validation_commands("demo", "trial_002")

            self.assertEqual(result["status"], "passed")
            self.assertTrue(marker.exists())
            self.assertEqual(result["commands"][0]["status"], "passed")
            self.assertTrue((trial / "validation_run.json").exists())
            self.assertTrue((trial / "validation_run.md").exists())
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            last = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["decision_type"], "validation_commands")
            self.assertEqual(last["decision"], "passed")

    def test_run_validation_commands_blocks_when_coding_result_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial, ["python -c \"raise SystemExit(0)\""])
            self._write_coding_validation(trial, "blocked")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_validation_commands("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("coding_result_not_accepted", result["issues"])
            self.assertEqual(result["commands"], [])

    def test_run_validation_commands_records_failed_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial, ["python -c \"raise SystemExit(3)\""])
            self._write_coding_validation(trial, "accepted")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_validation_commands("demo", "trial_002")

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["commands"][0]["returncode"], 3)
            self.assertEqual(result["commands"][0]["status"], "failed")

    def _write_handoff(self, trial: Path, commands: list[str]) -> None:
        (trial / "coding_handoff.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": "ready",
                    "validation_commands": commands,
                    "allowed_write_files": ["experiments/demo/trial_002/config.yaml"],
                    "create_files": [],
                    "forbidden_paths": [],
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

    def _write_coding_validation(self, trial: Path, status: str) -> None:
        (trial / "coding_result_validation.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": status,
                    "issues": [],
                    "next_action": "run-validation-commands" if status == "accepted" else "revise-code-result",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
