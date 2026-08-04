import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.agents.coding_result_validator import (
    create_dry_run_coding_result,
    validate_coding_result,
)


class CodingResultValidatorTest(unittest.TestCase):
    def test_validate_coding_result_accepts_completed_result_within_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Updated trial config.",
                        "changed_files": ["experiments/demo/trial_002/config.yaml"],
                        "validation_results": [
                            {"command": "python -B -m unittest discover -s tests -v", "status": "passed"}
                        ],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = validate_coding_result("demo", "trial_002")

            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["issues"], [])
            self.assertTrue((trial / "coding_result_validation.json").exists())
            self.assertTrue((trial / "coding_result_validation.md").exists())
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            last = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["decision_type"], "coding_result_validation")
            self.assertEqual(last["decision"], "accepted")

    def test_validate_coding_result_accepts_validation_results_commands_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Updated trial config.",
                        "changed_files": ["experiments/demo/trial_002/config.yaml"],
                        "validation_results": {
                            "commands": [
                                {"command": "python -B -m unittest discover -s tests -v", "status": "not_run"}
                            ]
                        },
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = validate_coding_result("demo", "trial_002")

            self.assertEqual("accepted", result["status"])
            self.assertEqual([], result["issues"])

    def test_validate_coding_result_blocks_out_of_scope_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Changed too much.",
                        "changed_files": [
                            "experiments/demo/trial_002/config.yaml",
                            "experiments/demo/trial_002/submission.csv",
                        ],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = validate_coding_result("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("changed_file_not_allowed:experiments/demo/trial_002/submission.csv", result["issues"])
            self.assertIn("forbidden_path_touched:experiments/demo/trial_002/submission.csv", result["issues"])

    def test_validate_coding_result_blocks_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps({"status": "done", "changed_files": []}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = validate_coding_result("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("invalid_status:done", result["issues"])
            self.assertIn("missing_required_field:summary", result["issues"])
            self.assertIn("missing_required_field:validation_results", result["issues"])

    def test_validate_coding_result_blocks_out_of_scope_file_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Returned unsafe file update.",
                        "changed_files": ["experiments/demo/trial_002/config.yaml"],
                        "file_updates": [
                            {
                                "path": "experiments/demo/trial_002/metrics.json",
                                "content": "{}",
                            }
                        ],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = validate_coding_result("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("file_update_not_allowed:experiments/demo/trial_002/metrics.json", result["issues"])
            self.assertIn("forbidden_path_touched:experiments/demo/trial_002/metrics.json", result["issues"])

    def test_dry_run_coding_result_writes_blocked_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)

            with patch("research_agent.paths.project_root", return_value=root):
                result = create_dry_run_coding_result("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("dry_run_no_code_writer", result["blocking_issues"])
            self.assertTrue((trial / "coding_result.json").exists())
            self.assertTrue((trial / "coding_result.md").exists())

    def _write_handoff(self, trial: Path) -> None:
        (trial / "coding_handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "request_id": "demo:trial_002:coding",
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": "ready",
                    "allowed_write_files": ["experiments/demo/trial_002/config.yaml"],
                    "create_files": ["experiments/demo/trial_002/new_feature.py"],
                    "forbidden_paths": [
                        "data/",
                        "submissions/",
                        "experiments/demo/trial_002/submission.csv",
                        "experiments/demo/trial_002/metrics.json",
                    ],
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
