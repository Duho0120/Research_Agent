import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.code_writer_adapter import (
    build_anthropic_messages_payload,
    normalize_anthropic_messages_response,
    run_code_writer,
)


class FakeCodeWriterClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def create_response(self, payload: dict) -> dict:
        self.calls.append(payload)
        return self.response


class CodeWriterAdapterTest(unittest.TestCase):
    def test_anthropic_payload_conversion_and_response_normalization(self):
        payload = {
            "model": "claude-sonnet-5",
            "input": [
                {"role": "developer", "content": "Return only JSON."},
                {"role": "user", "content": "Write a pipeline."},
            ],
        }

        converted = build_anthropic_messages_payload(payload, max_tokens=2048)
        normalized = normalize_anthropic_messages_response(
            {
                "id": "msg_123",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "{\"status\":\"completed\"}"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            fallback_model="claude-sonnet-5",
        )

        self.assertEqual("claude-sonnet-5", converted["model"])
        self.assertEqual(2048, converted["max_tokens"])
        self.assertEqual("Return only JSON.", converted["system"])
        self.assertEqual([{"role": "user", "content": "Write a pipeline."}], converted["messages"])
        self.assertEqual("{\"status\":\"completed\"}", normalized["output_text"])
        self.assertEqual(15, normalized["usage"]["total_tokens"])

    def test_run_code_writer_applies_allowed_file_update_and_validates_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_handoff(trial)
            client = FakeCodeWriterClient(
                {
                    "id": "resp_test",
                    "usage": {
                        "input_tokens": 1200,
                        "output_tokens": 300,
                        "total_tokens": 1500,
                    },
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
                            "validation_results": [
                                {"command": "python -B -m unittest discover -s tests -v", "status": "not_run"}
                            ],
                            "blocking_issues": [],
                        }
                    ),
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual(result["status"], "accepted")
            self.assertEqual(len(client.calls), 1)
            self.assertIn("balanced", (trial / "config.yaml").read_text(encoding="utf-8"))
            self.assertTrue((trial / "coding_api_request.json").exists())
            self.assertTrue((trial / "coding_api_response.json").exists())
            self.assertTrue((trial / "coding_result_validation.json").exists())
            usage_log = root / "memory" / "demo" / "token_usage.jsonl"
            self.assertTrue(usage_log.exists())
            usage = json.loads(usage_log.read_text(encoding="utf-8").strip())
            self.assertEqual(usage["provider"], "openai_responses")
            self.assertEqual(usage["model"], "gpt-5")
            self.assertEqual(usage["call_type"], "code_writing")
            self.assertEqual(usage["input_tokens"], 1200)
            self.assertEqual(usage["output_tokens"], 300)
            self.assertEqual(usage["total_tokens"], 1500)

    def test_run_code_writer_blocks_when_token_policy_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_handoff(trial)
            client = FakeCodeWriterClient({"output_text": "{}"})

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_code_writer("demo", "trial_002", client=client, allow_api=True, trial_llm_calls=4)

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(client.calls, [])
            self.assertIn("token_policy_blocked", result["blocking_issues"])
            saved = json.loads((trial / "coding_result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "blocked")

    def test_run_code_writer_blocks_forbidden_file_update_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            forbidden = trial / "submission.csv"
            forbidden.write_text("id,target\n1,0\n", encoding="utf-8")
            self._write_handoff(trial)
            client = FakeCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Tried to edit submission.",
                            "changed_files": ["experiments/demo/trial_002/submission.csv"],
                            "file_updates": [
                                {
                                    "path": "experiments/demo/trial_002/submission.csv",
                                    "content": "id,target\n1,1\n",
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual(result["status"], "blocked")
            self.assertIn("file_update_not_allowed:experiments/demo/trial_002/submission.csv", result["blocking_issues"])
            self.assertEqual(forbidden.read_text(encoding="utf-8"), "id,target\n1,0\n")

    def _write_handoff(self, trial: Path) -> None:
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
                    "validation_commands": ["python -B -m unittest discover -s tests -v"],
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
