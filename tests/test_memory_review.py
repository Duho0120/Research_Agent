import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.agents.memory import log_token_usage, normalize_token_usage, record_user_feedback, request_user_review


class MemoryReviewTest(unittest.TestCase):
    def test_request_user_review_writes_request_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_001").mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_001",
                "issues": ["CV/LB movement may be inconsistent."],
                "user_questions": [
                    "\uc0ac\uc6a9\uc790\uc5d0\uac8c validation split \uc758\uacac\uc744 \uc694\uccad\ud569\ub2c8\ub2e4."
                ],
                "improvement_candidates": ["Try a model-family change."],
            }

            with patch("research_agent.paths.project_root", return_value=root):
                path = request_user_review("demo", "trial_001", diagnosis)

            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("# trial_001 User Review Request", text)
            self.assertIn("CV/LB", text)

    def test_record_user_feedback_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                row = record_user_feedback(
                    "demo",
                    "trial_001",
                    topic="validation",
                    question="Is the split appropriate?",
                    user_feedback="Group split looks safer.",
                    decision="change_validation",
                    follow_up_action="plan validation review trial",
                )

            path = root / "memory" / "demo" / "user_feedback.jsonl"
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["topic"], "validation")
            self.assertEqual(saved["decision"], "change_validation")
            self.assertEqual(row["follow_up_action"], "plan validation review trial")

    def test_record_user_feedback_updates_review_pack_and_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "experiments" / "demo" / "trial_001" / "review_pack"
            pack.mkdir(parents=True)
            (pack / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "competition": "demo",
                        "trial_id": "trial_001",
                        "status": "pending_user_feedback",
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                record_user_feedback(
                    "demo",
                    "trial_001",
                    topic="validation",
                    question="Is the split appropriate?",
                    user_feedback="Use group split before large model changes.",
                    decision="change_validation",
                    follow_up_action="plan validation review trial",
                )

            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            feedback = json.loads((pack / "human_feedback.json").read_text(encoding="utf-8"))
            decision = json.loads((root / "memory" / "demo" / "decision_log.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(manifest["status"], "feedback_recorded")
            self.assertEqual(feedback["overall_decision"], "change_validation")
            self.assertEqual(decision["decision_type"], "human_feedback")
            self.assertTrue(decision["user_input_used"])

    def test_log_token_usage_appends_normalized_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                row = log_token_usage(
                    "demo",
                    "trial_001",
                    provider="openai_responses",
                    model="gpt-5",
                    call_type="code_writing",
                    usage={"prompt_tokens": "10", "completion_tokens": 5},
                    request_id="resp_001",
                )

            path = root / "memory" / "demo" / "token_usage.jsonl"
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["input_tokens"], 10)
            self.assertEqual(saved["output_tokens"], 5)
            self.assertEqual(saved["total_tokens"], 15)
            self.assertEqual(saved["request_id"], "resp_001")

    def test_normalize_token_usage_preserves_total_when_present(self):
        normalized = normalize_token_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 99})

        self.assertEqual(normalized["input_tokens"], 10)
        self.assertEqual(normalized["output_tokens"], 5)
        self.assertEqual(normalized["total_tokens"], 99)


if __name__ == "__main__":
    unittest.main()


