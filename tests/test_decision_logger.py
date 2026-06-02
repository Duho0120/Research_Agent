import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.memory import log_decision


class DecisionLoggerTest(unittest.TestCase):
    def test_log_decision_appends_competition_scoped_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                row = log_decision(
                    "demo",
                    "trial_010",
                    decision_type="diagnosis",
                    decision="request_user_review",
                    reason="CV improved but LB worsened.",
                    evidence={"cv_score": 0.82, "lb_score": 0.79},
                    user_input_used=False,
                    next_action="write_user_review_request",
                )

                path = root / "memory" / "demo" / "decision_log.jsonl"
                self.assertTrue(path.exists())
                saved = json.loads(path.read_text(encoding="utf-8").strip())
                self.assertEqual(saved["competition"], "demo")
                self.assertEqual(saved["trial_id"], "trial_010")
                self.assertEqual(saved["decision_type"], "diagnosis")
                self.assertEqual(saved["decision"], "request_user_review")
                self.assertEqual(saved["reason"], "CV improved but LB worsened.")
                self.assertEqual(saved["evidence"]["cv_score"], 0.82)
                self.assertFalse(saved["user_input_used"])
                self.assertEqual(saved["next_action"], "write_user_review_request")
                self.assertEqual(row["decision"], saved["decision"])


if __name__ == "__main__":
    unittest.main()


