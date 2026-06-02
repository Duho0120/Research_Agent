import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.review_pack import prepare_review_pack


class ReviewPackTest(unittest.TestCase):
    def test_prepare_review_pack_writes_schema_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_001",
                "objective": "maximize",
                "cv_score": 0.8,
                "lb_score": None,
                "best_cv_before": 0.79,
                "issues": ["Errors are concentrated in one segment."],
                "user_questions": ["Is this segment meaningful?"],
                "needs_user_review": True,
                "strategy_recommendation": "continue_refinement",
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_review_pack("demo", "trial_001", diagnosis)

            pack = trial / "review_pack"
            self.assertEqual(result["case_count"], 1)
            self.assertTrue((pack / "manifest.json").exists())
            self.assertTrue((pack / "summary.ko.md").exists())
            self.assertTrue((pack / "questions.ko.md").exists())
            self.assertTrue((pack / "cases.jsonl").exists())
            self.assertTrue((pack / "metrics_snapshot.json").exists())
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["status"], "pending_user_feedback")


if __name__ == "__main__":
    unittest.main()
