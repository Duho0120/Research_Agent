import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.result_analyst import diagnose_trial


class ResultAnalystTest(unittest.TestCase):
    def test_diagnosis_writes_markdown_and_flags_user_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.82,
                        "lb_score": 0.78,
                        "objective": "maximize",
                        "prediction_correlation_with_best": 0.997,
                        "segment_errors": {"fold_3": 0.31},
                        "notes": "CV up but LB down.",
                    }
                ),
                encoding="utf-8",
            )
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 2\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n"
                "    lb_score: 0.8\n",
                encoding="utf-8",
            )
            config_dir = root / "configs" / "demo"
            config_dir.mkdir(parents=True)
            (config_dir / "research_policy.yaml").write_text(
                "leaderboard_tracking:\n  enabled: true\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = diagnose_trial("demo", "trial_001")

            self.assertEqual(result["trial_id"], "trial_001")
            self.assertEqual(result["best_lb_before"], 0.8)
            self.assertFalse(result["cv_improved"])
            self.assertTrue(result["needs_user_review"])
            self.assertIn("CV/LB", " ".join(result["issues"]))
            self.assertIn("\uc0ac\uc6a9\uc790", " ".join(result["user_questions"]))
            diagnosis_path = trial / "diagnosis.md"
            self.assertTrue(diagnosis_path.exists())
            text = diagnosis_path.read_text(encoding="utf-8")
            self.assertIn("# trial_001 Diagnosis", text)
            self.assertIn("- best_lb_before: 0.8", text)
            self.assertIn("User Review", text)

    def test_no_cv_lb_conflict_when_best_lb_score_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.82,
                        "lb_score": 0.78,
                        "objective": "maximize",
                    }
                ),
                encoding="utf-8",
            )
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = diagnose_trial("demo", "trial_002")

            self.assertIsNone(result["best_lb_before"])
            self.assertTrue(result["cv_improved"])
            self.assertNotIn("CV/LB", " ".join(result["issues"]))


if __name__ == "__main__":
    unittest.main()


