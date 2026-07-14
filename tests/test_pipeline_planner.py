import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.pipeline_planner import plan_pipeline_improvement


class PipelinePlannerTest(unittest.TestCase):
    def test_prioritizes_validation_when_cv_lb_conflict_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n"
                "    lb_score: 0.79\n",
                encoding="utf-8",
            )
            config_dir = root / "configs" / "demo"
            config_dir.mkdir(parents=True)
            (config_dir / "research_policy.yaml").write_text(
                "leaderboard_tracking:\n  enabled: true\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.82, "lb_score": 0.78, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = plan_pipeline_improvement("demo", "trial_001")

            self.assertEqual(plan["primary_axis"], "validation")
            self.assertIn("model_family", plan["protected_axes"])
            self.assertIn("data split", " ".join(plan["candidate_actions"]))
            self.assertTrue((trial / "pipeline_improvement_plan.json").exists())
            self.assertTrue((trial / "pipeline_improvement_plan.md").exists())

    def test_prioritizes_data_and_human_review_for_concentrated_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.74,
                        "objective": "maximize",
                        "segment_errors": {"fold_3": 0.31, "view_C1": 0.25},
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = plan_pipeline_improvement("demo", "trial_001")

            self.assertEqual(plan["primary_axis"], "error_analysis")
            self.assertTrue(plan["requires_human_review"])
            self.assertIn("representative error cases", " ".join(plan["candidate_actions"]))

    def test_prioritizes_model_family_after_repeated_saturation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 3\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.71,
                        "objective": "maximize",
                        "prediction_correlation_with_best": 0.997,
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = plan_pipeline_improvement("demo", "trial_004")

            self.assertEqual(plan["primary_axis"], "model_family")
            self.assertIn("pretraining_strategy", plan["secondary_axes"])
            self.assertIn("model family", " ".join(plan["candidate_actions"]))

    def test_uses_latest_submission_score_when_metrics_have_no_lb_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.83, "objective": "maximize"}),
                encoding="utf-8",
            )
            submissions = root / "submissions" / "demo"
            submissions.mkdir(parents=True)
            (submissions / "submission_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "submitted_lb_score": 0.77511,
                        "submitted_rank": 1234,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = plan_pipeline_improvement("demo", "trial_001")

            self.assertEqual(plan["evidence_used"]["lb_score"], 0.77511)
            self.assertEqual(plan["evidence_used"]["rank"], 1234)
            self.assertEqual(plan["evidence_used"]["leaderboard_source"], "submission_log")


if __name__ == "__main__":
    unittest.main()
