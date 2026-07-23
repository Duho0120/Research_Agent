import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.research_planner import propose_next_experiment


class ResearchPlannerNextExperimentTest(unittest.TestCase):
    def test_propose_refinement_when_diagnosis_allows_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "diagnosis.md").write_text(
                "# trial_001 Diagnosis\n\n## Strategy Recommendation\n\ncontinue_refinement\n",
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.71, "lb_score": 0.7, "objective": "maximize"}),
                encoding="utf-8",
            )
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "decision_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "decision_type": "diagnosis",
                        "evidence": {
                            "strategy_recommendation": "continue_refinement",
                            "issues": [],
                            "cv_improved": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_001", "trial_002")

            self.assertEqual(plan["strategy"], "controlled_refinement")
            self.assertEqual(plan["next_trial_id"], "trial_002")
            self.assertIn("Keep validation unchanged", " ".join(plan["guardrails"]))
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "next_experiment.md").exists())

    def test_escalates_to_sota_attempt_after_repeated_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 3\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.69, "lb_score": 0.68, "objective": "maximize"}),
                encoding="utf-8",
            )
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "decision_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_004",
                        "decision_type": "diagnosis",
                        "evidence": {
                            "strategy_recommendation": "strategy_escalation",
                            "issues": ["Recent failures suggest strategy escalation is needed."],
                            "cv_improved": False,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "submissions" / "demo").mkdir(parents=True)
            (root / "submissions" / "demo" / "submission_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_004",
                        "version_name": "demo_trial_004_v01",
                        "score_delta": -0.01,
                        "rank_delta": -12,
                        "is_best": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_004", "trial_005")

            self.assertEqual(plan["strategy"], "sota_architecture_attempt")
            self.assertIn("model-family", " ".join(plan["changes"]))
            self.assertTrue(plan["requires_user_review_before_submit"])
            text = (root / "experiments" / "demo" / "trial_005" / "next_experiment.md").read_text(encoding="utf-8")
            self.assertIn("SOTA", text)

    def test_propose_next_experiment_uses_recent_user_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "decision_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "decision_type": "diagnosis",
                        "evidence": {
                            "strategy_recommendation": "continue_refinement",
                            "issues": [],
                            "cv_improved": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (mem / "user_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "topic": "validation",
                        "question": "Is the split appropriate?",
                        "user_feedback": "Use group split before large model changes.",
                        "decision": "change_validation",
                        "follow_up_action": "plan validation review trial",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_001", "trial_002")

            self.assertEqual(plan["strategy"], "validation_review")
            self.assertTrue(plan["evidence_used"]["latest_user_feedback"])
            self.assertIn("User feedback", plan["rationale"])

    def test_propose_next_experiment_maps_ensemble_user_insight_to_ensemble_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_005"
            trial.mkdir(parents=True)
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "user_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "topic": "user_insight",
                        "scope": "next_trial",
                        "user_feedback": "2~3개의 서로 다른 모델을 앙상블하자",
                        "decision": "user_insight_for_planner",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_005", "trial_006")

            self.assertEqual("model_ensemble", plan["strategy"])
            self.assertIn("ensemble", " ".join(plan["changes"]).casefold())
            self.assertIn("ensemble", plan["rationale"].casefold())


if __name__ == "__main__":
    unittest.main()


