import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.agents.research_planner import propose_next_experiment


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

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_001", "trial_002")

            self.assertEqual(plan["strategy"], "controlled_refinement")
            self.assertEqual(plan["next_trial_id"], "trial_002")
            self.assertIn("Keep validation unchanged", " ".join(plan["guardrails"]))
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "next_experiment.md").exists())

    def test_escalates_to_model_family_change_after_repeated_failures_not_sota(self):
        # Repeated failures must escalate to a concrete, bounded axis change
        # (model_family_change) -- never to an undirected "SOTA" strategy switch
        # picked without the user's explicit request/approval.
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

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_004", "trial_005")

            self.assertEqual(plan["strategy"], "model_family_change")
            self.assertIn("model family", " ".join(plan["changes"]))
            self.assertTrue(plan["requires_user_review_before_submit"])
            text = (root / "experiments" / "demo" / "trial_005" / "next_experiment.md").read_text(encoding="utf-8")
            self.assertNotIn("SOTA", text)

    def test_continues_active_axis_instead_of_escalating_before_it_is_exhausted(self):
        # An axis that still has attempts left (axis_attempt_count < axis_attempt_limit)
        # must keep being pursued via strategy_for_axis, even if the global
        # consecutive_failures counter has climbed past 3 for unrelated reasons
        # (e.g. a different axis failed earlier). The per-axis budget takes
        # priority over the axis-agnostic failure count.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: minimize\ncurrent_state:\n  consecutive_failures: 3\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_006"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.45, "lb_score": 0.41, "objective": "minimize"}),
                encoding="utf-8",
            )
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "decision_cards.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_006",
                        "objective": "minimize",
                        "decision": "reject_or_hold_cv_lb_mismatch",
                        "change_axis": "validation_review",
                        "active_axis": "validation_review",
                        "axis_attempt_count": 1,
                        "axis_attempt_limit": 3,
                        "recommended_base_trial": "trial_003",
                        "lb_score": 0.41,
                        "local_score": 0.45,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_006", "trial_007")

            self.assertEqual("validation_review", plan["strategy"])

    def test_prepare_sota_research_feedback_still_reaches_sota_strategy(self):
        # SOTA exploration must remain reachable, but only via an explicit user
        # decision -- never auto-selected by the rule-based planner on its own.
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
            (mem / "user_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "topic": "strategy",
                        "user_feedback": "Go ahead and try a SOTA-style architecture.",
                        "decision": "prepare_sota_research",
                        "follow_up_action": "plan sota trial",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_001", "trial_002")

            self.assertEqual("sota_architecture_attempt", plan["strategy"])

    def test_ignores_user_feedback_already_consumed_by_an_earlier_insight_cycle(self):
        # user_feedback.jsonl rows are keyed by the trial_id they were originally
        # attached to, and that same trial_id can remain the "source_trial_id" for
        # more than one later planning round if the insight axis gets continued
        # (see the active-axis continuation above). Once the referenced insight
        # has finished its lifecycle (superseded here), it must not be replanned
        # from the same raw feedback text again for a later trial.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "competitions" / "demo"
            state_dir.mkdir(parents=True)
            (state_dir / "state.yaml").write_text(
                "competition:\n  objective: minimize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_008"
            trial.mkdir(parents=True)
            mem = root / "memory" / "demo"
            mem.mkdir(parents=True)
            (mem / "user_feedback.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_008",
                        "topic": "user_insight",
                        "scope": "next_trial",
                        "user_feedback": "Use a pseudo-test validation split.",
                        "decision": "user_insight_for_planner",
                        "insight_id": "insight_stale_0001",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (mem / "user_insights.jsonl").write_text(
                json.dumps(
                    {
                        "insight_id": "insight_stale_0001",
                        "status": "superseded",
                        "axis": "validation_review",
                        "target_trial": "trial_006",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_008", "trial_009")

            self.assertNotEqual("validation_review", plan["strategy"])
            self.assertIsNone(plan["evidence_used"]["latest_user_feedback"])

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

            with patch("research_agent.paths.project_root", return_value=root):
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

            with patch("research_agent.paths.project_root", return_value=root):
                plan = propose_next_experiment("demo", "trial_005", "trial_006")

            self.assertEqual("model_ensemble", plan["strategy"])
            self.assertIn("ensemble", " ".join(plan["changes"]).casefold())
            self.assertIn("ensemble", plan["rationale"].casefold())


if __name__ == "__main__":
    unittest.main()


