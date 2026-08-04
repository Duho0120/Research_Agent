import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.feedback_policy import (
    create_feedback_request,
    evaluate_feedback_policy,
)
from research_agent.state_db import list_pending_actions, list_review_evidence


class FeedbackPolicyTest(unittest.TestCase):
    def test_structured_issue_codes_trigger_without_english_message_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = evaluate_feedback_policy(
                "demo",
                "trial_004",
                {
                    "issues": [{"code": "validation_mismatch", "message": "검증 결과를 신뢰하기 어렵습니다."}],
                    "axis_attempt_count": 3,
                },
                db_path=Path(tmp) / "state.sqlite3",
            )

        self.assertIn("cv_lb_conflict", decision["triggers"])

    def test_one_off_nonurgent_evidence_is_accumulated_without_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite3"
            diagnosis = self._diagnosis("trial_001", axis="error_analysis", segment=True)

            decision = evaluate_feedback_policy(
                "demo",
                "trial_001",
                diagnosis,
                db_path=db_path,
            )

            self.assertEqual("accumulate", decision["action"])
            self.assertEqual("defer", decision["timing"])
            self.assertEqual([], list_pending_actions("demo", db_path))
            self.assertEqual(1, len(list_review_evidence("demo", db_path=db_path)))

    def test_repeated_high_value_evidence_creates_one_structured_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite3"
            decision = {}
            for index in range(1, 4):
                trial_id = f"trial_{index:03d}"
                diagnosis = self._diagnosis(trial_id, axis="error_analysis", segment=True)
                decision = evaluate_feedback_policy(
                    "demo",
                    trial_id,
                    diagnosis,
                    db_path=db_path,
                )

            self.assertEqual("request_now", decision["action"])
            self.assertGreaterEqual(decision["score"], decision["score_threshold"])
            with patch("research_agent.paths.project_root", return_value=root):
                action = create_feedback_request(
                    "demo",
                    "trial_003",
                    diagnosis,
                    decision,
                    db_path=db_path,
                )
            pending = list_pending_actions("demo", db_path)
            self.assertEqual([action["action_id"]], [item["action_id"] for item in pending])
            self.assertEqual("use_default", pending[0]["payload"]["options"][-1]["value"])
            self.assertIn("최근 3회", pending[0]["payload"]["evidence_summary"][0])

    def test_leakage_requests_immediately_without_trial_count_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            diagnosis = self._diagnosis("trial_001", axis="validation")
            diagnosis["issues"] = ["Leakage warning is present in metrics."]

            decision = evaluate_feedback_policy(
                "demo",
                "trial_001",
                diagnosis,
                db_path=db_path,
            )

            self.assertEqual("request_now", decision["action"])
            self.assertTrue(decision["urgent"])
            self.assertTrue(decision["blocking"])

    def test_autonomous_axis_does_not_ask_user_even_when_strategy_is_saturated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            diagnosis = self._diagnosis("trial_009", axis="model_family")
            diagnosis["strategy_recommendation"] = "strategy_escalation"
            diagnosis["consecutive_failures"] = 4

            decision = evaluate_feedback_policy(
                "demo",
                "trial_009",
                diagnosis,
                db_path=db_path,
            )

            self.assertEqual("autonomous_continue", decision["action"])
            self.assertEqual("no_review", decision["timing"])

    def test_repeated_non_improvement_alone_does_not_trigger_a_review_request(self):
        # trial_decision.py already rotates the improvement axis automatically
        # once axis_attempt_count hits MAX_AXIS_NO_IMPROVE_ATTEMPTS (3), with no
        # user input needed. The strategy_saturation trigger used to fire a
        # second, redundant blocking human-review request for the exact same
        # "N consecutive failures" signal -- and its copy text was written for
        # an unrelated scenario (ambiguous ground-truth labels), not for
        # "should we keep pushing this axis or switch". It must not fire.
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            diagnosis = self._diagnosis("trial_009", axis="validation")
            diagnosis["strategy_recommendation"] = "strategy_escalation"
            diagnosis["consecutive_failures"] = 5
            diagnosis["issues"] = [
                "CV did not improve against the current best trial.",
                "Recent failures suggest strategy escalation is needed.",
            ]

            decision = evaluate_feedback_policy(
                "demo",
                "trial_009",
                diagnosis,
                db_path=db_path,
            )

            self.assertNotIn("strategy_saturation", decision["triggers"])
            self.assertEqual("no_review", decision["action"])
            self.assertEqual("no_review", decision["timing"])
            self.assertEqual([], list_pending_actions("demo", db_path))

    @staticmethod
    def _diagnosis(trial_id: str, *, axis: str, segment: bool = False) -> dict:
        return {
            "competition": "demo",
            "trial_id": trial_id,
            "primary_axis": axis,
            "cv_score": 0.80,
            "best_cv_before": 0.81,
            "lb_score": 0.76,
            "best_lb_before": 0.77,
            "issues": (
                ["Errors are concentrated in one or more segments/groups/folds/features/patterns."]
                if segment
                else []
            ),
            "user_questions": [],
            "improvement_candidates": ["Inspect the evidence."],
            "needs_user_review": segment,
            "strategy_recommendation": "continue_refinement",
            "consecutive_failures": 0,
        }


if __name__ == "__main__":
    unittest.main()
