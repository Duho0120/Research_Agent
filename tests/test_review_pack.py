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
            self.assertTrue((pack / "decision_support.json").exists())
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["status"], "pending_user_feedback")
            decision_support = json.loads((pack / "decision_support.json").read_text(encoding="utf-8"))
            self.assertEqual("error_analysis", decision_support["axis"])
            self.assertIn("에이전트 추천", (pack / "summary.ko.md").read_text(encoding="utf-8"))
            self.assertIn("선택지와 반영 결과", (pack / "questions.ko.md").read_text(encoding="utf-8"))

    def test_compute_backend_is_record_only_until_execution_route_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_002",
                "change_axis": "compute_backend",
                "issues": ["GPU memory is insufficient for the planned model."],
                "needs_user_review": True,
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_review_pack("demo", "trial_002", diagnosis)

            card = result["decision_support"]
            self.assertFalse(card["execution_supported"])
            self.assertIn("자동 실행하지 않습니다", card["execution_note"])
            self.assertTrue(all(item["follow_up_action"] == "record_constraint_only" for item in card["options"]))

    def test_leakage_request_asks_about_prediction_time_and_uses_safe_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_003"
            trial.mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_003",
                "primary_axis": "validation",
                "issues": ["Leakage warning is present in metrics."],
                "leakage_warning": True,
                "leakage_features": ["post_event_status"],
                "cv_score": 0.99,
                "lb_score": 0.74,
                "needs_user_review": True,
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_review_pack("demo", "trial_003", diagnosis)

            card = result["decision_support"]
            self.assertIn("post_event_status", card["problem"])
            self.assertIn("예측 시점", card["question"])
            self.assertEqual("post_prediction", card["options"][1]["value"])
            self.assertIn("제외", card["default_if_no_response"])

    def test_segment_request_contains_segment_rate_count_and_representative_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)
            diagnosis = {
                "competition": "demo",
                "trial_id": "trial_004",
                "primary_axis": "error_analysis",
                "issues": ["Errors are concentrated in one segment."],
                "segment_errors": {
                    "Pclass=3,Sex=male": {"error_rate": 0.36, "sample_count": 50}
                },
                "overall_error_rate": 0.18,
                "representative_error_cases": [
                    {
                        "sample_id": "PassengerId=17",
                        "segment": "Pclass=3,Sex=male",
                        "true_label": 0,
                        "pred_label": 1,
                    }
                ],
                "needs_user_review": True,
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_review_pack("demo", "trial_004", diagnosis)

            card = result["decision_support"]
            self.assertIn("Pclass=3,Sex=male", card["problem"])
            evidence = {item["label"]: item for item in card["evidence_snapshot"]}
            self.assertEqual("36.0%", evidence["고오류 구간: Pclass=3,Sex=male"]["value"])
            self.assertIn("표본 50개", evidence["고오류 구간: Pclass=3,Sex=male"]["meaning"])
            self.assertEqual("PassengerId=17", evidence["대표 오류 사례"]["value"])
            cases = (trial / "review_pack" / "cases.jsonl").read_text(encoding="utf-8")
            self.assertIn("PassengerId=17", cases)


if __name__ == "__main__":
    unittest.main()
