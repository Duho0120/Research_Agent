import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.experiment_qa import answer_experiment_question, collect_experiment_evidence


class FakeClient:
    def create_response(self, payload):
        return {
            "id": "response_1",
            "model": "test-low-cost",
            "output_text": "trial_003의 제출 점수는 0.77272입니다. 근거: 03_scores.ko.md",
            "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        }


class ExperimentQuestionTest(unittest.TestCase):
    def test_llm_answer_uses_collected_evidence_and_records_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "runs" / "titanic" / "trial_003" / "03_scores.ko.md"
            scores.parent.mkdir(parents=True)
            scores.write_text("Kaggle 제출 점수: 0.77272", encoding="utf-8")
            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.agents.memory.competition_memory_dir",
                    return_value=root / "memory" / "titanic",
                ):
                    result = answer_experiment_question(
                        "titanic", "trial_003", "제출 점수는?", client=FakeClient()
                    )
            self.assertEqual("low_cost_llm", result["mode"])
            self.assertIn("0.77272", result["answer"])
            self.assertTrue((root / "memory" / "titanic" / "token_usage.jsonl").exists())

    def test_api_failure_falls_back_to_local_evidence(self):
        class FailingClient:
            def create_response(self, payload):
                raise RuntimeError("quota exceeded")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "runs" / "titanic" / "trial_003" / "03_scores.ko.md"
            scores.parent.mkdir(parents=True)
            scores.write_text("- 제출 점수: 0.77272\n- 로컬 점수: 0.854749", encoding="utf-8")
            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                result = answer_experiment_question(
                    "titanic", "trial_003", "제출 점수는?", client=FailingClient()
                )
            self.assertEqual("local_evidence", result["mode"])
            self.assertIn("0.77272", result["answer"])
            self.assertIn("quota exceeded", result["warning"])

    def test_evidence_includes_all_manual_trial_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "demo_workspaces" / "titanic" / "manual_trials"
            for trial_id, local, lb in [
                ("trial_001", 0.8044692737430168, 0.76555),
                ("trial_002", 0.8044692737430168, 0.76315),
                ("trial_003", 0.8547486033519553, 0.77272),
            ]:
                path = manual / trial_id / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "local_score": local,
                            "kaggle_submitted": True,
                            "kaggle_lb_score": lb,
                        }
                    ),
                    encoding="utf-8",
                )

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence("titanic", "trial_003")

            summary = "\n".join(content for _, content in evidence)
            self.assertIn("trial_001", summary)
            self.assertIn("trial_002", summary)
            self.assertIn("trial_003", summary)
            self.assertIn("0.76315", summary)


if __name__ == "__main__":
    unittest.main()
