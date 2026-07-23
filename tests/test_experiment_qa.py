import tempfile
import unittest
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.experiment_qa import (
    _question_payload,
    answer_experiment_question,
    collect_experiment_evidence,
)
from kaggle_research_agent.experiment_qa_retrieval import MAX_CONTEXT_CHARS


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

    def test_structured_score_question_uses_sqlite_without_document_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "memory" / "research_agent.sqlite3"
            db_path.parent.mkdir(parents=True)
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE trials (
                    competition_id TEXT,
                    trial_id TEXT,
                    status TEXT,
                    source_trial_id TEXT,
                    plan_type TEXT,
                    plan_summary TEXT,
                    primary_change_axis TEXT
                );
                CREATE TABLE trial_scores (
                    competition_id TEXT,
                    trial_id TEXT,
                    metric TEXT,
                    objective TEXT,
                    local_score REAL,
                    lb_score REAL,
                    is_best_lb INTEGER
                );
                CREATE TABLE trial_decisions (
                    competition_id TEXT,
                    trial_id TEXT,
                    decision TEXT,
                    change_axis TEXT,
                    active_axis TEXT,
                    axis_attempt_count INTEGER,
                    axis_attempt_limit INTEGER
                );
                INSERT INTO trials VALUES
                    ('titanic', 'trial_001', 'completed', NULL, 'initial', 'baseline', 'baseline'),
                    ('titanic', 'trial_002', 'completed', 'trial_001', 'delta', 'features', 'feature');
                INSERT INTO trial_scores VALUES
                    ('titanic', 'trial_001', 'accuracy', 'maximize', 0.80, 0.76, 0),
                    ('titanic', 'trial_002', 'accuracy', 'maximize', 0.82, 0.78, 1);
                """
            )
            connection.commit()
            connection.close()
            irrelevant = root / "runs" / "titanic" / "trial_002" / "01_plan.ko.md"
            irrelevant.parent.mkdir(parents=True)
            irrelevant.write_text("질문과 관계없는 매우 긴 계획", encoding="utf-8")

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence(
                    "titanic",
                    "trial_002",
                    "지금까지 각 실험의 로컬 점수와 제출 점수, 베스트를 알려줘",
                )

            self.assertEqual(["sqlite:trial_summary"], [source for source, _ in evidence])
            content = evidence[0][1]
            self.assertIn("trial_001", content)
            self.assertIn("trial_002", content)
            self.assertIn('"lb_score": 0.78', content)

    def test_document_retrieval_excludes_index_and_deduplicates_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = "VotingClassifier 앙상블 실제 실행 모델"
            run_doc = root / "runs" / "titanic" / "trial_007" / "02_pipeline_structure.ko.md"
            user_doc = root / "experiments" / "titanic" / "trial_007" / "user_view" / "02_pipeline_structure.ko.md"
            index_doc = root / "memory" / "titanic" / "document_index.jsonl"
            for path, content in (
                (run_doc, duplicate),
                (user_doc, duplicate),
                (index_doc, "VotingClassifier 앙상블 실제 실행 모델 " * 20),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence(
                    "titanic",
                    "trial_007",
                    "trial_007 앙상블이 실제 실행 모델에 반영됐어?",
                )

            sources = [source for source, _ in evidence]
            self.assertFalse(any("document_index" in source for source in sources))
            self.assertEqual(1, sum(content == duplicate for _, content in evidence))

    def test_retrieval_context_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(20):
                path = root / "runs" / "titanic" / "trial_007" / f"note_{index}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"앙상블 모델 실행 기록 {index} " + ("내용 " * 2000), encoding="utf-8")

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence(
                    "titanic",
                    "trial_007",
                    "trial_007 앙상블 모델 실행 기록을 알려줘",
                )

            self.assertLessEqual(sum(len(content) for _, content in evidence), MAX_CONTEXT_CHARS)
            self.assertLessEqual(len(evidence), 7)

    def test_question_payload_uses_larger_but_bounded_output_limit(self):
        with patch.dict("os.environ", {}, clear=False):
            payload = _question_payload(
                "test-model",
                "titanic",
                "trial_007",
                "결과를 알려줘",
                [("sqlite:trial_summary", "{}")],
            )
        self.assertEqual(1000, payload["max_output_tokens"])


if __name__ == "__main__":
    unittest.main()
