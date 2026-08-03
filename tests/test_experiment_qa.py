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
from kaggle_research_agent.experiment_qa_retrieval import MAX_CONTEXT_CHARS, _requested_trials


class FakeClient:
    def create_response(self, payload):
        return {
            "id": "response_1",
            "model": "test-low-cost",
            "output_text": "trial_003의 제출 점수는 0.77272입니다. 근거: 03_scores.ko.md",
            "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        }


class ExperimentQuestionTest(unittest.TestCase):
    def test_question_channel_is_read_only_and_does_not_mutate_research_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            plan.parent.mkdir(parents=True)
            plan.write_text("local_score: 0.81\nplan: baseline", encoding="utf-8")
            state = root / "memory" / "demo" / "decision.json"
            state.parent.mkdir(parents=True)
            state.write_text('{"decision":"accept"}', encoding="utf-8")
            before = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                result = answer_experiment_question(
                    "demo",
                    "trial_001",
                    "local_score?",
                    use_llm=False,
                )

            after = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual("read_only", result["interaction"]["access"])
            self.assertEqual([], result["interaction"]["research_state_mutations"])
            self.assertFalse(result["interaction"]["requires_explicit_submit"])

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

    def test_applied_change_retrieval_includes_code_derived_execution_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            facts = (
                root
                / "experiments"
                / "bike"
                / "trial_005"
                / "internal"
                / "executed_trial_facts.json"
            )
            facts.parent.mkdir(parents=True)
            facts.write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "source_trial_id": "trial_004",
                        "primary_change_axis": "model_family",
                        "change_details": ["Replace Ridge with HGBR"],
                        "model": {
                            "estimator": "HistGradientBoostingRegressor",
                            "parameters": {"learning_rate": "0.05"},
                        },
                        "scores": {"local": 0.33549, "submission": 0.41066},
                    }
                ),
                encoding="utf-8",
            )
            stale_plan = (
                root
                / "experiments"
                / "bike"
                / "trial_005"
                / "user_view"
                / "01_plan.ko.md"
            )
            stale_plan.parent.mkdir(parents=True)
            stale_plan.write_text("plan only: set max_iter=300", encoding="utf-8")

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence(
                    "bike",
                    "trial_005",
                    "trial_005에서 실제 적용한 모델 개선은 무엇인가?",
                )

            self.assertTrue(
                any(source.replace("\\", "/").endswith("trial_005/internal/executed_trial_facts.json")
                    for source, _ in evidence)
            )
            executed = next(content for source, content in evidence if "executed_trial_facts" in source)
            self.assertIn('"evidence_kind": "executed_facts"', executed)
            self.assertIn("HistGradientBoostingRegressor", executed)
            self.assertIn("Replace Ridge with HGBR", executed)
            self.assertNotIn("max_iter=300", "\n".join(content for _, content in evidence))

    def test_evidence_broadens_instead_of_narrowing_to_current_trial_when_question_is_unparsed(self):
        # Regression: a user asked about "14회차" (a phrasing the trial-
        # reference regex didn't recognize) while trial_027 was the
        # currently open/viewed trial. The evidence used to silently narrow
        # to trial_027's data, so the chatbot answered "not in the docs"
        # about trial_014 while never even seeing it. It should now consider
        # every trial rather than guessing the open one.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for trial_id, estimator in [("trial_014", "HistGradientBoostingRegressor"), ("trial_027", "Ridge")]:
                facts = root / "experiments" / "bike" / trial_id / "internal" / "executed_trial_facts.json"
                facts.parent.mkdir(parents=True)
                facts.write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "model": {"estimator": estimator},
                            "scores": {"local": 0.39},
                        }
                    ),
                    encoding="utf-8",
                )

            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                evidence = collect_experiment_evidence(
                    "bike",
                    "trial_027",
                    "이 대회에서 지금까지 실행에 사용한 모델이 뭔지 알려줘",
                )

            combined = "\n".join(content for _, content in evidence)
            self.assertIn("HistGradientBoostingRegressor", combined)

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
        self.assertIn("Plan documents describe intent only", payload["input"])
        self.assertIn("executed_facts", payload["input"])

    def test_question_payload_includes_only_recent_bounded_chat_context(self):
        conversation = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index} " + ("x" * 900)}
            for index in range(10)
        ]
        payload = _question_payload(
            "test-model",
            "demo",
            "trial_003",
            "그 결과를 설명해줘",
            [("scores.md", "score: 0.77")],
            conversation=conversation,
        )

        self.assertNotIn("message-3", payload["input"])
        self.assertIn("message-4", payload["input"])
        self.assertIn("message-9", payload["input"])
        self.assertLess(len(payload["input"]), 6000)

    def test_demo_mode_never_calls_api_and_renders_structured_score_table(self):
        class UnexpectedClient:
            def create_response(self, payload):
                raise AssertionError("Demo mode must not call the API")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "demo_workspaces" / "titanic" / "manual_trials" / "trial_001" / "metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "local_score": 0.81,
                        "kaggle_lb_score": 0.77,
                        "kaggle_submitted": True,
                    }
                ),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.experiment_qa.project_root", return_value=root):
                with patch.dict(
                    "os.environ",
                    {
                        "RESEARCH_AGENT_CHAT_DEMO_MODE": "1",
                        "OPENAI_API_KEY": "must-not-be-used",
                    },
                    clear=False,
                ):
                    result = answer_experiment_question(
                        "titanic",
                        "trial_001",
                        "전체 실험 점수와 베스트를 알려줘",
                        client=UnexpectedClient(),
                    )

        self.assertEqual("demo_local_rag", result["mode"])
        self.assertEqual("DEMO · 로컬 근거 모드", result["mode_label"])
        self.assertIn("| trial | 로컬 점수 | 제출 점수 |", result["answer"])
        self.assertIn("0.77000", result["answer"])


class RequestedTrialsTest(unittest.TestCase):
    # Regression: a user asking "Trial 23, 24, 25는 점수가 망가졌는데 원인이
    # 뭐야?" got a chatbot answer claiming the docs had nothing on it. The
    # real cause was that _requested_trials only matched the exact
    # "trial_023" form, so it silently fell back to the currently-viewed
    # trial and retrieved the wrong evidence entirely.
    def test_matches_leading_trial_keyword_with_enumerated_list(self):
        self.assertEqual(
            {"trial_023", "trial_024", "trial_025"},
            _requested_trials("Trial 23, 24, 25는 점수가 망가졌는데 원인이 뭐야?"),
        )

    def test_matches_repeated_trial_keyword_without_zero_padding(self):
        self.assertEqual(
            {"trial_024", "trial_025", "trial_026"},
            _requested_trials("Trial_24, Trial_25, Trial_26은 점수가 망가졌는데 원인이 뭐야?"),
        )

    def test_matches_korean_ordinal_counter_cha(self):
        self.assertEqual({"trial_023", "trial_024"}, _requested_trials("23차, 24차 결과 비교해줘"))

    def test_matches_korean_ordinal_beonjjae(self):
        self.assertEqual({"trial_023"}, _requested_trials("23번째 실험은 왜 그래?"))

    def test_matches_korean_ordinal_hoecha(self):
        # Regression: "14회차" ("round 14") was missed because "회" sits
        # between the digits and "차", so the "cha"-only pattern never
        # matched it -- the chatbot silently fell back to whatever trial was
        # currently being viewed instead of trial_014.
        self.assertEqual(
            {"trial_014"}, _requested_trials("14회차 실험에서 사용한 모델은 뭔지 한 번 찾아봐줘")
        )
        self.assertEqual({"trial_014"}, _requested_trials("14회 실험 결과 알려줘"))

    def test_matches_exact_zero_padded_form(self):
        self.assertEqual({"trial_023", "trial_024"}, _requested_trials("trial_023과 trial_024 비교"))

    def test_no_match_returns_empty_set(self):
        self.assertEqual(set(), _requested_trials("전체적으로 어떻게 진행되고 있어?"))


if __name__ == "__main__":
    unittest.main()
