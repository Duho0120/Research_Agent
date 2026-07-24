import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.policy_gate import (
    classify_local_failure,
    count_llm_calls_from_decision_log,
    decide_execution,
    decide_human_review,
    log_llm_decision,
    should_call_llm,
)
from kaggle_research_agent.agents.memory import log_decision
from kaggle_research_agent.policies import load_policy, resolve_model_for_call, select_model_for_call


class PolicyGateTest(unittest.TestCase):
    def test_load_policy_reads_default_policy_file(self):
        policy = load_policy("execution_policy")

        self.assertEqual(policy["default_backend"], "local")
        self.assertTrue(policy["local_first"])
        self.assertTrue(policy["ask_before_colab"])

    def test_model_policy_selects_high_and_low_cost_models(self):
        policy = load_policy("model_policy")

        high_cost = select_model_for_call("experiment_planning", policy=policy)
        low_cost = select_model_for_call("status_summary", policy=policy)

        self.assertEqual("openai", high_cost["provider"])
        self.assertEqual("gpt-5.5", high_cost["model"])
        self.assertEqual("openai", low_cost["provider"])
        self.assertEqual("gpt-5.6-luna", low_cost["model"])

    def test_low_cost_model_can_be_overridden_centrally_or_per_feature(self):
        with patch.dict("os.environ", {"RESEARCH_AGENT_LOW_COST_MODEL": "central-low-cost"}, clear=False):
            selected = resolve_model_for_call("experiment_question")
        self.assertEqual("central-low-cost", selected["model"])

        with patch.dict(
            "os.environ",
            {
                "RESEARCH_AGENT_LOW_COST_MODEL": "central-low-cost",
                "RESEARCH_AGENT_CHAT_MODEL": "chat-specific",
            },
            clear=False,
        ):
            selected = resolve_model_for_call(
                "experiment_question",
                model_env_var="RESEARCH_AGENT_CHAT_MODEL",
            )
        self.assertEqual("chat-specific", selected["model"])

    def test_decide_execution_chooses_local_run_when_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_execution(
                    "demo",
                    "trial_001",
                    run_now=True,
                    command="python train.py",
                    log=True,
                )

            self.assertEqual(result["decision"], "run_local")
            self.assertTrue((root / "memory" / "demo" / "decision_log.jsonl").exists())

    def test_decide_execution_asks_user_after_resource_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            (trial / "local_run.log").write_text("STDERR:\nCUDA out of memory\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_execution("demo", "trial_001", run_now=True, command="python train.py")

            self.assertEqual(result["decision"], "ask_user")
            self.assertEqual(result["evidence"]["previous_local_failure_type"], "resource_cpu_memory")

    def test_decide_execution_uses_local_failure_artifact_for_next_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            (trial / "local_run.log").write_text("STDERR:\nCUDA out of memory\n", encoding="utf-8")
            (trial / "local_failure.json").write_text(
                json.dumps(
                    {
                        "failure_type": "missing_dependency",
                        "matched_pattern": "ModuleNotFoundError",
                        "suggested_next_action": "fix_dependency",
                        "artifact_path": "experiments/demo/trial_001/local_failure.json",
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_execution("demo", "trial_001", run_now=True, command="python train.py")

            self.assertEqual(result["decision"], "ask_user")
            self.assertEqual(result["next_action"], "fix-dependency")
            self.assertEqual(result["evidence"]["previous_local_failure_type"], "missing_dependency")
            self.assertEqual(
                result["evidence"]["local_failure_artifact_path"],
                "experiments/demo/trial_001/local_failure.json",
            )

    def test_classify_local_failure_detects_missing_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_run.log"
            path.write_text("STDERR:\nModuleNotFoundError: No module named 'x'\n", encoding="utf-8")

            result = classify_local_failure(path)

            self.assertEqual(result["failure_type"], "missing_dependency")

    def test_decide_human_review_requests_review_pack_from_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Errors are concentrated in one segment."],
                        "user_questions": ["Check this segment."],
                    },
                )

            self.assertEqual(result["decision"], "prepare_review_pack")
            self.assertIn("high_error_concentration", result["triggers"])

    def test_nonurgent_human_review_is_deferred_before_pipeline_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Errors are concentrated in one segment."],
                        "user_questions": ["Check this segment."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=1),
                )

            self.assertEqual("defer_review", result["decision"])
            self.assertEqual("defer", result["timing"])

    def test_nonurgent_human_review_is_released_after_two_completed_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_002",
                    {
                        "needs_user_review": True,
                        "issues": ["Errors are concentrated in one segment."],
                        "user_questions": ["Check this segment."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=2),
                )

            self.assertEqual("prepare_review_pack", result["decision"])
            self.assertEqual("request_now", result["timing"])

    def test_nonurgent_review_is_deferred_while_previous_feedback_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readiness = self._readiness(completed_trial_count=3)
            readiness["pending_user_review"] = True
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_003",
                    {
                        "needs_user_review": True,
                        "issues": ["Errors are concentrated in one segment."],
                        "user_questions": ["Check this segment."],
                    },
                    pipeline_readiness=readiness,
                )

            self.assertEqual("defer_review", result["decision"])
            self.assertEqual("defer", result["timing"])

    def test_leakage_review_bypasses_pipeline_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Leakage warning is present in metrics."],
                        "user_questions": ["Confirm the validation boundary."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=1),
                )

            self.assertEqual("prepare_review_pack", result["decision"])
            self.assertEqual("request_now", result["timing"])
            self.assertTrue(result["urgent"])

    def test_label_boundary_ambiguity_bypasses_pipeline_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Label boundary is ambiguous for adjacent classes."],
                        "user_questions": ["Confirm the label boundary."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=1),
                )

            self.assertIn("label_boundary_ambiguous", result["triggers"])
            self.assertEqual("request_now", result["timing"])

    def test_safety_false_negative_bypasses_pipeline_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Fall safety false negative was detected."],
                        "user_questions": ["Inspect the missed Fall case."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=1),
                )

            self.assertIn("safety_false_negative", result["triggers"])
            self.assertEqual("request_now", result["timing"])

    def test_missing_metric_definition_bypasses_pipeline_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = decide_human_review(
                    "demo",
                    "trial_001",
                    {
                        "needs_user_review": True,
                        "issues": ["Required metric definition is missing."],
                        "user_questions": ["Confirm the primary metric."],
                    },
                    pipeline_readiness=self._readiness(completed_trial_count=1),
                )

            self.assertIn("blocking_information_missing", result["triggers"])
            self.assertEqual("request_now", result["timing"])

    def test_should_call_llm_respects_reason_and_budget(self):
        allowed = should_call_llm("human_review_needed", trial_llm_calls=0, strategy_calls_today=0)
        blocked = should_call_llm("human_review_needed", trial_llm_calls=4, strategy_calls_today=0)
        unnecessary = should_call_llm("parse_metrics", trial_llm_calls=0, strategy_calls_today=0)

        self.assertEqual(allowed["decision"], "call_llm")
        self.assertEqual(blocked["decision"], "skip_llm")
        self.assertEqual(unnecessary["decision"], "skip_llm")

    def test_should_call_llm_counts_existing_decision_log_when_counts_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                for _ in range(4):
                    log_decision(
                        "demo",
                        "trial_001",
                        decision_type="llm_call",
                        decision="call_llm",
                        reason="Existing LLM call.",
                    )

                result = should_call_llm("human_review_needed", competition="demo", trial_id="trial_001")

        self.assertEqual(result["decision"], "skip_llm")
        self.assertEqual(result["trial_llm_calls"], 4)
        self.assertEqual(result["counts_source"], "decision_log")

    def test_should_call_llm_manual_counts_override_decision_log_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                for _ in range(4):
                    log_decision(
                        "demo",
                        "trial_001",
                        decision_type="llm_call",
                        decision="call_llm",
                        reason="Existing LLM call.",
                    )

                result = should_call_llm(
                    "human_review_needed",
                    competition="demo",
                    trial_id="trial_001",
                    trial_llm_calls=0,
                    strategy_calls_today=0,
                )

        self.assertEqual(result["decision"], "call_llm")
        self.assertEqual(result["trial_llm_calls"], 0)
        self.assertEqual(result["strategy_calls_today"], 0)

    def test_count_llm_calls_includes_code_writer_token_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                log_decision(
                    "demo",
                    "trial_001",
                    decision_type="code_writer_api",
                    decision="accepted",
                    reason="Code writer ran.",
                    evidence={"token_decision": {"decision": "call_llm"}},
                )
                log_decision(
                    "demo",
                    "trial_002",
                    decision_type="code_writer_api",
                    decision="blocked",
                    reason="Code writer blocked.",
                    evidence={"token_decision": {"decision": "skip_llm"}},
                )

                counts = count_llm_calls_from_decision_log("demo", "trial_001")

        self.assertEqual(counts["trial_llm_calls"], 1)
        self.assertGreaterEqual(counts["strategy_calls_today"], 1)

    def test_log_llm_decision_records_token_policy_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = log_llm_decision(
                    "demo",
                    "trial_001",
                    "human_review_needed",
                    trial_llm_calls=1,
                    strategy_calls_today=2,
                    prompt_summary_path="experiments/demo/trial_001/review_pack/summary.ko.md",
                )

            self.assertEqual(result["decision"], "call_llm")
            path = root / "memory" / "demo" / "decision_log.jsonl"
            saved = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["decision_type"], "llm_call")
            self.assertEqual(saved["decision"], "call_llm")
            self.assertEqual(saved["evidence"]["llm_reason"], "human_review_needed")
            self.assertEqual(saved["evidence"]["trial_llm_calls"], 1)

    @staticmethod
    def _readiness(*, completed_trial_count: int) -> dict:
        return {
            "execution_profile_ready": True,
            "workspace_run_completed": True,
            "metrics_collected": True,
            "completed_trial_count": completed_trial_count,
        }


if __name__ == "__main__":
    unittest.main()
