import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.policy_gate import (
    classify_local_failure,
    decide_execution,
    decide_human_review,
    log_llm_decision,
    should_call_llm,
)
from kaggle_research_agent.policies import load_policy


class PolicyGateTest(unittest.TestCase):
    def test_load_policy_reads_default_policy_file(self):
        policy = load_policy("execution_policy")

        self.assertEqual(policy["default_backend"], "local")
        self.assertTrue(policy["local_first"])
        self.assertTrue(policy["ask_before_colab"])

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

    def test_should_call_llm_respects_reason_and_budget(self):
        allowed = should_call_llm("human_review_needed", trial_llm_calls=0, strategy_calls_today=0)
        blocked = should_call_llm("human_review_needed", trial_llm_calls=4, strategy_calls_today=0)
        unnecessary = should_call_llm("parse_metrics", trial_llm_calls=0, strategy_calls_today=0)

        self.assertEqual(allowed["decision"], "call_llm")
        self.assertEqual(blocked["decision"], "skip_llm")
        self.assertEqual(unnecessary["decision"], "skip_llm")

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


if __name__ == "__main__":
    unittest.main()
