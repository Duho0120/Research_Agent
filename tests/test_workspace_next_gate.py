import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from research_agent.cli import main
from research_agent.workspace_next_gate import plan_next_workspace_trial


class WorkspaceNextGateTest(unittest.TestCase):
    def test_nonurgent_pending_review_registers_request_and_plans_with_caution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(root, status="awaiting_human_review", urgent=False)

            with patch("research_agent.paths.project_root", return_value=root):
                result = plan_next_workspace_trial("demo", "trial_001", "trial_002")

            source = root / "experiments" / "demo" / "trial_001"
            next_trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("planned_with_pending_review", result["status"])
            self.assertEqual("continue_with_caution", result["continuation_mode"])
            self.assertTrue(result["pending_human_review"])
            self.assertTrue((source / "user_review_request.md").exists())
            self.assertTrue((next_trial / "next_experiment.md").exists())
            context = json.loads((next_trial / "continuation_context.json").read_text(encoding="utf-8"))
            self.assertEqual("continue_with_caution", context["continuation_mode"])
            self.assertTrue(context["pending_human_review"])
            self.assertEqual("trial_001", context["review_source_trial"])

    def test_urgent_pending_review_registers_request_and_blocks_next_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(
                root,
                status="awaiting_human_review",
                urgent=True,
                issues=["Metric definition is missing and blocks a valid next experiment."],
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = plan_next_workspace_trial("demo", "trial_001", "trial_002")

            source = root / "experiments" / "demo" / "trial_001"
            next_trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("blocked_human_review", result["status"])
            self.assertEqual("must_wait", result["continuation_mode"])
            self.assertTrue((source / "user_review_request.md").exists())
            self.assertFalse((next_trial / "next_experiment.md").exists())

    def test_completed_result_plans_next_trial_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(root, status="completed", urgent=False, human_review_timing="no_review")

            with patch("research_agent.paths.project_root", return_value=root):
                result = plan_next_workspace_trial("demo", "trial_001", "trial_002")

            self.assertEqual("planned", result["status"])
            self.assertEqual("can_continue", result["continuation_mode"])
            self.assertFalse(result["pending_human_review"])
            self.assertTrue(
                (root / "experiments" / "demo" / "trial_002" / "next_experiment.md").exists()
            )

    def test_next_trial_user_insight_overrides_axis_and_uses_best_submitted_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(
                root,
                trial_id="trial_005",
                status="completed",
                urgent=False,
                human_review_timing="no_review",
            )
            manual = root / "demo_workspaces" / "demo" / "manual_trials"
            (manual / "trial_003").mkdir(parents=True)
            (manual / "trial_003" / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_003", "local_score": 0.85, "kaggle_lb_score": 0.77272}),
                encoding="utf-8",
            )
            (manual / "trial_005").mkdir(parents=True)
            (manual / "trial_005" / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_005", "local_score": 0.82, "kaggle_lb_score": 0.76076}),
                encoding="utf-8",
            )
            memory = root / "memory" / "demo"
            with (memory / "user_feedback.jsonl").open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "trial_id": "trial_005",
                            "topic": "user_insight",
                            "scope": "next_trial",
                            "user_feedback": "모델 학습은 금방하니까 2~3개의 서로 다른 모델을 앙상블하자",
                            "decision": "user_insight_for_planner",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with patch("research_agent.paths.project_root", return_value=root):
                result = plan_next_workspace_trial("demo", "trial_005", "trial_006")

            context = json.loads(
                (root / "experiments" / "demo" / "trial_006" / "continuation_context.json").read_text(
                    encoding="utf-8"
                )
            )
            override = context["decision_context"]["user_insight_override"]
            self.assertEqual("planned", result["status"])
            self.assertEqual("active", override["status"])
            self.assertEqual("trial_003", override["base_trial_id"])
            self.assertEqual("model_ensemble", override["active_axis"])
            self.assertEqual(1, override["axis_attempt_count"])
            self.assertEqual(3, override["axis_attempt_limit"])
            self.assertEqual("trial_003", context["recommended_base_trial"])
            text = (root / "experiments" / "demo" / "trial_006" / "next_experiment.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("User Insight Override Policy", text)
            self.assertIn("model_ensemble", text)

    def test_next_trial_user_insight_attempt_count_continues_after_non_improving_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(
                root,
                trial_id="trial_006",
                status="completed",
                urgent=False,
                human_review_timing="no_review",
            )
            manual = root / "demo_workspaces" / "demo" / "manual_trials"
            (manual / "trial_003").mkdir(parents=True)
            (manual / "trial_003" / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_003", "local_score": 0.85, "kaggle_lb_score": 0.77272}),
                encoding="utf-8",
            )
            (manual / "trial_006").mkdir(parents=True)
            (manual / "trial_006" / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_006", "local_score": 0.80, "kaggle_lb_score": 0.76315}),
                encoding="utf-8",
            )
            memory = root / "memory" / "demo"
            with (memory / "user_feedback.jsonl").open("a", encoding="utf-8") as file:
                file.write(
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
                    + "\n"
                )
            prior = root / "experiments" / "demo" / "trial_006"
            (prior / "continuation_context.json").write_text(
                json.dumps(
                    {
                        "decision_context": {
                            "user_insight_override": {
                                "status": "active",
                                "active_axis": "model_ensemble",
                                "axis_attempt_count": 1,
                                "axis_attempt_limit": 3,
                                "base_trial_id": "trial_003",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan_next_workspace_trial("demo", "trial_006", "trial_007")

            context = json.loads(
                (root / "experiments" / "demo" / "trial_007" / "continuation_context.json").read_text(
                    encoding="utf-8"
                )
            )
            override = context["decision_context"]["user_insight_override"]
            self.assertEqual("trial_003", override["base_trial_id"])
            self.assertEqual("model_ensemble", override["active_axis"])
            self.assertEqual(2, override["axis_attempt_count"])
            self.assertFalse(override["source_improved_submission_score"])

    def test_missing_workspace_result_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)

            with patch("research_agent.paths.project_root", return_value=root):
                result = plan_next_workspace_trial("demo", "trial_001", "trial_002")

            self.assertEqual("blocked_missing_result_cycle", result["status"])
            self.assertFalse((root / "experiments" / "demo" / "trial_002").exists())

    def test_plan_next_workspace_trial_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_state(root)
            self._write_source_trial(root, status="completed", urgent=False, human_review_timing="no_review")

            with patch("research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "plan-next-workspace-trial",
                            "--competition",
                            "demo",
                            "--source-trial",
                            "trial_001",
                            "--next-trial",
                            "trial_002",
                        ]
                    )

            self.assertEqual(0, code)
            self.assertTrue(
                (root / "experiments" / "demo" / "trial_002" / "continuation_context.json").exists()
            )

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "state.yaml").write_text(
            "competition:\n"
            "  objective: maximize\n"
            "current_state:\n"
            "  consecutive_failures: 0\n"
            "  best_trial:\n"
            "    trial_id: trial_001\n"
            "    cv_score: 0.70\n",
            encoding="utf-8",
        )

    def _write_source_trial(
        self,
        root: Path,
        *,
        trial_id: str = "trial_001",
        status: str,
        urgent: bool,
        human_review_timing: str = "request_now",
        issues: list[str] | None = None,
    ) -> None:
        source = root / "experiments" / "demo" / trial_id
        source.mkdir(parents=True)
        diagnosis = {
            "competition": "demo",
            "trial_id": trial_id,
            "needs_user_review": status == "awaiting_human_review",
            "strategy_recommendation": "continue_refinement",
            "issues": issues or ["Representative errors would benefit from human interpretation."],
            "user_questions": ["Do these error cases suggest a data issue or a safe refinement?"],
            "improvement_candidates": ["Try one controlled refinement while waiting for feedback."],
            "cv_score": 0.70,
            "objective": "maximize",
        }
        (source / "metrics.json").write_text(
            json.dumps({"cv_score": 0.70, "objective": "maximize"}),
            encoding="utf-8",
        )
        (source / "workspace_result_cycle.json").write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "trial_id": trial_id,
                    "status": status,
                    "steps": ["evaluated", "diagnosed", "remembered"],
                    "diagnosis": diagnosis,
                    "human_review": {
                        "timing": human_review_timing,
                        "urgent": urgent,
                        "triggers": ["diagnosis_requested_review"],
                    },
                    "next_action": "request-user-review" if status == "awaiting_human_review" else "plan-next-experiment",
                }
            ),
            encoding="utf-8",
        )
        memory = root / "memory" / "demo"
        memory.mkdir(parents=True)
        (memory / "decision_log.jsonl").write_text(
            json.dumps(
                {
                    "trial_id": trial_id,
                    "decision_type": "diagnosis",
                    "evidence": {
                        "strategy_recommendation": "continue_refinement",
                        "issues": diagnosis["issues"],
                        "cv_improved": True,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
