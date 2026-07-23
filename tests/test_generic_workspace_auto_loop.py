import csv
import json
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from kaggle_research_agent.workspace_preparer import prepare_workspace
from scripts import generic_workspace_auto_loop


class GenericWorkspaceAutoLoopTest(unittest.TestCase):
    def test_submission_gate_blocks_missing_or_unscored_kaggle_submission(self):
        self.assertEqual(
            "submission_submit_failed",
            generic_workspace_auto_loop._submission_blocking_issue({"status": "submit_failed"}, requires_score=True),
        )
        self.assertEqual(
            "submission_missing_leaderboard_score",
            generic_workspace_auto_loop._submission_blocking_issue(
                {"status": "submitted", "submitted_lb_score": None},
                requires_score=True,
            ),
        )
        self.assertIsNone(
            generic_workspace_auto_loop._submission_blocking_issue(
                {"status": "submitted", "submitted_lb_score": 0.77272},
                requires_score=True,
            )
        )

    def test_run_one_trial_executes_workspace_without_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            with patch("kaggle_research_agent.paths.ROOT", root):
                prepare_workspace(
                    "demo",
                    topic="Demo",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="target",
                    id_column="id",
                    required_data_files=["train.csv", "test.csv"],
                )
                data_dir = root / "demo_workspaces" / "demo" / "data"
                self._write_csv(data_dir / "train.csv", [{"id": "1", "x": "a", "target": "0"}, {"id": "2", "x": "b", "target": "1"}])
                self._write_csv(data_dir / "test.csv", [{"id": "3", "x": "c"}, {"id": "4", "x": "d"}])
                with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                    with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                        with patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", runtime / "pause.request"):
                            result = generic_workspace_auto_loop.run_one_trial(
                                "demo",
                                "trial_001",
                                submit=False,
                                kaggle_slug=None,
                                poll_attempts=1,
                                poll_interval_seconds=0,
                            )

            self.assertEqual("completed", result["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "metrics.json").exists())

    def test_run_one_trial_uses_code_writer_when_next_plan_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            with patch("kaggle_research_agent.paths.ROOT", root):
                prepare_workspace(
                    "demo",
                    topic="Demo",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="target",
                    id_column="id",
                    required_data_files=["train.csv", "test.csv"],
                )
                data_dir = root / "demo_workspaces" / "demo" / "data"
                self._write_csv(data_dir / "train.csv", [{"id": "1", "x": "a", "target": "0"}, {"id": "2", "x": "b", "target": "1"}])
                self._write_csv(data_dir / "test.csv", [{"id": "3", "x": "c"}, {"id": "4", "x": "d"}])
                trial = root / "experiments" / "demo" / "trial_002"
                trial.mkdir(parents=True)
                (trial / "next_experiment.md").write_text("# trial_002 Next Experiment\n\nKeep baseline stable.\n", encoding="utf-8")
                (trial / "continuation_context.json").write_text(
                    json.dumps(
                        {
                            "competition": "demo",
                            "source_trial_id": "trial_001",
                            "next_trial_id": "trial_002",
                            "continuation_mode": "can_continue",
                            "pending_human_review": False,
                        }
                    ),
                    encoding="utf-8",
                )

                def fake_code_writer(*args, **kwargs):
                    (trial / "workspace_coding_result_validation.json").write_text(
                        json.dumps(
                            {
                                "competition": "demo",
                                "trial_id": "trial_002",
                                "status": "accepted",
                                "issues": [],
                                "coding_result_status": "completed",
                                "changed_files": ["src/baseline.py"],
                                "next_action": "run-workspace-validation-commands",
                            }
                        ),
                        encoding="utf-8",
                    )
                    return {"status": "accepted", "changed_files": ["src/baseline.py"]}

                with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                    with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                        with patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", runtime / "pause.request"):
                            with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value={"status": "ready"}):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "run_workspace_code_writer",
                                    side_effect=fake_code_writer,
                                ):
                                    result = generic_workspace_auto_loop.run_one_trial(
                                        "demo",
                                        "trial_002",
                                        submit=False,
                                        kaggle_slug=None,
                                        poll_attempts=1,
                                        poll_interval_seconds=0,
                                        code_writer=True,
                                        allow_api=True,
                                    )

            self.assertEqual("completed", result["status"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["after_coding"]["status"])

    def test_run_code_writer_retries_once_with_expanded_handoff_for_missing_snapshot(self):
        handoffs = [
            {"status": "ready", "snapshot_mode": "standard"},
            {"status": "ready", "snapshot_mode": "expanded_after_code_writer_blocked"},
        ]
        code_writer_results = [
            {
                "status": "blocked",
                "blocking_issues": [
                    "patch_only_mode_requires_exact_find_text",
                    "no_code_snapshot_provided_for_required_files: need current contents of src/baseline.py",
                ],
            },
            {"status": "accepted", "changed_files": ["src/baseline.py"]},
        ]
        after_coding = {
            "status": "completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
            "workspace_result_cycle": {"status": "completed"},
        }

        with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", side_effect=handoffs) as handoff:
            with patch.object(generic_workspace_auto_loop, "run_workspace_code_writer", side_effect=code_writer_results) as writer:
                with patch.object(generic_workspace_auto_loop, "run_workspace_after_coding", return_value=after_coding):
                    result = generic_workspace_auto_loop.run_code_writer_trial(
                        "demo",
                        "trial_002",
                        model="gpt-5",
                        provider="openai",
                        allow_api=True,
                        trial_llm_calls=None,
                        strategy_calls_today=None,
                    )

        self.assertEqual("completed", result["status"])
        self.assertEqual(2, handoff.call_count)
        self.assertFalse(handoff.call_args_list[0].kwargs["expanded_snapshot"])
        self.assertTrue(handoff.call_args_list[1].kwargs["expanded_snapshot"])
        self.assertEqual("code_writer_blocked_snapshot_context", handoff.call_args_list[1].kwargs["retry_reason"])
        self.assertEqual(2, writer.call_count)
        self.assertEqual(2, result["code_writer_attempt"])

    def test_code_writer_retry_detects_truncated_context_blocks(self):
        result = {
            "status": "code_writer_blocked",
            "code_writer": {
                "blocking_issues": [
                    "src/titanic_pipeline.py is truncated in the provided snapshot",
                    "Blocked due to missing full code context to implement an ensemble change safely.",
                ]
            },
        }

        self.assertTrue(generic_workspace_auto_loop._should_retry_code_writer_block(result))

    def test_run_loop_plans_start_trial_before_code_writer_when_plan_is_missing(self):
        args = argparse.Namespace(
            competition="demo",
            start_trial="trial_006",
            max_trials=1,
            submit=False,
            kaggle_slug=None,
            poll_attempts=1,
            poll_interval_seconds=0,
            code_writer=True,
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(generic_workspace_auto_loop, "LOCK_PATH", runtime / "auto_loop.lock"):
                        with patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", runtime / "pause.request"):
                            with patch.object(generic_workspace_auto_loop, "has_coding_plan", return_value=False):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "plan_next_workspace_trial",
                                    return_value={"status": "planned"},
                                ) as planner:
                                    with patch.object(
                                        generic_workspace_auto_loop,
                                        "run_one_trial",
                                        return_value={"status": "completed"},
                                    ):
                                        with patch.object(generic_workspace_auto_loop, "sync_state_db"):
                                            result = generic_workspace_auto_loop.run_loop(args)

        self.assertEqual("completed", result["status"])
        self.assertEqual(
            [
                call("demo", "trial_005", "trial_006", allow_api=True),
                call("demo", "trial_006", "trial_007", allow_api=True),
            ],
            planner.call_args_list,
        )

    def test_run_loop_pauses_after_successor_plan_and_resumes_that_trial(self):
        args = argparse.Namespace(
            competition="demo",
            start_trial="trial_006",
            max_trials=1,
            submit=False,
            kaggle_slug=None,
            poll_attempts=1,
            poll_interval_seconds=0,
            code_writer=True,
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            state_path = runtime / "auto_loop_state.json"
            pause_path = runtime / "pause.request"

            def complete_and_request_pause(*args, **kwargs):
                pause_path.write_text("requested\n", encoding="utf-8")
                return {"status": "completed"}

            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", state_path):
                    with patch.object(generic_workspace_auto_loop, "LOCK_PATH", runtime / "auto_loop.lock"):
                        with patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", pause_path):
                            with patch.object(generic_workspace_auto_loop, "has_coding_plan", return_value=True):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "run_one_trial",
                                    side_effect=complete_and_request_pause,
                                ):
                                    with patch.object(
                                        generic_workspace_auto_loop,
                                        "plan_next_workspace_trial",
                                        return_value={"status": "planned"},
                                    ) as planner:
                                        with patch.object(generic_workspace_auto_loop, "sync_state_db"):
                                            result = generic_workspace_auto_loop.run_loop(args)
            state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual("paused", result["status"])
        self.assertEqual("trial_007", state["next_trial"])
        self.assertEqual("planned", state["phase"])
        planner.assert_called_once_with("demo", "trial_006", "trial_007", allow_api=True)

    def test_trial_001_gets_initial_plan_before_code_and_trial_002_plan_after_completion(self):
        args = argparse.Namespace(
            competition="demo",
            start_trial="trial_001",
            max_trials=1,
            submit=False,
            kaggle_slug=None,
            poll_attempts=1,
            poll_interval_seconds=0,
            code_writer=True,
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(generic_workspace_auto_loop, "LOCK_PATH", runtime / "auto_loop.lock"):
                        with patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", runtime / "pause.request"):
                            with patch.object(generic_workspace_auto_loop, "has_coding_plan", return_value=False):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "prepare_workspace_trial_plan",
                                    return_value={"status": "planned"},
                                ) as initial_planner:
                                    with patch.object(
                                        generic_workspace_auto_loop,
                                        "plan_next_workspace_trial",
                                        return_value={"status": "planned"},
                                    ) as next_planner:
                                        with patch.object(
                                            generic_workspace_auto_loop,
                                            "run_one_trial",
                                            return_value={"status": "completed"},
                                        ):
                                            with patch.object(generic_workspace_auto_loop, "sync_state_db"):
                                                result = generic_workspace_auto_loop.run_loop(args)

        self.assertEqual("completed", result["status"])
        initial_planner.assert_called_once()
        self.assertEqual("trial_001", initial_planner.call_args.args[1])
        next_planner.assert_called_once_with("demo", "trial_001", "trial_002", allow_api=True)

    def test_pending_insight_revises_existing_planned_trial_before_code(self):
        args = argparse.Namespace(
            competition="demo",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        override = {
            "status": "active",
            "insight_id": "insight-1",
            "active_axis": "model_ensemble",
        }
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            trial_path = Path(tmp) / "trial_008"
            trial_path.mkdir(parents=True)
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(generic_workspace_auto_loop, "trial_dir", return_value=trial_path):
                        with patch.object(
                            generic_workspace_auto_loop,
                            "latest_user_insight_record",
                            return_value={
                                "insight_id": "insight-1",
                                "source_trial_id": "trial_007",
                                "target_trial": "trial_008",
                            },
                        ):
                            with patch.object(
                                generic_workspace_auto_loop,
                                "build_next_trial_user_insight_override",
                                return_value=override,
                            ):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "prepare_workspace_trial_plan",
                                    return_value={"status": "planned"},
                                ) as planner:
                                    result = generic_workspace_auto_loop.revise_planned_trial_for_pending_insight(
                                        args,
                                        "trial_008",
                                    )

        self.assertEqual("planned", result["status"])
        planner.assert_called_once_with(
            "demo",
            "trial_008",
            source_trial_id="trial_007",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
            user_insight_override=override,
            force_replan=True,
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
