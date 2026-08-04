import contextlib
import csv
import json
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from research_agent.workspace_preparer import prepare_workspace
from scripts import generic_workspace_auto_loop


class GenericWorkspaceAutoLoopTest(unittest.TestCase):
    def setUp(self):
        # run_code_writer_trial() now generates the one-time scoring harness
        # before a trial's own code-writer attempt when it does not already
        # exist -- default it to "already there" for every test so existing
        # trial-level behavior tests aren't coupled to harness generation.
        # Tests that specifically exercise harness generation override this.
        for name in ("generate_scoring_harness", "generate_data_loader"):
            patcher = patch.object(generic_workspace_auto_loop, name, return_value={"status": "already_exists"})
            self.addCleanup(patcher.stop)
            patcher.start()

    def test_only_an_interrupted_langgraph_process_is_resumable(self):
        state = {
            "graph_runtime": "langgraph",
            "status": "running",
            "competition": "demo",
            "next_trial": "trial_003",
            "graph_thread_id": "thread-1",
        }
        self.assertTrue(
            generic_workspace_auto_loop._can_resume_graph_process(
                state,
                "demo",
                "trial_003",
            )
        )
        self.assertFalse(
            generic_workspace_auto_loop._can_resume_graph_process(
                state | {"status": "failed"},
                "demo",
                "trial_003",
            )
        )
        self.assertFalse(
            generic_workspace_auto_loop._can_resume_graph_process(
                state | {"status": "starting", "resume_from_status": "failed"},
                "demo",
                "trial_003",
            )
        )
        self.assertTrue(
            generic_workspace_auto_loop._can_resume_graph_process(
                state | {"status": "starting", "resume_from_status": "running"},
                "demo",
                "trial_003",
            )
        )

    def test_run_one_trial_preserves_execute_analyze_submit_artifact_order(self):
        events: list[str] = []
        profile = {
            "project_root": "C:/workspace",
            "objective": "maximize",
            "artifacts": {"submission": ["outputs/submission.csv"]},
        }

        def event(name, result):
            events.append(name)
            return result

        with patch.object(
            generic_workspace_auto_loop,
            "validate_execution_profile",
            return_value={"status": "ready"},
        ):
            with patch.object(generic_workspace_auto_loop, "load_execution_profile", return_value=profile):
                with patch.object(
                    generic_workspace_auto_loop,
                    "run_workspace_pipeline",
                    side_effect=lambda *args, **kwargs: event("execute", {"status": "completed"}),
                ):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "collect_workspace_metrics",
                        side_effect=lambda *args, **kwargs: event(
                            "collect",
                            {"status": "collected", "competition": "demo", "cv_score": 0.81},
                        ),
                    ):
                        with patch.object(
                            generic_workspace_auto_loop,
                            "process_workspace_result",
                            side_effect=lambda *args, **kwargs: event("analyze", {"status": "completed"}),
                        ):
                            with patch.object(
                                generic_workspace_auto_loop,
                                "reconcile_trial_execution_metadata",
                                side_effect=lambda *args, **kwargs: event("consistency", {"status": "ready"}),
                            ):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "submit_trial",
                                    side_effect=lambda **kwargs: event(
                                        "submit",
                                        {"status": "submitted", "submitted_lb_score": 0.77},
                                    ),
                                ):
                                    with patch.object(
                                        generic_workspace_auto_loop,
                                        "organize_trial_artifacts",
                                        side_effect=lambda *args, **kwargs: event("artifacts", {"status": "completed"}),
                                    ):
                                        with patch.object(
                                            generic_workspace_auto_loop,
                                            "submission_artifact_path",
                                            return_value="C:/workspace/outputs/submission.csv",
                                        ):
                                            with patch.object(generic_workspace_auto_loop, "write_loop_trial_result"):
                                                with patch.object(generic_workspace_auto_loop, "save_loop_state"):
                                                    result = generic_workspace_auto_loop.run_one_trial(
                                                        "demo",
                                                        "trial_001",
                                                        submit=True,
                                                        kaggle_slug="demo",
                                                        poll_attempts=1,
                                                        poll_interval_seconds=0,
                                                    )

        self.assertEqual("completed", result["status"])
        self.assertEqual(["execute", "collect", "consistency", "submit", "analyze", "artifacts"], events)

    def test_skips_auto_submit_when_daily_limit_known_and_auto_submit_disabled(self):
        # Real-world concern this addresses: without this skip, the loop
        # kept auto-submitting every trial and silently stalled once DACON
        # rejected a submission for exceeding the daily limit.
        events: list[str] = []
        profile = {
            "project_root": "C:/workspace",
            "objective": "maximize",
            "artifacts": {"submission": ["outputs/submission.csv"]},
        }

        def event(name, result):
            events.append(name)
            return result

        with patch.object(generic_workspace_auto_loop, "validate_execution_profile", return_value={"status": "ready"}):
            with patch.object(generic_workspace_auto_loop, "load_execution_profile", return_value=profile):
                with patch.object(
                    generic_workspace_auto_loop, "run_workspace_pipeline",
                    side_effect=lambda *a, **k: event("execute", {"status": "completed"}),
                ):
                    with patch.object(
                        generic_workspace_auto_loop, "collect_workspace_metrics",
                        side_effect=lambda *a, **k: event(
                            "collect", {"status": "collected", "competition": "demo", "cv_score": 0.591}
                        ),
                    ):
                        with patch.object(
                            generic_workspace_auto_loop, "reconcile_trial_execution_metadata",
                            side_effect=lambda *a, **k: event("consistency", {"status": "ready"}),
                        ):
                            with patch.object(
                                generic_workspace_auto_loop, "process_workspace_result",
                                side_effect=lambda *a, **k: event("analyze", {"status": "completed"}),
                            ):
                                with patch.object(
                                    generic_workspace_auto_loop, "organize_trial_artifacts",
                                    side_effect=lambda *a, **k: event("artifacts", {"status": "completed"}),
                                ):
                                    with patch.object(
                                        generic_workspace_auto_loop, "submit_trial",
                                        side_effect=AssertionError("must not call the real submit path when skipped"),
                                    ):
                                        with patch.object(
                                            generic_workspace_auto_loop, "check_dacon_submission_limit",
                                                return_value={"daily_submission_limit": 5}
                                        ):
                                            with patch.object(
                                                generic_workspace_auto_loop, "dacon_auto_submit_allowed", return_value=False
                                            ):
                                                with patch.object(generic_workspace_auto_loop, "write_loop_trial_result"):
                                                    with patch.object(generic_workspace_auto_loop, "save_loop_state"):
                                                        result = generic_workspace_auto_loop.run_one_trial(
                                                            "demo",
                                                            "trial_004",
                                                            submit=True,
                                                            kaggle_slug=None,
                                                            poll_attempts=1,
                                                            poll_interval_seconds=0,
                                                            dacon_competition_id="236716",
                                                            dacon_team_name="뚜로",
                                                        )

        self.assertEqual("completed", result["status"])
        self.assertEqual("skipped_daily_limit_known", result["submission_run"]["status"])
        self.assertEqual(["execute", "collect", "consistency", "analyze", "artifacts"], events)

    def test_auto_submits_when_limit_known_but_auto_submit_is_enabled(self):
        profile = {
            "project_root": "C:/workspace",
            "objective": "maximize",
            "artifacts": {"submission": ["outputs/submission.csv"]},
        }
        with patch.object(generic_workspace_auto_loop, "validate_execution_profile", return_value={"status": "ready"}):
            with patch.object(generic_workspace_auto_loop, "load_execution_profile", return_value=profile):
                with patch.object(
                    generic_workspace_auto_loop, "run_workspace_pipeline", return_value={"status": "completed"}
                ):
                    with patch.object(
                        generic_workspace_auto_loop, "collect_workspace_metrics",
                        return_value={"status": "collected", "competition": "demo", "cv_score": 0.591},
                    ):
                        with patch.object(
                            generic_workspace_auto_loop, "reconcile_trial_execution_metadata",
                            return_value={"status": "ready"},
                        ):
                            with patch.object(
                                generic_workspace_auto_loop, "process_workspace_result",
                                return_value={"status": "completed"},
                            ):
                                with patch.object(
                                    generic_workspace_auto_loop, "organize_trial_artifacts",
                                    return_value={"status": "completed"},
                                ):
                                    with patch.object(
                                        generic_workspace_auto_loop, "submit_trial",
                                        return_value={"status": "submitted", "submitted_lb_score": None},
                                    ) as submit_mock:
                                        with patch.object(
                                            generic_workspace_auto_loop, "submission_artifact_path",
                                            return_value="C:/workspace/outputs/submission.csv",
                                        ):
                                            with patch.object(
                                                generic_workspace_auto_loop, "check_dacon_submission_limit",
                                                return_value={"daily_submission_limit": 5}
                                            ):
                                                with patch.object(
                                                    generic_workspace_auto_loop, "dacon_auto_submit_allowed", return_value=True
                                                ):
                                                    with patch.object(generic_workspace_auto_loop, "write_loop_trial_result"):
                                                        with patch.object(generic_workspace_auto_loop, "save_loop_state"):
                                                            result = generic_workspace_auto_loop.run_one_trial(
                                                                "demo",
                                                                "trial_004",
                                                                submit=True,
                                                                kaggle_slug=None,
                                                                poll_attempts=1,
                                                                poll_interval_seconds=0,
                                                                dacon_competition_id="236716",
                                                                dacon_team_name="뚜로",
                                                            )

        submit_mock.assert_called_once()
        self.assertEqual("completed", result["status"])
        self.assertEqual("submitted", result["submission_run"]["status"])

    def test_recovers_completed_execution_without_calling_code_writer_again(self):
        recovered = {
            "workspace_run": {"status": "completed"},
            "metrics_collection": {
                "status": "collected",
                "competition": "demo",
                "cv_score": 0.81,
            },
        }
        profile = {
            "project_root": "C:/workspace",
            "objective": "maximize",
            "artifacts": {"submission": ["outputs/submission.csv"]},
        }
        with patch.object(generic_workspace_auto_loop, "save_loop_state"):
            with patch.object(
                generic_workspace_auto_loop,
                "validate_execution_profile",
                return_value={"status": "ready"},
            ):
                with patch.object(generic_workspace_auto_loop, "load_execution_profile", return_value=profile):
                    with patch.object(generic_workspace_auto_loop, "_recover_completed_execution", return_value=recovered):
                        with patch.object(generic_workspace_auto_loop, "run_code_writer_trial") as writer:
                            with patch.object(
                                generic_workspace_auto_loop,
                                "reconcile_trial_execution_metadata",
                                return_value={"status": "ready"},
                            ):
                                with patch.object(
                                    generic_workspace_auto_loop,
                                    "process_workspace_result",
                                    return_value={"status": "completed"},
                                ):
                                    with patch.object(
                                        generic_workspace_auto_loop,
                                        "organize_trial_artifacts",
                                        return_value={"status": "completed"},
                                    ):
                                        with patch.object(generic_workspace_auto_loop, "write_loop_trial_result"):
                                            result = generic_workspace_auto_loop.run_one_trial(
                                                "demo",
                                                "trial_001",
                                                submit=False,
                                                kaggle_slug=None,
                                                poll_attempts=1,
                                                poll_interval_seconds=0,
                                                code_writer=True,
                                            )

        self.assertEqual("completed", result["status"])
        self.assertEqual("resumed", result["code_writer"]["status"])
        writer.assert_not_called()

    def test_recovery_reads_artifacts_after_they_are_organized_under_internal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            internal = root / "experiments" / "demo" / "trial_005" / "internal"
            internal.mkdir(parents=True)
            (internal / "workspace_coding_result_validation.json").write_text(
                json.dumps({"status": "accepted"}),
                encoding="utf-8",
            )
            (internal / "workspace_run.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            (internal / "metrics_collection.json").write_text(
                json.dumps({"status": "collected", "cv_score": 0.33}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                recovered = generic_workspace_auto_loop._recover_completed_execution(
                    "demo",
                    "trial_005",
                )

        self.assertIsNotNone(recovered)
        self.assertEqual("completed", recovered["workspace_run"]["status"])
        self.assertEqual(0.33, recovered["metrics_collection"]["cv_score"])

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
            with patch("research_agent.paths.ROOT", root):
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
            with patch("research_agent.paths.ROOT", root):
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
            self.assertEqual("execution_completed", result["after_coding"]["status"])

    def test_run_code_writer_generates_scoring_harness_after_a_successful_code_writer_result(self):
        with patch.object(
            generic_workspace_auto_loop, "generate_scoring_harness", return_value={"status": "already_exists"}
        ) as harness_gen:
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value={"status": "ready"}
            ):
                with patch.object(
                    generic_workspace_auto_loop,
                    "run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["predict_step.py"]},
                ):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "run_workspace_after_coding",
                        return_value={
                            "status": "execution_completed",
                            "workspace_run": {"status": "completed"},
                            "metrics_collection": {"status": "collected"},
                        },
                    ):
                        generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_002",
                            model="gpt-5",
                            provider="openai",
                            allow_api=True,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        self.assertEqual(1, harness_gen.call_count)
        self.assertEqual("demo", harness_gen.call_args.args[0])

    def test_run_code_writer_continues_when_harness_generation_is_blocked(self):
        # Ordering fix: harness generation is best-effort and must never
        # interrupt the trial's own progress -- a locked one-time asset
        # failing to generate is not the trial's fault, and the pipeline
        # already degrades gracefully (no "score" stage) when the harness
        # doesn't exist yet.
        with patch.object(
            generic_workspace_auto_loop,
            "generate_scoring_harness",
            return_value={"status": "blocked", "reason": "predict_interface_not_ready"},
        ):
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value={"status": "ready"}
            ):
                with patch.object(
                    generic_workspace_auto_loop,
                    "run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["predict_step.py"]},
                ):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "run_workspace_after_coding",
                        return_value={
                            "status": "execution_completed",
                            "workspace_run": {"status": "completed"},
                            "metrics_collection": {"status": "collected"},
                        },
                    ):
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

    def test_run_code_writer_skips_harness_generation_without_api_access(self):
        with patch.object(generic_workspace_auto_loop, "generate_scoring_harness") as harness_gen:
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value={"status": "ready"}
            ):
                with patch.object(
                    generic_workspace_auto_loop,
                    "run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["predict_step.py"]},
                ):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "run_workspace_after_coding",
                        return_value={
                            "status": "execution_completed",
                            "workspace_run": {"status": "completed"},
                            "metrics_collection": {"status": "collected"},
                        },
                    ):
                        generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_002",
                            model="gpt-5",
                            provider="openai",
                            allow_api=False,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        harness_gen.assert_not_called()

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
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
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

    def test_run_code_writer_retries_with_corrective_feedback_after_guardrail_block(self):
        # Real incident: the code writer skipped local validation
        # ("validation_method": "none"), got mechanically blocked, and on
        # retry produced the *exact same* rejected code -- because nothing
        # ever told it which check it had tripped. The retry must carry the
        # specific rejected issues into the next handoff, not just more base
        # code context (that's the missing-snapshot retry path, which is
        # different and must not be conflated with this one).
        handoffs = [
            {"status": "ready", "snapshot_mode": "standard"},
            {"status": "ready", "snapshot_mode": "standard"},
        ]
        code_writer_results = [
            {
                "status": "blocked",
                "blocking_issues": [],
                "issues": ["local_validation_not_computed:validation_method_none"],
                "changed_files": ["predict_step.py"],
            },
            {"status": "accepted", "changed_files": ["predict_step.py", "test_step.py"]},
        ]
        after_coding = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
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
        self.assertEqual(2, writer.call_count)
        second_call_kwargs = handoff.call_args_list[1].kwargs
        self.assertEqual("code_writer_blocked_review_feedback", second_call_kwargs["retry_reason"])
        self.assertEqual(
            ["local_validation_not_computed:validation_method_none"],
            second_call_kwargs["coding_feedback"]["rejected_issues"],
        )
        self.assertEqual(["predict_step.py"], second_call_kwargs["coding_feedback"]["changed_files"])

    def test_run_code_writer_marks_feedback_ignored_when_retry_repeats_same_violation(self):
        # A retry that gets told exactly what it did wrong but does the same
        # thing again is a different situation than a first-time block: it
        # should be tagged so a later cycle doesn't just retry the same way.
        handoffs = [
            {"status": "ready", "snapshot_mode": "standard"},
            {"status": "ready", "snapshot_mode": "standard"},
        ]
        code_writer_results = [
            {
                "status": "blocked",
                "blocking_issues": ["forbidden_path_touched:scoring_harness.py"],
                "issues": [],
                "changed_files": ["scoring_harness.py"],
            },
            {
                "status": "blocked",
                "blocking_issues": ["forbidden_path_touched:scoring_harness.py"],
                "issues": [],
                "changed_files": ["scoring_harness.py"],
            },
        ]

        with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", side_effect=handoffs):
            with patch.object(
                generic_workspace_auto_loop, "run_workspace_code_writer", side_effect=code_writer_results
            ):
                with patch.object(
                    generic_workspace_auto_loop, "_previous_cycle_ignored_feedback_source_trial", return_value=None
                ):
                    with patch.object(generic_workspace_auto_loop, "log_decision") as log_decision:
                        result = generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_002",
                            model="gpt-5",
                            provider="openai",
                            allow_api=True,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        self.assertTrue(result.get("feedback_ignored"))
        self.assertEqual(1, log_decision.call_count)
        self.assertEqual("code_writer_feedback_ignored", log_decision.call_args.kwargs["decision_type"])

    def test_run_code_writer_forces_replan_when_previous_cycle_ignored_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "workspace_coding_handoff.json").write_text(
                json.dumps(
                    {
                        "source_trial_id": "trial_001",
                        "coding_feedback": {
                            "rejected_issues": ["forbidden_path_touched:scoring_harness.py"],
                            "changed_files": ["scoring_harness.py"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "blocking_issues": ["forbidden_path_touched:scoring_harness.py"],
                    }
                ),
                encoding="utf-8",
            )
            handoff = {"status": "ready", "snapshot_mode": "standard"}
            code_writer_result = {"status": "accepted", "changed_files": ["train_step.py"]}
            after_coding = {
                "status": "execution_completed",
                "workspace_run": {"status": "completed"},
                "metrics_collection": {"status": "collected"},
            }

            with patch("research_agent.paths.project_root", return_value=root):
                with patch.object(
                    generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value=handoff
                ):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "prepare_workspace_trial_plan",
                        return_value={"status": "planned"},
                    ) as replanner:
                        with patch.object(
                            generic_workspace_auto_loop,
                            "run_workspace_code_writer",
                            return_value=code_writer_result,
                        ) as writer:
                            with patch.object(
                                generic_workspace_auto_loop, "run_workspace_after_coding", return_value=after_coding
                            ):
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
        self.assertEqual(1, replanner.call_count)
        self.assertEqual("trial_001", replanner.call_args.kwargs["source_trial_id"])
        self.assertEqual(1, writer.call_count)

    def test_run_code_writer_replans_instead_of_retrying_when_handoff_finds_plan_code_mismatch(self):
        # Regression for trial_027: the plan assumed a model-family change
        # from an earlier, unaccepted attempt at the same axis was already
        # in the base trial's code. Showing more code context (the ordinary
        # retry path) cannot fix a wrong plan -- the plan itself needs to be
        # regenerated.
        handoffs = [
            {
                "status": "blocked",
                "blocking_issues": ["plan_find_target_missing_in_base_code:train_step.py"],
                "source_trial_id": "trial_026",
            },
            {"status": "ready", "snapshot_mode": "standard"},
        ]
        code_writer_result = {"status": "accepted", "changed_files": ["train_step.py"]}
        after_coding = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
        }

        with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", side_effect=handoffs):
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_trial_plan", return_value={"status": "planned"}
            ) as replanner:
                with patch.object(
                    generic_workspace_auto_loop, "run_workspace_code_writer", return_value=code_writer_result
                ) as writer:
                    with patch.object(generic_workspace_auto_loop, "run_workspace_after_coding", return_value=after_coding):
                        result = generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_027",
                            model="gpt-5",
                            provider="openai",
                            allow_api=True,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        self.assertEqual("completed", result["status"])
        self.assertEqual(1, replanner.call_count)
        self.assertEqual("trial_027", replanner.call_args.args[1])
        self.assertEqual("trial_026", replanner.call_args.kwargs["source_trial_id"])
        self.assertTrue(replanner.call_args.kwargs["force_replan"])
        self.assertEqual(1, writer.call_count)

    def test_run_code_writer_replans_instead_of_retrying_when_patch_target_not_found(self):
        handoff = {"status": "ready", "snapshot_mode": "standard", "source_trial_id": "trial_026"}
        code_writer_results = [
            {"status": "blocked", "blocking_issues": ["patch_find_not_found:train_step.py"]},
            {"status": "accepted", "changed_files": ["train_step.py"]},
        ]
        after_coding = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
        }

        with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value=handoff):
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_trial_plan", return_value={"status": "planned"}
            ) as replanner:
                with patch.object(
                    generic_workspace_auto_loop, "run_workspace_code_writer", side_effect=code_writer_results
                ) as writer:
                    with patch.object(generic_workspace_auto_loop, "run_workspace_after_coding", return_value=after_coding):
                        result = generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_027",
                            model="gpt-5",
                            provider="openai",
                            allow_api=True,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        self.assertEqual("completed", result["status"])
        self.assertEqual(1, replanner.call_count)
        self.assertTrue(replanner.call_args.kwargs["force_replan"])
        self.assertEqual(2, writer.call_count)

    def test_run_code_writer_replans_when_block_reason_is_a_structured_dict_mismatch(self):
        # Regression: the code writer LLM does not always phrase a plan/code
        # mismatch as one of the known marker strings -- this real example
        # returned a structured dict describing a "Model mismatch between
        # delta plan and authoritative code" instead. The catch-all must
        # still recognize this as a plan problem worth replanning, without
        # needing this exact phrasing enumerated anywhere.
        handoff = {"status": "ready", "snapshot_mode": "standard", "source_trial_id": "trial_026"}
        code_writer_results = [
            {
                "status": "blocked",
                "changed_files": [],
                "blocking_issues": [
                    {
                        "issue": "Model mismatch between delta plan and authoritative code",
                        "details": (
                            "The delta_plan targets train_step.py to set Ridge(alpha=1.0), but the "
                            "authoritative base snapshot for trial_014 shows the pipeline uses "
                            "HistGradientBoostingRegressor with poisson loss and no Ridge step."
                        ),
                        "evidence": "(\"model\", HistGradientBoostingRegressor(loss='poisson', ...))",
                        "request": "Please adjust the delta plan to target the current estimator.",
                    }
                ],
            },
            {"status": "accepted", "changed_files": ["train_step.py"]},
        ]
        after_coding = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
        }

        with patch.object(generic_workspace_auto_loop, "prepare_workspace_coding_handoff", return_value=handoff):
            with patch.object(
                generic_workspace_auto_loop, "prepare_workspace_trial_plan", return_value={"status": "planned"}
            ) as replanner:
                with patch.object(
                    generic_workspace_auto_loop, "run_workspace_code_writer", side_effect=code_writer_results
                ) as writer:
                    with patch.object(generic_workspace_auto_loop, "run_workspace_after_coding", return_value=after_coding):
                        result = generic_workspace_auto_loop.run_code_writer_trial(
                            "demo",
                            "trial_027",
                            model="gpt-5",
                            provider="openai",
                            allow_api=True,
                            trial_llm_calls=None,
                            strategy_calls_today=None,
                        )

        self.assertEqual("completed", result["status"])
        self.assertEqual(1, replanner.call_count)
        self.assertTrue(replanner.call_args.kwargs["force_replan"])
        self.assertEqual(2, writer.call_count)

    def test_should_replan_after_code_writer_mismatch_skips_when_files_were_changed(self):
        result = {
            "status": "code_writer_blocked",
            "code_writer": {"changed_files": ["train_step.py"], "blocking_issues": ["some validation issue"]},
        }
        self.assertFalse(generic_workspace_auto_loop._should_replan_after_code_writer_mismatch(result))

    def test_should_replan_after_code_writer_mismatch_skips_non_recoverable_token_block(self):
        result = {
            "status": "code_writer_blocked",
            "code_writer": {"changed_files": [], "blocking_issues": ["token_policy_blocked"]},
        }
        self.assertFalse(generic_workspace_auto_loop._should_replan_after_code_writer_mismatch(result))

    def test_should_replan_after_code_writer_mismatch_skips_known_missing_context_case(self):
        # Missing-context blocks already have a dedicated, more targeted
        # recovery path (retry the same plan with an expanded snapshot); the
        # catch-all should defer to that instead of also firing.
        result = {
            "status": "code_writer_blocked",
            "code_writer": {"changed_files": [], "blocking_issues": ["missing full code context"]},
        }
        self.assertFalse(generic_workspace_auto_loop._should_replan_after_code_writer_mismatch(result))

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

    def test_code_writer_retry_detects_missing_explicit_target_in_base_snapshot(self):
        result = {
            "status": "code_writer_blocked",
            "code_writer": {
                "blocking_issues": [
                    "The authoritative base snapshot does not include train_step.py."
                ]
            },
        }

        self.assertTrue(generic_workspace_auto_loop._should_retry_code_writer_block(result))

    def test_code_writer_retry_detects_missing_authoritative_source_phrasing(self):
        result = {
            "status": "code_writer_blocked",
            "code_writer": {
                "blocking_issues": [
                    "Missing authoritative source for train_step.py from trial_005 to locate "
                    "train_validate/run_experiment. Exact find/replace text is required for "
                    "patch-only mode."
                ]
            },
        }

        self.assertTrue(generic_workspace_auto_loop._should_retry_code_writer_block(result))

    def test_run_code_writer_repairs_runtime_failure_once_in_same_trial(self):
        handoffs = [
            {"status": "ready", "snapshot_mode": "standard"},
            {"status": "ready", "snapshot_mode": "expanded_runtime_repair"},
        ]
        failed = {
            "status": "workspace_run_failed",
            "issues": ["workspace_run_not_completed:failed"],
            "workspace_run": {
                "status": "failed",
                "failure": {
                    "failure_type": "artifact_serialization",
                    "matched_pattern": "is not JSON serializable",
                },
                "command_results": [
                    {
                        "stage": "train",
                        "command": "python train_step.py",
                        "returncode": 1,
                        "log_path": "",
                    }
                ],
            },
        }
        completed = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
        }

        with patch.object(
            generic_workspace_auto_loop,
            "prepare_workspace_coding_handoff",
            side_effect=handoffs,
        ) as handoff:
            with patch.object(
                generic_workspace_auto_loop,
                "run_workspace_code_writer",
                side_effect=[
                    {"status": "accepted", "changed_files": ["train_step.py"]},
                    {"status": "accepted", "changed_files": ["train_step.py"]},
                ],
            ) as writer:
                with patch.object(
                    generic_workspace_auto_loop,
                    "run_workspace_after_coding",
                    side_effect=[failed, completed],
                ):
                    result = generic_workspace_auto_loop.run_code_writer_trial(
                        "demo",
                        "trial_005",
                        model="gpt-5",
                        provider="openai",
                        allow_api=True,
                        trial_llm_calls=None,
                        strategy_calls_today=None,
                    )

        self.assertEqual("completed", result["status"])
        self.assertEqual(2, writer.call_count)
        repair_call = handoff.call_args_list[1]
        self.assertEqual("workspace_runtime_failure", repair_call.kwargs["retry_reason"])
        self.assertEqual(
            "artifact_serialization",
            repair_call.kwargs["runtime_failure_context"]["failure_type"],
        )
        self.assertEqual(2, result["code_writer_attempt"])

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

    def test_run_loop_starts_fresh_thread_when_saved_thread_already_finished(self):
        # Regression: after a run failed, the state kept
        # resume_from_status="running" plus that run's graph_thread_id. Every
        # relaunch then "resumed" a thread whose graph had already reached
        # END, so LangGraph replayed the terminal result without executing a
        # single node -- no work, no state write, instant exit, and the
        # launcher's running state reconciled into "process_not_running".
        # The loop could never move forward again.
        def make_args():
            return argparse.Namespace(
                competition="demo",
                start_trial="trial_028",
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
            patches = lambda: (  # noqa: E731
                patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime),
                patch.object(generic_workspace_auto_loop, "STATE_PATH", state_path),
                patch.object(generic_workspace_auto_loop, "LOCK_PATH", runtime / "auto_loop.lock"),
                patch.object(generic_workspace_auto_loop, "PAUSE_REQUEST_PATH", runtime / "pause.request"),
                patch.object(generic_workspace_auto_loop, "has_coding_plan", return_value=True),
                patch.object(generic_workspace_auto_loop, "sync_state_db"),
            )

            with contextlib.ExitStack() as stack:
                for item in patches():
                    stack.enter_context(item)
                stack.enter_context(
                    patch.object(
                        generic_workspace_auto_loop,
                        "run_one_trial",
                        return_value={"status": "code_writer_blocked"},
                    )
                )
                first = generic_workspace_auto_loop.run_loop(make_args())

            self.assertEqual("failed", first["status"])
            first_thread = json.loads(state_path.read_text(encoding="utf-8"))["graph_thread_id"]

            # The loop lock is normally released by atexit when the launcher
            # process ends; both runs share one process here, so release it
            # explicitly to model two separate launches.
            with patch.object(generic_workspace_auto_loop, "LOCK_PATH", runtime / "auto_loop.lock"):
                generic_workspace_auto_loop.release_loop_lock()

            # Reproduce the stuck state the dead-process reconciler leaves behind.
            stuck = json.loads(state_path.read_text(encoding="utf-8"))
            stuck["resume_from_status"] = "running"
            state_path.write_text(json.dumps(stuck), encoding="utf-8")

            with contextlib.ExitStack() as stack:
                for item in patches():
                    stack.enter_context(item)
                runner = stack.enter_context(
                    patch.object(
                        generic_workspace_auto_loop,
                        "run_one_trial",
                        return_value={"status": "completed"},
                    )
                )
                stack.enter_context(
                    patch.object(
                        generic_workspace_auto_loop,
                        "plan_next_workspace_trial",
                        return_value={"status": "planned"},
                    )
                )
                second = generic_workspace_auto_loop.run_loop(make_args())

            second_thread = json.loads(state_path.read_text(encoding="utf-8"))["graph_thread_id"]

        # The second launch must do real work on a fresh thread, not replay
        # the finished one.
        self.assertEqual(1, runner.call_count)
        self.assertEqual("completed", second["status"])
        self.assertNotEqual(first_thread, second_thread)

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
        # prepare_workspace_trial_plan is now called for both the first trial
        # (no source) and again for trial_002 (via plan_following_trial, which
        # always asks the LLM-driven planner for a concrete plan even without an
        # active user insight).
        self.assertEqual(2, initial_planner.call_count)
        self.assertEqual("trial_001", initial_planner.call_args_list[0].args[1])
        self.assertEqual("trial_002", initial_planner.call_args_list[1].args[1])
        self.assertEqual(
            "trial_001", initial_planner.call_args_list[1].kwargs["source_trial_id"]
        )
        self.assertIsNone(initial_planner.call_args_list[1].kwargs["user_insight_override"])
        self.assertTrue(initial_planner.call_args_list[1].kwargs["force_replan"])
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

    def test_plan_following_trial_asks_llm_planner_even_without_active_insight(self):
        args = argparse.Namespace(
            competition="demo",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "plan_next_workspace_trial",
                        return_value={
                            "status": "planned",
                            "next_experiment": {"user_insight_override": None},
                        },
                    ):
                        with patch.object(
                            generic_workspace_auto_loop,
                            "prepare_workspace_trial_plan",
                            return_value={"status": "planned", "trial_id": "trial_009"},
                        ) as planner:
                            result = generic_workspace_auto_loop.plan_following_trial(
                                args, "trial_008", "trial_009"
                            )

        self.assertEqual("planned", result["status"])
        planner.assert_called_once_with(
            "demo",
            "trial_009",
            source_trial_id="trial_008",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
            user_insight_override=None,
            force_replan=True,
        )

    def test_plan_following_trial_falls_back_to_rule_based_plan_when_llm_blocked(self):
        # If the LLM-driven planner is unavailable (budget/API), the rule-based
        # plan that plan_next_workspace_trial already wrote must remain the final
        # result instead of the whole planning step failing.
        args = argparse.Namespace(
            competition="demo",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        rule_based_result = {
            "status": "planned",
            "trial_id": "trial_009",
            "next_experiment": {"user_insight_override": None, "strategy": "hyperparameter_tuning"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "plan_next_workspace_trial",
                        return_value=rule_based_result,
                    ):
                        with patch.object(
                            generic_workspace_auto_loop,
                            "prepare_workspace_trial_plan",
                            return_value={"status": "blocked_planning", "issues": ["token_policy_blocked"]},
                        ):
                            result = generic_workspace_auto_loop.plan_following_trial(
                                args, "trial_008", "trial_009"
                            )

        self.assertEqual(rule_based_result, result)

    def test_plan_following_trial_surfaces_duplicate_candidate_block_instead_of_rule_based_fallback(self):
        # Unlike a budget/API block, duplicate_candidate_blocked means the LLM
        # planner actually produced a plan -- one that repeats a candidate
        # already rejected for the same axis, even after one forced retry.
        # Falling back to the rule-based plan here would silently bypass the
        # duplicate check (the rule-based planner is never checked against
        # rejected_candidates_by_axis), so this must be surfaced as a block.
        args = argparse.Namespace(
            competition="demo",
            model="gpt-5",
            provider="openai",
            allow_api=True,
            trial_llm_calls=None,
            strategy_calls_today=None,
        )
        rule_based_result = {
            "status": "planned",
            "trial_id": "trial_009",
            "next_experiment": {"user_insight_override": None, "strategy": "validation_review"},
        }
        duplicate_blocked_result = {
            "status": "duplicate_candidate_blocked",
            "issues": ["Repeated already-rejected candidate: validation_review: method=time_series_cv"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            with patch.object(generic_workspace_auto_loop, "RUNTIME_DIR", runtime):
                with patch.object(generic_workspace_auto_loop, "STATE_PATH", runtime / "auto_loop_state.json"):
                    with patch.object(
                        generic_workspace_auto_loop,
                        "plan_next_workspace_trial",
                        return_value=rule_based_result,
                    ):
                        with patch.object(
                            generic_workspace_auto_loop,
                            "prepare_workspace_trial_plan",
                            return_value=duplicate_blocked_result,
                        ):
                            result = generic_workspace_auto_loop.plan_following_trial(
                                args, "trial_008", "trial_009"
                            )

        self.assertEqual(duplicate_blocked_result, result)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
