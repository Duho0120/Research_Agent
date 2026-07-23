import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kaggle_research_agent import cli_app
from kaggle_research_agent.state_db import (
    initialize_state_db,
    upsert_competition,
    upsert_trial,
    upsert_trial_artifact,
    upsert_trial_decision,
    upsert_trial_score,
)


class CliAppTest(unittest.TestCase):
    def test_snapshot_uses_manual_submission_scores_and_infers_next_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            self._write_metrics(root, "trial_001", 0.80, 0.76)
            self._write_metrics(root, "trial_002", 0.81, 0.77)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        snapshot = cli_app.experiment_snapshot("titanic", sync=False)

            self.assertEqual("trial_002", snapshot["last_completed_trial"])
            self.assertEqual("trial_003", snapshot["next_trial"])
            self.assertEqual("대기 중", snapshot["state"])
            self.assertEqual(0.77, snapshot["latest"]["lb_score"])
            self.assertEqual("trial_002", snapshot["best"]["trial_id"])

    def test_snapshot_prefers_database_next_trial_over_manual_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            for index in range(1, 6):
                self._write_metrics(root, f"trial_{index:03d}", 0.80, 0.76)
            response = {
                "ok": True,
                "data": {
                    "experiment": {
                        "competition": "titanic",
                        "topic": "Titanic",
                        "state": "completed",
                        "next_trial_id": "trial_007",
                    },
                    "trials": [
                        {"trial_id": "trial_006", "status": "completed", "local_score": 0.82, "lb_score": 0.76076}
                    ],
                },
            }
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
                        snapshot = cli_app.experiment_snapshot("titanic", sync=False)

            self.assertEqual("trial_007", snapshot["next_trial"])
            self.assertEqual("trial_006", snapshot["last_completed_trial"])
            self.assertEqual("trial_006", snapshot["latest"]["trial_id"])

    def test_snapshot_renders_progress_status_and_recent_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            log = runtime / "auto_loop.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("\n".join(["계획서 생성 중...", "코드 실행 중...", "Kaggle 제출 중..."]), encoding="utf-8")
            snapshot = {
                "competition": "titanic",
                "topic": "Titanic",
                "state": "실행 중",
                "current_trial": "trial_006",
                "next_trial": "trial_006",
                "latest": {},
                "best": {},
                "pending_request_count": 0,
                "loop": {"status": "running", "current_trial": "trial_006", "log_path": str(log)},
            }

            rendered = cli_app.render_snapshot(snapshot)

            self.assertIn("진행 상태:", rendered)
            self.assertIn("trial_006 진행 중", rendered)
            self.assertIn("최근 로그:", rendered)
            self.assertIn("코드 실행 중...", rendered)

    def test_snapshot_summarizes_structured_recent_log_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            log = runtime / "auto_loop.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                "\n".join(
                    [
                        "=== titanic / trial_006 ===",
                        "{",
                        '  "competition": "titanic",',
                        '  "status": "completed",',
                        '  "trials": [',
                        "    {",
                        '      "trial_id": "trial_006",',
                        '      "status": "completed"',
                        "    }",
                        "  ]",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )
            snapshot = {
                "competition": "titanic",
                "topic": "Titanic",
                "state": "?湲?以?",
                "current_trial": None,
                "next_trial": "trial_007",
                "latest": {},
                "best": {},
                "pending_request_count": 0,
                "loop": {"status": "completed", "last_completed_trial": "trial_006", "log_path": str(log)},
            }

            rendered = cli_app.render_snapshot(snapshot)

            self.assertIn("titanic / trial_006", rendered)
            self.assertIn("loop status: completed", rendered)
            self.assertIn("trial_006: completed", rendered)
            self.assertNotIn("trials: [", rendered)
            self.assertNotIn("trial_id: trial_006", rendered)

    def test_render_snapshot_shows_next_base_trial(self):
        snapshot = {
            "competition": "titanic",
            "topic": "Titanic",
            "state": "대기 중",
            "next_trial": "trial_007",
            "last_completed_trial": "trial_006",
            "latest": {"trial_id": "trial_006", "local_score": 0.80, "lb_score": 0.76},
            "best": {"trial_id": "trial_003", "local_score": 0.85, "lb_score": 0.77},
            "pending_request_count": 0,
        }

        rendered = cli_app.render_snapshot(snapshot)

        self.assertIn("다음 실험 기준 base: trial_003 (제출 점수 기준 베스트)", rendered)

    def test_start_refuses_duplicate_running_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": tmp}):
                cli_app.save_json_atomic(
                    cli_app.loop_state_path(),
                    {"competition": "titanic", "status": "running", "current_trial": "trial_004"},
                )
                with patch("kaggle_research_agent.cli_app.subprocess.Popen") as popen:
                    message = cli_app.start_experiment("titanic")
            self.assertIn("이미 자동 실험이 실행 중입니다.", message)
            self.assertIn("trial_004", message)
            popen.assert_not_called()

    def test_start_uses_first_trial_without_recorded_lb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            self._write_metrics(root, "trial_001", 0.80, 0.76)
            self._write_metrics(root, "trial_002", 0.81, 0.77)
            self._write_execution_profile(root, "titanic")
            process = Mock(pid=321)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                            message = cli_app.start_experiment("titanic")
                state = cli_app.load_json(cli_app.loop_state_path())
            command = popen.call_args.args[0]
            self.assertIn("generic_workspace_auto_loop.py", " ".join(command))
            self.assertNotIn("titanic_auto_submit_loop.py", " ".join(command))
            self.assertIn("trial_003", command)
            self.assertEqual("starting", state["status"])
            self.assertEqual(321, state["pid"])
            self.assertIn("trial_003", message)

    def test_start_dialog_uses_requested_trial_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            self._write_metrics(root, "trial_001", 0.80, 0.76)
            self._write_metrics(root, "trial_002", 0.81, 0.77)
            self._write_execution_profile(root, "titanic")
            process = Mock(pid=321)
            output = []
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                            message = cli_app._start_experiment_dialog(
                                "titanic",
                                lambda _: "1",
                                output.append,
                            )

            command = popen.call_args.args[0]
            self.assertIn("generic_workspace_auto_loop.py", " ".join(command))
            self.assertNotIn("titanic_auto_submit_loop.py", " ".join(command))
            self.assertIn("--start-trial", command)
            self.assertIn("trial_003", command)
            self.assertIn("--max-trials", command)
            self.assertEqual("1", command[command.index("--max-trials") + 1])
            self.assertIn("실험을 시작하겠습니다", "\n".join(output))
            self.assertIn("실행 범위: 1회", message)

    def test_start_dialog_can_cancel_or_continue(self):
        self.assertIn(
            "취소",
            cli_app._start_experiment_dialog("titanic", lambda _: "q", lambda _: None),
        )
        with patch("kaggle_research_agent.cli_app.start_experiment", return_value="continued") as start:
            message = cli_app._start_experiment_dialog("titanic", lambda _: "c", lambda _: None)
        self.assertEqual("continued", message)
        start.assert_called_once_with("titanic", continuous=True)

    def test_titanic_after_manual_trials_continues_with_generic_workspace_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            workspace = root / "demo_workspaces" / "titanic"
            workspace.mkdir(parents=True)
            competition_dir = root / "competitions" / "titanic"
            competition_dir.mkdir(parents=True)
            (competition_dir / "execution_profile.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: '1.0'",
                        "competition: titanic",
                        "platform: kaggle",
                        f"project_root: {workspace}",
                        f"python: {Path(sys.executable)}",
                        "commands:",
                        "  test:",
                        "    - '{python} test_step.py'",
                        "  train:",
                        "    - '{python} train_step.py'",
                        "artifacts:",
                        "  metrics:",
                        "    - outputs/metrics.json",
                        "  submission:",
                        "    - outputs/submission.csv",
                        "write_scope:",
                        "  allowed:",
                        "    - src/",
                        "  forbidden:",
                        "    - data/",
                    ]
                ),
                encoding="utf-8",
            )
            for index in range(1, 6):
                self._write_metrics(root, f"trial_{index:03d}", 0.8, 0.76)
            process = Mock(pid=987)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.paths.ROOT", root):
                    with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                        with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                            snapshot = cli_app.experiment_snapshot("titanic", sync=False)
                            with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                                message = cli_app.start_experiment("titanic", trial_count=1)

            command = popen.call_args.args[0]
            self.assertEqual("trial_006", snapshot["next_trial"])
            self.assertIn("generic_workspace_auto_loop.py", " ".join(command))
            self.assertIn("--start-trial", command)
            self.assertEqual("trial_006", command[command.index("--start-trial") + 1])
            self.assertIn("--max-trials", command)
            self.assertEqual("1", command[command.index("--max-trials") + 1])
            self.assertIn("--submit", command)
            self.assertIn("trial_006", message)

    def test_start_non_titanic_uses_generic_workspace_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            workspace = root / "demo_workspaces" / "demo"
            workspace.mkdir(parents=True)
            competition_dir = root / "competitions" / "demo"
            competition_dir.mkdir(parents=True)
            (competition_dir / "execution_profile.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: '1.0'",
                        "competition: demo",
                        "platform: kaggle",
                        f"project_root: {workspace}",
                        f"python: {Path(sys.executable)}",
                        "commands:",
                        "  test:",
                        "    - '{python} test_step.py'",
                        "  train:",
                        "    - '{python} train_step.py'",
                        "artifacts:",
                        "  metrics:",
                        "    - outputs/metrics.json",
                        "  submission:",
                        "    - outputs/submission.csv",
                        "write_scope:",
                        "  allowed:",
                        "    - src/",
                        "  forbidden:",
                        "    - data/",
                    ]
                ),
                encoding="utf-8",
            )
            process = Mock(pid=654)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.paths.ROOT", root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                            message = cli_app.start_experiment("demo")
                state = cli_app.load_json(cli_app.loop_state_path())

            command = popen.call_args.args[0]
            env = popen.call_args.kwargs["env"]
            self.assertIn("generic_workspace_auto_loop.py", " ".join(command))
            self.assertIn("--code-writer", command)
            self.assertIn("--allow-api", command)
            self.assertIn("--submit", command)
            self.assertEqual(str(runtime), env["RESEARCH_AGENT_RUNTIME_DIR"])
            self.assertEqual("starting", state["status"])
            self.assertEqual(654, state["pid"])
            self.assertIn("trial_001", message)

    def test_stop_request_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": tmp}):
                cli_app.save_json_atomic(cli_app.loop_state_path(), {"competition": "titanic", "status": "running"})
                first = cli_app.request_experiment_stop("titanic")
                second = cli_app.request_experiment_stop("titanic")
                state = cli_app.load_json(cli_app.loop_state_path())
            self.assertIn("중단을 요청", first)
            self.assertIn("이미 중단 대기 중", second)
            self.assertTrue(state["pause_requested"])

    def test_menu_can_exit_without_external_services(self):
        answers = iter(["11"])
        output = []
        with patch("kaggle_research_agent.cli_app.load_experiments", return_value=[]):
            with patch(
                "kaggle_research_agent.cli_app.experiment_snapshot",
                return_value={"competition": "titanic", "topic": "Titanic", "state": "대기 중", "latest": {}, "best": {}},
            ):
                code = cli_app.run_menu(sync_on_start=False, input_fn=lambda _: next(answers), output=output.append)
        self.assertEqual(0, code)
        self.assertTrue(any("종료합니다" in line for line in output))

    def test_open_paths_dialog_opens_user_view_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_view = root / "experiments" / "demo" / "trial_001" / "user_view"
            user_view.mkdir(parents=True)
            snapshot = {"last_completed_trial": "trial_001"}
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                answers = iter(["1", "5", "q"])
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(user_view.resolve())
        self.assertIn("사용자용 산출물", message)

    def test_open_paths_dialog_opens_best_trial_user_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest_view = root / "experiments" / "demo" / "trial_006" / "user_view"
            best_view = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_003" / "user_view"
            latest_view.mkdir(parents=True)
            best_view.mkdir(parents=True)
            snapshot = {
                "last_completed_trial": "trial_006",
                "best": {"trial_id": "trial_003"},
            }
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                answers = iter(["1", "4", "q"])
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(best_view.resolve())
        self.assertIn("베스트 trial 사용자용 산출물", message)

    def test_user_artifact_summary_previews_core_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_view = root / "experiments" / "demo" / "trial_003" / "user_view"
            user_view.mkdir(parents=True)
            plan = user_view / "01_plan.ko.md"
            pipeline = user_view / "02_pipeline_structure.ko.md"
            scores = user_view / "03_scores.ko.md"
            plan.write_text("# Plan\nUse title feature and family size.", encoding="utf-8")
            pipeline.write_text("# Pipeline\nload -> feature -> model -> submit", encoding="utf-8")
            scores.write_text("# Scores\nlocal 0.85475\nsubmit 0.77272", encoding="utf-8")
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    db_path = initialize_state_db()
                    upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                    upsert_trial(
                        {
                            "competition_id": "demo",
                            "trial_id": "trial_003",
                            "status": "completed",
                            "plan_type": "manual_import",
                            "primary_change_axis": "feature_engineering_name_title",
                            "source_trial_id": "trial_001",
                        },
                        db_path,
                    )
                    upsert_trial_score(
                        {
                            "competition_id": "demo",
                            "trial_id": "trial_003",
                            "local_score": 0.8547486033519553,
                            "lb_score": 0.77272,
                            "is_best_lb": True,
                        },
                        db_path,
                    )
                    for artifact_type, path in [
                        ("plan_ko", plan),
                        ("pipeline_structure_ko", pipeline),
                        ("scores_ko", scores),
                    ]:
                        upsert_trial_artifact(
                            {
                                "competition_id": "demo",
                                "trial_id": "trial_003",
                                "artifact_type": artifact_type,
                                "path": str(path),
                                "is_user_facing": True,
                            },
                            db_path,
                        )

                    rendered = cli_app.render_user_artifact_summary("demo", "trial_003", label="베스트 trial")

        self.assertIn("trial_003 사용자용 산출물 요약", rendered)
        self.assertIn("로컬 점수: 0.85475", rendered)
        self.assertIn("제출 점수: 0.77272", rendered)
        self.assertIn("개선축: feature_engineering_name_title", rendered)
        self.assertIn("base trial: trial_001", rendered)
        self.assertIn("판단: BEST", rendered)
        self.assertIn("Use title feature and family size.", rendered)
        self.assertIn("load -> feature -> model -> submit", rendered)
        self.assertIn("submit 0.77272", rendered)

    def test_open_paths_dialog_keeps_submenu_after_folder_open_until_q(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "demo_workspaces" / "demo"
            workspace.mkdir(parents=True)
            snapshot = {"last_completed_trial": "trial_001"}
            answers = iter(["2", "", "q"])
            output = []
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder"):
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), output.append)

        rendered = "\n".join(output)
        self.assertGreaterEqual(rendered.count("어떤 폴더/DB를 볼까요?"), 2)
        self.assertIn("실행 워크스페이스", message)
        self.assertIn("outputs는 trial별 아카이브가 아니라 최신 작업 결과", message)

    def test_open_paths_dialog_explains_experiment_record_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment_root = root / "experiments" / "demo"
            experiment_root.mkdir(parents=True)
            snapshot = {"last_completed_trial": "trial_001"}
            answers = iter(["3", "q"])
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(experiment_root.resolve())
        self.assertIn("공식 기록 공간", message)
        self.assertIn("사용자용 요약만 보려면", message)

    def test_open_paths_dialog_can_choose_recent_trial_internal_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            internal = root / "experiments" / "demo" / "trial_006" / "internal"
            internal.mkdir(parents=True)
            snapshot = {"last_completed_trial": "trial_006"}
            answers = iter(["4", "2", "q"])
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(internal.resolve())
        self.assertIn("최근 완료 trial 내부 기록", message)

    def test_open_paths_dialog_falls_back_to_trial_root_when_internal_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_root = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_003"
            trial_root.mkdir(parents=True)
            snapshot = {"best": {"trial_id": "trial_003"}}
            answers = iter(["4", "3", "q"])
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(trial_root.resolve())
        self.assertIn("internal 폴더가 없어 trial 기록 루트를 열었습니다", message)

    def test_open_user_artifacts_roots_opens_only_canonical_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs" / "demo"
            experiments = root / "experiments" / "demo"
            manual_trials = root / "demo_workspaces" / "demo" / "manual_trials"
            runs.mkdir(parents=True)
            experiments.mkdir(parents=True)
            manual_trials.mkdir(parents=True)

            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_user_artifacts_roots("demo")

            opener.assert_called_once_with(experiments.resolve())
        self.assertIn(str(experiments.resolve()), message)
        self.assertIn(str(runs.resolve()), message)
        self.assertIn(str(manual_trials.resolve()), message)

    def test_open_paths_dialog_can_open_best_submission_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best_submission = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_003" / "submission.csv"
            best_submission.parent.mkdir(parents=True)
            best_submission.write_text("id,pred\n1,0\n", encoding="utf-8")
            snapshot = {
                "last_completed_trial": "trial_006",
                "best": {"trial_id": "trial_003"},
            }
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                answers = iter(["5", "2", "q"])
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(best_submission.parent.resolve())
        self.assertIn("베스트 trial 제출 파일 폴더", message)

    def test_open_paths_dialog_opens_recent_submission_file_folder_from_run_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = root / "demo_workspaces" / "demo" / "outputs" / "submission.csv"
            submission.parent.mkdir(parents=True)
            submission.write_text("id,pred\n1,0\n", encoding="utf-8")
            run_json = root / "experiments" / "demo" / "trial_006" / "submission_run.json"
            run_json.parent.mkdir(parents=True)
            run_json.write_text(json.dumps({"submission_file": str(submission)}), encoding="utf-8")
            snapshot = {"last_completed_trial": "trial_006"}
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                answers = iter(["5", "1", "q"])
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(submission.parent.resolve())
        self.assertIn("최근 완료 trial 제출 파일 폴더", message)

    def test_submission_folder_dialog_can_open_direct_trial_submission_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = root / "experiments" / "demo" / "trial_004" / "submission.csv"
            submission.parent.mkdir(parents=True)
            submission.write_text("id,pred\n1,0\n", encoding="utf-8")
            snapshot = {}
            answers = iter(["5", "4", "trial_004", "q"])
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(submission.parent.resolve())
        self.assertIn("trial_004 제출 파일 폴더", message)

    def test_submission_folder_dialog_labels_workspace_outputs_as_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "demo_workspaces" / "demo" / "outputs"
            outputs.mkdir(parents=True)
            snapshot = {}
            answers = iter(["5", "3", "q"])
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app._open_folder") as opener:
                    message = cli_app._open_paths_dialog("demo", snapshot, lambda _: next(answers), lambda _: None)

            opener.assert_called_once_with(outputs.resolve())
        self.assertIn("실행 워크스페이스 최신 outputs 폴더", message)
        self.assertIn("trial별 제출 아카이브가 아니라", message)

    def test_open_paths_dialog_can_expand_sqlite_trial_detail_before_returning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()
                upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                upsert_trial(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_001",
                        "status": "completed",
                        "plan_type": "continuation_delta_plan_with_long_name",
                        "primary_change_axis": "feature_engineering_family_structure",
                    },
                    db_path,
                )
                upsert_trial_score(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_001",
                        "local_score": 0.8044692737430168,
                        "lb_score": 0.75,
                        "is_best_lb": True,
                    },
                    db_path,
                )
                answers = iter(["6", "trial_001", "q"])
                output = []
                message = cli_app._open_paths_dialog("demo", {"last_completed_trial": "trial_001"}, lambda _: next(answers), output.append)

        rendered = "\n".join(output)
        self.assertIn("SQLite DB trial 요약", rendered)
        self.assertIn("trial : trial_001", rendered)
        self.assertIn("local : 0.80447", rendered)
        self.assertIn("axis : feature_engineering_family_structure", rendered)
        self.assertIn("plan : continuation_delta_plan_with_long_name", rendered)
        self.assertIn("best : BEST", rendered)
        self.assertEqual("SQLite DB 요약을 표시했습니다.", message)

    def test_render_sqlite_trial_table_joins_trial_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()
                upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                upsert_trial(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_001",
                        "status": "completed",
                        "plan_type": "initial_pipeline_plan",
                        "primary_change_axis": "feature_engineering",
                    },
                    db_path,
                )
                upsert_trial_score(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_001",
                        "metric": "accuracy",
                        "objective": "maximize",
                        "local_score": 0.82,
                        "lb_score": 0.75,
                        "is_best_local": False,
                        "is_best_lb": True,
                    },
                    db_path,
                )
                upsert_trial_decision(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_001",
                        "decision": "accept",
                        "change_axis": "feature_engineering",
                        "axis_attempt_count": 1,
                        "axis_attempt_limit": 3,
                    },
                    db_path,
                )
                rendered = cli_app.render_sqlite_trial_table("demo")

        self.assertIn("SQLite DB trial 요약", rendered)
        self.assertIn("trial", rendered)
        self.assertIn("trial_001", rendered)
        self.assertIn("0.82000", rendered)
        self.assertIn("0.75000", rendered)
        self.assertIn("feature_engineeri…", rendered)
        self.assertIn("Best", rendered)
        self.assertIn("조인", rendered)

    def test_render_sqlite_trial_table_falls_back_to_primary_axis_when_decision_axis_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()
                upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                upsert_trial(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_004",
                        "status": "completed",
                        "plan_type": "reject_or_hold",
                        "primary_change_axis": "model_family_random_forest",
                    },
                    db_path,
                )
                upsert_trial_score(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_004",
                        "local_score": 0.80,
                        "lb_score": 0.76,
                    },
                    db_path,
                )
                upsert_trial_decision(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_004",
                        "decision": "reject_or_hold",
                        "change_axis": "",
                        "axis_attempt_count": 0,
                        "axis_attempt_limit": 3,
                    },
                    db_path,
                )
                rendered = cli_app.render_sqlite_trial_table("demo")

        self.assertIn("model_family_rand…", rendered)
        self.assertNotIn("| -                  | reject_or_hold", rendered)

    def test_render_trial_comparison_table_shows_scores_deltas_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()
                upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                for trial_id, source, local, submit, axis, decision, is_best in [
                    ("trial_001", None, 0.80, 0.76, "baseline", "baseline_established", False),
                    ("trial_002", "trial_001", 0.81, 0.75, "feature_family", "reject_or_hold", False),
                    ("trial_003", "trial_001", 0.85, 0.78, "feature_title", "accept", True),
                ]:
                    upsert_trial(
                        {
                            "competition_id": "demo",
                            "trial_id": trial_id,
                            "status": "completed",
                            "source_trial_id": source,
                            "primary_change_axis": axis,
                        },
                        db_path,
                    )
                    upsert_trial_score(
                        {
                            "competition_id": "demo",
                            "trial_id": trial_id,
                            "local_score": local,
                            "lb_score": submit,
                            "is_best_lb": is_best,
                        },
                        db_path,
                    )
                    upsert_trial_decision(
                        {
                            "competition_id": "demo",
                            "trial_id": trial_id,
                            "decision": decision,
                            "change_axis": axis,
                        },
                        db_path,
                    )

                rendered = cli_app.render_trial_comparison_table("demo")

        self.assertIn("Trial 비교표", rendered)
        self.assertIn("trial_001", rendered)
        self.assertIn("trial_002", rendered)
        self.assertIn("trial_003", rendered)
        self.assertIn("+0.01000", rendered)
        self.assertIn("-0.01000", rendered)
        self.assertIn("+0.03000", rendered)
        self.assertIn("feature_title", rendered)
        self.assertIn("Best", rendered)

    def test_render_sqlite_trial_detail_shows_untruncated_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                db_path = initialize_state_db()
                upsert_competition({"competition_id": "demo", "objective": "maximize"}, db_path)
                upsert_trial(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_002",
                        "status": "completed",
                        "plan_type": "continuation_delta_plan_with_long_name",
                        "primary_change_axis": "feature_engineering_family_structure",
                    },
                    db_path,
                )
                upsert_trial_score(
                    {
                        "competition_id": "demo",
                        "trial_id": "trial_002",
                        "local_score": 0.8044692737430168,
                        "lb_score": None,
                        "is_best_local": True,
                    },
                    db_path,
                )
                detail = cli_app.render_sqlite_trial_detail("demo", "trial_002")

        self.assertIn("trial : trial_002", detail)
        self.assertIn("local : 0.80447", detail)
        self.assertIn("submit : -", detail)
        self.assertIn("axis : feature_engineering_family_structure", detail)
        self.assertIn("plan : continuation_delta_plan_with_long_name", detail)
        self.assertIn("best : -", detail)

    def test_new_experiment_dialog_scaffolds_workspace_and_selects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            answers = iter(
                [
                    "https://www.kaggle.com/competitions/playground-series-s4e1",
                    "제출 점수 기준으로 개선하고 싶어",
                    "",
                    "3",
                    "Survived",
                    "PassengerId",
                    "4",
                    "accuracy",
                    "",
                    "5",
                    "train.csv,test.csv",
                    "",
                ]
            )
            output = []
            prompts = []
            def ask(prompt):
                prompts.append(prompt)
                return next(answers)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.paths.ROOT", root):
                    with patch("kaggle_research_agent.cli_app.sync_state", return_value={"ok": True}):
                        created = cli_app._new_experiment_dialog(ask, output.append)
                selected = cli_app.selected_competition()

            self.assertEqual("URL 또는 실험을 설명해주세요.\n> ", prompts[0])
            self.assertEqual("playground-series-s4e1", created)
            self.assertTrue((root / "competitions" / "playground-series-s4e1" / "workspace_source.json").exists())
            config_path = root / "demo_workspaces" / "playground-series-s4e1" / "workspace_config.json"
            self.assertTrue(config_path.exists())
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual("Survived", config["target_column"])
            self.assertEqual("PassengerId", config["id_column"])
            self.assertEqual("accuracy", config["metric"])
            self.assertEqual(["train.csv", "test.csv"], config["required_data_files"])
            self.assertEqual("playground-series-s4e1", selected)
            self.assertTrue(any("에이전트 분석 결과" in line for line in output))
            self.assertTrue(any("등록하고 선택했습니다" in line for line in output))

    def test_new_experiment_settings_infers_schema_from_project_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "workspace"
            data = project / "data"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("PassengerId,Survived,Fare\n1,1,7.25\n", encoding="utf-8")
            (data / "test.csv").write_text("PassengerId,Fare\n2,8.05\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("PassengerId,Survived\n2,0\n", encoding="utf-8")

            settings = cli_app._propose_new_experiment_settings(
                "Titanic survival prediction accuracy",
                "",
                str(project),
            )

        self.assertEqual(str(project), settings["source_path"])
        self.assertFalse(settings["create_workspace"])
        self.assertEqual("Survived", settings["target_column"])
        self.assertEqual("PassengerId", settings["id_column"])
        self.assertEqual("accuracy", settings["metric"])
        self.assertEqual(["train.csv", "test.csv", "sample_submission.csv"], settings["required_data_files"])

    def test_new_experiment_settings_normalizes_direct_data_folder_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "workspace"
            data = project / "data"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("id,target,value\n1,0,10\n", encoding="utf-8")
            (data / "test.csv").write_text("id,value\n2,11\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("id,target\n2,0\n", encoding="utf-8")

            settings = cli_app._propose_new_experiment_settings(
                "binary classification accuracy",
                "",
                str(data),
            )

        self.assertEqual(str(project), settings["source_path"])
        self.assertEqual("target", settings["target_column"])
        self.assertEqual("id", settings["id_column"])
        self.assertEqual(["train.csv", "test.csv", "sample_submission.csv"], settings["required_data_files"])

    def test_new_experiment_settings_uses_known_kaggle_url_preset(self):
        settings = cli_app._propose_new_experiment_settings(
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
            "",
            None,
        )

        self.assertEqual("house-prices-advanced-regression-techniques", settings["competition"])
        self.assertEqual("House Prices - Advanced Regression Techniques", settings["topic"])
        self.assertEqual("SalePrice", settings["target_column"])
        self.assertEqual("Id", settings["id_column"])
        self.assertEqual("rmsle", settings["metric"])
        self.assertEqual("minimize", settings["objective"])

    def test_new_experiment_settings_extracts_description_column_hints(self):
        settings = cli_app._propose_new_experiment_settings(
            "Customer churn classification. target column is Exited. ID column is CustomerId. metric is AUC.",
            "",
            None,
        )

        self.assertEqual("Exited", settings["target_column"])
        self.assertEqual("CustomerId", settings["id_column"])
        self.assertEqual("roc_auc", settings["metric"])
        self.assertEqual("maximize", settings["objective"])
        self.assertEqual("customer-churn-classification", settings["competition"])

    def test_new_experiment_settings_infers_regression_metric_from_target_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "workspace"
            data = project / "data"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("Id,SalePrice,Area\n1,120000.5,80\n2,230500.0,120\n", encoding="utf-8")
            (data / "test.csv").write_text("Id,Area\n3,90\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("Id,SalePrice\n3,0\n", encoding="utf-8")

            settings = cli_app._propose_new_experiment_settings(
                "Predict target column SalePrice using tabular data",
                "",
                str(project),
            )

        self.assertEqual("SalePrice", settings["target_column"])
        self.assertEqual("rmse", settings["metric"])
        self.assertEqual("minimize", settings["objective"])

    def test_insight_message_shows_original_text_next_trial_and_planned_improvement(self):
        snapshot = {"current_trial": None, "last_completed_trial": "trial_003", "next_trial": "trial_004"}
        answers = iter(["다음 실험은 Kaggle 제출 점수 개선을 우선해줘."])
        with patch("kaggle_research_agent.cli_app.submit_human_insight", return_value={"ok": True}):
            message = cli_app._insight_dialog("titanic", snapshot, lambda _: next(answers), lambda _: None)

        self.assertIn("다음 실험은 Kaggle 제출 점수 개선을 우선해줘.", message)
        self.assertIn("- trial_004 실험 반영 예정", message)
        self.assertIn("- 적용 개선안 :", message)
        self.assertIn("제출 점수", message)

    def test_insight_dialog_shows_existing_insight_and_overwrites_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feedback_path = root / "memory" / "titanic" / "user_feedback.jsonl"
            feedback_path.parent.mkdir(parents=True)
            feedback_path.write_text(
                json.dumps(
                    {
                        "competition": "titanic",
                        "trial_id": "trial_005",
                        "topic": "user_insight",
                        "scope": "next_trial",
                        "user_feedback": "기존 앙상블 인사이트",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            snapshot = {"last_completed_trial": "trial_005", "next_trial": "trial_006"}
            answers = iter(["앙상블 대신 제출 점수 안정성을 우선하자"])
            output = []
            new_feedback = {
                "competition": "titanic",
                "trial_id": "trial_005",
                "topic": "user_insight",
                "scope": "next_trial",
                "user_feedback": "앙상블 대신 제출 점수 안정성을 우선하자",
            }
            with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.cli_app.submit_human_insight",
                    return_value={"ok": True, "data": {"feedback": new_feedback}},
                ):
                    message = cli_app._insight_dialog("titanic", snapshot, lambda _: next(answers), output.append)

            self.assertTrue(any("이미 인사이트가 제공 되었습니다" in line for line in output))
            self.assertTrue(any("기존 앙상블 인사이트" in line for line in output))
            self.assertTrue(any("다음 실험 계획 단계" in line for line in output))
            self.assertTrue(any("기존 인사이트 유지: q" in line for line in output))
            preview = "\n".join(output)
            self.assertEqual(1, preview.count("기존 앙상블 인사이트"))
            self.assertIn("- trial_006 실험 반영 예정", preview)
            self.assertIn("- 적용 개선안 :", preview)
            self.assertIn("앙상블 대신 제출 점수 안정성을 우선하자", message)
            saved = feedback_path.read_text(encoding="utf-8")
            self.assertNotIn("기존 앙상블 인사이트", saved)
            self.assertIn("앙상블 대신 제출 점수 안정성을 우선하자", saved)

    @staticmethod
    def _write_metrics(root: Path, trial_id: str, local: float, lb: float) -> None:
        path = root / "demo_workspaces" / "titanic" / "manual_trials" / trial_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(
            json.dumps({"trial_id": trial_id, "local_score": local, "kaggle_lb_score": lb}),
            encoding="utf-8",
        )

    @staticmethod
    def _write_execution_profile(root: Path, competition: str) -> None:
        workspace = root / "demo_workspaces" / competition
        workspace.mkdir(parents=True, exist_ok=True)
        competition_dir = root / "competitions" / competition
        competition_dir.mkdir(parents=True, exist_ok=True)
        (competition_dir / "execution_profile.yaml").write_text(
            "\n".join(
                [
                    "schema_version: '1.0'",
                    f"competition: {competition}",
                    "platform: kaggle",
                    f"project_root: {workspace}",
                    f"python: {Path(sys.executable)}",
                    "commands:",
                    "  test:",
                    "    - '{python} test_step.py'",
                    "  train:",
                    "    - '{python} train_step.py'",
                    "artifacts:",
                    "  metrics:",
                    "    - outputs/metrics.json",
                    "  submission:",
                    "    - outputs/submission.csv",
                    "write_scope:",
                    "  allowed:",
                    "    - src/",
                    "  forbidden:",
                    "    - data/",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
