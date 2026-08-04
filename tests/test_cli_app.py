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
    def test_current_loop_state_repairs_dead_process_without_losing_failure_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                cli_app.save_json_atomic(
                    cli_app.loop_state_path(),
                    {
                        "competition": "demo",
                        "status": "running",
                        "pid": 999999,
                        "next_trial": "trial_001",
                        "error": "recoverable_after_metrics_collection",
                    },
                )
                with patch("kaggle_research_agent.cli_app.os.kill", side_effect=OSError):
                    state = cli_app._current_loop_state("demo")

                saved = cli_app.load_json(cli_app.loop_state_path())

        self.assertEqual("failed", state["status"])
        self.assertIsNone(state["pid"])
        self.assertEqual("failed", state["resume_from_status"])
        self.assertEqual("recoverable_after_metrics_collection", state["error"])
        self.assertEqual(state, saved)

    def test_snapshot_prefers_human_readable_filesystem_topic_over_slug(self):
        response = {
            "ok": True,
            "data": {
                "experiment": {
                    "competition": "bike-sharing-demand",
                    "topic": "bike-sharing-demand",
                    "objective": "minimize",
                    "state": "ready_next_trial",
                },
                "trials": [],
            },
        }
        with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
            with patch("kaggle_research_agent.cli_app._filesystem_topic", return_value="Bike Sharing Demand"):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("bike-sharing-demand", sync=False)

        self.assertEqual("Bike Sharing Demand", snapshot["topic"])

    def test_snapshot_labels_post_metrics_failure_as_recovery_pending(self):
        with patch(
            "kaggle_research_agent.cli_app.get_experiment",
            return_value={
                "ok": True,
                "data": {
                    "experiment": {
                        "competition": "demo",
                        "topic": "Demo",
                        "state": "ready_next_trial",
                    },
                    "trials": [],
                },
            },
        ):
            with patch(
                "kaggle_research_agent.cli_app._current_loop_state",
                return_value={
                    "competition": "demo",
                    "status": "failed",
                    "next_trial": "trial_001",
                    "error": "recoverable_after_metrics_collection",
                },
            ):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("demo", sync=False)

        self.assertEqual("후처리 복구 대기", snapshot["state"])

    def test_running_snapshot_separates_current_next_and_last_completed_trials(self):
        response = {
            "ok": True,
            "data": {
                "experiment": {
                    "competition": "demo",
                    "topic": "Demo",
                    "objective": "maximize",
                    "state": "ready_next_trial",
                },
                "trials": [
                    {
                        "trial_id": "trial_001",
                        "status": "completed",
                        "local_score": 0.8,
                        "lb_score": None,
                    }
                ],
            },
        }
        with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
            with patch(
                "kaggle_research_agent.cli_app._current_loop_state",
                return_value={
                    "competition": "demo",
                    "status": "running",
                    "phase": "coding",
                    "current_trial": "trial_001",
                    "next_trial": "trial_001",
                },
            ):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("demo", sync=False)

        self.assertEqual("trial_001", snapshot["current_trial"])
        self.assertEqual("trial_002", snapshot["next_trial"])
        self.assertIsNone(snapshot["last_completed_trial"])
        self.assertIsNone(snapshot["latest"])

    def test_running_later_trial_keeps_previous_trial_as_latest_completed(self):
        response = {
            "ok": True,
            "data": {
                "experiment": {
                    "competition": "demo",
                    "topic": "Demo",
                    "objective": "maximize",
                    "state": "ready_next_trial",
                },
                "trials": [
                    {"trial_id": "trial_001", "status": "completed", "local_score": 0.8, "lb_score": 0.7},
                    {"trial_id": "trial_002", "status": "completed", "local_score": 0.81, "lb_score": None},
                ],
            },
        }
        with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
            with patch(
                "kaggle_research_agent.cli_app._current_loop_state",
                return_value={
                    "competition": "demo",
                    "status": "running",
                    "phase": "coding",
                    "current_trial": "trial_002",
                    "next_trial": "trial_002",
                    "last_completed_trial": "trial_001",
                },
            ):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("demo", sync=False)

        self.assertEqual("trial_002", snapshot["current_trial"])
        self.assertEqual("trial_003", snapshot["next_trial"])
        self.assertEqual("trial_001", snapshot["last_completed_trial"])
        self.assertEqual("trial_001", snapshot["latest"]["trial_id"])

    def test_feedback_request_lines_explain_problem_recommendation_and_execution_boundary(self):
        lines = cli_app._feedback_request_lines(
            {
                "interaction_label": "외부 계산 자원 승인",
                "problem": "현재 환경의 메모리가 부족합니다.",
                "evidence_snapshot": [
                    {"label": "최대 메모리", "value": "11.6GB", "meaning": "실행 중 사용량"}
                ],
                "interpretation": "외부 자원 또는 경량화가 필요합니다.",
                "recommendation": "로컬 경량화를 기본값으로 둡니다.",
                "why_user_needed": "외부 비용은 사용자 승인이 필요합니다.",
                "question": "외부 환경 사용을 검토할까요?",
                "options": [
                    {
                        "value": "estimate_first",
                        "label": "비용 추정 후 결정",
                        "impact": "실행 없이 비용만 추정합니다.",
                    }
                ],
                "default_if_no_response": "로컬 경량화를 적용합니다.",
                "execution_supported": False,
                "execution_note": "현재 버전에서는 외부 환경을 자동 실행하지 않습니다.",
            }
        )
        rendered = "\n".join(lines)
        self.assertIn("11.6GB", rendered)
        self.assertIn("로컬 경량화", rendered)
        self.assertIn("비용 추정 후 결정", rendered)
        self.assertIn("자동 실행하지 않습니다", rendered)

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

    def test_snapshot_uses_lowest_submission_score_for_minimize_objective(self):
        response = {
            "ok": True,
            "data": {
                "experiment": {
                    "competition": "bike-sharing-demand",
                    "topic": "Bike Sharing Demand",
                    "objective": "minimize",
                    "state": "completed",
                },
                "trials": [
                    {"trial_id": "trial_001", "status": "completed", "lb_score": 1.01},
                    {"trial_id": "trial_002", "status": "completed", "lb_score": 0.92},
                ],
            },
        }
        with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
            with patch("kaggle_research_agent.cli_app._current_loop_state", return_value={}):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("bike-sharing-demand", sync=False)

        self.assertEqual("trial_002", snapshot["best"]["trial_id"])

    def test_empty_install_does_not_inject_titanic_experiment(self):
        with patch("kaggle_research_agent.cli_app.list_experiments", return_value={"ok": True, "data": {"experiments": []}}):
            with patch("kaggle_research_agent.cli_app._filesystem_experiments", return_value=[]):
                self.assertEqual([], cli_app.load_experiments(sync=False))

    def test_snapshot_does_not_treat_discovered_scaffold_as_completed(self):
        response = {
            "ok": True,
            "data": {
                "experiment": {
                    "competition": "bike-sharing-demand",
                    "topic": "Bike Sharing Demand",
                    "state": "ready_first_trial",
                    "next_trial_id": "trial_001",
                },
                "trials": [
                    {
                        "trial_id": "trial_001",
                        "status": "discovered",
                        "local_score": None,
                        "lb_score": None,
                    }
                ],
            },
        }
        with patch("kaggle_research_agent.cli_app.get_experiment", return_value=response):
            with patch("kaggle_research_agent.cli_app._current_loop_state", return_value={}):
                with patch("kaggle_research_agent.cli_app._manual_trial_rows", return_value=[]):
                    snapshot = cli_app.experiment_snapshot("bike-sharing-demand", sync=False)

        self.assertIsNone(snapshot["last_completed_trial"])
        self.assertEqual("trial_001", snapshot["next_trial"])
        self.assertIsNone(snapshot["latest"])

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

    def test_recent_logs_only_show_selected_competition(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "auto_loop.log"
            log.write_text(
                "\n".join(
                    [
                        "=== titanic / trial_007 ===",
                        '{"status":"code_writer_blocked"}',
                        "=== bike-sharing-demand / trial_001 ===",
                        "Planning baseline...",
                    ]
                ),
                encoding="utf-8",
            )

            lines = cli_app._recent_log_lines(
                log,
                limit=4,
                competition="bike-sharing-demand",
            )

        self.assertIn("bike-sharing-demand / trial_001", "\n".join(lines))
        self.assertIn("Planning baseline", "\n".join(lines))
        self.assertNotIn("titanic", "\n".join(lines))
        self.assertNotIn("code_writer_blocked", "\n".join(lines))

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

    def test_resolve_start_trial_ignores_failed_loop_pointing_at_missing_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.ROOT", root):
                with patch("kaggle_research_agent.cli_app._infer_start_trial", return_value="trial_006"):
                    start_trial = cli_app._resolve_start_trial(
                        "bike-sharing-demand",
                        {"status": "failed", "next_trial": "trial_009", "error": "blocked_missing_result_cycle"},
                    )
        self.assertEqual("trial_006", start_trial)

    def test_resolve_start_trial_keeps_failed_loop_when_trial_still_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "bike-sharing-demand" / "trial_002").mkdir(parents=True)
            with patch("kaggle_research_agent.paths.ROOT", root):
                with patch("kaggle_research_agent.cli_app._infer_start_trial", return_value="trial_099"):
                    start_trial = cli_app._resolve_start_trial(
                        "bike-sharing-demand",
                        {
                            "status": "failed",
                            "next_trial": "trial_002",
                            "error": "recoverable_after_metrics_collection",
                        },
                    )
        self.assertEqual("trial_002", start_trial)

    def test_resolve_start_trial_trusts_non_failed_loop_regardless_of_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.ROOT", root):
                with patch("kaggle_research_agent.cli_app._infer_start_trial", return_value="trial_006"):
                    start_trial = cli_app._resolve_start_trial(
                        "bike-sharing-demand",
                        {"status": "paused", "next_trial": "trial_009"},
                    )
        self.assertEqual("trial_009", start_trial)

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

    def _write_dacon_profile(self, root: Path, *, team_name: str | None) -> None:
        workspace = root / "demo_workspaces" / "236716"
        workspace.mkdir(parents=True)
        competition_dir = root / "competitions" / "236716"
        competition_dir.mkdir(parents=True)
        lines = [
            "schema_version: '1.0'",
            "competition: '236716'",
            "platform: dacon",
            "dacon_competition_id: '236716'",
        ]
        if team_name is not None:
            lines.append(f"dacon_team_name: {team_name}")
        lines.extend(
            [
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
        )
        (competition_dir / "execution_profile.yaml").write_text("\n".join(lines), encoding="utf-8")

    def test_start_dacon_experiment_passes_dacon_submit_flags(self):
        # Until this was wired, only platform == kaggle got --submit, so a
        # DACON competition could finish every trial locally and never reach
        # the submission adapter that already existed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            self._write_dacon_profile(root, team_name="뚜로")
            process = Mock(pid=321)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.paths.ROOT", root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                            cli_app.start_experiment("236716")

            command = popen.call_args.args[0]
            self.assertIn("--submit", command)
            self.assertIn("--dacon-competition-id", command)
            self.assertEqual("236716", command[command.index("--dacon-competition-id") + 1])
            self.assertIn("--dacon-team-name", command)
            self.assertEqual("뚜로", command[command.index("--dacon-team-name") + 1])
            self.assertNotIn("--kaggle-slug", command)

    def test_start_dacon_experiment_without_team_name_runs_without_submitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            self._write_dacon_profile(root, team_name=None)
            process = Mock(pid=322)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(runtime)}):
                with patch("kaggle_research_agent.paths.ROOT", root):
                    with patch("kaggle_research_agent.cli_app.get_experiment", return_value={"ok": False}):
                        with patch("kaggle_research_agent.cli_app.subprocess.Popen", return_value=process) as popen:
                            cli_app.start_experiment("236716")

            command = popen.call_args.args[0]
            self.assertNotIn("--submit", command)
            self.assertNotIn("--dacon-team-name", command)

    def test_current_loop_state_drops_a_failure_the_trial_has_since_recovered_from(self):
        # auto_loop_state.json keeps the previous run's outcome until some
        # later run overwrites it. Once the trial it failed on completed,
        # the dashboard kept surfacing that stale error and its stale
        # next_trial instead of the real, recovered state.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "workspace_result_cycle.json").write_text(
                json.dumps({"competition": "demo", "trial_id": "trial_001", "status": "completed"}),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "runtime")}):
                with patch("kaggle_research_agent.paths.project_root", return_value=root):
                    cli_app.save_json_atomic(
                        cli_app.loop_state_path(),
                        {
                            "competition": "demo",
                            "status": "failed",
                            "next_trial": "trial_001",
                            "error": "after_coding_metrics_needs_review",
                        },
                    )
                    self.assertEqual({}, cli_app._current_loop_state("demo"))

    def test_current_loop_state_keeps_a_failure_that_was_never_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_001").mkdir(parents=True)
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "runtime")}):
                with patch("kaggle_research_agent.paths.project_root", return_value=root):
                    cli_app.save_json_atomic(
                        cli_app.loop_state_path(),
                        {
                            "competition": "demo",
                            "status": "failed",
                            "next_trial": "trial_001",
                            "error": "blocked_context",
                        },
                    )
                    loop = cli_app._current_loop_state("demo")

            self.assertEqual("failed", loop["status"])
            self.assertEqual("blocked_context", loop["error"])

    def test_current_loop_state_keeps_a_failure_when_the_trial_is_still_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "workspace_result_cycle.json").write_text(
                json.dumps({"competition": "demo", "trial_id": "trial_001", "status": "blocked"}),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "runtime")}):
                with patch("kaggle_research_agent.paths.project_root", return_value=root):
                    cli_app.save_json_atomic(
                        cli_app.loop_state_path(),
                        {"competition": "demo", "status": "failed", "next_trial": "trial_001"},
                    )
                    loop = cli_app._current_loop_state("demo")

            self.assertEqual("failed", loop["status"])

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

    def test_new_experiment_settings_accepts_camel_case_sample_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "workspace"
            data = project / "data"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("datetime,count,temp\n2020-01-01,1,10\n", encoding="utf-8")
            (data / "test.csv").write_text("datetime,temp\n2020-01-02,11\n", encoding="utf-8")
            (data / "sampleSubmission.csv").write_text("datetime,count\n2020-01-02,0\n", encoding="utf-8")

            settings = cli_app._propose_new_experiment_settings(
                "Bike sharing demand RMSLE regression",
                "",
                str(project),
            )

        self.assertEqual("count", settings["target_column"])
        self.assertEqual("datetime", settings["id_column"])
        self.assertEqual("rmsle", settings["metric"])
        self.assertIn("sampleSubmission.csv", settings["required_data_files"])

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

    def test_new_experiment_settings_infers_competition_id_and_platform_from_dacon_url(self):
        settings = cli_app._propose_new_experiment_settings(
            "https://dacon.io/competitions/official/236716/overview/description",
            "",
            None,
        )

        self.assertEqual("236716", settings["competition"])
        self.assertEqual("dacon", settings["platform"])

    def test_new_experiment_settings_defaults_to_kaggle_platform_for_kaggle_url(self):
        settings = cli_app._propose_new_experiment_settings(
            "https://www.kaggle.com/competitions/titanic",
            "",
            None,
        )

        self.assertEqual("kaggle", settings["platform"])

    def test_new_experiment_settings_defaults_to_kaggle_platform_without_a_url(self):
        settings = cli_app._propose_new_experiment_settings(
            "고객 이탈 예측 실험을 하고 싶어",
            "",
            None,
        )

        self.assertEqual("kaggle", settings["platform"])

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


class DeleteExperimentTest(unittest.TestCase):
    def test_delete_experiment_removes_db_row_and_created_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "_runtime")}):
                        db_path = initialize_state_db()
                        upsert_competition({"competition_id": "demo-delete", "platform": "dacon"}, db_path)
                        cli_app.prepare_workspace(
                            "demo-delete",
                            topic="Demo Delete",
                            platform="dacon",
                            create_workspace=True,
                        )
                        cli_app.select_competition("demo-delete")
                        self.assertTrue((root / "demo_workspaces" / "demo-delete").is_dir())
                        self.assertTrue((root / "competitions" / "demo-delete").is_dir())

                        result = cli_app.delete_experiment("demo-delete")

                        self.assertTrue(result["ok"])
                        self.assertFalse((root / "demo_workspaces" / "demo-delete").exists())
                        self.assertFalse((root / "competitions" / "demo-delete").exists())
                        from kaggle_research_agent.state_db import list_competitions

                        remaining_ids = [row["competition_id"] for row in list_competitions(db_path)]
                        self.assertNotIn("demo-delete", remaining_ids)
                        self.assertNotEqual("demo-delete", cli_app.selected_competition())

    def test_delete_experiment_removes_demo_workspace_even_with_stale_or_missing_source_record(self):
        # Real incident: a demo_workspaces/<competition>/ folder (with old
        # code and stale outputs/metrics.json from a previous, buggy run)
        # survived a delete because workspace_source.json's created_workspace
        # bookkeeping didn't reflect reality. Removal by convention -- not
        # by trusting that record -- is what actually fixes this.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "_runtime")}):
                        db_path = initialize_state_db()
                        upsert_competition({"competition_id": "demo-stale", "platform": "dacon"}, db_path)
                        workspace = root / "demo_workspaces" / "demo-stale"
                        (workspace / "outputs").mkdir(parents=True)
                        (workspace / "predict_step.py").write_text("stale code", encoding="utf-8")
                        (workspace / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
                        (root / "competitions" / "demo-stale").mkdir(parents=True)
                        # No workspace_source.json at all -- simulates the
                        # record being missing/stale, not just False.
                        cli_app.select_competition("demo-stale")

                        result = cli_app.delete_experiment("demo-stale")

                        self.assertTrue(result["ok"])
                        self.assertFalse(workspace.exists())

    def test_delete_experiment_keeps_external_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = Path(tmp) / "outside_project"
            external.mkdir()
            (external / "keep.txt").write_text("keep me", encoding="utf-8")
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "_runtime")}):
                        db_path = initialize_state_db()
                        upsert_competition({"competition_id": "demo-external", "platform": "kaggle"}, db_path)
                        cli_app.prepare_workspace(
                            "demo-external",
                            topic="Demo External",
                            platform="kaggle",
                            source_path=str(external),
                            create_workspace=False,
                        )

                        cli_app.delete_experiment("demo-external")

            self.assertTrue(external.is_dir())
            self.assertTrue((external / "keep.txt").exists())

    def test_delete_experiment_clears_matching_stale_loop_state(self):
        # Real bug: auto_loop_state.json is a single global file, not
        # namespaced per competition. Deleting and re-registering an
        # experiment under the same id (numeric DACON ids are especially
        # prone to reuse) left the old run's failure status/next_trial
        # showing up immediately on the freshly re-registered experiment.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "_runtime")}):
                        db_path = initialize_state_db()
                        upsert_competition({"competition_id": "236716", "platform": "dacon"}, db_path)
                        cli_app.prepare_workspace(
                            "236716",
                            topic="Demo",
                            platform="dacon",
                            create_workspace=True,
                        )
                        cli_app.save_json_atomic(
                            cli_app.loop_state_path(),
                            {
                                "competition": "236716",
                                "status": "failed",
                                "error": "blocked_missing_result_cycle",
                                "next_trial": "trial_002",
                            },
                        )
                        cli_app.pause_request_path().parent.mkdir(parents=True, exist_ok=True)
                        cli_app.pause_request_path().write_text("requested\n", encoding="utf-8")

                        cli_app.delete_experiment("236716")

                        self.assertFalse(cli_app.loop_state_path().exists())
                        self.assertFalse(cli_app.pause_request_path().exists())

    def test_delete_experiment_keeps_unrelated_loop_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch.dict("os.environ", {"RESEARCH_AGENT_RUNTIME_DIR": str(root / "_runtime")}):
                        db_path = initialize_state_db()
                        upsert_competition({"competition_id": "demo-delete", "platform": "dacon"}, db_path)
                        cli_app.prepare_workspace(
                            "demo-delete",
                            topic="Demo Delete",
                            platform="dacon",
                            create_workspace=True,
                        )
                        cli_app.save_json_atomic(
                            cli_app.loop_state_path(),
                            {"competition": "some-other-experiment", "status": "running"},
                        )

                        cli_app.delete_experiment("demo-delete")

                        self.assertTrue(cli_app.loop_state_path().exists())
                        self.assertEqual(
                            "some-other-experiment",
                            cli_app.load_json(cli_app.loop_state_path()).get("competition"),
                        )


class SubmitTrialManuallyTest(unittest.TestCase):
    def _write_profile(self, root: Path, project: Path, *, team_name: str = "뚜로") -> None:
        from kaggle_research_agent import simple_yaml

        comp_dir = root / "competitions" / "demo"
        comp_dir.mkdir(parents=True)
        simple_yaml.dump(
            {
                "platform": "dacon",
                "dacon_competition_id": "236716",
                "dacon_team_name": team_name,
                "project_root": str(project),
                "objective": "maximize",
                "artifacts": {"submission": ["outputs/submission.csv"]},
            },
            comp_dir / "execution_profile.yaml",
        )

    def test_submits_using_execution_profile_and_trial_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "workspace"
            (project / "outputs").mkdir(parents=True)
            (project / "outputs" / "submission.csv").write_text("id,x,y,z\n1,0,0,0\n", encoding="utf-8")
            self._write_profile(root, project)
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.591}), encoding="utf-8")

            captured = {}

            def fake_submit_trial(**kwargs):
                captured.update(kwargs)
                return {"status": "submitted", "trial_id": kwargs["trial_id"]}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    with patch("kaggle_research_agent.cli_app._submit_trial", side_effect=fake_submit_trial):
                        result = cli_app.submit_trial_manually("demo", "trial_004")

            self.assertTrue(result["ok"])
            self.assertEqual("236716", captured["dacon_competition_id"])
            self.assertEqual("뚜로", captured["dacon_team_name"])
            self.assertEqual("trial_004_manual", captured["version_name"])
            self.assertIn("0.591000", captured["dacon_message"])
            self.assertTrue(captured["submission_file"].endswith("submission.csv"))

    def test_rejects_non_dacon_competition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from kaggle_research_agent import simple_yaml

            comp_dir = root / "competitions" / "demo"
            comp_dir.mkdir(parents=True)
            simple_yaml.dump({"platform": "kaggle"}, comp_dir / "execution_profile.yaml")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    result = cli_app.submit_trial_manually("demo", "trial_004")

            self.assertFalse(result["ok"])
            self.assertEqual("not_dacon", result["status"])

    def test_rejects_when_team_name_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "workspace"
            project.mkdir()
            self._write_profile(root, project, team_name="")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    result = cli_app.submit_trial_manually("demo", "trial_004")

            self.assertFalse(result["ok"])
            self.assertEqual("missing_team_name", result["status"])

    def test_rejects_when_submission_file_missing_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "workspace"
            project.mkdir()
            self._write_profile(root, project)
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    result = cli_app.submit_trial_manually("demo", "trial_004")

            self.assertFalse(result["ok"])
            self.assertEqual("submission_file_missing", result["status"])


class RefreshDaconCompetitionDocsTest(unittest.TestCase):
    def test_writes_overview_and_data_notes_from_scraped_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    comp_dir = root / "competitions" / "demo"
                    comp_dir.mkdir(parents=True)
                    (comp_dir / "overview.md").write_text(
                        "# demo\n\nDescribe the competition, data, and important constraints here.\n",
                        encoding="utf-8",
                    )

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_competition_overview",
                        return_value={"ok": True, "status": "found", "text": "실제 대회 설명입니다."},
                    ):
                        with patch(
                            "kaggle_research_agent.cli_app.dacon_api.fetch_competition_data_description",
                            return_value={"ok": True, "status": "found", "text": "실제 데이터 설명입니다."},
                        ):
                            result = cli_app.refresh_dacon_competition_docs("demo", dacon_competition_id="236716")

                    overview_text = (comp_dir / "overview.md").read_text(encoding="utf-8")
                    data_notes_text = (comp_dir / "data_notes.md").read_text(encoding="utf-8")

            self.assertEqual(["overview.md", "data_notes.md"], result["written"])
            self.assertIn("실제 대회 설명입니다.", overview_text)
            self.assertNotIn("Describe the competition", overview_text)
            self.assertIn("실제 데이터 설명입니다.", data_notes_text)

    def test_a_scrape_failure_does_not_raise_or_overwrite_the_other_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    comp_dir = root / "competitions" / "demo"
                    comp_dir.mkdir(parents=True)

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_competition_overview",
                        return_value={"ok": False, "status": "fetch_error", "error": "timeout"},
                    ):
                        with patch(
                            "kaggle_research_agent.cli_app.dacon_api.fetch_competition_data_description",
                            return_value={"ok": True, "status": "found", "text": "실제 데이터 설명입니다."},
                        ):
                            result = cli_app.refresh_dacon_competition_docs("demo", dacon_competition_id="236716")

            self.assertEqual(["data_notes.md"], result["written"])
            self.assertFalse((comp_dir / "overview.md").exists())
            self.assertTrue((comp_dir / "data_notes.md").exists())


class DaconSubmissionLimitTest(unittest.TestCase):
    # fetch_my_submissions is patched in every test below (even ones only
    # about the limit number) so no test can ever make a real network call
    # to DACON -- a real DACON_SESSION_COOKIE happening to be set in the dev
    # environment must not turn a unit test into a live API call.
    _NO_SUBMISSION_HISTORY = {"ok": False, "status": "session_token_missing", "error": "no token"}

    def test_check_uses_manual_override_without_hitting_the_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"dacon_competition_id": "236716", "dacon_daily_submission_limit": 3},
                        profile_path,
                    )

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_daily_submission_limit",
                        side_effect=AssertionError("should not fetch when a manual override is set"),
                    ):
                        with patch(
                            "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                            return_value=self._NO_SUBMISSION_HISTORY,
                        ):
                            result = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual("manual_override", result["status"])
            self.assertEqual(3, result["daily_submission_limit"])

    def test_check_fetches_once_then_caches_into_execution_profile(self):
        # "처음에만 조회하면 돼" -- the rules page must only be scraped the
        # first time; every later call must read the cached value instead.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"dacon_competition_id": "236716"}, profile_path)

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_daily_submission_limit",
                        return_value={"ok": True, "status": "found", "daily_submission_limit": 5},
                    ) as fetch_limit:
                        with patch(
                            "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                            return_value=self._NO_SUBMISSION_HISTORY,
                        ):
                            first = cli_app.check_dacon_submission_limit("demo")

                            with patch(
                                "kaggle_research_agent.cli_app.dacon_api.fetch_daily_submission_limit",
                                side_effect=AssertionError("should not re-fetch once cached"),
                            ):
                                second = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual(1, fetch_limit.call_count)
            self.assertEqual("auto_detected", first["status"])
            self.assertEqual(5, first["daily_submission_limit"])
            self.assertEqual("auto_detected", second["status"])
            self.assertEqual(5, second["daily_submission_limit"])
            cached = simple_yaml.load(profile_path, default={})
            self.assertEqual(5, cached["dacon_daily_submission_limit_detected"])

    def test_check_reports_unknown_when_rules_page_has_no_stated_limit(self):
        # A miss must surface as "unknown", not silently be treated as "no
        # limit" -- that distinction is what tells the user a manual
        # override is worth setting.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"dacon_competition_id": "236716"}, profile_path)

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_daily_submission_limit",
                        return_value={"ok": False, "status": "not_found", "error": "..."},
                    ):
                        with patch(
                            "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                            side_effect=AssertionError("must not fetch submission history when limit is unknown"),
                        ):
                            result = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual("unknown", result["status"])
            self.assertIsNone(result["daily_submission_limit"])
            self.assertIsNone(result["remaining"])

    def test_check_computes_remaining_using_kst_midnight_boundary(self):
        # DACON's daily limit is understood to reset around local midnight
        # KST -- unlike Kaggle's 24h-after-first-submission model (see
        # RollingWindowSubmissionStatsTest below, kept for that future use).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from datetime import datetime, timedelta, timezone

                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"dacon_competition_id": "236716", "dacon_daily_submission_limit": 5},
                        profile_path,
                    )
                    kst = timezone(timedelta(hours=9))
                    now_kst = datetime.now(timezone.utc).astimezone(kst).replace(tzinfo=None)
                    midnight_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
                    today_a = (midnight_kst + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                    today_b = (midnight_kst + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
                    yesterday = (midnight_kst - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                        return_value={
                            "ok": True,
                            "submissions": [
                                {"c_time": today_a},
                                {"c_time": today_b},
                                {"c_time": yesterday},  # before KST midnight, must not count
                            ],
                        },
                    ):
                        result = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual(5, result["daily_submission_limit"])
            self.assertEqual(3, result["remaining"])  # 5 - 2 counted since KST midnight
            expected_reset = midnight_kst + timedelta(days=1)
            self.assertEqual(expected_reset.strftime("%Y-%m-%d %H:%M KST"), result["next_reset_estimate"])

    def test_check_omits_reset_estimate_when_no_submissions_are_counted_today(self):
        # Nothing pending against the limit -- there is nothing to reset.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"dacon_competition_id": "236716", "dacon_daily_submission_limit": 5},
                        profile_path,
                    )

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                        return_value={"ok": True, "submissions": []},
                    ):
                        result = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual(5, result["remaining"])
            self.assertIsNone(result["next_reset_estimate"])

    def test_check_shows_limit_alone_when_remaining_cannot_be_computed(self):
        # Per explicit instruction: when the limit is known but the
        # remaining count can't be (e.g. no/expired session token), show the
        # limit alone rather than a broken "?/5" or hiding it entirely.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"dacon_competition_id": "236716", "dacon_daily_submission_limit": 5},
                        profile_path,
                    )

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
                        return_value=self._NO_SUBMISSION_HISTORY,
                    ):
                        result = cli_app.check_dacon_submission_limit("demo")

            self.assertEqual(5, result["daily_submission_limit"])
            self.assertIsNone(result["remaining"])
            self.assertIsNone(result["next_reset_estimate"])

    def test_set_override_writes_and_clear_removes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"dacon_competition_id": "236716"}, profile_path)

                    cli_app.set_dacon_submission_limit_override("demo", 7)
                    written = simple_yaml.load(profile_path, default={})
                    self.assertEqual(7, written["dacon_daily_submission_limit"])

                    cli_app.set_dacon_submission_limit_override("demo", None)
                    cleared = simple_yaml.load(profile_path, default={})
                    self.assertNotIn("dacon_daily_submission_limit", cleared)

    def test_set_dacon_team_name_writes_and_clears(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"dacon_competition_id": "236716"}, profile_path)

                    cli_app.set_dacon_team_name("demo", "뚜로")
                    written = simple_yaml.load(profile_path, default={})
                    self.assertEqual("뚜로", written["dacon_team_name"])

                    cli_app.set_dacon_team_name("demo", "")
                    cleared = simple_yaml.load(profile_path, default={})
                    self.assertNotIn("dacon_team_name", cleared)


class BestTrialLocalFallbackTest(unittest.TestCase):
    def test_best_trial_by_local_score_picks_highest_for_maximize(self):
        rows = [
            {"trial_id": "trial_001", "local_score": 0.5},
            {"trial_id": "trial_002", "local_score": 0.8},
        ]
        best = cli_app._best_trial_by_local_score(rows, [], objective="maximize")
        self.assertEqual("trial_002", best["trial_id"])

    def test_best_trial_by_local_score_picks_lowest_for_minimize(self):
        rows = [
            {"trial_id": "trial_001", "local_score": 0.5},
            {"trial_id": "trial_002", "local_score": 0.8},
        ]
        best = cli_app._best_trial_by_local_score(rows, [], objective="minimize")
        self.assertEqual("trial_001", best["trial_id"])

    def test_best_trial_by_local_score_ignores_rows_without_a_local_score(self):
        best = cli_app._best_trial_by_local_score([{"trial_id": "trial_001", "local_score": None}], [])
        self.assertIsNone(best)

    def test_dacon_daily_limit_is_known_false_for_non_dacon_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"platform": "kaggle"}, profile_path)

                    self.assertFalse(cli_app._dacon_daily_limit_is_known("demo"))

    def test_dacon_daily_limit_is_known_true_for_manual_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"platform": "dacon", "dacon_daily_submission_limit": 5},
                        profile_path,
                    )

                    self.assertTrue(cli_app._dacon_daily_limit_is_known("demo"))

    def test_dacon_daily_limit_is_known_true_for_cached_auto_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump(
                        {"platform": "dacon", "dacon_daily_submission_limit_detected": 5},
                        profile_path,
                    )

                    self.assertTrue(cli_app._dacon_daily_limit_is_known("demo"))

    def test_dacon_daily_limit_is_known_false_when_neither_set(self):
        # Must not attempt a live fetch to answer this -- it's checked on
        # every dashboard render via experiment_snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.cli_app.project_root", return_value=root):
                    from kaggle_research_agent import simple_yaml

                    profile_path = root / "competitions" / "demo" / "execution_profile.yaml"
                    profile_path.parent.mkdir(parents=True)
                    simple_yaml.dump({"platform": "dacon", "dacon_competition_id": "236716"}, profile_path)

                    with patch(
                        "kaggle_research_agent.cli_app.dacon_api.fetch_daily_submission_limit",
                        side_effect=AssertionError("must not fetch live from experiment_snapshot's hot path"),
                    ):
                        self.assertFalse(cli_app._dacon_daily_limit_is_known("demo"))


class InferNextTrialTest(unittest.TestCase):
    def test_a_locally_completed_but_unsubmitted_trial_counts_as_done(self):
        # Submission is now optional (daily-limited competitions skip it),
        # so a trial with only a local_score must still be treated as
        # completed for numbering -- otherwise its number could be reused.
        manual = [
            {"trial_id": "trial_001", "local_score": 0.9, "lb_score": None},
        ]
        self.assertEqual("trial_002", cli_app._infer_next_trial(manual))

    def test_a_submitted_trial_still_counts_as_done(self):
        manual = [
            {"trial_id": "trial_001", "local_score": None, "lb_score": 0.9},
        ]
        self.assertEqual("trial_002", cli_app._infer_next_trial(manual))


class RollingWindowSubmissionStatsTest(unittest.TestCase):
    # Not wired into any live code path -- kept working and tested because
    # Kaggle resets its daily submission limit 24h after a team's first
    # submission of the day, unlike DACON's local-midnight model, so this is
    # the function to reach for if/when Kaggle gets the same display feature.
    def test_computes_remaining_and_reset_from_rolling_24h_window(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        recent = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        also_recent = (now - timedelta(hours=20)).strftime("%Y-%m-%d %H:%M:%S")
        stale = (now - timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")

        with patch(
            "kaggle_research_agent.cli_app.dacon_api.fetch_my_submissions",
            return_value={
                "ok": True,
                "submissions": [
                    {"c_time": recent},
                    {"c_time": also_recent},
                    {"c_time": stale},  # outside the 24h window, must not count
                ],
            },
        ):
            result = cli_app._rolling_24h_submission_window_stats("236716", 5)

        self.assertEqual(3, result["remaining"])  # 5 - 2 counted-in-window submissions
        expected_reset = (now - timedelta(hours=20) + timedelta(hours=24)).replace(microsecond=0)
        self.assertEqual(expected_reset, result["next_reset_estimate"])


if __name__ == "__main__":
    unittest.main()
