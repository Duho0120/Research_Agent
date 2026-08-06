import http.client
import json
import tempfile
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from research_agent import web_app


def _multipart_body(boundary: str, files: list[tuple[str, bytes]]) -> bytes:
    chunks: list[bytes] = []
    for filename, content in files:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
        chunks.append(header + content + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


class _QuietHandler(web_app.ResearchAgentHandler):
    """ResearchAgentHandler, minus per-request stderr logging.

    BaseHTTPRequestHandler logs each request via sys.stderr from the
    request-handling thread. Under pytest's output capture that write races
    the test thread's own capture teardown and hangs the run rather than
    failing it -- silently, with no traceback, which is worse than not
    testing this at all. The dispatch logic under test is unaffected.
    """

    def log_message(self, format, *args):  # noqa: A002 - matches base signature
        pass


class SubmitTrialEndpointTest(unittest.TestCase):
    """Real incident: a submission's score landed correctly in
    submission_run.json, but the dashboard reads from the state DB and
    nothing synced it there -- the score sat on disk, invisible, until
    someone happened to click the manual refresh link. This starts a real
    server rather than calling the handler function directly, because the
    bug was in the HTTP dispatch wiring itself, not in any function a direct
    call could exercise in isolation."""

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.thread.join, 5)

    def _post_submit(self, competition: str, trial_id: str) -> dict:
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=10)
        body = f"competition={competition}&trial_id={trial_id}"
        connection.request(
            "POST", "/api/submit-trial", body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return payload

    def test_a_successful_submission_is_synced_into_the_state_db(self):
        with patch.object(
            web_app, "submit_trial_manually",
            return_value={"ok": True, "status": "submitted", "submitted_lb_score": 0.472},
        ):
            with patch.object(web_app, "sync_state_db") as sync_fn:
                self._post_submit("demo", "trial_004")
        sync_fn.assert_called_once_with("demo")

    def test_a_blocked_submission_is_still_reported_even_if_sync_fails(self):
        """The submission's own result must not be swallowed by a display
        refresh that happens to fail."""
        with patch.object(
            web_app, "submit_trial_manually",
            return_value={"ok": False, "status": "missing_team_name", "message": "x"},
        ):
            with patch.object(web_app, "sync_state_db", side_effect=RuntimeError("db locked")):
                result = self._post_submit("demo", "trial_004")
        self.assertEqual("missing_team_name", result["status"])

    def test_missing_trial_id_never_reaches_submit_or_sync(self):
        with patch.object(web_app, "submit_trial_manually") as submit_fn:
            with patch.object(web_app, "sync_state_db") as sync_fn:
                result = self._post_submit("demo", "")
        self.assertFalse(result["ok"])
        submit_fn.assert_not_called()
        sync_fn.assert_not_called()


class WebAppTest(unittest.TestCase):
    def test_app_state_renders_snapshot(self):
        with patch("research_agent.web_app.selected_competition", return_value="demo"):
            with patch(
                "research_agent.web_app.experiment_snapshot",
                return_value={"competition": "demo", "topic": "Demo", "state": "대기 중", "latest": {}, "best": {}},
            ):
                with patch("research_agent.web_app.load_experiments", return_value=[]):
                    with patch("research_agent.web_app.list_pending_requests", return_value={"data": {"requests": []}}):
                        payload = web_app.app_state()
        self.assertTrue(payload["ok"])
        self.assertIn("선택된 실험: demo", payload["text"])

    def test_render_home_renders_dashboard_actions(self):
        html = web_app.render_home(
            {
                "ok": True,
                "snapshot": {"competition": "demo", "state": "대기 중", "next_trial": "trial_001", "latest": {}, "best": {}},
                "experiments": [{"competition": "demo", "topic": "Demo", "state": "대기 중"}],
                "pending_requests": [],
                "text": "선택된 실험: demo",
            }
        )
        for label in [
            "현재 실험 대시보드",
            "Trial 목록",
            "실험 제어",
            "실험 바꾸기",
            "새 실험 등록",
            "자동 실험 시작",
            "현재 실험 중단 요청",
            "다음 실험 인사이트",
            "피드백 요청",
            "폴더 / DB 위치",
        ]:
            self.assertIn(label, html)
        self.assertNotIn("<h2>사용자용 산출물</h2>", html)
        self.assertNotIn("관리자 / 개발자 정보", html)
        self.assertNotIn("에이전트가 연구 판단을 요청한 경우 여기에 표시됩니다.", html)
        self.assertNotIn('class="context-grid"', html)
        self.assertIn('data-open-modal="control-modal"', html)
        self.assertIn('data-open-modal="new-experiment-modal"', html)
        self.assertIn('data-open-modal="insight-modal"', html)
        self.assertIn('id="control-modal"', html)
        self.assertIn('id="dashboard-experiment"', html)
        self.assertIn('class="header-experiment-switch"', html)
        self.assertEqual(1, html.count("실험 바꾸기"))
        self.assertIn("data-continuous-toggle", html)
        self.assertIn("data-trial-count", html)
        self.assertIn("countInput.disabled = toggle.checked", html)
        self.assertIn('id="start-experiment-form"', html)
        self.assertIn('class="control-actions divided"', html)
        self.assertIn('form="start-experiment-form"', html)
        self.assertIn('id="insight-modal"', html)
        self.assertIn('class="modal modal-large insight-modal"', html)
        self.assertIn("trial_001은 자동으로 실행되지 않습니다.", html)
        self.assertIn('id="question-form"', html)
        self.assertIn('action="/api/question"', html)
        self.assertIn('id="chat-log"', html)
        self.assertIn('id="chat-fab"', html)
        self.assertIn('id="chat-widget"', html)
        self.assertIn('id="chat-resize-handle"', html)
        self.assertIn('class="chat-resize-handle"', html)
        self.assertIn('id="chat-session-select"', html)
        self.assertIn('id="chat-new-session"', html)
        self.assertIn("/api/chat/history", html)
        self.assertIn("/api/chat/session", html)
        self.assertIn("renderChatHistory(payload.history)", html)
        self.assertIn("읽기 전용 · 대화는 실험 계획, 코드, 점수, 연구 판단을 변경하지 않습니다.", html)
        self.assertIn("명시적으로 저장한 인사이트만 다음 계획 단계의 입력으로 기록됩니다.", html)
        self.assertIn("읽기 전용 · 대화는 실험 계획, 코드, 점수, 연구 판단을 변경하지 않습니다.", html)
        self.assertIn("명시적으로 저장한 인사이트만 다음 계획 단계의 입력으로 기록됩니다.", html)
        self.assertIn("min-width: 390px", html)
        self.assertIn('chatResizeHandle.addEventListener("pointerdown"', html)
        self.assertIn('localStorage.setItem(chatSizeStorageKey', html)
        self.assertIn('data-open-chat', html)
        self.assertIn('event.key === "Enter" && !event.shiftKey', html)
        self.assertIn("URL 또는 실험을 설명해주세요.", html)
        self.assertIn("분석하기", html)

    def test_render_home_connects_pending_feedback_card_to_modal(self):
        request = {
            "request_id": "review_demo",
            "interaction_label": "도메인 정보 질문",
            "title": "예측 시점 정보 확인",
            "problem": "post_event_status에 leakage 경고가 감지됐습니다.",
            "question": "post_event_status는 실제 예측 시점에도 사용할 수 있나요?",
            "evidence_snapshot": [
                {
                    "label": "leakage 의심 정보",
                    "value": "post_event_status",
                    "meaning": "예측 시점 사용 가능 여부를 확인해야 합니다.",
                }
            ],
            "interpretation": "예측 이후 정보라면 로컬 점수를 신뢰할 수 없습니다.",
            "recommendation": "의심 피처를 제외하고 다시 검증합니다.",
            "why_user_needed": "정보 생성 시점은 사용자 확인이 필요합니다.",
            "options": [
                {
                    "value": "unknown_quarantine",
                    "label": "사용 가능 여부를 모름",
                    "impact": "의심 피처를 제외하고 재검증합니다.",
                }
            ],
            "default_if_no_response": "의심 피처를 제외하고 다시 검증합니다.",
        }
        html = web_app.render_home(
            {
                "ok": True,
                "snapshot": {
                    "competition": "demo",
                    "state": "피드백 대기",
                    "next_trial": "trial_002",
                    "latest": {},
                    "best": {},
                },
                "experiments": [{"competition": "demo", "topic": "Demo", "state": "ready"}],
                "pending_requests": [request],
            }
        )

        self.assertIn('data-open-modal="feedback-modal"', html)
        self.assertIn('id="feedback-modal"', html)
        self.assertIn("피드백 요청 (1)", html)
        self.assertIn("post_event_status에 leakage 경고", html)
        self.assertIn("사용 가능 여부를 모름", html)

    def test_resolve_project_file_blocks_paths_outside_project(self):
        with self.assertRaises(ValueError):
            web_app.resolve_project_file("../outside.txt")

    def test_artifact_href_normalizes_absolute_project_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            artifact = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            with patch("research_agent.web_app.project_root", return_value=root):
                href = web_app.artifact_href(artifact)

        self.assertEqual("/artifact?path=runs%2Fdemo%2Ftrial_001%2F01_plan.ko.md", href)

    def test_trial_artifact_links_hide_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            existing = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            existing.parent.mkdir(parents=True)
            existing.write_text("# plan\n", encoding="utf-8")
            rows = [
                {"artifact_type": "plan_ko", "path": str(existing)},
                {
                    "artifact_type": "scores_ko",
                    "path": str(root / "runs" / "demo" / "trial_001" / "03_scores.ko.md"),
                },
            ]
            connection = unittest.mock.MagicMock()
            connection.execute.return_value = rows
            manager = unittest.mock.MagicMock()
            manager.__enter__.return_value = connection
            with patch("research_agent.web_app.project_root", return_value=root):
                with patch("research_agent.web_app.state_db_connection", return_value=manager):
                    html = web_app.trial_artifact_links("demo", "trial_001")

        self.assertIn(">계획</button>", html)
        self.assertNotIn(">점수</button>", html)

    def test_trial_artifact_links_prefers_korean_preview_for_planned_trial(self):
        # Once prepare_workspace_trial_plan writes user_view/01_plan.ko.md
        # ahead of execution, the "계획" chip for a not-yet-run trial should
        # open that Korean-labeled preview instead of the raw English
        # next_experiment.md/demo_experiment_plan.md fallback.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            out_dir = root / "experiments" / "demo" / "trial_013"
            out_dir.mkdir(parents=True)
            (out_dir / "next_experiment.md").write_text("# trial_013 Demo Experiment Plan\n", encoding="utf-8")
            (out_dir / "user_view").mkdir()
            (out_dir / "user_view" / "01_plan.ko.md").write_text("# trial_013 실험 계획\n", encoding="utf-8")
            with patch("research_agent.web_app.project_root", return_value=root):
                html = web_app.trial_artifact_links("demo", "trial_013", planned=True)

        self.assertIn("data-open-artifact=\"experiments/demo/trial_013/user_view/01_plan.ko.md\"", html)

    def test_trial_artifact_links_open_in_modal_instead_of_navigating_away(self):
        # 구조/계획/점수 chips must open the artifact inside the in-page modal
        # (like the 실험 제어/새 실험 등록 buttons) instead of navigating the
        # dashboard away to a separate page.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            existing = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            existing.parent.mkdir(parents=True)
            existing.write_text("# plan\n", encoding="utf-8")
            rows = [{"artifact_type": "plan_ko", "path": str(existing)}]
            connection = unittest.mock.MagicMock()
            connection.execute.return_value = rows
            manager = unittest.mock.MagicMock()
            manager.__enter__.return_value = connection
            with patch("research_agent.web_app.project_root", return_value=root):
                with patch("research_agent.web_app.state_db_connection", return_value=manager):
                    html = web_app.trial_artifact_links("demo", "trial_001")

        self.assertIn("data-open-artifact=", html)
        self.assertNotIn("<a ", html)
        self.assertNotIn('target="_blank"', html)

    def test_artifact_panel_links_open_in_modal_instead_of_navigating_away(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            user_dir = root / "runs" / "demo" / "trial_001" / "user_view"
            user_dir.mkdir(parents=True)
            (user_dir / "01_plan.ko.md").write_text("# plan\n", encoding="utf-8")
            with patch("research_agent.web_app.project_root", return_value=root):
                with patch(
                    "research_agent.web_app.safe_artifact_locations",
                    return_value={"user_view": user_dir},
                ):
                    html = web_app.artifact_panel("demo", {})

        self.assertIn("data-open-artifact=", html)
        self.assertNotIn("<a ", html)

    def test_api_artifact_returns_json_content_for_modal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            existing = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            existing.parent.mkdir(parents=True)
            existing.write_text("# plan body\n", encoding="utf-8")
            with patch("research_agent.web_app.project_root", return_value=root):
                loaded = web_app.load_artifact_content("runs/demo/trial_001/01_plan.ko.md")

        self.assertEqual("# plan body\n", loaded["content"])
        self.assertEqual("실험 계획서", loaded["title"])

    def test_trial_table_marks_local_completion_waiting_for_recovery(self):
        rows = [
            {
                "trial_id": "trial_001",
                "status": "completed",
                "local_score": 1.0,
                "lb_score": None,
                "change_axis": "baseline",
                "improvement_plan": "initial",
                "is_best_local": True,
                "is_best_lb": False,
            }
        ]
        snapshot = {
            "loop": {
                "status": "failed",
                "next_trial": "trial_001",
                "error": "recoverable_after_metrics_collection",
            }
        }
        with patch("research_agent.web_app._sqlite_trial_rows", return_value=rows):
            with patch("research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                with patch("research_agent.web_app.user_insight_target_trial_ids", return_value=set()):
                    with patch("research_agent.web_app.trial_artifact_links", return_value="-"):
                        html = web_app.trial_table("demo", snapshot=snapshot)

        self.assertIn("로컬 완료 · 후처리 대기", html)

    def test_trial_table_shows_source_trial_column(self):
        rows = [
            {
                "trial_id": "trial_004",
                "status": "completed",
                "source_trial_id": "trial_003",
                "local_score": 0.591,
                "lb_score": None,
                "change_axis": "model_family_switch",
                "improvement_plan": "try ExtraTrees",
                "is_best_local": True,
                "is_best_lb": False,
            }
        ]
        with patch("research_agent.web_app._sqlite_trial_rows", return_value=rows):
            with patch("research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                with patch("research_agent.web_app.user_insight_target_trial_ids", return_value=set()):
                    with patch("research_agent.web_app.trial_artifact_links", return_value="-"):
                        html = web_app.trial_table("demo", snapshot={})

        self.assertIn("<th>기준</th>", html)
        self.assertIn("<td>trial_003</td>", html)

    def test_trial_table_shows_submit_button_only_when_not_yet_submitted(self):
        rows = [
            {"trial_id": "trial_004", "status": "completed", "local_score": 0.591, "lb_score": None},
            {"trial_id": "trial_003", "status": "completed", "local_score": 0.591, "lb_score": 0.6006},
            {"trial_id": "trial_005", "status": "planned", "local_score": None, "lb_score": None},
        ]
        with patch("research_agent.web_app._sqlite_trial_rows", return_value=rows):
            with patch("research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                with patch("research_agent.web_app.user_insight_target_trial_ids", return_value=set()):
                    with patch("research_agent.web_app.trial_artifact_links", return_value="-"):
                        html = web_app.trial_table("demo", snapshot={})

        self.assertIn("<th>제출</th>", html)
        self.assertIn('data-trial-id="trial_004"', html)
        self.assertNotIn('data-trial-id="trial_003"', html)  # already submitted
        self.assertNotIn('data-trial-id="trial_005"', html)  # only planned, not run yet

    def test_resolve_project_file_accepts_absolute_path_inside_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            artifact = root / "runs" / "demo" / "trial_001" / "01_plan.ko.md"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("plan", encoding="utf-8")
            with patch("research_agent.web_app.project_root", return_value=root):
                resolved = web_app.resolve_project_file(str(artifact))

        self.assertEqual(artifact, resolved)

    def test_chat_header_identifies_api_free_demo_mode(self):
        with patch.dict("os.environ", {"RESEARCH_AGENT_CHAT_DEMO_MODE": "1"}, clear=False):
            html = web_app.floating_chat()
        self.assertIn("DEMO · API 없이 로컬 근거로 답변", html)
        self.assertIn("읽기 전용", html)
        self.assertIn("읽기 전용", html)

    def test_start_from_form_uses_trial_count_or_continuous_mode(self):
        with patch("research_agent.web_app.selected_competition", return_value="demo"):
            with patch("research_agent.web_app.start_experiment", return_value="started") as start:
                self.assertEqual("started", web_app.start_from_form({"trial_count": ["3"]}))
                start.assert_called_once_with("demo", trial_count=3)

        with patch("research_agent.web_app.selected_competition", return_value="demo"):
            with patch("research_agent.web_app.start_experiment", return_value="running") as start:
                self.assertEqual("running", web_app.start_from_form({"continuous": ["1"], "trial_count": ["2"]}))
                start.assert_called_once_with("demo", continuous=True)

    def test_record_insight_requires_text(self):
        result = web_app.record_insight("")
        self.assertFalse(result["ok"])
        self.assertIn("인사이트", result["message"])

    def test_record_insight_returns_readable_summary(self):
        insight = "서로 다른 모델을 비교하고 앙상블하자"
        with patch(
            "research_agent.web_app.app_state",
            return_value={
                "snapshot": {
                    "competition": "demo",
                    "last_completed_trial": "trial_007",
                    "next_trial": "trial_008",
                }
            },
        ):
            with patch(
                "research_agent.web_app.submit_human_insight",
                return_value={
                    "ok": True,
                    "data": {
                        "feedback": {},
                        "insight": {
                            "status": "pending",
                            "axis": "model_ensemble",
                            "interpretation": {
                                "implementation_intent": {"change": "Build a diverse ensemble."}
                            },
                        },
                    },
                },
            ):
                with patch("research_agent.web_app._keep_latest_user_insight"):
                    result = web_app.record_insight(insight)

        self.assertTrue(result["ok"])
        self.assertEqual(
            [
                "인사이트:",
                insight,
                "",
                "- 반영 예정: trial_008",
                "- 적용 개선안: Build a diverse ensemble.",
                "- 상태: pending",
                "- 개선축: model_ensemble",
            ],
            result["message"].splitlines(),
        )

    def test_feedback_response_records_answer(self):
        with patch("research_agent.web_app.respond_to_request", return_value={"ok": True}) as respond:
            message = web_app.record_feedback_response({"request_id": ["req-1"], "answer": ["반영해주세요"]})
        respond.assert_called_once_with("req-1", answers={}, free_text="반영해주세요")
        self.assertIn("답변을 기록했습니다", message)

    def test_feedback_response_records_structured_choice_and_optional_note(self):
        with patch("research_agent.web_app.respond_to_request", return_value={"ok": True}) as respond:
            message = web_app.record_feedback_response(
                {
                    "request_id": ["req-2"],
                    "decision": ["estimate_first"],
                    "answer": [""],
                }
            )
        respond.assert_called_once_with("req-2", answers={"decision": "estimate_first"}, free_text="")
        self.assertIn("답변을 기록했습니다", message)

    def test_pending_list_shows_evidence_recommendation_option_impact_and_execution_boundary(self):
        rendered = web_app.pending_list(
            [
                {
                    "request_id": "req-compute",
                    "interaction_label": "외부 계산 자원 승인",
                    "title": "계산 자원 확인",
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
            ]
        )
        self.assertIn("11.6GB", rendered)
        self.assertIn("로컬 경량화", rendered)
        self.assertIn("비용 추정 후 결정", rendered)
        self.assertIn("자동 실행하지 않습니다", rendered)

    def test_create_experiment_from_form_registers_workspace(self):
        with patch("research_agent.web_app.prepare_workspace", return_value={"status": "needs_data"}) as prepare:
            with patch("research_agent.web_app.select_competition") as select:
                message, competition = web_app.create_experiment_from_form(
                    {
                        "description": ["https://www.kaggle.com/competitions/playground-series-s4e1"],
                        "competition": ["playground-series-s4e1"],
                        "topic": ["Playground"],
                        "platform": ["kaggle"],
                        "metric": ["accuracy"],
                        "objective": ["maximize"],
                        "target_column": ["target"],
                        "id_column": ["id"],
                        "required_data_files": ["train.csv,test.csv"],
                        "create_workspace": ["1"],
                    }
                )
        prepare.assert_called_once()
        self.assertEqual("playground-series-s4e1", prepare.call_args.args[0])
        self.assertEqual("Playground", prepare.call_args.kwargs["topic"])
        self.assertEqual("accuracy", prepare.call_args.kwargs["metric"])
        self.assertEqual("target", prepare.call_args.kwargs["target_column"])
        self.assertEqual(["train.csv", "test.csv"], prepare.call_args.kwargs["required_data_files"])
        select.assert_called_once_with("playground-series-s4e1")
        self.assertIn("등록하고 선택했습니다", message)
        self.assertEqual("playground-series-s4e1", competition)

    def test_create_experiment_from_form_returns_none_competition_on_error(self):
        message, competition = web_app.create_experiment_from_form({})
        self.assertIsNone(competition)

    def test_create_experiment_from_form_sets_dacon_team_name_when_provided(self):
        with patch("research_agent.web_app.prepare_workspace", return_value={"status": "needs_data"}):
            with patch("research_agent.web_app.select_competition"):
                with patch("research_agent.web_app.set_dacon_team_name") as set_team:
                    with patch("research_agent.web_app.refresh_dacon_competition_docs") as refresh_docs:
                        web_app.create_experiment_from_form(
                            {
                                "description": ["https://dacon.io/competitions/official/236716"],
                                "competition": ["236716"],
                                "topic": ["Mosquito"],
                                "platform": ["dacon"],
                                "metric": ["r_hit"],
                                "objective": ["maximize"],
                                "create_workspace": ["1"],
                                "dacon_team_name": ["뚜로"],
                            }
                        )
        set_team.assert_called_once_with("236716", "뚜로")
        refresh_docs.assert_called_once_with("236716")

    def test_create_experiment_from_form_skips_doc_refresh_for_non_dacon_platform(self):
        with patch("research_agent.web_app.prepare_workspace", return_value={"status": "needs_data"}):
            with patch("research_agent.web_app.select_competition"):
                with patch("research_agent.web_app.refresh_dacon_competition_docs") as refresh_docs:
                    web_app.create_experiment_from_form(
                        {
                            "description": ["https://www.kaggle.com/competitions/titanic"],
                            "competition": ["titanic"],
                            "topic": ["Titanic"],
                            "platform": ["kaggle"],
                            "metric": ["accuracy"],
                            "objective": ["maximize"],
                            "create_workspace": ["1"],
                        }
                    )
        refresh_docs.assert_not_called()

    def test_create_experiment_from_form_skips_team_name_when_blank(self):
        with patch("research_agent.web_app.prepare_workspace", return_value={"status": "needs_data"}):
            with patch("research_agent.web_app.select_competition"):
                with patch("research_agent.web_app.set_dacon_team_name") as set_team:
                    web_app.create_experiment_from_form(
                        {
                            "description": ["https://www.kaggle.com/competitions/titanic"],
                            "competition": ["titanic"],
                            "topic": ["Titanic"],
                            "platform": ["kaggle"],
                            "metric": ["accuracy"],
                            "objective": ["maximize"],
                            "create_workspace": ["1"],
                        }
                    )
        set_team.assert_not_called()

    def test_delete_experiment_from_form_blocks_on_mismatched_confirmation_text(self):
        with patch("research_agent.web_app._filesystem_topic", return_value="모기 비행 궤적 예측"):
            with patch("research_agent.web_app.delete_experiment") as delete:
                message = web_app.delete_experiment_from_form(
                    {"competition": ["236716"], "confirm_text": ["모기 비행 궤적 예측"]}
                )
        delete.assert_not_called()
        self.assertIn("일치하지 않습니다", message)
        self.assertIn("모기 비행 궤적 예측 지우기", message)

    def test_delete_experiment_from_form_deletes_on_exact_confirmation_text(self):
        with patch("research_agent.web_app._filesystem_topic", return_value="모기 비행 궤적 예측"):
            with patch("research_agent.web_app.delete_experiment", return_value={"ok": True}) as delete:
                message = web_app.delete_experiment_from_form(
                    {"competition": ["236716"], "confirm_text": ["모기 비행 궤적 예측 지우기"]}
                )
        delete.assert_called_once_with("236716")
        self.assertEqual("236716 | 모기 비행 궤적 예측 실험을 삭제했습니다.", message)

    def test_delete_experiment_modal_shows_expected_confirmation_text(self):
        html = web_app.delete_experiment_modal({"competition": "236716", "topic": "모기 비행 궤적 예측"})

        self.assertIn("모기 비행 궤적 예측 지우기", html)
        self.assertIn('data-expected="모기 비행 궤적 예측 지우기"', html)
        self.assertIn('id="delete-confirm-submit" disabled', html)

    def test_analyze_new_experiment_from_form_returns_editable_settings(self):
        settings, message = web_app.analyze_new_experiment_from_form(
            {
                "description": ["https://www.kaggle.com/competitions/titanic"],
                "research_direction": ["앙상블을 시도하고 싶어"],
            }
        )

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual("titanic", settings["competition"])
        self.assertEqual("앙상블을 시도하고 싶어", settings["research_direction"])
        self.assertIn("분석했습니다", message)

    def test_new_experiment_panel_renders_analysis_edit_form(self):
        html = web_app.new_experiment_panel(
            {
                "description": "https://www.kaggle.com/competitions/titanic",
                "competition": "titanic",
                "topic": "Titanic",
                "platform": "kaggle",
                "metric": "accuracy",
                "objective": "maximize",
                "target_column": "Survived",
                "id_column": "PassengerId",
                "required_data_files": ["train.csv", "test.csv"],
                "create_workspace": True,
                "research_direction": "",
            }
        )

        self.assertIn("에이전트 분석 결과", html)
        self.assertIn('name="competition" value="titanic"', html)
        self.assertIn('name="target_column" value="Survived"', html)
        self.assertIn('value="new_experiment_analyze"', html)
        self.assertIn('value="new_experiment"', html)

    def test_new_experiment_panel_platform_select_offers_kaggle_and_dacon_with_local_disabled(self):
        html = web_app.new_experiment_panel(
            {
                "description": "고객 이탈 예측",
                "competition": "churn",
                "topic": "Churn",
                "platform": "kaggle",
                "metric": "accuracy",
                "objective": "maximize",
                "target_column": "",
                "id_column": "",
                "required_data_files": [],
                "create_workspace": True,
                "research_direction": "",
            }
        )

        self.assertIn('<label for="new-experiment-platform">플랫폼</label>', html)
        self.assertIn('<select id="new-experiment-platform" name="platform">', html)
        self.assertIn('<option value="kaggle" selected >캐글</option>', html)
        self.assertIn('value="dacon"', html)
        local_option_start = html.index('value="local_research"')
        local_option = html[local_option_start : html.index("</option>", local_option_start)]
        self.assertIn("disabled", local_option)
        self.assertNotIn("disabled", html[html.index('value="kaggle"') : html.index("</option>", html.index('value="kaggle"'))])
        self.assertNotIn("disabled", html[html.index('value="dacon"') : html.index("</option>", html.index('value="dacon"'))])

    def test_trial_table_shows_only_submitted_best_badge(self):
        rows = [
            {
                "trial_id": "trial_001",
                "local_score": 0.82,
                "lb_score": 0.75,
                "is_best_local": True,
                "is_best_lb": False,
            },
            {
                "trial_id": "trial_002",
                "local_score": 0.81,
                "lb_score": 0.77,
                "change_axis": "model_ensemble_with_multiple_model_families",
                "improvement_plan": "compare_diverse_models_and_blend_predictions",
                "is_best_local": False,
                "is_best_lb": True,
            },
        ]
        with patch("research_agent.web_app._sqlite_trial_rows", return_value=rows):
            with patch("research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                with patch(
                    "research_agent.web_app.user_insight_target_trial_ids",
                    return_value={"trial_002"},
                ):
                    html = web_app.trial_table("demo")

        self.assertIn(">Best<", html)
        self.assertNotIn("best_local", html)
        self.assertNotIn("best_submit", html)
        self.assertIn('title="insight: model_ensemble_with_multiple_model_families"', html)
        self.assertIn('title="compare_diverse_models_and_blend_predictions"', html)
        self.assertIn("insight: model_ensemble", html)
        self.assertIn('class="cell-tooltip"', html)
        self.assertIn('data-page-size="5"', html)
        self.assertEqual(2, html.count("data-trial-row"))
        self.assertLess(html.index("trial_002"), html.index("trial_001"))

    def test_notice_formats_insight_summary_as_multiline_content(self):
        html = web_app.notice("인사이트:\n모델 비교\n\n- 반영 예정: trial_008")
        self.assertIn('<strong class="notice-title">인사이트:</strong>', html)
        self.assertIn('class="notice-body"', html)
        self.assertIn("모델 비교\n\n- 반영 예정", html)

    def test_trial_table_shows_planned_trial_with_only_plan_artifact_enabled(self):
        rows = [
            {
                "trial_id": "trial_008",
                "status": "planned",
                "local_score": None,
                "lb_score": None,
                "change_axis": "model_ensemble",
                "improvement_plan": "Build a diverse two-model ensemble",
                "is_best_local": False,
                "is_best_lb": False,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "experiments" / "demo" / "trial_008" / "next_experiment.md"
            plan.parent.mkdir(parents=True)
            plan.write_text("# trial_008 plan\n", encoding="utf-8")
            with patch("research_agent.web_app._sqlite_trial_rows", return_value=rows):
                with patch("research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                    with patch("research_agent.web_app.user_insight_target_trial_ids", return_value=set()):
                        with patch("research_agent.web_app.project_root", return_value=root):
                            html = web_app.trial_table("demo")

        self.assertIn("계획 완료", html)
        self.assertIn(">Build a diverse two-model ensemble<", html)
        self.assertNotIn(">model_ensemble<", html)
        self.assertIn('class="artifact-chip disabled">구조</span>', html)
        self.assertIn('class="artifact-chip disabled">점수</span>', html)
        self.assertIn(">계획</button>", html)

    def test_render_home_keeps_saved_insight_visible_after_refresh(self):
        payload = {
            "ok": True,
            "snapshot": {
                "competition": "demo",
                "state": "대기 중",
                "last_completed_trial": "trial_007",
                "next_trial": "trial_008",
                "latest": {},
                "best": {},
            },
            "experiments": [],
            "pending_requests": [],
            "existing_insight": "서로 다른 모델을 비교하고 앙상블하자",
        }
        with patch(
            "research_agent.web_app.latest_user_insight_record",
            return_value={
                "status": "pending",
                "axis": "model_ensemble",
                "target_trial": "trial_008",
                "interpretation": {
                    "implementation_intent": {"change": "Build a diverse ensemble."}
                },
            },
        ):
            html = web_app.render_home(payload)

        self.assertIn('<strong class="notice-title">인사이트:</strong>', html)
        self.assertIn("서로 다른 모델을 비교하고 앙상블하자", html)
        self.assertIn("- 반영 예정: trial_008", html)
        self.assertIn("- 적용 개선안: Build a diverse ensemble.", html)

    def test_render_home_hides_insight_notice_once_resolved(self):
        # existing_insight only carries raw feedback text (from user_feedback.jsonl)
        # and does not know whether that insight already finished its lifecycle.
        # latest_user_insight_record is the source of truth for that -- once an
        # insight is superseded/completed/exhausted it returns None, and the
        # banner must disappear instead of rendering placeholder "pending"/
        # "해석 대기" text for a resolved, unrelated old insight.
        payload = {
            "ok": True,
            "snapshot": {
                "competition": "demo",
                "state": "대기 중",
                "last_completed_trial": "trial_008",
                "next_trial": "trial_009",
                "latest": {},
                "best": {},
            },
            "experiments": [],
            "pending_requests": [],
            "existing_insight": "매월 1~19일을 학습하고 20~31일을 검증하자",
        }
        with patch("research_agent.web_app.latest_user_insight_record", return_value=None):
            html = web_app.render_home(payload)

        self.assertNotIn('class="notice-title">인사이트:</strong>', html)
        self.assertNotIn("해석 대기", html)

    def test_render_home_does_not_duplicate_just_recorded_insight_notice(self):
        payload = {
            "ok": True,
            "snapshot": {
                "competition": "demo",
                "state": "대기 중",
                "last_completed_trial": "trial_007",
                "next_trial": "trial_008",
                "latest": {},
                "best": {},
            },
            "experiments": [],
            "pending_requests": [],
            "existing_insight": "앙상블하자",
        }
        html = web_app.render_home(payload, message="인사이트:\n앙상블하자")
        self.assertEqual(1, html.count('class="notice-title"'))

    def test_render_home_includes_dacon_submission_limit_card(self):
        html = web_app.render_home(
            {
                "ok": True,
                "snapshot": {"competition": "demo", "state": "대기 중", "latest": {}, "best": {}},
                "experiments": [],
                "pending_requests": [],
            }
        )
        self.assertIn('id="dacon-submission-limit-value"', html)
        self.assertIn("/api/dacon-submission-limit", html)
        # "remaining / limit" formatting, e.g. "3 / 5"
        self.assertIn("data.remaining", html)
        self.assertIn("data.daily_submission_limit", html)
        self.assertIn("data.next_reset_estimate", html)
        self.assertIn('id="dacon-submission-limit-edit-toggle"', html)
        # Editing opens the shared modal (no backdrop-click-to-close, same as
        # every other modal in this app) instead of an inline form the user
        # is forced to save or abandon awkwardly.
        self.assertIn('data-open-modal="dacon-limit-modal"', html)
        self.assertIn('id="dacon-limit-modal"', html)
        self.assertIn('id="dacon-submission-limit-input"', html)
        self.assertIn('id="dacon-submission-limit-save"', html)
        self.assertIn('id="dacon-auto-submit-checkbox"', html)
        self.assertIn("/api/dacon-auto-submit", html)


class DaconSubmissionLimitSnapshotTest(unittest.TestCase):
    def test_not_applicable_for_non_dacon_platform(self):
        with patch("research_agent.web_app._load_profile_safely", return_value={"platform": "kaggle"}):
            result = web_app.dacon_submission_limit_snapshot("demo")
        self.assertEqual({"ok": True, "applicable": False}, result)

    def test_includes_auto_submit_flag_when_applicable(self):
        with patch("research_agent.web_app._load_profile_safely", return_value={"platform": "dacon"}):
            with patch(
                "research_agent.web_app.check_dacon_submission_limit",
                return_value={"status": "auto_detected", "daily_submission_limit": 5, "remaining": 3, "message": "..."},
            ):
                with patch("research_agent.web_app.dacon_auto_submit_allowed", return_value=True):
                    result = web_app.dacon_submission_limit_snapshot("demo")
        self.assertTrue(result["auto_submit"])

    def test_applicable_for_dacon_platform_returns_check_result(self):
        with patch("research_agent.web_app._load_profile_safely", return_value={"platform": "dacon"}):
            with patch(
                "research_agent.web_app.check_dacon_submission_limit",
                return_value={
                    "competition": "demo",
                    "status": "auto_detected",
                    "daily_submission_limit": 5,
                    "message": "규칙 페이지에서 자동으로 확인한 일일 제출 한도: 5회",
                },
            ):
                result = web_app.dacon_submission_limit_snapshot("demo")
        self.assertTrue(result["applicable"])
        self.assertEqual("auto_detected", result["status"])
        self.assertEqual(5, result["daily_submission_limit"])

    def test_network_error_does_not_crash_the_dashboard(self):
        with patch("research_agent.web_app._load_profile_safely", return_value={"platform": "dacon"}):
            with patch(
                "research_agent.web_app.check_dacon_submission_limit",
                side_effect=RuntimeError("network unreachable"),
            ):
                result = web_app.dacon_submission_limit_snapshot("demo")
        self.assertTrue(result["ok"])
        self.assertTrue(result["applicable"])
        self.assertEqual("unknown", result["status"])
        self.assertIsNone(result["daily_submission_limit"])


class UploadDataTest(unittest.TestCase):
    def test_parse_multipart_files_extracts_filename_and_content(self):
        body = _multipart_body(
            "BOUNDARY123",
            [("train.csv", b"id,target\n1,0\n"), ("sample_submission.csv", b"id,target\n1,0\n")],
        )
        files = web_app.parse_multipart_files("multipart/form-data; boundary=BOUNDARY123", body)

        self.assertEqual(
            [("train.csv", b"id,target\n1,0\n"), ("sample_submission.csv", b"id,target\n1,0\n")],
            files,
        )

    def test_parse_multipart_files_returns_empty_for_non_multipart_content_type(self):
        self.assertEqual([], web_app.parse_multipart_files("application/x-www-form-urlencoded", b"a=1"))

    def test_save_uploaded_files_writes_plain_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            result = web_app.save_uploaded_files([("train.csv", b"id,target\n1,0\n")], target)

            self.assertTrue(result["ok"])
            self.assertEqual(["train.csv"], result["saved"])
            self.assertEqual([], result["skipped"])
            self.assertEqual(b"id,target\n1,0\n", (target / "train.csv").read_bytes())

    def test_save_uploaded_files_strips_directory_components_from_plain_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            result = web_app.save_uploaded_files([("../../evil.csv", b"x")], target)

            self.assertEqual(["evil.csv"], result["saved"])
            self.assertTrue((target / "evil.csv").is_file())
            self.assertFalse((Path(tmp) / "evil.csv").exists())

    def test_save_uploaded_files_extracts_zip_preserving_folder_structure(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("sample_submission.csv", "id,target\n1,0\n")
            archive.writestr("train/TRAIN_0001.csv", "x,y,z\n1,2,3\n")
            archive.writestr("train/TRAIN_0002.csv", "x,y,z\n4,5,6\n")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            result = web_app.save_uploaded_files([("data.zip", buffer.getvalue())], target)

            self.assertEqual(
                {"sample_submission.csv", "train/TRAIN_0001.csv", "train/TRAIN_0002.csv"},
                {path.replace("\\", "/") for path in result["saved"]},
            )
            self.assertEqual("x,y,z\n1,2,3\n", (target / "train" / "TRAIN_0001.csv").read_text(encoding="utf-8"))

    def test_save_uploaded_files_skips_zip_slip_entries(self):
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../../evil.csv", "x")
            archive.writestr("safe.csv", "ok")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            result = web_app.save_uploaded_files([("data.zip", buffer.getvalue())], target)

            self.assertEqual(["safe.csv"], result["saved"])
            self.assertEqual(1, len(result["skipped"]))
            self.assertEqual("unsafe_path", result["skipped"][0]["reason"])
            self.assertFalse((Path(tmp) / "evil.csv").exists())

    def test_save_uploaded_files_reports_bad_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data"
            result = web_app.save_uploaded_files([("data.zip", b"not a real zip")], target)

        self.assertEqual([], result["saved"])
        self.assertEqual("bad_zip", result["skipped"][0]["reason"])

    def test_workspace_data_dir_reads_source_path_from_workspace_source_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp_dir = root / "competitions" / "demo"
            comp_dir.mkdir(parents=True)
            (comp_dir / "workspace_source.json").write_text(
                json.dumps({"source_path": str(root / "demo_workspaces" / "demo")}), encoding="utf-8"
            )
            with patch("research_agent.web_app.competition_dir", return_value=comp_dir):
                data_dir = web_app.workspace_data_dir("demo")

        self.assertEqual(root / "demo_workspaces" / "demo" / "data", data_dir)

    def test_workspace_data_dir_returns_none_when_source_record_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            comp_dir = Path(tmp) / "competitions" / "demo"
            with patch("research_agent.web_app.competition_dir", return_value=comp_dir):
                self.assertIsNone(web_app.workspace_data_dir("demo"))


if __name__ == "__main__":
    unittest.main()
