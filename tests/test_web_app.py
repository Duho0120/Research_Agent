import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import web_app


class WebAppTest(unittest.TestCase):
    def test_app_state_renders_snapshot(self):
        with patch("kaggle_research_agent.web_app.selected_competition", return_value="demo"):
            with patch(
                "kaggle_research_agent.web_app.experiment_snapshot",
                return_value={"competition": "demo", "topic": "Demo", "state": "대기 중", "latest": {}, "best": {}},
            ):
                with patch("kaggle_research_agent.web_app.load_experiments", return_value=[]):
                    with patch("kaggle_research_agent.web_app.list_pending_requests", return_value={"data": {"requests": []}}):
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
        self.assertIn("min-width: 390px", html)
        self.assertIn('chatResizeHandle.addEventListener("pointerdown"', html)
        self.assertIn('localStorage.setItem(chatSizeStorageKey', html)
        self.assertIn('data-open-chat', html)
        self.assertIn('event.key === "Enter" && !event.shiftKey', html)
        self.assertIn("URL 또는 실험을 설명해주세요.", html)
        self.assertIn("분석하기", html)

    def test_resolve_project_file_blocks_paths_outside_project(self):
        with self.assertRaises(ValueError):
            web_app.resolve_project_file("../outside.txt")

    def test_start_from_form_uses_trial_count_or_continuous_mode(self):
        with patch("kaggle_research_agent.web_app.selected_competition", return_value="demo"):
            with patch("kaggle_research_agent.web_app.start_experiment", return_value="started") as start:
                self.assertEqual("started", web_app.start_from_form({"trial_count": ["3"]}))
                start.assert_called_once_with("demo", trial_count=3)

        with patch("kaggle_research_agent.web_app.selected_competition", return_value="demo"):
            with patch("kaggle_research_agent.web_app.start_experiment", return_value="running") as start:
                self.assertEqual("running", web_app.start_from_form({"continuous": ["1"], "trial_count": ["2"]}))
                start.assert_called_once_with("demo", continuous=True)

    def test_record_insight_requires_text(self):
        result = web_app.record_insight("")
        self.assertFalse(result["ok"])
        self.assertIn("인사이트", result["message"])

    def test_record_insight_returns_readable_summary(self):
        insight = "서로 다른 모델을 비교하고 앙상블하자"
        with patch(
            "kaggle_research_agent.web_app.app_state",
            return_value={
                "snapshot": {
                    "competition": "demo",
                    "last_completed_trial": "trial_007",
                    "next_trial": "trial_008",
                }
            },
        ):
            with patch(
                "kaggle_research_agent.web_app.submit_human_insight",
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
                with patch("kaggle_research_agent.web_app._keep_latest_user_insight"):
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
        with patch("kaggle_research_agent.web_app.respond_to_request", return_value={"ok": True}) as respond:
            message = web_app.record_feedback_response({"request_id": ["req-1"], "answer": ["반영해주세요"]})
        respond.assert_called_once_with("req-1", free_text="반영해주세요")
        self.assertIn("답변을 기록했습니다", message)

    def test_create_experiment_from_form_registers_workspace(self):
        with patch("kaggle_research_agent.web_app.prepare_workspace", return_value={"status": "needs_data"}) as prepare:
            with patch("kaggle_research_agent.web_app.select_competition") as select:
                message = web_app.create_experiment_from_form(
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
        with patch("kaggle_research_agent.web_app._sqlite_trial_rows", return_value=rows):
            with patch("kaggle_research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                with patch(
                    "kaggle_research_agent.web_app.user_insight_target_trial_ids",
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
            with patch("kaggle_research_agent.web_app._sqlite_trial_rows", return_value=rows):
                with patch("kaggle_research_agent.web_app.render_sqlite_trial_detail", return_value="detail"):
                    with patch("kaggle_research_agent.web_app.user_insight_target_trial_ids", return_value=set()):
                        with patch("kaggle_research_agent.web_app.project_root", return_value=root):
                            html = web_app.trial_table("demo")

        self.assertIn("계획 완료", html)
        self.assertIn(">Build a diverse two-model ensemble<", html)
        self.assertNotIn(">model_ensemble<", html)
        self.assertIn('class="artifact-chip disabled">구조</span>', html)
        self.assertIn('class="artifact-chip disabled">점수</span>', html)
        self.assertIn(">계획</a>", html)

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
            "kaggle_research_agent.web_app.latest_user_insight_record",
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


if __name__ == "__main__":
    unittest.main()
