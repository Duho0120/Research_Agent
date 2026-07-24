import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.interface_contract import (
    get_experiment,
    get_pending_request,
    get_trial,
    list_experiments,
    list_pending_requests,
    preview_next_trial,
    respond_to_request,
    submit_human_insight,
)
from kaggle_research_agent.state_db import (
    create_pending_action,
    initialize_state_db,
    list_pending_actions,
    record_token_usage,
    upsert_competition,
    upsert_trial,
    upsert_trial_artifact,
    upsert_trial_decision,
    upsert_trial_score,
)


class InterfaceContractTest(unittest.TestCase):
    def test_list_experiments_returns_ui_ready_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_contract_fixture(db_path)

            result = list_experiments(sync=False, db_path=db_path)

            self.assertTrue(result["ok"])
            self.assertEqual("list_experiments", result["action"])
            self.assertFalse(result["source"]["synced"])
            self.assertEqual(1, result["data"]["experiment_count"])
            experiment = result["data"]["experiments"][0]
            self.assertEqual("demo", experiment["competition"])
            self.assertEqual("ready_next_trial", experiment["state"])
            self.assertEqual("trial_003", experiment["next_trial_id"])
            self.assertEqual({"name": "feature_engineering", "attempt_count": 2, "attempt_limit": 3}, experiment["active_axis"])

    def test_get_experiment_returns_trials_and_next_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_contract_fixture(db_path)

            result = get_experiment("demo", sync=False, db_path=db_path)

            self.assertTrue(result["ok"])
            self.assertEqual("get_experiment", result["action"])
            self.assertEqual("trial_002", result["data"]["experiment"]["best_trial"]["trial_id"])
            self.assertEqual(2, len(result["data"]["trials"]))
            self.assertEqual("feature_engineering", result["data"]["trials"][1]["active_axis"])
            self.assertTrue(any(item.get("action") == "preview_next_trial" for item in result["next_actions"]))
            self.assertTrue(any(item.get("action") == "run_next_trial" for item in result["next_actions"]))
            self.assertTrue(any(item.get("type") == "cli_command" for item in result["next_actions"]))

    def test_get_trial_returns_artifact_labels_and_token_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_contract_fixture(db_path)

            result = get_trial("demo", "trial_002", sync=False, db_path=db_path)

            self.assertTrue(result["ok"])
            trial = result["data"]["trial"]
            self.assertEqual("trial_002", trial["trial_id"])
            self.assertEqual("continue_axis_refinement", trial["decision"])
            self.assertEqual("trial_002", trial["recommended_base_trial"])
            self.assertEqual(240, trial["token_total"])
            self.assertEqual("실험 계획서", trial["user_artifacts"][0]["label"])

    def test_preview_next_trial_reports_runnable_policy_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_contract_fixture(db_path)

            result = preview_next_trial("demo", sync=False, db_path=db_path)

            self.assertTrue(result["ok"])
            self.assertEqual("preview_next_trial", result["action"])
            self.assertTrue(result["data"]["can_run"])
            self.assertEqual("trial_003", result["data"]["next_trial_id"])
            self.assertEqual({"name": "feature_engineering", "attempt_count": 2, "attempt_limit": 3}, result["data"]["active_axis"])
            self.assertTrue(any(item.get("action") == "run_next_trial" for item in result["next_actions"]))

    def test_missing_experiment_returns_stable_error_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            initialize_state_db(db_path)

            result = get_experiment("missing", sync=False, db_path=db_path)

            self.assertFalse(result["ok"])
            self.assertEqual("not_found", result["status"])
            self.assertEqual("experiment_not_found", result["errors"][0]["code"])

    def test_pending_requests_contract_lists_and_loads_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            _write_contract_fixture(db_path)
            _write_pending_request_fixture(db_path)

            result = list_pending_requests("demo", sync=False, db_path=db_path)

            self.assertTrue(result["ok"])
            self.assertEqual("list_pending_requests", result["action"])
            self.assertEqual(1, result["data"]["request_count"])
            request = result["data"]["requests"][0]
            self.assertEqual("review_001", request["request_id"])
            self.assertEqual("Feature review", request["title"])
            self.assertEqual("feature_review", request["topic"])
            self.assertEqual("runs/demo/trial_002/02_pipeline_structure.ko.md", request["context_files"][0]["path"])

            detail = get_pending_request("review_001", db_path=db_path)

            self.assertTrue(detail["ok"])
            self.assertEqual("pending", detail["status"])
            self.assertEqual("Keep the engineered family feature?", detail["data"]["request"]["question"])

    def test_respond_to_request_records_feedback_and_resolves_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite3"
            _write_contract_fixture(db_path)
            _write_pending_request_fixture(db_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = respond_to_request(
                    "review_001",
                    answers={"continue_axis": "continue", "follow_up_action": "try a smaller variant"},
                    free_text="The feature is plausible, but keep the next change narrow.",
                    db_path=db_path,
                )

            self.assertTrue(result["ok"])
            self.assertEqual("completed", result["status"])
            self.assertEqual("resolved", result["data"]["request"]["status"])
            interaction = result["data"]["interaction"]
            self.assertEqual("resolve_request", interaction["access"])
            self.assertEqual("requested_trial", interaction["scope"])
            self.assertTrue(interaction["requires_pending_request"])
            self.assertTrue(interaction["requires_explicit_submit"])
            self.assertEqual([], list_pending_actions("demo", db_path))
            self.assertEqual(["review_001"], [item["action_id"] for item in list_pending_actions("demo", db_path, status="resolved")])
            feedback_path = root / "memory" / "demo" / "user_feedback.jsonl"
            self.assertTrue(feedback_path.exists())
            self.assertIn("keep the next change narrow", feedback_path.read_text(encoding="utf-8"))

    def test_respond_to_request_requires_an_existing_pending_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite3"
            _write_contract_fixture(db_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = respond_to_request(
                    "missing_review",
                    free_text="This must not be recorded.",
                    db_path=db_path,
                )

            self.assertFalse(result["ok"])
            self.assertEqual("not_found", result["status"])
            self.assertEqual("human_review_response", result["data"]["interaction"]["channel"])
            self.assertFalse((root / "memory" / "demo" / "user_feedback.jsonl").exists())
            self.assertEqual([], list_pending_actions("demo", db_path, status="resolved"))

    def test_submit_human_insight_records_feedback_without_pending_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite3"
            _write_contract_fixture(db_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_human_insight(
                    "demo",
                    "trial_002",
                    insight="Avoid changing model family before finishing this feature axis.",
                    db_path=db_path,
                )

            self.assertTrue(result["ok"])
            self.assertEqual("submit_human_insight", result["action"])
            interaction = result["data"]["interaction"]
            self.assertEqual("write_intent", interaction["access"])
            self.assertEqual("next_trial", interaction["scope"])
            self.assertEqual(["user_feedback", "user_insight"], interaction["research_state_mutations"])
            self.assertTrue(interaction["requires_explicit_submit"])
            feedback_path = root / "memory" / "demo" / "user_feedback.jsonl"
            self.assertTrue(feedback_path.exists())
            self.assertIn("Avoid changing model family", feedback_path.read_text(encoding="utf-8"))


def _write_contract_fixture(db_path: Path) -> None:
    initialize_state_db(db_path)
    upsert_competition(
        {
            "competition_id": "demo",
            "platform": "kaggle",
            "topic": "Demo Competition",
            "metric": "accuracy",
            "objective": "maximize",
            "status": "has_trials",
            "workspace_path": "demo_workspaces/demo",
        },
        db_path,
    )
    for trial_id, score, is_best in [("trial_001", 0.81, False), ("trial_002", 0.84, True)]:
        upsert_trial(
            {
                "competition_id": "demo",
                "trial_id": trial_id,
                "status": "completed",
                "primary_change_axis": "feature_engineering",
                "recommended_base_trial": "trial_002",
            },
            db_path,
        )
        upsert_trial_score(
            {
                "competition_id": "demo",
                "trial_id": trial_id,
                "metric": "accuracy",
                "objective": "maximize",
                "local_score": score,
                "is_best_local": is_best,
            },
            db_path,
        )
    upsert_trial_decision(
        {
            "competition_id": "demo",
            "trial_id": "trial_002",
            "decision": "continue_axis_refinement",
            "change_axis": "feature_engineering",
            "active_axis": "feature_engineering",
            "axis_attempt_count": 2,
            "axis_attempt_limit": 3,
            "recommended_base_trial": "trial_002",
        },
        db_path,
    )
    upsert_trial_artifact(
        {
            "competition_id": "demo",
            "trial_id": "trial_002",
            "artifact_type": "plan_ko",
            "path": "runs/demo/trial_002/01_plan.ko.md",
            "is_user_facing": True,
        },
        db_path,
    )
    record_token_usage(
        {
            "source_key": "memory/demo/token_usage.jsonl:1",
            "competition_id": "demo",
            "trial_id": "trial_002",
            "provider": "openai",
            "model": "gpt-5.5",
            "call_type": "experiment_planning",
            "input_tokens": 180,
            "output_tokens": 60,
            "total_tokens": 240,
        },
        db_path,
    )


def _write_pending_request_fixture(db_path: Path) -> None:
    create_pending_action(
        {
            "action_id": "review_001",
            "competition_id": "demo",
            "trial_id": "trial_002",
            "action_type": "human_review",
            "priority": 5,
            "message": "Please review whether to keep this feature-engineering axis.",
            "payload": {
                "title": "Feature review",
                "topic": "feature_review",
                "question": "Keep the engineered family feature?",
                "context_files": [
                    {
                        "label": "Pipeline structure",
                        "path": "runs/demo/trial_002/02_pipeline_structure.ko.md",
                    }
                ],
                "questions": [
                    {
                        "id": "continue_axis",
                        "label": "Continue the current axis?",
                        "answer_type": "choice",
                        "choices": [
                            {"value": "continue", "label": "Continue"},
                            {"value": "switch_axis", "label": "Switch axis"},
                        ],
                        "required": True,
                    }
                ],
            },
        },
        db_path,
    )


if __name__ == "__main__":
    unittest.main()
