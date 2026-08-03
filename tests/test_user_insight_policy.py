from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.interface_contract import submit_human_insight
from kaggle_research_agent.user_insight_policy import (
    build_next_trial_user_insight_override,
    evaluate_user_insight_submission,
    interpret_user_insight,
    latest_user_insight_record,
    mark_user_insight_applied,
    update_user_insight_record,
    validate_user_insight_code_result,
)


class FakeInsightClient:
    def __init__(self) -> None:
        self.calls = 0

    def create_response(self, payload):
        self.calls += 1
        return {
            "id": "resp_insight",
            "usage": {"input_tokens": 40, "output_tokens": 30, "total_tokens": 70},
            "output": [
                {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "axis": "feature_engineering",
                                    "implementation_intent": {
                                        "change": "Add interaction features",
                                        "keep_unchanged": ["validation"],
                                        "verification": "Feature metadata changes",
                                    },
                                }
                            )
                        }
                    ]
                }
            ],
        }


class UserInsightPolicyTest(unittest.TestCase):
    def test_clear_insight_is_rule_based_without_llm(self):
        client = FakeInsightClient()
        result = interpret_user_insight(
            "2~3개의 서로 다른 모델을 앙상블하자",
            allow_api=True,
            client=client,
        )
        self.assertEqual("model_ensemble", result["axis"])
        self.assertEqual("rules", result["source"])
        self.assertEqual(0, client.calls)

    def test_ambiguous_insight_uses_low_cost_llm_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeInsightClient()
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = interpret_user_insight(
                    "관계를 조금 더 섬세하게 활용해보자",
                    competition="demo",
                    trial_id="trial_001",
                    allow_api=True,
                    client=client,
                )
        self.assertEqual("feature_engineering", result["axis"])
        self.assertEqual("low_cost_llm", result["source"])
        self.assertEqual(1, client.calls)

    def test_new_insight_preserves_history_and_supersedes_previous_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                first = submit_human_insight("demo", "trial_001", insight="앙상블을 사용하자")
                second = submit_human_insight("demo", "trial_001", insight="검증 누수를 먼저 확인하자")
                latest = latest_user_insight_record("demo", "trial_001")
            rows = [
                json.loads(line)
                for line in (root / "memory" / "demo" / "user_insights.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        self.assertNotEqual(first["data"]["insight"]["insight_id"], second["data"]["insight"]["insight_id"])
        self.assertEqual(2, len(rows))
        self.assertEqual("superseded", rows[0]["status"])
        self.assertEqual("validation_review", latest["axis"])

    def test_improvement_compares_against_fixed_pre_insight_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "demo_workspaces" / "demo" / "manual_trials"
            for trial_id, score in [("trial_003", 0.77), ("trial_006", 0.75)]:
                path = manual / trial_id
                path.mkdir(parents=True)
                (path / "metrics.json").write_text(
                    json.dumps({"trial_id": trial_id, "kaggle_lb_score": score}),
                    encoding="utf-8",
                )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_006", insight="앙상블을 사용하자")
                first = build_next_trial_user_insight_override("demo", "trial_006", "trial_007")
                trial_007 = manual / "trial_007"
                trial_007.mkdir()
                (trial_007 / "metrics.json").write_text(
                    json.dumps({"trial_id": "trial_007", "kaggle_lb_score": 0.78}),
                    encoding="utf-8",
                )
                context_dir = root / "experiments" / "demo" / "trial_007"
                context_dir.mkdir(parents=True)
                (context_dir / "continuation_context.json").write_text(
                    json.dumps({"decision_context": {"user_insight_override": first}}),
                    encoding="utf-8",
                )
                second = build_next_trial_user_insight_override("demo", "trial_007", "trial_008")
        self.assertEqual(0.77, first["baseline_lb_score"])
        self.assertTrue(second["source_improved_submission_score"])
        self.assertEqual("trial_007", second["base_trial_id"])
        self.assertEqual(1, second["axis_attempt_count"])

    def test_ensemble_application_is_validated_and_lifecycle_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="앙상블을 사용하자")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                missing = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {"status": "completed", "patch_updates": [{"replace": "LogisticRegression()"}]},
                )
                present = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {"status": "completed", "patch_updates": [{"replace": "VotingClassifier(estimators=members)"}]},
                )
                applied = mark_user_insight_applied(
                    "demo",
                    "trial_002",
                    {
                        "primary_change_axis": "model_ensemble",
                        "model": {
                            "estimator": "VotingClassifier",
                            "members": [{"estimator": "LR"}, {"estimator": "RF"}],
                        },
                    },
                )
                evaluated = evaluate_user_insight_submission("demo", "trial_002", 0.78)
                next_override = build_next_trial_user_insight_override("demo", "trial_002", "trial_003")
        self.assertEqual(
            ["user_insight_not_implemented:model_ensemble:model_change_present_in_code"],
            missing,
        )
        self.assertEqual([], present)
        self.assertEqual("applied", applied["status"])
        self.assertEqual("completed", evaluated["status"])
        self.assertIsNone(next_override)

    def test_application_check_is_still_recorded_when_submission_resolves_first(self):
        # The real pipeline runs submission (evaluate_user_insight_submission, via
        # agents/submission.py) before the result cycle (mark_user_insight_applied,
        # via workspace_result_cycle.py) -- the opposite order from the test above.
        # By the time mark_user_insight_applied looks the record up for the same
        # trial, evaluate_user_insight_submission may have already moved its status
        # to a terminal one ("completed"/"exhausted"). That must not make
        # _insight_for_target_trial treat it as a finished, unrelated insight and
        # silently skip recording applied_trials/application_check for the exact
        # trial that just resolved it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="앙상블을 사용하자")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")

                evaluated = evaluate_user_insight_submission("demo", "trial_002", 0.78)
                applied = mark_user_insight_applied(
                    "demo",
                    "trial_002",
                    {
                        "primary_change_axis": "model_ensemble",
                        "model": {
                            "estimator": "VotingClassifier",
                            "members": [{"estimator": "LR"}, {"estimator": "RF"}],
                        },
                    },
                )

        self.assertEqual("completed", evaluated["status"])
        self.assertIsNotNone(applied)
        # The terminal status set by evaluate_user_insight_submission must survive
        # -- mark_user_insight_applied should not downgrade it back to "applied".
        self.assertEqual("completed", applied["status"])
        self.assertIn("trial_002", applied["applied_trials"])
        self.assertTrue(applied["application_check"]["verification"]["passed"])

    def test_feature_insight_uses_the_same_contract_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            base_internal = root / "experiments" / "demo" / "trial_001" / "internal"
            base_internal.mkdir(parents=True)
            (base_internal / "executed_trial_facts.json").write_text(
                json.dumps(
                    {
                        "primary_change_axis": "initial_pipeline",
                        "features": {"final_feature_columns": ["age"]},
                        "preprocessing": {"steps": ["median"]},
                        "validation": {"method": "holdout"},
                        "submission": {"prediction_column": "target"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="family feature를 추가하자")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                missing = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {"status": "completed", "patch_updates": [{"replace": "model = ExistingModel()"}]},
                )
                present = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {"status": "completed", "patch_updates": [{"replace": "data['family_feature'] = 1"}]},
                )
                applied = mark_user_insight_applied(
                    "demo",
                    "trial_002",
                    {
                        "primary_change_axis": "feature_engineering",
                        "features": {"final_feature_columns": ["age", "family_feature"]},
                        "preprocessing": {"steps": ["median"]},
                        "validation": {"method": "holdout"},
                        "submission": {"prediction_column": "target"},
                    },
                )
        self.assertEqual(
            ["user_insight_not_implemented:feature_engineering:feature_delta_present_in_code"],
            missing,
        )
        self.assertEqual([], present)
        self.assertEqual("applied", applied["status"])
        self.assertTrue(applied["application_check"]["verification"]["passed"])

    def test_blocked_code_writer_result_does_not_report_insight_as_unimplemented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="Try a different model family.")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                issues = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {
                        "status": "blocked",
                        "blocking_issues": ["missing code snapshot"],
                        "patch_updates": [],
                    },
                )

        self.assertEqual([], issues)

    def test_finished_insight_with_stale_target_trial_does_not_block_later_trial(self):
        # A resolved insight's target_trial can linger unchanged after it reaches a
        # terminal status. If a later, unrelated trial happens to reuse that same
        # trial id, it must not be re-validated against this closed-out insight's
        # (possibly unrelated) verification contract.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="Try a different model family.")
                override = build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                update_user_insight_record(
                    "demo",
                    override["insight_id"],
                    status="completed",
                )
                issues = validate_user_insight_code_result(
                    "demo",
                    "trial_002",
                    {"status": "completed", "patch_updates": [{"replace": "no model change here"}]},
                )

        self.assertEqual([], issues)

    def test_model_family_axis_accepts_planner_and_strategy_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            base_internal = root / "experiments" / "demo" / "trial_001" / "internal"
            base_internal.mkdir(parents=True)
            (base_internal / "executed_trial_facts.json").write_text(
                json.dumps(
                    {
                        "primary_change_axis": "initial_pipeline",
                        "model": {"estimator": "Ridge"},
                        "features": {"final_feature_columns": ["hour"]},
                        "validation": {"method": "time_order_split"},
                        "submission": {"prediction_column": "count"},
                    }
                ),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="Try another boosting model.")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                result = mark_user_insight_applied(
                    "demo",
                    "trial_002",
                    {
                        "primary_change_axis": "model_family",
                        "model": {"estimator": "HistGradientBoostingRegressor"},
                        "features": {"final_feature_columns": ["hour"]},
                        "validation": {"method": "time_order_split"},
                        "submission": {"prediction_column": "count"},
                    },
                )

        self.assertEqual("applied", result["status"])
        self.assertTrue(result["application_check"]["verification"]["passed"])

    def test_execution_contract_rejects_unrelated_axis_or_unchanged_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "demo_workspaces" / "demo" / "manual_trials" / "trial_001"
            baseline.mkdir(parents=True)
            (baseline / "metrics.json").write_text(
                json.dumps({"trial_id": "trial_001", "kaggle_lb_score": 0.77}),
                encoding="utf-8",
            )
            base_internal = root / "experiments" / "demo" / "trial_001" / "internal"
            base_internal.mkdir(parents=True)
            base_facts = {
                "primary_change_axis": "initial_pipeline",
                "model": {"estimator": "LogisticRegression", "parameters": {"C": 1.0}},
                "features": {"final_feature_columns": ["age"]},
                "validation": {"method": "holdout"},
                "submission": {"prediction_column": "target"},
            }
            (base_internal / "executed_trial_facts.json").write_text(json.dumps(base_facts), encoding="utf-8")
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                submit_human_insight("demo", "trial_001", insight="RandomForest model로 바꾸자")
                build_next_trial_user_insight_override("demo", "trial_001", "trial_002")
                result = mark_user_insight_applied(
                    "demo",
                    "trial_002",
                    {
                        **base_facts,
                        "primary_change_axis": "feature_engineering",
                    },
                )
        self.assertEqual("application_mismatch", result["status"])
        failed = result["application_check"]["verification"]["failed_rule_ids"]
        self.assertIn("model_definition_changed", failed)
        self.assertIn("planned_axis_executed", failed)


class EnsembleVerificationTest(unittest.TestCase):
    # Regression for trial_028: the code writer correctly implemented the
    # user's ensemble insight with a BaggingRegressor, but verification only
    # recognized voting/stacking/blend, so a valid implementation was
    # rejected and the trial reported code_writer_blocked.
    _CODE_RESULT = {
        "status": "completed",
        "summary": (
            "Wrapped the final HistGradientBoostingRegressor in a BaggingRegressor "
            "(n_estimators=15, max_samples=0.75)."
        ),
        "changed_files": ["train_step.py"],
    }

    def _insight(self, root: Path, *, contract: dict | None = None) -> None:
        memory = root / "memory" / "demo"
        memory.mkdir(parents=True, exist_ok=True)
        record = {
            "insight_id": "insight_1",
            "competition": "demo",
            "target_trial": "trial_002",
            "axis": "model_ensemble",
            "status": "planned",
            "interpretation": {"verification_contract": contract} if contract else {},
        }
        (memory / "user_insights.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def test_bagging_counts_as_an_ensemble_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._insight(root)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                issues = validate_user_insight_code_result("demo", "trial_002", self._CODE_RESULT)

        self.assertEqual([], issues)

    def test_stored_contract_with_outdated_markers_is_refreshed(self):
        # The contract is snapshotted onto the insight when it is created, so
        # an insight written before the marker list was fixed must not keep
        # verifying against the stale list forever.
        stale_contract = {
            "expected_changes": ["model"],
            "protected_components": ["features"],
            "rules": [
                {
                    "id": "model_change_present_in_code",
                    "phase": "code",
                    "field": "serialized",
                    "operator": "contains_any",
                    "values": ["votingclassifier", "stackingclassifier", "blend"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._insight(root, contract=stale_contract)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                issues = validate_user_insight_code_result("demo", "trial_002", self._CODE_RESULT)

        self.assertEqual([], issues)

    def test_unrelated_change_still_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._insight(root)
            unrelated = {
                "status": "completed",
                "summary": "Tuned the learning rate on the existing single estimator.",
                "changed_files": ["train_step.py"],
            }
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                issues = validate_user_insight_code_result("demo", "trial_002", unrelated)

        self.assertIn("user_insight_not_implemented:model_ensemble:model_change_present_in_code", issues)

    def test_ensemble_size_reads_both_member_list_and_n_estimators(self):
        from kaggle_research_agent.user_insight_policy import _ensemble_size

        self.assertEqual(15, _ensemble_size({"estimator": "BaggingRegressor", "parameters": {"n_estimators": "15"}}))
        self.assertEqual(3, _ensemble_size({"estimator": "VotingRegressor", "members": ["a", "b", "c"]}))
        self.assertEqual(0, _ensemble_size({"estimator": "Ridge", "parameters": {"alpha": "1.0"}}))
        self.assertEqual(0, _ensemble_size(None))


if __name__ == "__main__":
    unittest.main()
