from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.execution_facts import resolve_trial_plan, write_executed_trial_facts
from research_agent.trial_decision import write_trial_decision_card
from research_agent.trial_memory_card import write_trial_memory_card


class ExecutionFactsTest(unittest.TestCase):
    def test_delta_and_executed_model_override_stale_metrics_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_ensemble",
                        "change_details": ["Use soft voting"],
                    }
                ),
                encoding="utf-8",
            )
            stale_metrics = {
                "cv_score": 0.84,
                "metric": "accuracy",
                "objective": "maximize",
                "model": "LogisticRegression",
            }
            (trial / "metrics.json").write_text(json.dumps(stale_metrics), encoding="utf-8")
            structure = {
                "consistency_issues": ["model_metadata_mismatch:metrics=LogisticRegression;code=VotingClassifier"],
                "stages": [
                    {
                        "id": "model_definition",
                        "structured_details": {
                            "estimator": "VotingClassifier",
                            "parameters": {"voting": "soft"},
                            "members": [
                                {"estimator": "LogisticRegression", "parameters": {}},
                                {"estimator": "RandomForestClassifier", "parameters": {}},
                            ],
                        },
                    }
                ],
            }
            summary = {
                "competition": "demo",
                "trial_id": "trial_002",
                "metric": "accuracy",
                "objective": "maximize",
                "local_score": 0.84,
                "metrics": stale_metrics,
            }

            with patch("research_agent.paths.project_root", return_value=root):
                facts = write_executed_trial_facts(
                    "demo",
                    "trial_002",
                    pipeline_structure=structure,
                    summary=summary,
                )
                decision = write_trial_decision_card("demo", "trial_002")
                memory = write_trial_memory_card("demo", "trial_002", decision_card=decision)

            self.assertEqual("model_ensemble", facts["primary_change_axis"])
            self.assertEqual("VotingClassifier", facts["model"]["estimator"])
            self.assertEqual("model_ensemble", decision["change_axis"])
            self.assertEqual("trial_001", decision["source_trial_id"])
            self.assertEqual("model_ensemble", memory["change_axis"])
            self.assertEqual("VotingClassifier", memory["model_type"])
            self.assertEqual(2, len(memory["model"]["members"]))
            self.assertTrue(memory["consistency_issues"])

    def test_plan_resolution_prefers_delta_over_legacy_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "demo_experiment_plan.json").write_text(
                json.dumps({"source_trial_id": "trial_old", "primary_change_axis": "feature_engineering"}),
                encoding="utf-8",
            )
            (trial / "delta_plan.json").write_text(
                json.dumps({"source_trial_id": "trial_001", "primary_change_axis": "model_ensemble"}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = resolve_trial_plan("demo", "trial_002")

            self.assertEqual("trial_001", plan["source_trial_id"])
            self.assertEqual("model_ensemble", plan["primary_change_axis"])

    def test_plan_resolution_rejects_self_referential_retry_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_005"
            internal = trial / "internal"
            revision = internal / "plan_revisions" / "20260727_113618"
            revision.mkdir(parents=True)
            (internal / "workspace_coding_handoff.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "source_trial_id": "trial_004",
                        "code_base_trial_id": "trial_004",
                    }
                ),
                encoding="utf-8",
            )
            (internal / "demo_experiment_plan.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "source_trial_id": "trial_005",
                        "plan_title": "HGBR max_iter 300",
                        "primary_change_axis": "hyperparameter",
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "source_trial_id": "trial_005",
                        "change_details": ["Set max_iter=300"],
                    }
                ),
                encoding="utf-8",
            )
            (revision / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_005",
                        "source_trial_id": "trial_004",
                        "primary_change_axis": "model_family",
                        "candidate": {"name": "HGBR with log target transform"},
                        "change_details": ["Replace Ridge with HGBR"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = resolve_trial_plan("demo", "trial_005")

            self.assertEqual("trial_004", plan["source_trial_id"])
            self.assertEqual("model_family", plan["primary_change_axis"])
            self.assertEqual("HGBR with log target transform", plan["plan_title"])
            self.assertEqual(["Replace Ridge with HGBR"], plan["change_details"])
            self.assertNotIn("max_iter", json.dumps(plan))

    def test_plan_resolution_uses_finalized_execution_snapshot_over_conflicting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_006"
            internal = trial / "internal"
            internal.mkdir(parents=True)
            (internal / "execution_plan_snapshot.json").write_text(
                json.dumps(
                    {
                        "status": "finalized",
                        "plan": {
                            "trial_id": "trial_006",
                            "source_trial_id": "trial_003",
                            "plan_title": "Adopt time series validation",
                            "primary_change_axis": "validation_review",
                            "change_details": ["Use TimeSeriesSplit"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_006",
                        "source_trial_id": "trial_005",
                        "primary_change_axis": "model_family",
                        "change_details": ["Use Poisson loss"],
                    }
                ),
                encoding="utf-8",
            )
            (internal / "workspace_coding_handoff.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_006",
                        "source_trial_id": "trial_005",
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = resolve_trial_plan("demo", "trial_006")

            self.assertEqual("trial_003", plan["source_trial_id"])
            self.assertEqual("validation_review", plan["primary_change_axis"])
            self.assertEqual(["Use TimeSeriesSplit"], plan["change_details"])
            self.assertEqual(
                ["internal/execution_plan_snapshot.json"],
                plan["_resolved_plan_sources"],
            )

    def test_plan_resolution_falls_back_when_snapshot_has_no_primary_axis(self):
        # A plan rendered with research_planner's "## Strategy" layout (rather than
        # demo_one_cycle's "## Primary Change Axis" layout) still produces a
        # non-empty execution_plan_snapshot.json, since the markdown parser only
        # understands the latter layout. That near-empty snapshot must not be
        # trusted over the richer next_experiment.json fallback, which already
        # knows how to recover primary_change_axis/change_details from either
        # layout.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            internal = trial / "internal"
            internal.mkdir(parents=True)
            (internal / "execution_plan_snapshot.json").write_text(
                json.dumps(
                    {
                        "status": "finalized",
                        "plan": {"trial_id": "trial_002", "plan_type": "initial_pipeline_plan"},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\nhyperparameter_tuning\n\n"
                "## Changes\n\n- tweak learning rate\n",
                encoding="utf-8",
            )
            (trial / "next_experiment.json").write_text(
                json.dumps(
                    {
                        "strategy": "hyperparameter_tuning",
                        "changes": ["tweak learning rate"],
                        "source_trial_id": "trial_001",
                    }
                ),
                encoding="utf-8",
            )
            (internal / "workspace_coding_handoff.json").write_text(
                json.dumps({"trial_id": "trial_002", "source_trial_id": "trial_001"}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                plan = resolve_trial_plan("demo", "trial_002")

            self.assertEqual("hyperparameter_tuning", plan["primary_change_axis"])
            self.assertNotIn("internal/execution_plan_snapshot.json", plan["_resolved_plan_sources"])


if __name__ == "__main__":
    unittest.main()
