from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.execution_facts import resolve_trial_plan, write_executed_trial_facts
from kaggle_research_agent.trial_decision import write_trial_decision_card
from kaggle_research_agent.trial_memory_card import write_trial_memory_card


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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = resolve_trial_plan("demo", "trial_002")

            self.assertEqual("trial_001", plan["source_trial_id"])
            self.assertEqual("model_ensemble", plan["primary_change_axis"])


if __name__ == "__main__":
    unittest.main()
