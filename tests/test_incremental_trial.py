from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.incremental_trial import (
    enrich_delta_plan,
    write_base_summary,
    write_effective_trial_artifacts,
)
from kaggle_research_agent.trial_user_view_effective import render_effective_pipeline_structure


class IncrementalTrialTest(unittest.TestCase):
    def test_enrich_delta_plan_maps_model_axis_to_affected_code(self):
        plan = enrich_delta_plan(
            {
                "source_trial_id": "trial_003",
                "primary_change_axis": "model_ensemble",
                "change_details": ["Use soft voting"],
            }
        )

        self.assertIn("model_definition", plan["affected_stages"])
        self.assertIn("test_inference_output", plan["affected_stages"])
        self.assertIn("build_pipeline", plan["required_code_symbols"])
        self.assertIn("model", plan["expected_metadata_changes"])

    def test_base_summary_uses_trial_artifacts_and_code_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            snapshot = source / "internal" / "code_snapshot"
            snapshot.mkdir(parents=True)
            (source / "metrics.json").write_text(
                json.dumps({"cv_score": 0.82, "metric": "accuracy", "model": "LogisticRegression"}),
                encoding="utf-8",
            )
            (source / "internal" / "pipeline_structure.json").write_text(
                json.dumps(
                    {
                        "metric": "accuracy",
                        "objective": "maximize",
                        "stages": [
                            {
                                "id": "model_definition",
                                "included": True,
                                "structured_details": {"estimator": "LogisticRegression"},
                                "actual_applied": ["LogisticRegression"],
                                "code_locations": ["src/pipeline.py"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "pipeline.py").write_text(
                "def build_pipeline():\n    return None\n\ndef train_validate():\n    return None\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                summary = write_base_summary("demo", "trial_002", "trial_001")

            self.assertEqual("trial_001", summary["base_trial_id"])
            self.assertEqual(0.82, summary["scores"]["local"])
            self.assertIn("build_pipeline", summary["code"]["symbols"]["pipeline.py"])
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "base_summary.json").exists())

    def test_effective_artifacts_record_delta_and_pipeline_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "experiments" / "demo" / "trial_001"
            current = root / "experiments" / "demo" / "trial_002"
            (base / "internal").mkdir(parents=True)
            (current / "internal").mkdir(parents=True)
            base_structure = {
                "stages": [
                    {"id": "model_definition", "included": True, "actual_applied": ["LR"], "structured_details": {"estimator": "LogisticRegression"}}
                ]
            }
            current_structure = {
                "stages": [
                    {"id": "model_definition", "included": True, "actual_applied": ["RF"], "structured_details": {"estimator": "RandomForestClassifier"}}
                ]
            }
            (base / "internal" / "pipeline_structure.json").write_text(json.dumps(base_structure), encoding="utf-8")
            (base / "internal" / "effective_plan.json").write_text(
                json.dumps({"baseline_trial_id": "trial_001", "baseline_plan": {"plan_title": "Baseline"}, "change_history": []}),
                encoding="utf-8",
            )
            (current / "demo_experiment_plan.json").write_text(
                json.dumps({"source_trial_id": "trial_001", "primary_change_axis": "model_family"}),
                encoding="utf-8",
            )
            (current / "delta_plan.json").write_text(
                json.dumps({"source_trial_id": "trial_001", "primary_change_axis": "model_family", "change_details": ["LR to RF"]}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_effective_trial_artifacts("demo", "trial_002", pipeline_structure=current_structure)

            effective = json.loads((current / "internal" / "effective_plan.json").read_text(encoding="utf-8-sig"))
            delta = json.loads((current / "internal" / "pipeline_delta_applied.json").read_text(encoding="utf-8-sig"))
            self.assertEqual("trial_001", effective["baseline_trial_id"])
            self.assertEqual("model_family", effective["change_history"][-1]["primary_change_axis"])
            self.assertEqual("model_definition", delta["changes"][0]["stage_id"])

    def test_delta_plan_carries_forward_code_change_targets(self):
        # workspace_coding_handoff.py prioritizes which base-trial file gets embedded in the
        # (size-limited) coding context by reading code_change_targets from delta_plan.json.
        # If this field is dropped when the delta plan is derived from the full plan, the
        # actually-targeted file can be silently excluded from the code writer's context.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "experiments" / "demo" / "trial_001"
            current = root / "experiments" / "demo" / "trial_002"
            (base / "internal").mkdir(parents=True)
            (current / "internal").mkdir(parents=True)
            (current / "demo_experiment_plan.json").write_text(
                json.dumps(
                    {
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "code_change_targets": ["train_step.py"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_effective_trial_artifacts("demo", "trial_002", pipeline_structure={"stages": []})

            delta = json.loads((current / "delta_plan.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(["train_step.py"], delta["code_change_targets"])

    def test_pipeline_view_shows_ensemble_members_and_stage_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "internal").mkdir()
            structure = {
                "metric": "accuracy",
                "stages": [
                    {
                        "id": "model_definition",
                        "included": True,
                        "structured_details": {
                            "estimator": "VotingClassifier",
                            "parameters": {"voting": "soft"},
                            "members": [
                                {"estimator": "LogisticRegression", "parameters": {"max_iter": "1000"}},
                                {"estimator": "RandomForestClassifier", "parameters": {"n_estimators": "300"}},
                            ],
                        },
                        "inputs": ["model-ready features"],
                        "outputs": ["untrained model"],
                        "code_locations": ["src/pipeline.py"],
                    }
                ],
            }
            (out_dir / "internal" / "pipeline_structure.json").write_text(json.dumps(structure), encoding="utf-8")
            text = render_effective_pipeline_structure(
                out_dir,
                {"trial_id": "trial_002", "metric": "accuracy", "local_score": 0.84},
            )

            self.assertIn("VotingClassifier(LogisticRegression, RandomForestClassifier)", text)
            self.assertIn("voting=soft", text)
            self.assertIn("관련 코드", text)


if __name__ == "__main__":
    unittest.main()
