import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.model_advisor import advise_model_candidates


class ModelAdvisorTest(unittest.TestCase):
    def test_recommends_tabular_baseline_friendly_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "titanic"
            comp.mkdir(parents=True)
            (comp / "data_profile.json").write_text(
                json.dumps(
                    {
                        "competition": "titanic",
                        "status": "ready",
                        "task_type": "tabular",
                        "target_candidates": ["Survived"],
                        "files": [{"name": "train.csv", "format": "csv", "role": "train"}],
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "titanic" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.72, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = advise_model_candidates("titanic", "trial_001")

            self.assertEqual(result["task_type"], "tabular")
            self.assertEqual(result["recommendation_scope"], "baseline_or_controlled_refinement")
            self.assertEqual(result["candidates"][0]["model_family"], "gradient_boosted_trees")
            self.assertEqual(result["candidates"][0]["training_strategy"], "train_from_scratch")
            self.assertTrue((trial / "model_candidates.json").exists())
            self.assertTrue((trial / "model_candidates.md").exists())

    def test_recommends_pretrained_vision_candidates_for_image_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "vision_demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 3\n",
                encoding="utf-8",
            )
            (comp / "data_profile.json").write_text(
                json.dumps(
                    {
                        "competition": "vision_demo",
                        "status": "ready",
                        "task_type": "image",
                        "target_candidates": ["label"],
                        "files": [{"name": "train_images/", "format": "jpg", "role": "train"}],
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "vision_demo" / "trial_003"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.81, "objective": "maximize", "prediction_correlation_with_best": 0.997}),
                encoding="utf-8",
            )
            (trial / "pipeline_improvement_plan.json").write_text(
                json.dumps({"primary_axis": "model_family", "protected_axes": ["validation"]}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = advise_model_candidates("vision_demo", "trial_003")

            self.assertEqual(result["recommendation_scope"], "model_family_exploration")
            self.assertIn("pretrained_finetune", {item["training_strategy"] for item in result["candidates"]})
            self.assertIn("self_supervised_vision_backbone", {item["model_family"] for item in result["candidates"]})

    def test_protects_model_change_when_validation_is_primary_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "data_profile.json").write_text(
                json.dumps({"competition": "demo", "status": "ready", "task_type": "image", "target_candidates": ["label"]}),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.8, "lb_score": 0.7, "objective": "maximize"}), encoding="utf-8")
            (trial / "pipeline_improvement_plan.json").write_text(
                json.dumps({"primary_axis": "validation", "protected_axes": ["model_family", "model_architecture"]}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = advise_model_candidates("demo", "trial_001")

            self.assertEqual(result["recommendation_scope"], "defer_model_change")
            self.assertTrue(result["model_change_protected"])
            self.assertIn("validation", " ".join(result["guardrails"]))


if __name__ == "__main__":
    unittest.main()
