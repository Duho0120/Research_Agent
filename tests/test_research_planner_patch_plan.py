import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.pipeline_patch_planner import prepare_patch_plan


class PipelinePatchPlannerTest(unittest.TestCase):
    def test_prepare_controlled_refinement_updates_next_config_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "lightgbm",
                            "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8},
                        },
                        "features": {
                            "use_frequency_encoding": False,
                            "use_target_encoding": False,
                            "use_interactions": False,
                            "use_missing_indicators": True,
                        },
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_002"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\ncontrolled_refinement\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = prepare_patch_plan("demo", "trial_001", "trial_002")

            self.assertEqual(plan["strategy"], "controlled_refinement")
            self.assertEqual(plan["config_changes"]["features.use_frequency_encoding"], True)
            self.assertTrue((next_trial / "config.yaml").exists())
            self.assertTrue((next_trial / "code_patch_plan.json").exists())
            self.assertTrue((next_trial / "code_patch_plan.md").exists())
            self.assertNotIn("validation_errors", plan)

    def test_prepare_sota_attempt_switches_model_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_004"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "lightgbm",
                            "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8},
                        },
                        "features": {"use_missing_indicators": True},
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm", "skeleton_transformer"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                                "d_model": {"min": 32, "max": 256},
                                "nhead": [2, 4, 8],
                                "num_layers": {"min": 1, "max": 6},
                                "dropout": {"min": 0.0, "max": 0.5},
                            },
                        },
                        "features": {
                            "use_missing_indicators": [True, False],
                            "use_view_aware_features": [True, False],
                            "use_bed_wandering_aux_head": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                        "training": {
                            "epochs": {"min": 1, "max": 200},
                            "batch_size": [16, 32, 64],
                            "warmup_epochs": {"min": 0, "max": 20},
                            "early_stopping_patience": {"min": 1, "max": 50},
                        },
                    }
                ),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_005"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_005 Next Experiment\n\n## Strategy\n\nsota_architecture_attempt\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = prepare_patch_plan("demo", "trial_004", "trial_005")

            self.assertEqual(plan["strategy"], "sota_architecture_attempt")
            self.assertEqual(plan["config"]["model"]["type"], "skeleton_transformer")
            self.assertIn("training", plan["config"])
            self.assertIn("scripts/demo_train.py", " ".join(plan["target_files"]))

    def test_prepare_patch_plan_translates_sampling_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (source / "pipeline_improvement_plan.json").write_text(
                json.dumps(
                    {
                        "primary_axis": "sampling",
                        "secondary_axes": ["loss_metric_alignment", "augmentation"],
                        "protected_axes": ["validation", "model_family"],
                    }
                ),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_002"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\ncontrolled_refinement\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = prepare_patch_plan("demo", "trial_001", "trial_002")

            self.assertEqual(plan["pipeline_axis"], "sampling")
            self.assertEqual(plan["config"]["training"]["sampler"], "balanced")
            self.assertIn("training.sampler", plan["config_changes"])
            self.assertIn("dataset", " ".join(plan["target_files"]))
            self.assertIn("kaggle_research_agent/pipeline/dataset.py", plan["create_files"])
            self.assertIn("protected_axes", plan)

    def test_prepare_patch_plan_translates_loss_metric_alignment_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (source / "pipeline_improvement_plan.json").write_text(
                json.dumps({"primary_axis": "loss_metric_alignment", "secondary_axes": ["post_processing"]}),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_002"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\ncontrolled_refinement\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = prepare_patch_plan("demo", "trial_001", "trial_002")

            self.assertEqual(plan["pipeline_axis"], "loss_metric_alignment")
            self.assertEqual(plan["config"]["training"]["loss"], "metric_aligned")
            self.assertTrue(plan["config"]["post_processing"]["threshold_sweep"])
            self.assertIn("training.loss", plan["config_changes"])

    def test_prepare_patch_plan_translates_pretraining_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps({"model": {"type": "efficientnet"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (source / "pipeline_improvement_plan.json").write_text(
                json.dumps(
                    {
                        "primary_axis": "pretraining_strategy",
                        "secondary_axes": ["model_architecture"],
                        "protected_axes": ["validation"],
                        "requires_human_review": True,
                    }
                ),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_002"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\nsota_architecture_attempt\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = prepare_patch_plan("demo", "trial_001", "trial_002")

            self.assertEqual(plan["pipeline_axis"], "pretraining_strategy")
            self.assertEqual(plan["config"]["model"]["pretraining"]["mode"], "partial_finetune")
            self.assertTrue(plan["requires_user_approval"])
            self.assertIn("external pretrained model permission", " ".join(plan["implementation_steps"]))


if __name__ == "__main__":
    unittest.main()


