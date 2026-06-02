import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.patch_validator import validate_patch_plan


class PatchValidatorTest(unittest.TestCase):
    def test_blocks_when_user_approval_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "sota_architecture_attempt",
                        "pipeline_axis": "pretraining_strategy",
                        "requires_user_approval": True,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {"model.pretraining.mode": "partial_finetune"},
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("user_approval_required", result["issues"])
            self.assertTrue((trial / "patch_validation.json").exists())
            self.assertTrue((trial / "patch_validation.md").exists())

    def test_blocks_protected_axis_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "protected_axes": ["validation", "model_family"],
                        "requires_user_approval": False,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {
                            "training.sampler": "balanced",
                            "cv.n_splits": 10,
                        },
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("protected_axis_violation:validation", result["issues"])

    def test_ready_for_safe_patch_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "protected_axes": ["validation", "model_family"],
                        "requires_user_approval": False,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {"training.sampler": "balanced"},
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["issues"], [])

    def test_records_patch_validation_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "protected_axes": ["validation", "model_family"],
                        "requires_user_approval": False,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {"training.sampler": "balanced"},
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "ready")
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            self.assertTrue(log_path.exists())
            row = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["decision_type"], "patch_validation")
            self.assertEqual(row["decision"], "ready")
            self.assertEqual(row["trial_id"], "trial_002")
            self.assertEqual(row["evidence"]["issues"], [])


if __name__ == "__main__":
    unittest.main()
