import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.experiment_runner import apply_patch_plan, run_local_job


class ExperimentRunnerTest(unittest.TestCase):
    def test_run_local_job_writes_structured_failure_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            command = f'{sys.executable} -c "import definitely_missing_package_for_agent_test"'

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                job = run_local_job("demo", "trial_001", command=command)

            self.assertEqual(job["status"], "failed")
            failure_path = trial / "local_failure.json"
            self.assertTrue(failure_path.exists())
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["failure_type"], "missing_dependency")
            self.assertEqual(failure["suggested_next_action"], "fix_dependency")
            self.assertEqual(failure["exit_code"], job["returncode"])
            self.assertTrue(failure["log_tail"])
            self.assertTrue((trial / "local_failure.md").exists())

    def test_apply_patch_plan_validates_targets_and_writes_result(self):
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
                        "competition": "demo",
                        "next_trial_id": "trial_002",
                        "strategy": "controlled_refinement",
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = apply_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["missing_targets"], [])
            self.assertTrue((trial / "code_edit_result.json").exists())
            self.assertTrue((trial / "code_edit_result.md").exists())

    def test_apply_patch_plan_uses_patch_validator_result_as_single_gate(self):
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
                        "competition": "demo",
                        "next_trial_id": "trial_002",
                        "strategy": "controlled_refinement",
                        "target_files": ["experiments/demo/trial_002/missing_target.py"],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = apply_patch_plan("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["missing_targets"], ["experiments/demo/trial_002/missing_target.py"])
            self.assertEqual(result["validation_errors"], [])
            self.assertEqual(result["patch_validation_issues"], ["missing_target:experiments/demo/trial_002/missing_target.py"])

    def test_apply_patch_plan_can_run_trial_and_evaluate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {"type": ["skeleton_transformer"]},
                        "features": {
                            "use_view_aware_features": [True, False],
                            "use_bed_wandering_aux_head": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {"type": "skeleton_transformer"},
                        "features": {
                            "use_view_aware_features": True,
                            "use_bed_wandering_aux_head": True,
                        },
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "plan.md").write_text("# trial_002 Plan\n", encoding="utf-8")
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "next_trial_id": "trial_002",
                        "strategy": "sota_architecture_attempt",
                        "target_files": [
                            "experiments/demo/trial_002/config.yaml",
                            str((Path(__file__).resolve().parents[1] / "scripts" / "demo_train.py")),
                        ],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "demo_train.py"
            command = (
                f"{sys.executable} {script} "
                f"--config {trial / 'config.yaml'} "
                f"--output {trial}"
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = apply_patch_plan("demo", "trial_002", run_command=command)

            self.assertEqual(result["status"], "executed")
            self.assertEqual(result["job_status"], "done")
            self.assertTrue((trial / "metrics.json").exists())
            self.assertTrue((trial / "evaluation.md").exists())
            self.assertTrue((trial / "diagnosis.md").exists())
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["model_type"], "skeleton_transformer")


if __name__ == "__main__":
    unittest.main()


