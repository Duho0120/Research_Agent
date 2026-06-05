import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.coding_handoff import prepare_coding_handoff


class CodingHandoffTest(unittest.TestCase):
    def test_prepare_coding_handoff_writes_agent_request_for_ready_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "next_experiment.md").write_text("# trial_002 Next Experiment\n\nImprove sampling.\n", encoding="utf-8")
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "protected_axes": ["validation", "model_family"],
                        "requires_user_approval": False,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "create_files": [],
                        "config_changes": {"training.sampler": "balanced"},
                        "implementation_steps": ["Add balanced sampler support."],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_coding_handoff("demo", "trial_002")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["handoff_type"], "coding_agent_request")
            self.assertEqual(result["schema_version"], "1.0")
            self.assertEqual(result["request_id"], "demo:trial_002:coding")
            self.assertEqual(result["target_files"], ["experiments/demo/trial_002/config.yaml"])
            self.assertEqual(result["create_files"], [])
            self.assertEqual(result["allowed_write_files"], result["target_files"])
            self.assertIn("experiments/demo/trial_002/code_patch_plan.json", result["context_files"])
            self.assertIn("experiments/demo/trial_002/config.yaml", result["context_files"])
            self.assertIn("submissions/", result["forbidden_paths"])
            self.assertTrue(result["execution_constraints"]["do_not_run_training"])
            self.assertTrue(result["execution_constraints"]["do_not_submit"])
            self.assertEqual(result["required_output"]["status_values"], ["completed", "blocked", "failed"])
            self.assertEqual(result["required_output"]["next_action"], "validate-code-change")
            self.assertTrue((trial / "coding_handoff.json").exists())
            request = trial / "coding_agent_request.md"
            self.assertTrue(request.exists())
            text = request.read_text(encoding="utf-8")
            self.assertIn("Add balanced sampler support.", text)
            self.assertIn("Do not edit submission artifacts.", text)
            self.assertIn("## Input Context Files", text)
            self.assertIn("## Allowed Write Files", text)
            self.assertIn("## Forbidden Paths", text)
            self.assertIn("## Required Result Contract", text)
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            last = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(last["decision_type"], "coding_handoff")
            self.assertEqual(last["decision"], "ready")

    def test_prepare_coding_handoff_blocks_when_patch_validation_blocks(self):
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
                        "implementation_steps": ["Switch to pretrained backbone."],
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_coding_handoff("demo", "trial_002")

            self.assertEqual(result["status"], "blocked")
            self.assertIn("user_approval_required", result["blocking_issues"])
            self.assertTrue((trial / "coding_handoff.json").exists())
            self.assertFalse((trial / "coding_agent_request.md").exists())

    def test_prepare_coding_handoff_reuses_existing_patch_validation(self):
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
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {"training.sampler": "balanced"},
                        "implementation_steps": ["Add balanced sampler support."],
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "patch_validation.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "trial_id": "trial_002",
                        "status": "ready",
                        "issues": [],
                        "missing_targets": [],
                        "config_errors": [],
                        "requires_user_approval": False,
                        "user_approved": False,
                        "pipeline_axis": "sampling",
                        "protected_axes": [],
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                    }
                ),
                encoding="utf-8",
            )
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                json.dumps({"decision_type": "patch_validation", "decision": "ready"}) + "\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_coding_handoff("demo", "trial_002")

            self.assertEqual(result["status"], "ready")
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["decision_type"] for row in rows].count("patch_validation"), 1)
            self.assertEqual(rows[-1]["decision_type"], "coding_handoff")


if __name__ == "__main__":
    unittest.main()
