import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from kaggle_research_agent.agents.orchestrator import run_auto_research_loop, run_cycle


class OrchestratorDiagnosisTest(unittest.TestCase):
    def test_auto_research_loop_runs_multiple_trials_without_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: trial_000\n"
                "    cv_score: 0.7\n"
                "strategy:\n"
                "  current_focus: baseline\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
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
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.71, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_auto_research_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=2,
                    submit_policy="never",
                    stop_no_improvement=2,
                    run_now=False,
                )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["submit_policy"], "never")
            self.assertEqual(len(result["trials"]), 2)
            self.assertEqual(result["trials"][0]["trial_id"], "trial_001")
            self.assertEqual(result["trials"][1]["trial_id"], "trial_002")
            self.assertIn("next_experiment_planned", result["trials"][0]["steps"])
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "code_patch_plan.md").exists())
            self.assertFalse((root / "submissions" / "demo" / "submission_log.jsonl").exists())

    def test_auto_research_loop_stops_after_no_improvement_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 2\n"
                "  best_trial:\n"
                "    trial_id: trial_000\n"
                "    cv_score: 0.9\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {}}), encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.7, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_auto_research_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=3,
                    submit_policy="never",
                    stop_no_improvement=1,
                    run_now=False,
                )

            self.assertEqual(result["status"], "stopped_no_improvement")
            self.assertEqual(len(result["trials"]), 1)

    def test_cycle_with_metrics_writes_diagnosis_and_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.7\n"
                "strategy:\n"
                "  current_focus: baseline\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            allowed_space = {
                "model": {
                    "type": ["lightgbm"],
                    "params": {
                        "learning_rate": {"min": 0.005, "max": 0.2},
                        "num_leaves": {"min": 16, "max": 256},
                        "max_depth": {"min": 3, "max": 12},
                    },
                },
                "features": {
                    "use_frequency_encoding": [True, False],
                    "use_target_encoding": [True, False],
                    "use_interactions": [True, False],
                    "use_missing_indicators": [True, False],
                },
                "cv": {"n_splits": [5, 10], "seed": {"min": 1, "max": 9999}},
            }
            (cfg / "allowed_space.yaml").write_text(json.dumps(allowed_space), encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                "model:\n"
                "  type: lightgbm\n"
                "features:\n"
                "  use_missing_indicators: True\n"
                "cv:\n"
                "  n_splits: 5\n",
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.72, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_cycle("demo", "trial_001", create_job_request=False)

            self.assertIn("diagnosed", result["steps"])
            self.assertLess(result["steps"].index("evaluated"), result["steps"].index("diagnosed"))
            self.assertLess(result["steps"].index("diagnosed"), result["steps"].index("remembered"))
            self.assertEqual(0.7, result["diagnosis"]["best_cv_before"])
            self.assertTrue(result["diagnosis"]["cv_improved"])
            self.assertEqual("trial_001", result["memory"]["trial_id"])
            self.assertTrue((trial / "diagnosis.md").exists())
            decision_log = root / "memory" / "demo" / "decision_log.jsonl"
            self.assertTrue(decision_log.exists())
            decision = json.loads(decision_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertTrue(decision["evidence"]["cv_improved"])

            state = (comp / "state.yaml").read_text(encoding="utf-8")
            self.assertIn("trial_id: trial_001", state)

    def test_cycle_can_plan_next_experiment_after_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 3\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n"
                "    lb_score: 0.79\n"
                "strategy:\n"
                "  current_focus: baseline\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            allowed_space = {
                "model": {
                    "type": ["lightgbm"],
                    "params": {
                        "learning_rate": {"min": 0.005, "max": 0.2},
                        "num_leaves": {"min": 16, "max": 256},
                        "max_depth": {"min": 3, "max": 12},
                    },
                },
                "features": {
                    "use_frequency_encoding": [True, False],
                    "use_target_encoding": [True, False],
                    "use_interactions": [True, False],
                    "use_missing_indicators": [True, False],
                },
                "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
            }
            (cfg / "allowed_space.yaml").write_text(json.dumps(allowed_space), encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                "model:\n"
                "  type: lightgbm\n"
                "features:\n"
                "  use_missing_indicators: True\n"
                "cv:\n"
                "  n_splits: 5\n",
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.78, "lb_score": 0.77, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_cycle(
                    "demo",
                    "trial_001",
                    create_job_request=False,
                    next_trial_id="trial_002",
                    prepare_next_patch=True,
                )

            self.assertIn("next_experiment_planned", result["steps"])
            self.assertIn("patch_plan_prepared", result["steps"])
            self.assertEqual("trial_002", result["next_experiment"]["next_trial_id"])
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "next_experiment.md").exists())
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "code_patch_plan.md").exists())

    def test_cycle_can_apply_next_patch_and_run_next_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: trial_001\n"
                "    cv_score: 0.7\n"
                "strategy:\n"
                "  current_focus: baseline\n",
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
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
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
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
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
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.69, "objective": "maximize"}),
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "demo_train.py"
            next_trial = root / "experiments" / "demo" / "trial_002"
            command = f"{sys.executable} {script} --config {next_trial / 'config.yaml'} --output {next_trial}"

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_cycle(
                    "demo",
                    "trial_001",
                    create_job_request=False,
                    next_trial_id="trial_002",
                    prepare_next_patch=True,
                    apply_next_patch=True,
                    next_run_command=command,
                )

            self.assertIn("next_patch_applied", result["steps"])
            self.assertEqual("executed", result["next_code_edit"]["status"])
            self.assertTrue((next_trial / "metrics.json").exists())
            self.assertTrue((next_trial / "code_edit_result.md").exists())

    def test_cycle_can_run_safe_execution_chain_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {}}), encoding="utf-8")
            self._write_coding_handoff(
                trial,
                validation_commands=[
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/cycle_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            response_path = root / "mock_response.json"
            self._write_mock_code_response(response_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_cycle(
                    "demo",
                    "trial_002",
                    run_safe_chain=True,
                    safe_chain_mock_response_file=str(response_path),
                    command="python train.py",
                )

            self.assertIn("safe_execution_chain_ran", result["steps"])
            self.assertEqual("job_created", result["safe_execution_chain"]["status"])
            self.assertTrue((trial / "safe_execution_chain.json").exists())
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def _write_coding_handoff(self, trial: Path, validation_commands: list[str]) -> None:
        (trial / "coding_handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "request_id": "demo:trial_002:coding",
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": "ready",
                    "objective": "Implement the validated code patch plan without expanding scope.",
                    "context_files": ["experiments/demo/trial_002/config.yaml"],
                    "allowed_write_files": ["experiments/demo/trial_002/config.yaml"],
                    "create_files": [],
                    "forbidden_paths": [
                        "data/",
                        "submissions/",
                        "experiments/demo/trial_002/submission.csv",
                        "experiments/demo/trial_002/metrics.json",
                    ],
                    "implementation_steps": ["Add balanced sampler config."],
                    "validation_commands": validation_commands,
                    "required_output": {
                        "required_fields": [
                            "status",
                            "summary",
                            "changed_files",
                            "validation_results",
                            "blocking_issues",
                        ],
                        "status_values": ["completed", "blocked", "failed"],
                    },
                }
            ),
            encoding="utf-8",
        )

    def _write_mock_code_response(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated config from mock response.",
                            "changed_files": ["experiments/demo/trial_002/config.yaml"],
                            "file_updates": [
                                {
                                    "path": "experiments/demo/trial_002/config.yaml",
                                    "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()


