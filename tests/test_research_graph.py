import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.graph.research_graph import run_graph_cycle


class ResearchGraphTest(unittest.TestCase):
    def _write_allowed_space(self, root: Path) -> None:
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

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "state.yaml").write_text(
            "competition:\n"
            "  objective: maximize\n"
            "current_state:\n"
            "  consecutive_failures: 0\n"
            "  best_trial:\n"
            "    trial_id: old\n"
            "    cv_score: 0.7\n",
            encoding="utf-8",
        )

    def test_graph_cycle_evaluates_completed_trial_and_updates_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_allowed_space(root)
            self._write_state(root)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {"type": "lightgbm", "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8}},
                        "features": {"use_missing_indicators": True},
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.72, "objective": "maximize"}), encoding="utf-8")

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_graph_cycle("demo", "trial_001", create_job_request=False)

            self.assertEqual(result["status"], "completed")
            self.assertIn("planned", result["steps"])
            self.assertIn("diagnosed", result["steps"])
            self.assertIn("remembered", result["steps"])
            self.assertEqual("trial_001", result["memory"]["trial_id"])
            self.assertTrue((trial / "diagnosis.md").exists())
            graph_state = json.loads((trial / "graph_state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", graph_state["status"])
            self.assertEqual("finalize", graph_state["last_completed_node"])
            self.assertEqual("experiments/demo/trial_001/graph_state.json", result["graph_state_file"])
            events = [
                json.loads(line)
                for line in (trial / "node_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("plan_trial", events[0]["node"])
            self.assertEqual("started", events[0]["event"])
            self.assertEqual("finalize", events[-1]["node"])
            self.assertEqual("completed", events[-1]["event"])

    def test_graph_cycle_creates_job_when_metrics_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_allowed_space(root)
            self._write_state(root)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {"type": "lightgbm", "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8}},
                        "features": {"use_missing_indicators": True},
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_graph_cycle("demo", "trial_001", create_job_request=True)

            self.assertEqual(result["status"], "waiting_for_metrics")
            self.assertIn("execution_decided", result["steps"])
            self.assertIn("local_job_created", result["steps"])
            self.assertEqual("local", result["job"]["backend"])
            graph_state = json.loads((trial / "graph_state.json").read_text(encoding="utf-8"))
            self.assertEqual("waiting_for_metrics", graph_state["status"])
            self.assertEqual("finalize", graph_state["last_completed_node"])
            self.assertTrue((trial / "node_events.jsonl").exists())

    def test_graph_cycle_can_run_safe_execution_chain_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_allowed_space(root)
            self._write_state(root)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {}}),
                encoding="utf-8",
            )
            self._write_coding_handoff(
                trial,
                [
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/graph_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            response_path = root / "mock_response.json"
            self._write_mock_code_response(response_path)

            with patch("research_agent.paths.project_root", return_value=root):
                result = run_graph_cycle(
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
