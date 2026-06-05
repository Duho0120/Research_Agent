import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.graph.research_graph import run_graph_cycle


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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_graph_cycle("demo", "trial_001", create_job_request=False)

            self.assertEqual(result["status"], "completed")
            self.assertIn("planned", result["steps"])
            self.assertIn("diagnosed", result["steps"])
            self.assertIn("remembered", result["steps"])
            self.assertEqual("trial_001", result["memory"]["trial_id"])
            self.assertTrue((trial / "diagnosis.md").exists())

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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_graph_cycle("demo", "trial_001", create_job_request=True)

            self.assertEqual(result["status"], "waiting_for_metrics")
            self.assertIn("execution_decided", result["steps"])
            self.assertIn("local_job_created", result["steps"])
            self.assertEqual("local", result["job"]["backend"])


if __name__ == "__main__":
    unittest.main()
