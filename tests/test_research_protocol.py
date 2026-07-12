import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.research_protocol import build_research_protocol
from kaggle_research_agent.cli import main


class ResearchProtocolTest(unittest.TestCase):
    def test_global_protocol_uses_simple_actions_without_leaderboard_or_candidate_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root)
            self._write_trial(
                root,
                "trial_001",
                {"cv_score": 0.61, "lb_score": 0.70, "objective": "minimize"},
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = build_research_protocol("demo", "trial_001", "trial_002")

            self.assertEqual("controlled_improvement", result["recommended_action"]["strategy"])
            self.assertIsInstance(result["candidate_actions"], list)
            self.assertNotIn("risk", result)
            self.assertNotIn("recommended_next_trial", result)
            self.assertNotIn("safe", result["candidate_actions"])
            self.assertNotIn("lb_score", result["evidence"])

    def test_competition_policy_can_enable_optional_leaderboard_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root)
            policy_dir = root / "configs" / "demo"
            policy_dir.mkdir(parents=True)
            (policy_dir / "research_policy.yaml").write_text(
                "leaderboard_tracking:\n"
                "  enabled: true\n"
                "  affects_strategy: true\n",
                encoding="utf-8",
            )
            self._write_trial(
                root,
                "trial_001",
                {"cv_score": 0.59, "lb_score": 0.70, "objective": "minimize"},
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = build_research_protocol("demo", "trial_001", "trial_002")

            self.assertEqual("validation_review", result["recommended_action"]["strategy"])
            self.assertEqual(0.70, result["optional_evidence"]["leaderboard_score"])
            self.assertIn("Local and leaderboard movement disagree.", result["issues"])

    def test_research_protocol_cli_writes_protocol_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root)
            self._write_trial(root, "trial_001", {"cv_score": 0.61, "objective": "minimize"})

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    ["research-protocol", "--competition", "demo", "--trial", "trial_001", "--next-trial", "trial_002"]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "research_protocol.json").exists())

    def _write_competition(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "state.yaml").write_text(
            "competition:\n"
            "  name: demo\n"
            "  platform: external\n"
            "  metric: score\n"
            "  objective: minimize\n"
            "current_state:\n"
            "  active_trial: trial_001\n"
            "  best_trial:\n"
            "    trial_id: trial_000\n"
            "    cv_score: 0.60\n"
            "    lb_score: 0.65\n"
            "  consecutive_failures: 0\n"
            "  validation_suspected: false\n",
            encoding="utf-8",
        )
        (comp / "data_profile.json").write_text(
            json.dumps({"task_type": "tabular", "train_rows": 500}),
            encoding="utf-8",
        )
        memory = root / "memory" / "demo"
        memory.mkdir(parents=True)
        (memory / "research_notes.md").write_text("Generic research notes.\n", encoding="utf-8")
        (memory / "rules.md").write_text("Change one primary axis.\n", encoding="utf-8")

    def _write_trial(self, root: Path, trial_id: str, metrics: dict) -> None:
        trial = root / "experiments" / "demo" / trial_id
        trial.mkdir(parents=True)
        (trial / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (trial / "plan.md").write_text(f"# {trial_id} Plan\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
