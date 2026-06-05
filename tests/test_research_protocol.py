import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.research_protocol import build_research_protocol
from kaggle_research_agent.cli import main


class ResearchProtocolTest(unittest.TestCase):
    def test_protocol_preserves_public_anchor_when_local_improves_but_lb_worsens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root, objective="minimize", best_cv=0.591306, best_lb=0.5984)
            self._write_trial(
                root,
                "trial_local_gain_public_worse",
                {
                    "cv_score": 0.586609,
                    "lb_score": 0.5994936146,
                    "objective": "minimize",
                    "metric": "mean_binary_logloss",
                    "target_scores": {"Q2": 0.634620, "S2": 0.560297},
                    "notes": "Local CV improved but Public LB worsened.",
                },
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = build_research_protocol("demo", "trial_local_gain_public_worse", "trial_next")

            self.assertEqual("validation_review", result["recommended_next_trial"]["strategy"])
            self.assertEqual("high", result["risk"]["level"])
            self.assertIn("public_anchor_preserved", result["risk"]["flags"])
            self.assertIn("safe", result["candidate_actions"])
            self.assertIn("main", result["candidate_actions"])
            self.assertIn("aggressive", result["candidate_actions"])
            self.assertIn("Need User Check", (root / "experiments" / "demo" / "trial_local_gain_public_worse" / "research_protocol.md").read_text(encoding="utf-8"))
            self.assertTrue((root / "experiments" / "demo" / "trial_local_gain_public_worse" / "research_protocol.json").exists())

    def test_protocol_marks_local_only_best_as_submit_safe_before_model_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root, objective="minimize", best_cv=0.591306, best_lb=0.5984)
            self._write_trial(
                root,
                "trial_local_best_unsubmitted",
                {
                    "cv_score": 0.584915,
                    "lb_score": None,
                    "objective": "minimize",
                    "metric": "mean_binary_logloss",
                    "notes": "Local best, Public LB unknown.",
                },
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = build_research_protocol("demo", "trial_local_best_unsubmitted", "trial_next")

            self.assertEqual("safe_submission_or_holdout_confirmation", result["recommended_next_trial"]["strategy"])
            self.assertIn("local_best_public_unknown", result["risk"]["flags"])
            self.assertIn("Record or request leaderboard evidence before promoting the local best.", result["need_user_check"])
            self.assertIn("Do not change model family before resolving public evidence.", result["do_not_change"])

    def test_research_protocol_cli_writes_protocol_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_competition(root, objective="minimize", best_cv=0.591306, best_lb=0.5984)
            self._write_trial(root, "trial_001", {"cv_score": 0.592, "lb_score": None, "objective": "minimize"})

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(["research-protocol", "--competition", "demo", "--trial", "trial_001", "--next-trial", "trial_002"])

            self.assertEqual(0, code)
            self.assertTrue((root / "experiments" / "demo" / "trial_001" / "research_protocol.json").exists())

    def _write_competition(self, root: Path, *, objective: str, best_cv: float, best_lb: float) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "state.yaml").write_text(
            "competition:\n"
            "  name: demo\n"
            "  platform: external\n"
            "  metric: mean_binary_logloss\n"
            f"  objective: {objective}\n"
            "current_state:\n"
            "  active_trial: trial_001\n"
            "  best_trial:\n"
            "    trial_id: public_anchor\n"
            f"    cv_score: {best_cv}\n"
            f"    lb_score: {best_lb}\n"
            "  consecutive_failures: 0\n"
            "  validation_suspected: true\n",
            encoding="utf-8",
        )
        profile = root / "competitions" / "demo" / "data_profile.json"
        profile.write_text(
            json.dumps(
                {
                    "competition": "demo",
                    "platform": "external",
                    "task_type": "multi_output_binary_classification",
                    "train_rows": 450,
                    "subjects": 10,
                    "target_columns": ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"],
                }
            ),
            encoding="utf-8",
        )
        memory = root / "memory" / "demo"
        memory.mkdir(parents=True)
        (memory / "research_notes.md").write_text("Public baseline must remain the anchor.\n", encoding="utf-8")
        (memory / "rules.md").write_text("Do not use random KFold as main validation.\n", encoding="utf-8")

    def _write_trial(self, root: Path, trial_id: str, metrics: dict) -> None:
        trial = root / "experiments" / "demo" / trial_id
        trial.mkdir(parents=True)
        (trial / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (trial / "plan.md").write_text(f"# {trial_id} Plan\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
