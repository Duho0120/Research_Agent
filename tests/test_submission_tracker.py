import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.submission import record_submission_result


class SubmissionTrackerTest(unittest.TestCase):
    def test_record_submission_result_logs_and_marks_best_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                row = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    cv_score=0.82,
                    previous_lb_score=0.80,
                    previous_rank=120,
                    submitted_lb_score=0.84,
                    submitted_rank=90,
                    objective="maximize",
                    notes="Manual dry-run result.",
                )

            self.assertTrue(row["is_best"])
            log_path = root / "submissions" / "demo" / "submission_log.jsonl"
            self.assertTrue(log_path.exists())
            saved = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["score_delta"], 0.04)
            self.assertEqual(saved["rank_delta"], 30)
            self.assertTrue((root / "experiments" / "demo" / "BEST_TRIAL.md").exists())
            self.assertTrue((root / "memory" / "demo" / "best_trial.json").exists())
            self.assertTrue((trial / "BEST_MARKER.md").exists())
            self.assertTrue((trial / "VERSION.md").exists())
            state = (root / "competitions" / "demo" / "state.yaml").read_text(encoding="utf-8")
            self.assertIn("source: leaderboard_submission", state)
            self.assertIn("lb_score: 0.84", state)

    def test_new_best_replaces_previous_best_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_001 = root / "experiments" / "demo" / "trial_001"
            trial_002 = root / "experiments" / "demo" / "trial_002"
            trial_001.mkdir(parents=True)
            trial_002.mkdir(parents=True)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    previous_lb_score=None,
                    submitted_lb_score=0.80,
                    objective="maximize",
                )
                record_submission_result(
                    competition="demo",
                    trial_id="trial_002",
                    version_name="demo_trial_002_baseline_v01",
                    submission_file="experiments/demo/trial_002/submission.csv",
                    previous_lb_score=0.80,
                    submitted_lb_score=0.85,
                    objective="maximize",
                )

            self.assertFalse((trial_001 / "BEST_MARKER.md").exists())
            self.assertTrue((trial_002 / "BEST_MARKER.md").exists())
            best = json.loads((root / "memory" / "demo" / "best_trial.json").read_text(encoding="utf-8"))
            self.assertEqual(best["trial_id"], "trial_002")

    def test_minimize_objective_marks_lower_score_as_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                better = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    previous_lb_score=0.5,
                    submitted_lb_score=0.4,
                    objective="minimize",
                )
                worse = record_submission_result(
                    competition="demo",
                    trial_id="trial_002",
                    version_name="demo_trial_002_baseline_v01",
                    submission_file="experiments/demo/trial_002/submission.csv",
                    previous_lb_score=0.5,
                    submitted_lb_score=0.6,
                    objective="minimize",
                )

            self.assertTrue(better["is_best"])
            self.assertFalse(worse["is_best"])

    def test_non_best_maximize_submission_writes_without_best_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                row = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    previous_lb_score=0.85,
                    submitted_lb_score=0.80,
                    objective="maximize",
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertFalse(row["is_best"])
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())
            self.assertTrue((trial / "submission_result.md").exists())
            self.assertTrue((trial / "VERSION.md").exists())
            self.assertFalse((trial / "BEST_MARKER.md").exists())

    def test_historical_best_submission_prevents_false_best_from_previous_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial_001 = root / "experiments" / "demo" / "trial_001"
            trial_002 = root / "experiments" / "demo" / "trial_002"
            trial_001.mkdir(parents=True)
            trial_002.mkdir(parents=True)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                first = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    previous_lb_score=None,
                    submitted_lb_score=0.90,
                    objective="maximize",
                )
                second = record_submission_result(
                    competition="demo",
                    trial_id="trial_002",
                    version_name="demo_trial_002_v01",
                    submission_file="experiments/demo/trial_002/submission.csv",
                    previous_lb_score=0.80,
                    submitted_lb_score=0.85,
                    objective="maximize",
                )

            self.assertTrue(first["is_best"])
            self.assertFalse(second["is_best"])
            self.assertEqual(0.90, second["best_reference_score"])
            self.assertTrue((trial_001 / "BEST_MARKER.md").exists())
            self.assertFalse((trial_002 / "BEST_MARKER.md").exists())

    def test_none_previous_and_current_values_have_none_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                row = record_submission_result(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_baseline_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    previous_lb_score=None,
                    previous_rank=None,
                    submitted_lb_score=None,
                    submitted_rank=None,
                    objective="maximize",
                )

            self.assertIsNone(row["score_delta"])
            self.assertIsNone(row["rank_delta"])

    def test_unsupported_objective_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with self.assertRaisesRegex(ValueError, "Unsupported objective: median"):
                    record_submission_result(
                        competition="demo",
                        trial_id="trial_001",
                        version_name="demo_trial_001_baseline_v01",
                        submission_file="experiments/demo/trial_001/submission.csv",
                        previous_lb_score=0.5,
                        submitted_lb_score=0.6,
                        objective="median",
                    )


if __name__ == "__main__":
    unittest.main()


