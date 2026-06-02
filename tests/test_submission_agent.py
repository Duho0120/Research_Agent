import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.agents.submission import prepare_submission, submit_trial


class SubmissionAgentTest(unittest.TestCase):
    def test_prepare_submission_writes_manifest_without_marking_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.83, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                manifest = prepare_submission(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    objective="maximize",
                    notes="Ready for approval",
                )

            self.assertEqual(manifest["status"], "ready")
            self.assertTrue(manifest["requires_user_approval"])
            self.assertEqual(manifest["cv_score"], 0.83)
            self.assertTrue((trial / "submit_manifest.json").exists())
            self.assertTrue((trial / "submit_manifest.md").exists())
            self.assertFalse((trial / "BEST_MARKER.md").exists())
            self.assertFalse((root / "submissions" / "demo" / "submission_log.jsonl").exists())

    def test_prepare_submission_blocks_when_metrics_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_missing_metrics"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                manifest = prepare_submission(
                    competition="demo",
                    trial_id="trial_missing_metrics",
                    version_name="demo_trial_missing_metrics_v01",
                    submission_file="experiments/demo/trial_missing_metrics/submission.csv",
                )

            self.assertEqual(manifest["status"], "blocked")
            self.assertIn("Missing metrics file", manifest["checks"])
            self.assertTrue((trial / "submit_manifest.json").exists())

    def test_submit_trial_records_before_and_after_leaderboard_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            submission = trial / "submission.csv"
            submission.write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.83, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_001",
                    version_name="demo_trial_001_v01",
                    submission_file="experiments/demo/trial_001/submission.csv",
                    before_score=0.8,
                    before_rank=120,
                    after_score=0.84,
                    after_rank=90,
                    objective="maximize",
                    submit_command=None,
                )

            self.assertEqual(result["status"], "recorded")
            self.assertEqual(result["previous_lb_score"], 0.8)
            self.assertEqual(result["submitted_lb_score"], 0.84)
            self.assertEqual(result["submitted_rank"], 90)
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())
            self.assertTrue((trial / "submission_run.md").exists())
            self.assertTrue((trial / "BEST_MARKER.md").exists())

    def test_submit_trial_runs_commands_and_parses_json_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.75, "objective": "maximize"}),
                encoding="utf-8",
            )
            before = root / "before.json"
            after = root / "after.json"
            marker = root / "submitted.txt"
            before.write_text(json.dumps({"score": 0.7, "rank": 200}), encoding="utf-8")
            after.write_text(json.dumps({"score": 0.76, "rank": 150}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_002",
                    version_name="demo_trial_002_v01",
                    submission_file="experiments/demo/trial_002/submission.csv",
                    before_command=f"python -c \"import pathlib; print(pathlib.Path(r'{before}').read_text())\"",
                    submit_command=f"python -c \"import pathlib; pathlib.Path(r'{marker}').write_text('ok')\"",
                    after_command=f"python -c \"import pathlib; print(pathlib.Path(r'{after}').read_text())\"",
                    objective="maximize",
                )

            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["previous_rank"], 200)
            self.assertEqual(result["submitted_lb_score"], 0.76)
            self.assertTrue(marker.exists())
            self.assertTrue((trial / "submission_run.json").exists())

    def test_submit_trial_uses_kaggle_adapter_and_polling_before_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_kaggle"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.86, "objective": "maximize"}),
                encoding="utf-8",
            )
            calls = []

            def runner(args, cwd):
                calls.append(args)
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "config", "view"]:
                    return {"returncode": 0, "stdout": "username: hidden\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submit"]:
                    return {"returncode": 0, "stdout": "Successfully submitted\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "leaderboard"]:
                    return {"returncode": 0, "stdout": "teamId,teamName,score\n22,my team,0.87\n", "stderr": ""}
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_kaggle",
                    version_name="demo_trial_kaggle_v01",
                    submission_file="experiments/demo/trial_kaggle/submission.csv",
                    objective="maximize",
                    kaggle_competition_slug="demo-competition",
                    kaggle_team_name="my team",
                    kaggle_message="demo trial kaggle v01",
                    poll_leaderboard=True,
                    poll_attempts=1,
                    poll_interval_seconds=0,
                    kaggle_runner=runner,
                )

            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["submitted_lb_score"], 0.87)
            self.assertEqual(result["submitted_rank"], 1)
            self.assertTrue(any(call[:3] == ["kaggle", "competitions", "submit"] for call in calls))
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())
            self.assertTrue((trial / "BEST_MARKER.md").exists())

    def test_submit_trial_does_not_record_when_kaggle_auth_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_auth_fail"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.86}), encoding="utf-8")

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "config", "view"]:
                    return {"returncode": 1, "stdout": "", "stderr": "Could not find kaggle.json\n"}
                return {"returncode": 1, "stdout": "", "stderr": "should not submit"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_auth_fail",
                    version_name="demo_trial_auth_fail_v01",
                    submission_file="experiments/demo/trial_auth_fail/submission.csv",
                    kaggle_competition_slug="demo-competition",
                    kaggle_team_name="my team",
                    poll_leaderboard=True,
                    poll_attempts=1,
                    poll_interval_seconds=0,
                    kaggle_runner=runner,
                )

            self.assertEqual(result["status"], "kaggle_auth_failed")
            self.assertFalse((root / "submissions" / "demo" / "submission_log.jsonl").exists())
            self.assertFalse((trial / "BEST_MARKER.md").exists())

    def test_submit_trial_does_not_record_when_leaderboard_polling_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_poll_timeout"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.86}), encoding="utf-8")

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "config", "view"]:
                    return {"returncode": 0, "stdout": "username: hidden\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submit"]:
                    return {"returncode": 0, "stdout": "Successfully submitted\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "leaderboard"]:
                    return {"returncode": 0, "stdout": "teamId,teamName,score\n11,other,0.88\n", "stderr": ""}
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_poll_timeout",
                    version_name="demo_trial_poll_timeout_v01",
                    submission_file="experiments/demo/trial_poll_timeout/submission.csv",
                    kaggle_competition_slug="demo-competition",
                    kaggle_team_name="my team",
                    poll_leaderboard=True,
                    poll_attempts=2,
                    poll_interval_seconds=0,
                    kaggle_runner=runner,
                )

            self.assertEqual(result["status"], "leaderboard_timeout")
            self.assertFalse((root / "submissions" / "demo" / "submission_log.jsonl").exists())
            self.assertFalse((trial / "BEST_MARKER.md").exists())

    def test_submit_trial_blocks_when_submission_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_003").mkdir(parents=True)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_003",
                    version_name="demo_trial_003_v01",
                    submission_file="experiments/demo/trial_003/submission.csv",
                    before_score=0.7,
                    before_rank=100,
                    after_score=0.71,
                    after_rank=90,
                )

            self.assertEqual(result["status"], "blocked")
            self.assertIn("Missing submission file", result["reason"])


if __name__ == "__main__":
    unittest.main()


