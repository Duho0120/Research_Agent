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
            metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(0.84, metrics["lb_score"])
            self.assertEqual(90, metrics["submitted_rank"])
            self.assertTrue(metrics["is_best_submission"])
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
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\nplayground,demo\n", "stderr": ""}
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

    def test_submit_trial_reads_kaggle_submission_public_score_without_leaderboard_polling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_kaggle_score"
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
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\nplayground,demo\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submit"]:
                    return {"returncode": 0, "stdout": "Successfully submitted\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submissions"]:
                    return {
                        "returncode": 0,
                        "stdout": (
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "2,submission.csv,2026-07-14 06:13:40,demo trial kaggle v01,"
                            "SubmissionStatus.COMPLETE,0.77511,\n"
                        ),
                        "stderr": "",
                    }
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_kaggle_score",
                    version_name="demo_trial_kaggle_v01",
                    submission_file="experiments/demo/trial_kaggle_score/submission.csv",
                    objective="maximize",
                    kaggle_competition_slug="demo-competition",
                    kaggle_message="demo trial kaggle v01",
                    kaggle_runner=runner,
                )

            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["submitted_lb_score"], 0.77511)
            self.assertIsNone(result["submitted_rank"])
            self.assertTrue(any(call[:3] == ["kaggle", "competitions", "submissions"] for call in calls))
            self.assertTrue((trial / "BEST_MARKER.md").exists())

    def test_submit_trial_reuses_existing_completed_kaggle_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_existing"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.86}), encoding="utf-8")
            calls = []

            def runner(args, cwd):
                calls.append(args)
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API\n", "stderr": ""}
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\ndemo,Demo\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submissions"]:
                    return {
                        "returncode": 0,
                        "stdout": (
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "2,submission.csv,2026-07-27,demo trial existing,"
                            "SubmissionStatus.COMPLETE,0.77511,\n"
                        ),
                        "stderr": "",
                    }
                if args[:3] == ["kaggle", "competitions", "submit"]:
                    raise AssertionError("existing submission must not be submitted again")
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_existing",
                    version_name="trial_existing_auto",
                    submission_file="experiments/demo/trial_existing/submission.csv",
                    kaggle_competition_slug="demo",
                    kaggle_message="demo trial existing",
                    poll_attempts=1,
                    poll_interval_seconds=0,
                    kaggle_runner=runner,
                )

            self.assertEqual("submitted", result["status"])
            self.assertEqual(0.77511, result["submitted_lb_score"])
            self.assertTrue(result["submit_result"]["reused_existing_submission"])
            self.assertFalse(any(call[:3] == ["kaggle", "competitions", "submit"] for call in calls))

    def test_submit_trial_waits_for_existing_pending_submission_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_pending"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,1\n", encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.86}), encoding="utf-8")
            calls = []

            def runner(args, cwd):
                calls.append(args)
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API\n", "stderr": ""}
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\ndemo,Demo\n", "stderr": ""}
                if args[:3] == ["kaggle", "competitions", "submissions"]:
                    return {
                        "returncode": 0,
                        "stdout": (
                            "ref,fileName,date,description,status,publicScore,privateScore\n"
                            "3,submission.csv,2026-07-27,demo trial pending,"
                            "SubmissionStatus.PENDING,,\n"
                        ),
                        "stderr": "",
                    }
                if args[:3] == ["kaggle", "competitions", "submit"]:
                    raise AssertionError("pending submission must not be submitted again")
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = submit_trial(
                    competition="demo",
                    trial_id="trial_pending",
                    version_name="trial_pending_auto",
                    submission_file="experiments/demo/trial_pending/submission.csv",
                    kaggle_competition_slug="demo",
                    kaggle_message="demo trial pending",
                    poll_attempts=2,
                    poll_interval_seconds=0,
                    kaggle_runner=runner,
                )

            self.assertEqual("submission_pending", result["status"])
            self.assertTrue(result["submit_result"]["reused_existing_submission"])
            self.assertFalse(any(call[:3] == ["kaggle", "competitions", "submit"] for call in calls))

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
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
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
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\nplayground,demo\n", "stderr": ""}
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


class DaconSubmitTrialDispatchTest(unittest.TestCase):
    def test_submit_trial_dispatches_to_dacon_when_dacon_competition_id_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.42, "objective": "minimize"}), encoding="utf-8"
            )
            calls = []

            def fake_submit(file_path, token, competition_id, team_name, memo):
                calls.append((file_path, token, competition_id, team_name, memo))
                return {"message": "ok"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {"DACON_PERSONAL_TOKEN": "test-token"}, clear=False):
                    result = submit_trial(
                        competition="demo",
                        trial_id="trial_001",
                        version_name="demo_trial_001_v01",
                        submission_file="experiments/demo/trial_001/submission.csv",
                        dacon_competition_id="235610",
                        dacon_team_name="my-team",
                        dacon_submit_fn=fake_submit,
                    )

        self.assertEqual("submitted", result["status"])
        self.assertEqual("235610", result["dacon_competition_id"])
        self.assertEqual(1, len(calls))
        self.assertEqual("test-token", calls[0][1])
        self.assertEqual("235610", calls[0][2])
        self.assertEqual("my-team", calls[0][3])

    def test_submit_trial_uploads_dacon_file_named_after_the_version_not_submission_csv(self):
        # Every trial's local file is named outputs/submission.csv, so without
        # renaming the upload, DACON's own submission-history page would show
        # an identical, indistinguishable file name for every single trial.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.42, "objective": "minimize"}), encoding="utf-8"
            )
            captured = {}

            def fake_submit(file_path, token, competition_id, team_name, memo):
                captured["file_path"] = file_path
                captured["existed_at_call_time"] = Path(file_path).exists()
                captured["content_at_call_time"] = Path(file_path).read_text(encoding="utf-8")
                return {"message": "ok"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {"DACON_PERSONAL_TOKEN": "test-token"}, clear=False):
                    submit_trial(
                        competition="demo",
                        trial_id="trial_002",
                        version_name="demo_trial_002_v01",
                        submission_file="experiments/demo/trial_002/submission.csv",
                        dacon_competition_id="235610",
                        dacon_team_name="my-team",
                        dacon_submit_fn=fake_submit,
                    )

            uploaded_path = Path(captured["file_path"])
            self.assertEqual("demo_trial_002_v01.csv", uploaded_path.name)
            self.assertTrue(captured["existed_at_call_time"])
            self.assertEqual("id,target\n1,0\n", captured["content_at_call_time"])
            # The temp copy is cleaned up afterward, and the original workspace
            # output file is untouched under its normal name.
            self.assertFalse(uploaded_path.exists())
            self.assertTrue((trial / "submission.csv").exists())

    def test_submit_trial_records_dacon_leaderboard_score_from_submission_history(self):
        # DACON's submit endpoint returns no score, but its submission
        # history does. Reading it back is what lets lb_score be recorded
        # automatically instead of staying null until someone checks the
        # leaderboard by hand and passes --after-score.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_003"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.591, "objective": "maximize"}), encoding="utf-8"
            )

            def fake_submit(*args):
                return {"message": "ok"}

            def fake_fetch(competition_id, **kwargs):
                return {
                    "ok": True,
                    "submissions": [
                        {"sub_id": 2, "memo": "trial_003 run", "score": 0.6006, "c_time": "2026-07-31 09:52:36"},
                        {"sub_id": 1, "memo": "an older unrelated submission", "score": 0.1},
                    ],
                }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {"DACON_PERSONAL_TOKEN": "test-token"}, clear=False):
                    result = submit_trial(
                        competition="demo",
                        trial_id="trial_003",
                        version_name="demo_trial_003_v01",
                        submission_file="experiments/demo/trial_003/submission.csv",
                        objective="maximize",
                        dacon_competition_id="236716",
                        dacon_team_name="my-team",
                        dacon_message="trial_003 run",
                        dacon_submit_fn=fake_submit,
                        dacon_fetch_submissions_fn=fake_fetch,
                    )

        self.assertEqual("submitted", result["status"])
        self.assertEqual(0.6006, result["submitted_lb_score"])

    def test_submit_trial_still_succeeds_when_dacon_score_lookup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_004"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.5, "objective": "maximize"}), encoding="utf-8"
            )

            def fake_submit(*args):
                return {"message": "ok"}

            def failing_fetch(competition_id, **kwargs):
                raise RuntimeError("session token expired")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {"DACON_PERSONAL_TOKEN": "test-token"}, clear=False):
                    result = submit_trial(
                        competition="demo",
                        trial_id="trial_004",
                        version_name="demo_trial_004_v01",
                        submission_file="experiments/demo/trial_004/submission.csv",
                        dacon_competition_id="236716",
                        dacon_team_name="my-team",
                        dacon_message="trial_004 run",
                        dacon_submit_fn=fake_submit,
                        dacon_fetch_submissions_fn=failing_fetch,
                    )

        # The upload itself already succeeded; a score lookup failure must
        # not turn that into a reported submission failure.
        self.assertEqual("submitted", result["status"])
        self.assertIsNone(result["submitted_lb_score"])

    def test_submit_trial_ignores_dacon_history_entries_from_other_submissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_005"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.5, "objective": "maximize"}), encoding="utf-8"
            )

            def fake_submit(*args):
                return {"message": "ok"}

            def fake_fetch(competition_id, **kwargs):
                return {
                    "ok": True,
                    "submissions": [
                        {"sub_id": 9, "memo": "someone else's manual upload", "score": 0.99},
                    ],
                }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {"DACON_PERSONAL_TOKEN": "test-token"}, clear=False):
                    result = submit_trial(
                        competition="demo",
                        trial_id="trial_005",
                        version_name="demo_trial_005_v01",
                        submission_file="experiments/demo/trial_005/submission.csv",
                        dacon_competition_id="236716",
                        dacon_team_name="my-team",
                        dacon_message="trial_005 run",
                        dacon_submit_fn=fake_submit,
                        dacon_fetch_submissions_fn=fake_fetch,
                    )

        self.assertEqual("submitted", result["status"])
        self.assertIsNone(result["submitted_lb_score"])

    def test_submit_trial_blocks_when_dacon_token_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.42, "objective": "minimize"}), encoding="utf-8"
            )
            calls = []

            def fake_submit(*args):
                calls.append(args)
                return {"message": "ok"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch.dict("os.environ", {}, clear=True):
                    result = submit_trial(
                        competition="demo",
                        trial_id="trial_002",
                        version_name="demo_trial_002_v01",
                        submission_file="experiments/demo/trial_002/submission.csv",
                        dacon_competition_id="235610",
                        dacon_team_name="my-team",
                        dacon_submit_fn=fake_submit,
                    )

        self.assertEqual("token_missing", result["status"])
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()


