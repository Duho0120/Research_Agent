import unittest
from pathlib import Path

from research_agent.integrations import kaggle_cli


class KaggleCliIntegrationTest(unittest.TestCase):
    def test_build_submit_args_preserves_values_without_shell_quoting(self):
        args = kaggle_cli.build_submit_args(
            competition_slug="demo-competition",
            submission_file="experiments/demo/trial_001/submission.csv",
            message="trial 001 v01",
        )

        self.assertEqual(
            args,
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                "demo-competition",
                "-f",
                "experiments/demo/trial_001/submission.csv",
                "-m",
                "trial 001 v01",
            ],
        )

    def test_normalize_competition_slug_from_url_or_name(self):
        self.assertEqual(
            kaggle_cli.normalize_competition_slug("https://www.kaggle.com/competitions/titanic"),
            "titanic",
        )
        self.assertEqual(kaggle_cli.normalize_competition_slug("titanic"), "titanic")

    def test_fetch_competition_files_returns_structured_command_result(self):
        def runner(args, cwd):
            return {"returncode": 0, "stdout": "name,size\ntrain.csv,10KB\n", "stderr": ""}

        result = kaggle_cli.fetch_competition_files(
            competition_slug="titanic",
            cwd=Path("workspace"),
            runner=runner,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "competition_files")
        self.assertEqual(result["args"], ["kaggle", "competitions", "files", "-c", "titanic"])

    def test_parse_competition_files_csv(self):
        text = "name,size,creationDate\ntrain.csv,10KB,2026-01-01\nsample_submission.csv,2KB,2026-01-01\n"

        files = kaggle_cli.parse_competition_files(text)

        self.assertEqual(files[0]["name"], "train.csv")
        self.assertEqual(files[0]["size"], "10KB")
        self.assertEqual(files[1]["name"], "sample_submission.csv")

    def test_check_cli_available_returns_structured_status(self):
        def runner(args, cwd):
            return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}

        result = kaggle_cli.check_cli_available(Path("workspace"), runner=runner)

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "check_cli")
        self.assertEqual(result["args"], ["kaggle", "--version"])
        self.assertEqual(result["stdout"], "Kaggle API 1.6.17\n")

    def test_check_cli_auth_returns_structured_status(self):
        def runner(args, cwd):
            return {"returncode": 1, "stdout": "", "stderr": "Could not find kaggle.json\n"}

        result = kaggle_cli.check_cli_auth(Path("workspace"), runner=runner)

        self.assertFalse(result["ok"])
        self.assertEqual(result["action"], "check_auth")
        self.assertEqual(result["args"], ["kaggle", "competitions", "list", "--page-size", "1"])
        self.assertIn("kaggle.json", result["stderr"])

    def test_submit_competition_returns_structured_command_result(self):
        calls = []

        def runner(args, cwd):
            calls.append((args, cwd))
            return {"returncode": 0, "stdout": "Successfully submitted\n", "stderr": ""}

        result = kaggle_cli.submit_competition(
            competition_slug="demo-competition",
            submission_file="experiments/demo/trial_001/submission.csv",
            message="trial 001 v01",
            cwd=Path("workspace"),
            runner=runner,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "submit")
        self.assertIn("kaggle competitions submit", result["command"])
        self.assertEqual(calls[0][0][0:3], ["kaggle", "competitions", "submit"])

    def test_fetch_leaderboard_returns_structured_command_result(self):
        def runner(args, cwd):
            return {"returncode": 0, "stdout": "team,score\nme,0.9\n", "stderr": ""}

        result = kaggle_cli.fetch_leaderboard(
            competition_slug="demo-competition",
            cwd=Path("workspace"),
            runner=runner,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "leaderboard")
        self.assertEqual(
            result["args"],
            ["kaggle", "competitions", "leaderboard", "demo-competition", "--show"],
        )

    def test_fetch_submissions_returns_structured_command_result(self):
        def runner(args, cwd):
            return {
                "returncode": 0,
                "stdout": "ref,fileName,date,description,status,publicScore,privateScore\n",
                "stderr": "",
            }

        result = kaggle_cli.fetch_submissions(
            competition_slug="demo-competition",
            cwd=Path("workspace"),
            runner=runner,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "submissions")
        self.assertEqual(
            result["args"],
            ["kaggle", "competitions", "submissions", "demo-competition", "--csv", "--page-size", "5"],
        )

    def test_parse_submissions_finds_latest_completed_public_score(self):
        text = (
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            "2,submission.csv,2026-07-14 06:13:40,trial_001,SubmissionStatus.COMPLETE,0.77511,\n"
            "1,old.csv,2026-07-13 06:13:40,old,SubmissionStatus.COMPLETE,0.76315,\n"
        )

        rows = kaggle_cli.parse_submissions(text)
        latest = kaggle_cli.latest_completed_submission(
            text,
            description="trial_001",
            file_name="submission.csv",
        )

        self.assertEqual(rows[0]["public_score"], 0.77511)
        self.assertEqual(latest["ref"], "2")
        self.assertEqual(latest["public_score"], 0.77511)

    def test_parse_leaderboard_csv_finds_team_score_and_rank(self):
        text = (
            "teamId,teamName,submissionDate,score\n"
            "11,alpha,2026-06-01,0.8123\n"
            "22,my team,2026-06-01,0.8456\n"
        )

        result = kaggle_cli.parse_leaderboard(text, team_name="my team")

        self.assertEqual(result["status"], "found")
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["score"], 0.8456)
        self.assertEqual(result["team_name"], "my team")

    def test_parse_leaderboard_table_finds_team_score_and_rank(self):
        text = (
            "teamName           score\n"
            "----------------  -------\n"
            "alpha             0.8123\n"
            "my team           0.8456\n"
        )

        result = kaggle_cli.parse_leaderboard(text, team_name="my team")

        self.assertEqual(result["status"], "found")
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["score"], 0.8456)

    def test_poll_leaderboard_retries_until_team_appears(self):
        calls = []
        outputs = [
            "teamId,teamName,score\n11,alpha,0.8123\n",
            "teamId,teamName,score\n11,alpha,0.8123\n22,my team,0.8456\n",
        ]

        def runner(args, cwd):
            calls.append(args)
            return {"returncode": 0, "stdout": outputs[len(calls) - 1], "stderr": ""}

        result = kaggle_cli.poll_leaderboard(
            competition_slug="demo-competition",
            team_name="my team",
            cwd=Path("workspace"),
            attempts=2,
            sleep_seconds=0,
            runner=runner,
        )

        self.assertEqual(result["status"], "found")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["score"], 0.8456)
        self.assertEqual(len(calls), 2)

    def test_poll_leaderboard_times_out_without_mutating_result(self):
        def runner(args, cwd):
            return {"returncode": 0, "stdout": "teamId,teamName,score\n11,alpha,0.8123\n", "stderr": ""}

        result = kaggle_cli.poll_leaderboard(
            competition_slug="demo-competition",
            team_name="my team",
            cwd=Path("workspace"),
            attempts=2,
            sleep_seconds=0,
            runner=runner,
        )

        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["score"])
        self.assertIsNone(result["rank"])


if __name__ == "__main__":
    unittest.main()
