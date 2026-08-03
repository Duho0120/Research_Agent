import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.data_onboarding import profile_competition_data


class DataOnboardingTest(unittest.TestCase):
    def test_camel_case_sample_submission_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "bike"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("datetime,count,temp\n2020-01-01,1,10\n", encoding="utf-8")
            (data / "test.csv").write_text("datetime,temp\n2020-01-02,11\n", encoding="utf-8")
            (data / "sampleSubmission.csv").write_text("datetime,count\n2020-01-02,0\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = profile_competition_data("bike")

        roles = {item["role"]: item["name"] for item in profile["files"]}
        self.assertEqual("sampleSubmission.csv", roles["sample_submission"])

    def test_profile_competition_data_reads_csv_schemas_and_detects_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "titanic"
            data.mkdir(parents=True)
            (data / "train.csv").write_text(
                "PassengerId,Survived,Pclass,Sex,Age\n1,0,3,male,22\n2,1,1,female,38\n",
                encoding="utf-8",
            )
            (data / "test.csv").write_text(
                "PassengerId,Pclass,Sex,Age\n3,2,male,30\n",
                encoding="utf-8",
            )
            (data / "sample_submission.csv").write_text(
                "PassengerId,Survived\n3,0\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = profile_competition_data("titanic")

            self.assertEqual(profile["status"], "ready")
            self.assertEqual(profile["task_type"], "tabular")
            self.assertEqual(profile["target_candidates"], ["Survived"])
            self.assertEqual(profile["id_candidates"], ["PassengerId"])
            roles = {item["role"]: item["name"] for item in profile["files"]}
            self.assertEqual(roles["train"], "train.csv")
            self.assertEqual(roles["test"], "test.csv")
            self.assertEqual(roles["sample_submission"], "sample_submission.csv")
            self.assertTrue((root / "competitions" / "titanic" / "data_profile.json").exists())
            self.assertTrue((root / "competitions" / "titanic" / "data_profile.md").exists())
            saved = json.loads((root / "competitions" / "titanic" / "data_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["target_candidates"], ["Survived"])

    def test_profile_competition_data_uses_inspection_when_local_data_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            competition = root / "competitions" / "titanic"
            competition.mkdir(parents=True)
            (competition / "competition_inspection.json").write_text(
                json.dumps(
                    {
                        "competition_slug": "titanic",
                        "status": "ready",
                        "files": [
                            {"name": "train.csv", "size": "10KB"},
                            {"name": "test.csv", "size": "5KB"},
                            {"name": "sample_submission.csv", "size": "2KB"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = profile_competition_data("titanic")

            self.assertEqual(profile["status"], "needs_data_download")
            self.assertEqual(profile["task_type"], "unknown")
            self.assertEqual(profile["local_data_dir"], "data/titanic")
            self.assertIn("Download competition data", profile["human_review_questions"][0])


if __name__ == "__main__":
    unittest.main()
