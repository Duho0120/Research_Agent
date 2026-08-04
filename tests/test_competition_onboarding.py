import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.competition_onboarding import start_competition


class CompetitionOnboardingTest(unittest.TestCase):
    def test_start_competition_inspects_initializes_and_plans_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 0, "stdout": "ref,title\nplayground,demo\n", "stderr": ""}
                if args == ["kaggle", "competitions", "files", "-c", "titanic"]:
                    return {
                        "returncode": 0,
                        "stdout": "name,size\ntrain.csv,10KB\ntest.csv,5KB\nsample_submission.csv,2KB\n",
                        "stderr": "",
                    }
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("research_agent.paths.project_root", return_value=root):
                result = start_competition(
                    "https://www.kaggle.com/competitions/titanic",
                    metric="accuracy",
                    objective="maximize",
                    runner=runner,
                )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["competition_slug"], "titanic")
            self.assertTrue((root / "competitions" / "titanic" / "competition_inspection.md").exists())
            self.assertTrue((root / "competitions" / "titanic" / "overview.md").exists())
            self.assertTrue((root / "competitions" / "titanic" / "data_notes.md").exists())
            self.assertTrue((root / "competitions" / "titanic" / "data_profile.md").exists())
            self.assertTrue((root / "experiments" / "titanic" / "trial_001" / "plan.md").exists())
            self.assertTrue((root / "experiments" / "titanic" / "trial_001" / "config.yaml").exists())
            overview = (root / "competitions" / "titanic" / "overview.md").read_text(encoding="utf-8")
            self.assertIn("https://www.kaggle.com/competitions/titanic", overview)
            self.assertIn("sample_submission.csv", overview)
            state = (root / "competitions" / "titanic" / "state.yaml").read_text(encoding="utf-8")
            self.assertIn("metric: accuracy", state)
            profile = json.loads((root / "competitions" / "titanic" / "data_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["status"], "needs_data_download")

    def test_start_competition_records_blocked_inspection_without_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 1, "stdout": "", "stderr": "Could not find kaggle.json\n"}
                return {"returncode": 1, "stdout": "", "stderr": "should not fetch"}

            with patch("research_agent.paths.project_root", return_value=root):
                result = start_competition("titanic", runner=runner)

            self.assertEqual(result["status"], "auth_failed")
            self.assertTrue((root / "competitions" / "titanic" / "competition_inspection.json").exists())
            self.assertFalse((root / "experiments" / "titanic" / "trial_001" / "plan.md").exists())
            saved = json.loads(
                (root / "competitions" / "titanic" / "competition_inspection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["status"], "auth_failed")


if __name__ == "__main__":
    unittest.main()
