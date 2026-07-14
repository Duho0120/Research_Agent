import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.competition_inspector import inspect_competition


class CompetitionInspectorTest(unittest.TestCase):
    def test_inspect_competition_writes_summary_files_from_kaggle_cli(self):
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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = inspect_competition("https://www.kaggle.com/competitions/titanic", runner=runner)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["competition_slug"], "titanic")
            self.assertEqual(len(result["files"]), 3)
            inspection_json = root / "competitions" / "titanic" / "competition_inspection.json"
            inspection_md = root / "competitions" / "titanic" / "competition_inspection.md"
            self.assertTrue(inspection_json.exists())
            self.assertTrue(inspection_md.exists())
            saved = json.loads(inspection_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["files"][0]["name"], "train.csv")

    def test_inspect_competition_blocks_when_auth_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "competitions", "list", "--page-size", "1"]:
                    return {"returncode": 1, "stdout": "", "stderr": "Could not find kaggle.json\n"}
                return {"returncode": 1, "stdout": "", "stderr": "should not fetch"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = inspect_competition("titanic", runner=runner)

            self.assertEqual(result["status"], "auth_failed")
            self.assertEqual(result["files"], [])
            self.assertTrue((root / "competitions" / "titanic" / "competition_inspection.json").exists())


if __name__ == "__main__":
    unittest.main()
