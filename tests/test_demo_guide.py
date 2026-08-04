import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.demo_guide import (
    inspect_competition_url,
    list_demo_experiments,
    render_demo_experiment_status,
    run_demo_guide,
    summarize_demo_experiment,
)
from research_agent.workspace_preparer import prepare_workspace


class DemoGuideTest(unittest.TestCase):
    def test_demo_guide_defaults_to_langgraph_cycle(self):
        self.assertEqual("run_demo_graph_cycle", run_demo_guide.__kwdefaults__["run_cycle_fn"].__name__)

    def test_demo_guide_can_exit_from_menu(self):
        output = io.StringIO()
        inputs = iter(["4"])

        code = run_demo_guide(input_fn=lambda prompt: next(inputs), output=output)

        self.assertEqual(0, code)
        text = output.getvalue()
        self.assertIn("Autonomous ML Research Agent", text)
        self.assertIn("종료합니다.", text)

    def test_url_inspection_infers_kaggle_slug_even_when_dynamic_page_is_unreadable(self):
        with patch("research_agent.demo_guide._fetch_url_text", return_value="<html><title>Titanic</title></html>"):
            result = inspect_competition_url("https://www.kaggle.com/competitions/titanic/overview")

        self.assertEqual("unreadable", result["status"])
        self.assertEqual("kaggle", result["inferred"]["platform"])
        self.assertEqual("titanic", result["inferred"]["competition"])

    def test_demo_guide_starts_new_experiment_saves_materials_and_runs_cycle_with_openai(self):
        output = io.StringIO()
        inputs = iter(
            [
                "1",
                "https://www.kaggle.com/competitions/titanic/overview",
                "",
                "",
                "Titanic - Machine Learning from Disaster",
                "accuracy",
                "maximize",
                "Survived",
                "PassengerId",
                "train.csv, test.csv",
                (
                    "Predict passenger survival as a binary classification task. "
                    "train.csv has labels, test.csv has unlabeled passengers, "
                    "and submissions use PassengerId and Survived columns."
                ),
                "",
                "y",
            ]
        )
        prepare_calls = []
        run_calls = []

        def fake_prepare(*args, **kwargs):
            prepare_calls.append((args, kwargs))
            return {
                "competition": args[0],
                "status": "ready",
                "source_path": r"C:\demo\titanic",
                "data_check": {"missing_files": []},
            }

        def fake_run(*args, **kwargs):
            run_calls.append((args, kwargs))
            return {"competition": args[0], "trial_id": args[1], "status": "completed"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                with patch("research_agent.demo_guide._fetch_url_text", return_value="<html><title>Titanic</title></html>"):
                    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                        code = run_demo_guide(
                            input_fn=lambda prompt: next(inputs),
                            output=output,
                            prepare_workspace_fn=fake_prepare,
                            run_cycle_fn=fake_run,
                        )
                materials = json.loads((root / "competitions" / "titanic" / "source_materials.json").read_text(encoding="utf-8"))
                overview = (root / "competitions" / "titanic" / "overview.md").read_text(encoding="utf-8")

        self.assertEqual(0, code)
        self.assertEqual(2, len(prepare_calls))
        self.assertEqual([("titanic", "trial_001")], [call[0] for call in run_calls])
        self.assertEqual("openai", run_calls[0][1]["provider"])
        self.assertEqual("gpt-5.5", run_calls[0][1]["model"])
        self.assertTrue(run_calls[0][1]["allow_api"])
        self.assertTrue(run_calls[0][1]["run_now"])
        self.assertTrue(run_calls[0][1]["show_progress"])
        self.assertEqual(0, run_calls[0][1]["trial_llm_calls"])
        self.assertEqual(0, run_calls[0][1]["strategy_calls_today"])
        self.assertEqual("https://www.kaggle.com/competitions/titanic/overview", materials["competition_url"])
        self.assertIn("Predict passenger survival", materials["source_materials"]["source_summary"])
        self.assertIn("Predict passenger survival", materials["source_materials"]["problem_description"])
        self.assertIn("Predict passenger survival", overview)
        text = output.getvalue()
        self.assertIn("대회 링크 내용을 자동으로 충분히 확인하지 못했습니다.", text)
        self.assertIn("workspace가 생성되었습니다.", text)
        self.assertIn("1회 실험 사이클을 시작합니다.", text)

    def test_demo_guide_blocks_real_api_without_openai_key(self):
        output = io.StringIO()
        inputs = iter(["2", "1", "y"])
        run_calls = []

        def fake_run(*args, **kwargs):
            run_calls.append((args, kwargs))
            return {"competition": args[0], "trial_id": args[1], "status": "completed"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                prepare_workspace(
                    "titanic",
                    topic="Titanic survival prediction",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="Survived",
                    id_column="PassengerId",
                    required_data_files=[],
                )
                with patch.dict("os.environ", {}, clear=True):
                    code = run_demo_guide(
                        input_fn=lambda prompt: next(inputs),
                        output=output,
                        run_cycle_fn=fake_run,
                    )

        self.assertEqual(1, code)
        self.assertEqual([], run_calls)
        self.assertIn("OPENAI_API_KEY가 설정되어 있지 않습니다.", output.getvalue())

    def test_demo_experiment_summary_detects_missing_and_found_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                prepare_workspace(
                    "titanic",
                    topic="Titanic survival prediction",
                    platform="kaggle",
                    metric="accuracy",
                    objective="maximize",
                    create_workspace=True,
                    target_column="Survived",
                    id_column="PassengerId",
                    required_data_files=["train.csv", "test.csv"],
                )
                source = root / "demo_workspaces" / "titanic"
                (source / "data" / "train.csv").write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")

                experiments = list_demo_experiments()
                summary = summarize_demo_experiment("titanic")
                status_text = render_demo_experiment_status(summary)

        self.assertEqual(["titanic"], [item["competition"] for item in experiments])
        self.assertEqual("needs_data", summary["status"])
        self.assertEqual(["train.csv"], summary["data_status"]["found_files"])
        self.assertEqual(["test.csv"], summary["data_status"]["missing_files"])
        self.assertIn("train.csv: found", status_text)
        self.assertIn("test.csv: missing", status_text)


if __name__ == "__main__":
    unittest.main()
