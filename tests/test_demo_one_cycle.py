import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.demo_one_cycle import build_demo_plan_payload, run_demo_one_cycle
from kaggle_research_agent.graph.demo_auto_loop import run_demo_graph_auto_loop
from kaggle_research_agent.graph.demo_cycle_graph import run_demo_graph_cycle
from kaggle_research_agent.trial_artifacts import trial_artifact_exists


class DemoOneCycleTest(unittest.TestCase):
    def test_demo_one_cycle_runs_mock_plan_code_execution_and_records_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("completed", result["status"])
            self.assertEqual("ready", result["plan"]["status"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertEqual(0.88, result["record"]["local_score"])
            self.assertEqual("ready", result["submission_manifest"]["status"])
            self.assertTrue((trial / "submit_manifest.json").exists())
            self.assertTrue((trial / "submit_manifest.md").exists())
            self.assertTrue((trial / "demo_context.md").exists())
            self.assertTrue((trial / "next_experiment.md").exists())
            self.assertTrue(trial_artifact_exists(trial, "workspace_coding_result.json"))
            self.assertTrue(trial_artifact_exists(trial, "workspace_run.json"))
            self.assertTrue(trial_artifact_exists(trial, "demo_cycle_record.json"))
            self.assertTrue((trial / "agent_status.json").exists())
            self.assertTrue((trial / "agent_events.jsonl").exists())
            status = json.loads((trial / "agent_status.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", status["status"])
            self.assertEqual("done", status["current_stage"])
            events = (trial / "agent_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "stage_started"' in line for line in events))
            self.assertTrue(any('"event": "cycle_completed"' in line for line in events))
            self.assertTrue((root / "memory" / "demo" / "demo_trial_index.jsonl").exists())
            usage_lines = (root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(usage_lines))
            usages = [json.loads(line) for line in usage_lines]
            self.assertEqual(["experiment_planning", "workspace_code_writing"], [row["call_type"] for row in usages])
            self.assertEqual({"openai_responses"}, {row["provider"] for row in usages})
            self.assertEqual({"gpt-5.5"}, {row["model"] for row in usages})
            self.assertEqual("gpt-5.6-luna", result["model_policy"]["low_cost"]["model"])
            self.assertFalse(result["context"]["artifact_policy"]["save_model"]["default"])
            self.assertIn("improved", (project / "train_step.py").read_text(encoding="utf-8"))
            plan_request = build_demo_plan_payload(result["context"], model="gpt-5.5")
            plan_prompt = plan_request["input"][1]["content"]
            self.assertIn("Model/checkpoint artifacts are optional", plan_prompt)
            self.assertIn("required_for_separate_predict_command", plan_prompt)
            self.assertIn("RAG Context Pack", plan_prompt)
            self.assertIn("retrieved documents", plan_prompt.lower())
            self.assertIn("Artifact Policy", (trial / "demo_context.md").read_text(encoding="utf-8"))
            self.assertTrue((trial / "context_pack_experiment_planning.json").exists())
            self.assertTrue((trial / "retrieval_manifest_experiment_planning.json").exists())

    def test_demo_graph_cycle_runs_same_one_cycle_with_graph_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("completed", result["status"])
            self.assertTrue(result["graph_execution"]["enabled"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual(0.88, result["record"]["local_score"])
            self.assertEqual("ready", result["submission_manifest"]["status"])
            self.assertTrue((trial / "demo_graph_cycle.json").exists())
            self.assertTrue((trial / "submit_manifest.json").exists())
            self.assertTrue(trial_artifact_exists(trial, "demo_one_cycle.json"))
            self.assertTrue((trial / "graph_state.json").exists())
            self.assertTrue((trial / "node_events.jsonl").exists())
            graph_state = json.loads((trial / "graph_state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", graph_state["status"])
            self.assertEqual("finalize", graph_state["last_completed_node"])
            events = [
                json.loads(line)
                for line in (trial / "node_events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual("load_context", events[0]["node"])
            self.assertTrue(any(item["node"] == "record_result" and item["event"] == "completed" for item in events))
            self.assertTrue(any(item["node"] == "prepare_submission" and item["event"] == "completed" for item in events))

    def test_demo_graph_auto_loop_runs_two_trials_and_records_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_auto_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=2,
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            self.assertEqual("completed", result["status"])
            self.assertEqual(["trial_001", "trial_002"], [row["trial_id"] for row in result["trials"]])
            self.assertEqual("trial_001", result["best_trial"]["trial_id"])
            self.assertTrue(result["trials"][0]["is_best"])
            self.assertFalse(result["trials"][1]["is_best"])
            memory = root / "memory" / "demo"
            self.assertTrue((memory / "demo_best_trial.json").exists())
            self.assertTrue((memory / "demo_graph_auto_loop.json").exists())
            self.assertTrue((memory / "document_index.jsonl").exists())
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "context_pack_experiment_planning.json").exists())
            index_text = (memory / "document_index.jsonl").read_text(encoding="utf-8")
            self.assertIn("experiments/demo/trial_001/user_view/02_pipeline_structure.ko.md", index_text)
            state = simple_yaml.load(root / "competitions" / "demo" / "state.yaml")
            self.assertEqual("trial_001", state["current_state"]["best_trial"]["trial_id"])

    def test_demo_graph_auto_loop_stops_after_no_improvement_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_auto_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=3,
                    stop_no_improvement=1,
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            self.assertEqual("stopped_no_improvement", result["status"])
            self.assertEqual(2, len(result["trials"]))
            self.assertEqual(1, result["no_improvement_count"])

    def test_demo_graph_auto_loop_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "demo-graph-auto-loop",
                        "--competition",
                        "demo",
                        "--start-trial",
                        "trial_001",
                        "--max-trials",
                        "2",
                        "--mock-plan-file",
                        str(plan_response),
                        "--mock-response-file",
                        str(code_response),
                        "--run-now",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "memory" / "demo" / "demo_graph_auto_loop.json").exists())

    def test_demo_one_cycle_resumes_from_completed_plan_and_code_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            plan = {
                "schema_version": "1.0",
                "competition": "demo",
                "trial_id": "trial_001",
                "status": "ready",
                "plan_title": "Existing first-cycle plan",
                "objective": "Resume from an already prepared plan.",
                "rationale": "The previous run completed planning.",
                "implementation_notes": ["Reuse existing code result."],
                "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                "issues": [],
                "next_action": "prepare-workspace-handoff",
            }
            (trial / "demo_experiment_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (trial / "next_experiment.md").write_text("# Existing first-cycle plan\n", encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Existing code change already accepted.",
                        "changed_files": ["train_step.py"],
                        "validation_results": {
                            "commands": [{"command": "{python} test_step.py", "status": "not_run"}]
                        },
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001", run_now=True)

            self.assertEqual("completed", result["status"])
            self.assertTrue(result["plan"]["resumed_from_existing_artifact"])
            self.assertTrue(result["code_writer"]["resumed_from_existing_artifact"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertEqual(0.5, result["record"]["local_score"])
            self.assertFalse((root / "memory" / "demo" / "token_usage.jsonl").exists())

    def test_demo_one_cycle_resumes_from_existing_accepted_code_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            plan = {
                "schema_version": "1.0",
                "competition": "demo",
                "trial_id": "trial_001",
                "status": "ready",
                "plan_title": "Existing first-cycle plan",
                "objective": "Resume from an already prepared plan.",
                "rationale": "The previous run completed planning.",
                "implementation_notes": ["Reuse existing accepted validation."],
                "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                "issues": [],
                "next_action": "prepare-workspace-handoff",
            }
            (trial / "demo_experiment_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (trial / "next_experiment.md").write_text("# Existing first-cycle plan\n", encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "summary": "A later retry was blocked by token policy.",
                        "changed_files": [],
                        "validation_results": [],
                        "blocking_issues": ["token_policy_blocked"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_result_validation.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "trial_id": "trial_001",
                        "status": "accepted",
                        "issues": [],
                        "coding_result_status": "completed",
                        "changed_files": ["train_step.py"],
                        "allowed_write_paths": ["src/", "tests/", "train_step.py"],
                        "forbidden_paths": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                        "next_action": "run-workspace-validation-commands",
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001", run_now=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual("workspace_coding_result_validation", result["code_writer"]["resume_source"])
            self.assertEqual(["train_step.py"], result["code_writer"]["changed_files"])
            self.assertEqual("completed", result["workspace_run"]["status"])

    def test_demo_one_cycle_blocks_without_plan_api_or_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001")

            self.assertEqual("blocked", result["status"])
            self.assertIn("api_call_not_enabled", result["issues"])

    def test_demo_one_cycle_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(
                        [
                            "demo-one-cycle",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--mock-plan-file",
                            str(plan_response),
                            "--mock-response-file",
                            str(code_response),
                            "--run-now",
                            "--show-progress",
                        ]
                    )

            self.assertEqual(0, code)
            trial = root / "experiments" / "demo" / "trial_001"
            self.assertTrue(trial_artifact_exists(trial, "demo_one_cycle.json"))
            self.assertIn("[진행] 1/5 대회와 데이터 정보 확인", output.getvalue())
            self.assertIn("1회 실험 사이클 완료", output.getvalue())
            self.assertIn("실험 요약", output.getvalue())

    def test_watch_demo_cycle_cli_prints_current_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["watch-demo-cycle", "--competition", "demo", "--trial", "trial_001"])

            text = output.getvalue()
            self.assertEqual(0, code)
            self.assertIn("데모 에이전트 상태: demo / trial_001", text)
            self.assertIn("- 전체 상태: 완료", text)
            self.assertIn("최근 진행 기록:", text)

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": "accuracy", "objective": "maximize", "platform": "external"},
                "current_state": {
                    "active_trial": None,
                    "best_trial": None,
                    "consecutive_failures": 0,
                    "submissions_today": 0,
                    "validation_suspected": False,
                },
                "strategy": {"current_focus": "baseline", "promising_directions": [], "forbidden_directions": []},
            },
            comp / "state.yaml",
        )
        (root / "memory" / "demo").mkdir(parents=True, exist_ok=True)
        (root / "memory" / "demo" / "trial_index.jsonl").write_text("", encoding="utf-8")

    def _write_profile(self, root: Path, project: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": sys.executable,
                "commands": {
                    "test": ["{python} test_step.py"],
                    "train": ["{python} train_step.py"],
                    "predict": ["{python} predict_step.py"],
                },
                "artifacts": {
                    "metrics": ["outputs/metrics.json"],
                    "submission": ["outputs/submission.csv"],
                },
                "write_scope": {
                    "allowed": ["src/", "tests/", "train_step.py"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            },
            comp / "execution_profile.yaml",
        )

    def _write_project_scripts(self, project: Path) -> None:
        (project / "outputs").mkdir()
        (project / "src").mkdir()
        (project / "tests").mkdir()
        (project / "test_step.py").write_text("from pathlib import Path\nPath('order.txt').write_text('test\\n')\n", encoding="utf-8")
        (project / "train_step.py").write_text(
            "from pathlib import Path\n"
            "Path('outputs').mkdir(exist_ok=True)\n"
            "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.5, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n",
            encoding="utf-8",
        )
        (project / "predict_step.py").write_text(
            "from pathlib import Path\n"
            "path = Path('order.txt')\n"
            "path.write_text(path.read_text() + 'predict\\n')\n"
            "Path('outputs/submission.csv').write_text('id,target\\n1,0\\n')\n",
            encoding="utf-8",
        )

    def _write_plan_response(self, root: Path) -> Path:
        path = root / "mock_plan_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_plan",
                    "usage": {"input_tokens": 300, "output_tokens": 90, "total_tokens": 390},
                    "output_text": json.dumps(
                        {
                            "plan_title": "First local baseline",
                            "objective": "Write a small train script and verify local metrics.",
                            "rationale": "A minimal runnable baseline proves the loop before deeper optimization.",
                            "implementation_notes": ["Update train_step.py to emit metrics."],
                            "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_code_response(self, root: Path) -> Path:
        path = root / "mock_code_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_code",
                    "usage": {"input_tokens": 500, "output_tokens": 120, "total_tokens": 620},
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Wrote improved first-cycle train script.",
                            "changed_files": ["train_step.py"],
                            "file_updates": [
                                {
                                    "path": "train_step.py",
                                    "content": (
                                        "from pathlib import Path\n"
                                        "# improved demo baseline\n"
                                        "path = Path('order.txt')\n"
                                        "path.write_text(path.read_text() + 'train\\n')\n"
                                        "Path('outputs').mkdir(exist_ok=True)\n"
                                        "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.88, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n"
                                    ),
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
