import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_code_writer import run_workspace_code_writer, validate_workspace_coding_result


class FakeWorkspaceCodeWriterClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def create_response(self, payload: dict) -> dict:
        self.calls.append(payload)
        return self.response


class WorkspaceCodeWriterTest(unittest.TestCase):
    def test_run_workspace_code_writer_applies_allowed_external_update_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            client = FakeWorkspaceCodeWriterClient(
                {
                    "id": "resp_workspace",
                    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated model feature flag.",
                            "changed_files": ["src/model.py"],
                            "file_updates": [
                                {
                                    "path": "src/model.py",
                                    "content": "FEATURE_FLAG = True\n",
                                }
                            ],
                            "validation_results": [{"command": "{python} -m pytest tests -q", "status": "not_run"}],
                            "blocking_issues": [],
                        }
                    ),
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("accepted", result["status"])
            self.assertEqual(1, len(client.calls))
            prompt = client.calls[0]["input"][1]["content"]
            self.assertIn("Artifact policy", prompt)
            self.assertIn("Do not persist trained model/checkpoint artifacts by default", prompt)
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))
            self.assertTrue((trial / "workspace_coding_api_request.json").exists())
            self.assertTrue((trial / "workspace_coding_api_response.json").exists())
            self.assertTrue((trial / "workspace_coding_result_validation.json").exists())
            usage = json.loads((root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual("workspace_code_writing", usage["call_type"])

    def test_run_workspace_code_writer_accepts_validation_results_commands_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated model feature flag.",
                            "changed_files": ["src/model.py"],
                            "file_updates": [{"path": "src/model.py", "content": "FEATURE_FLAG = True\n"}],
                            "validation_results": {
                                "commands": [{"command": "{python} -m pytest tests -q", "status": "not_run"}]
                            },
                            "blocking_issues": [],
                        }
                    ),
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            self.assertEqual([], result["blocking_issues"])
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_run_workspace_code_writer_blocks_forbidden_artifact_update_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            original_metrics = (project / "outputs" / "metrics.json").read_text(encoding="utf-8")
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Tried to rewrite metrics.",
                            "changed_files": ["outputs/metrics.json"],
                            "file_updates": [
                                {
                                    "path": "outputs/metrics.json",
                                    "content": "{\"cv_score\": 1.0}",
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertIn("file_update_not_allowed:outputs/metrics.json", result["blocking_issues"])
            self.assertEqual(original_metrics, (project / "outputs" / "metrics.json").read_text(encoding="utf-8"))

    def test_run_workspace_code_writer_blocks_without_client_or_api_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("api_call_not_enabled", result["blocking_issues"])

    def test_validate_workspace_coding_result_blocks_out_of_scope_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Changed data.",
                        "changed_files": ["data/train.csv"],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("changed_file_not_allowed:data/train.csv", result["issues"])
            self.assertIn("forbidden_path_touched:data/train.csv", result["issues"])

    def test_validate_workspace_coding_result_accepts_validation_results_commands_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Updated model.",
                        "changed_files": ["src/model.py"],
                        "validation_results": {
                            "commands": [{"command": "{python} -m pytest tests -q", "status": "not_run"}]
                        },
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("accepted", result["status"])
            self.assertEqual([], result["issues"])

    def test_run_workspace_code_writer_cli_uses_mock_response_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            response_path = root / "mock_response.json"
            response_path.write_text(
                json.dumps(
                    {
                        "output_text": json.dumps(
                            {
                                "status": "completed",
                                "summary": "Updated train script.",
                                "changed_files": ["train.py"],
                                "file_updates": [{"path": "train.py", "content": "print('updated')\n"}],
                                "validation_results": [],
                                "blocking_issues": [],
                            }
                        )
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "run-workspace-code-writer",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--mock-response-file",
                            str(response_path),
                        ]
                    )

            self.assertEqual(0, code)
            self.assertEqual("print('updated')\n", (project / "train.py").read_text(encoding="utf-8"))

    def _write_project(self, root: Path) -> Path:
        project = root / "external_project"
        (project / "src").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "outputs").mkdir()
        (project / "data").mkdir()
        (project / "src" / "model.py").write_text("FEATURE_FLAG = False\n", encoding="utf-8")
        (project / "train.py").write_text("print('baseline')\n", encoding="utf-8")
        (project / "tests" / "test_model.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        (project / "outputs" / "metrics.json").write_text("{\"cv_score\": 0.5}", encoding="utf-8")
        (project / "outputs" / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
        return project

    def _write_handoff(self, root: Path, project: Path) -> None:
        trial = root / "experiments" / "demo" / "trial_002"
        trial.mkdir(parents=True)
        (trial / "next_experiment.md").write_text("# trial_002\n\nImprove model feature.\n", encoding="utf-8")
        (trial / "workspace_coding_handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "request_id": "demo:trial_002:workspace-coding",
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "handoff_type": "workspace_coding_agent_request",
                    "status": "ready",
                    "objective": "Implement the next workspace experiment.",
                    "project_root": str(project),
                    "context_files": ["experiments/demo/trial_002/next_experiment.md"],
                    "allowed_write_paths": ["src/", "tests/", "train.py"],
                    "forbidden_paths": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                    "validation_commands": ["{python} -m pytest tests -q"],
                    "artifact_policy": {
                        "save_model": {
                            "default": False,
                            "allowed_when": ["required_for_separate_predict_command"],
                            "require_reason": True,
                        }
                    },
                    "required_output": {
                        "required_fields": [
                            "status",
                            "summary",
                            "changed_files",
                            "validation_results",
                            "blocking_issues",
                        ],
                        "status_values": ["completed", "blocked", "failed"],
                    },
                    "next_action": "send-to-workspace-coding-agent",
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
