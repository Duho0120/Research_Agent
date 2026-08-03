import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_code_writer import (
    _normalize_coding_result,
    _validate_predict_test_sync,
    run_workspace_code_writer,
    validate_workspace_coding_result,
)


class FakeWorkspaceCodeWriterClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def create_response(self, payload: dict) -> dict:
        self.calls.append(payload)
        return self.response


class WorkspaceCodeWriterTest(unittest.TestCase):
    def test_normalize_coding_result_keeps_string_blocking_issue_as_one_item(self):
        result = _normalize_coding_result(
            {
                "status": "blocked",
                "blocking_issues": "authoritative base snapshot does not include train_step.py",
            },
            "demo",
            "trial_002",
            {"request_id": "demo:trial_002"},
        )

        self.assertEqual(
            ["authoritative base snapshot does not include train_step.py"],
            result["blocking_issues"],
        )

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
            self.assertIn("Use the RAG context pack", prompt)
            self.assertIn("continuation_delta_plan", prompt)
            self.assertIn("modify the existing pipeline incrementally", prompt)
            self.assertIn("primary_change_axis", prompt)
            self.assertIn("context_pack_workspace_code_writing.md", prompt)
            self.assertIn("Data card summary", prompt)
            self.assertIn("Survived", prompt)
            self.assertIn("Pclass", prompt)
            self.assertIn("You are not expected to access the filesystem or run tools directly.", prompt)
            self.assertIn("Return code changes in patch_updates when edit_policy.mode is patch_only", prompt)
            self.assertIn("Do not block merely because this API response environment has no file read/write tools.", prompt)
            self.assertIn("Output budget for continuation/patch-only coding", prompt)
            self.assertIn("Return at most 2 patch_updates", prompt)
            self.assertIn("validation_results must be []", prompt)
            self.assertIn("copy the complete existing line or block from the authoritative code snapshot", prompt)
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))
            self.assertEqual(
                "FEATURE_FLAG = True\n",
                (trial / "internal" / "code_snapshot" / "src" / "model.py").read_text(encoding="utf-8"),
            )
            snapshot_manifest = json.loads(
                (trial / "internal" / "code_snapshot_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["src/model.py"], [item["path"] for item in snapshot_manifest["files"]])
            self.assertTrue((trial / "workspace_coding_api_request.json").exists())
            self.assertTrue((trial / "workspace_coding_api_response.json").exists())
            self.assertTrue((trial / "workspace_coding_result_validation.json").exists())
            execution_plan = json.loads(
                (trial / "internal" / "execution_plan_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual("finalized", execution_plan["status"])
            self.assertEqual("trial_002", execution_plan["plan"]["trial_id"])
            self.assertIn("Improve model feature", execution_plan["plan_markdown"])
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

    def test_run_workspace_code_writer_accepts_single_validation_result_object(self):
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
                                "status": "not_run_in_response_environment",
                                "intended_command": "{python} -m pytest tests -q",
                            },
                            "blocking_issues": [],
                        }
                    ),
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            saved = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsInstance(saved["validation_results"], list)
            self.assertEqual("not_run_in_response_environment", saved["validation_results"][0]["status"])

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

    def test_validate_workspace_coding_result_blocks_fabricated_data_fallback_in_file_update(self):
        # Real incident: a code writer, unable to find the conventional
        # data/train.csv, silently invented a `_synthesize_train()` fallback
        # instead of reading the real (differently-shaped) data the
        # workspace actually had, producing a fake-but-plausible metric.
        # This must be caught regardless of which competition/file layout
        # triggered it -- the check is on the generated code text itself.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Added training script.",
                        "changed_files": ["train.py"],
                        "file_updates": [
                            {
                                "path": "train.py",
                                "content": (
                                    "def _synthesize_train(n=64):\n"
                                    "    return [{'id': f'syn_{i}'} for i in range(n)]\n"
                                ),
                            }
                        ],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("possible_fabricated_data_fallback:synthesize_train", result["issues"])

    def test_validate_workspace_coding_result_blocks_fabricated_data_fallback_in_patch_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Patched training script.",
                        "changed_files": ["train.py"],
                        "patch_updates": [
                            {
                                "path": "train.py",
                                "find": "rows = read_csv(train_path)",
                                "replace": "rows = generate_dummy_rows(64)",
                            }
                        ],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("possible_fabricated_data_fallback:generate_dummy", result["issues"])

    def test_validate_workspace_coding_result_blocks_generate_synthetic_data_fallback(self):
        # Real incident (DACON competition 236716, per-sample-file dataset
        # layout): unable to handle the more complex real data shape, the
        # code writer generated a `_generate_synthetic()` fallback that
        # trained/scored entirely on np.random noise, and separately wrote
        # its output under a different file name than the declared artifact
        # ("avoid reserved filename"), leaving the real declared metrics.json
        # untouched. The original marker list only caught a handful of exact
        # names (synthesize_train, fake_data, ...) and missed this shape
        # entirely.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Added training script.",
                        "changed_files": ["train_step.py"],
                        "file_updates": [
                            {
                                "path": "train_step.py",
                                "content": (
                                    "def _generate_synthetic(n_samples=800, seed=42):\n"
                                    "    # Fallback to synthetic wide+targets for this demo baseline\n"
                                    "    rng = np.random.default_rng(seed)\n"
                                    "    return rng.normal(size=(n_samples, 3))\n"
                                    "\n"
                                    "# Save local metrics (avoid reserved filename)\n"
                                    "(outputs / 'metrics_local.json').write_text(...)\n"
                                ),
                            }
                        ],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("possible_fabricated_data_fallback:generate_synthetic", result["issues"])
            self.assertIn("possible_fabricated_data_fallback:reserved filename", result["issues"])

    def test_validate_predict_test_sync_blocks_predict_change_left_alone(self):
        # Real incident: predict_step.py's prediction algorithm changed but
        # test_step.py (which independently reimplements the same formula
        # for local validation) did not, so the local CV score kept
        # reporting the OLD algorithm's result while the real submission
        # used the NEW one. A prompt instruction alone missed this on a
        # second isolated run of the same scenario, so this mechanical
        # check is the actual backstop.
        handoff = {
            "predict_commands": ["{python} predict_step.py"],
            "validation_commands": ["{python} test_step.py"],
        }
        coding_result = {"changed_files": ["predict_step.py"]}

        issues = _validate_predict_test_sync(coding_result, handoff)

        self.assertEqual(["predict_script_changed_without_test_script_sync:predict_step.py"], issues)

    def test_validate_predict_test_sync_allows_both_scripts_changed_together(self):
        handoff = {
            "predict_commands": ["{python} predict_step.py"],
            "validation_commands": ["{python} test_step.py"],
        }
        coding_result = {"changed_files": ["predict_step.py", "test_step.py"]}

        self.assertEqual([], _validate_predict_test_sync(coding_result, handoff))

    def test_validate_predict_test_sync_allows_shared_module_refactor(self):
        # A refactor into a shared module (e.g. src/baseline.py) that both
        # predict_step.py and test_step.py already call is the intended way
        # to keep them in sync going forward -- it must not be blocked just
        # because predict_step.py itself was not touched this trial.
        handoff = {
            "predict_commands": ["{python} predict_step.py"],
            "validation_commands": ["{python} test_step.py"],
        }
        coding_result = {"changed_files": ["src/baseline.py"]}

        self.assertEqual([], _validate_predict_test_sync(coding_result, handoff))

    def test_validate_predict_test_sync_ignores_unrelated_changes(self):
        handoff = {
            "predict_commands": ["{python} predict_step.py"],
            "validation_commands": ["{python} test_step.py"],
        }
        coding_result = {"changed_files": ["README.md"]}

        self.assertEqual([], _validate_predict_test_sync(coding_result, handoff))

    def test_validate_predict_test_sync_is_a_noop_without_declared_commands(self):
        coding_result = {"changed_files": ["predict_step.py"]}

        self.assertEqual([], _validate_predict_test_sync(coding_result, {}))

    def test_validate_predict_test_sync_checks_patch_updates_too(self):
        handoff = {
            "predict_commands": ["{python} predict_step.py"],
            "validation_commands": ["{python} test_step.py"],
        }
        coding_result = {
            "patch_updates": [{"path": "predict_step.py", "find": "a", "replace": "b"}],
        }

        issues = _validate_predict_test_sync(coding_result, handoff)

        self.assertEqual(["predict_script_changed_without_test_script_sync:predict_step.py"], issues)

    def test_validate_workspace_coding_result_blocks_predict_change_without_test_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(
                root,
                project,
                allowed_write_paths=["src/", "tests/", "predict_step.py", "test_step.py"],
                predict_commands=["{python} predict_step.py"],
                validation_commands=["{python} test_step.py"],
            )
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Improved prediction algorithm.",
                        "changed_files": ["predict_step.py"],
                        "file_updates": [{"path": "predict_step.py", "content": "def predict(): ...\n"}],
                        "validation_results": [],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = validate_workspace_coding_result("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("predict_script_changed_without_test_script_sync:predict_step.py", result["issues"])

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

    def test_patch_is_rejected_when_it_introduces_invalid_python_syntax(self):
        # Regression: the code writer LLM wrote a JSON literal ("null")
        # into Python code instead of None. That patch passed every
        # scope/schema check that existed, got written to disk, ran
        # partway through train_step.py, and only failed with a bare
        # NameError mid-execution -- burning a full execution attempt on a
        # mistake a syntax check would have caught for free.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Add a weights field.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = False\nWEIGHTS = null\n",
                                    "reason": "Record default weights.",
                                }
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            coding_result = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                any("invalid_python_syntax" in item for item in coding_result["blocking_issues"])
            )
            self.assertEqual(
                "FEATURE_FLAG = False\n",
                (project / "src" / "model.py").read_text(encoding="utf-8"),
            )

    def test_patch_is_rejected_when_it_introduces_a_real_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Broken parenthesis.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = (False\n",
                                    "reason": "Oops.",
                                }
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            coding_result = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                any("invalid_python_syntax" in item for item in coding_result["blocking_issues"])
            )
            self.assertEqual(
                "FEATURE_FLAG = False\n",
                (project / "src" / "model.py").read_text(encoding="utf-8"),
            )

    def test_full_file_update_is_rejected_when_it_contains_a_json_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project)
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Rewrite the model file.",
                            "changed_files": ["src/model.py"],
                            "file_updates": [
                                {"path": "src/model.py", "content": "FEATURE_FLAG = False\nWEIGHTS = null\n"}
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
            coding_result = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                any("invalid_python_syntax" in item for item in coding_result["blocking_issues"])
            )
            self.assertEqual(
                "FEATURE_FLAG = False\n",
                (project / "src" / "model.py").read_text(encoding="utf-8"),
            )

    def test_run_workspace_code_writer_applies_patch_updates_in_patch_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Patched feature flag only.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = True\n",
                                    "reason": "Enable the planned flag without rewriting the file.",
                                }
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))
            self.assertEqual(
                "FEATURE_FLAG = True\n",
                (
                    root
                    / "experiments"
                    / "demo"
                    / "trial_002"
                    / "internal"
                    / "code_snapshot"
                    / "src"
                    / "model.py"
                ).read_text(encoding="utf-8"),
            )

    def test_patch_batch_is_all_or_nothing_when_one_patch_fails_to_find_its_target(self):
        # Regression for trial_023: two patches targeted the same file. The
        # first found its exact text and used to be written to disk
        # immediately; the second had a whitespace/indentation mismatch and
        # failed. The overall attempt was correctly marked blocked, but the
        # first patch's write was never rolled back -- leaving the workspace
        # file in a state that matched no trial's clean snapshot, so the next
        # retry's patches (computed against a clean base) kept failing too.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Two patches to the same file; second one has a bad find string.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = True\n",
                                    "reason": "First patch matches exactly.",
                                },
                                {
                                    "path": "src/model.py",
                                    "find": "    FEATURE_FLAG = True\n",
                                    "replace": "    FEATURE_FLAG = True  # enabled\n",
                                    "reason": "Second patch has an indentation mismatch and will not be found.",
                                },
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            coding_result = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("patch_find_not_found:src/model.py", coding_result["blocking_issues"])
            self.assertEqual(
                "FEATURE_FLAG = False\n",
                (project / "src" / "model.py").read_text(encoding="utf-8"),
            )

    def test_run_workspace_code_writer_restores_base_snapshot_before_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            (project / "src" / "model.py").write_text("FEATURE_FLAG = FAILED_TRIAL\n", encoding="utf-8")
            source_code = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src"
            source_code.mkdir(parents=True)
            (source_code / "model.py").write_text("FEATURE_FLAG = False\n", encoding="utf-8")
            self._write_handoff(
                root,
                project,
                edit_policy={
                    "mode": "patch_only",
                    "allow_full_file_updates": False,
                    "restore_base_before_patch": True,
                    "base_code_source": "experiments/demo/trial_001/user_view/code",
                },
                source_trial_id="trial_001",
            )
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Patched from source trial snapshot.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = True\n",
                                    "reason": "Apply the next delta on the recommended base trial code.",
                                }
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_run_workspace_code_writer_prefers_internal_base_snapshot_before_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            (project / "src" / "model.py").write_text("FEATURE_FLAG = FAILED_TRIAL\n", encoding="utf-8")
            source_internal = root / "experiments" / "demo" / "trial_001" / "internal"
            source_snapshot = source_internal / "code_snapshot" / "src"
            source_snapshot.mkdir(parents=True)
            (source_snapshot / "model.py").write_text("FEATURE_FLAG = False\n", encoding="utf-8")
            (source_internal / "code_snapshot_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "snapshot_dir": "internal/code_snapshot",
                        "files": [{"path": "src/model.py"}],
                    }
                ),
                encoding="utf-8",
            )
            stale_user_view = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src"
            stale_user_view.mkdir(parents=True)
            (stale_user_view / "model.py").write_text("FEATURE_FLAG = POLLUTED\n", encoding="utf-8")
            self._write_handoff(
                root,
                project,
                edit_policy={
                    "mode": "patch_only",
                    "allow_full_file_updates": False,
                    "restore_base_before_patch": True,
                    "base_code_source": "experiments/demo/trial_001/internal/code_snapshot",
                },
                source_trial_id="trial_001",
            )
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Patched from internal source snapshot.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "FEATURE_FLAG = True\n",
                                    "reason": "Apply the next delta on the durable base snapshot.",
                                }
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            self.assertEqual("FEATURE_FLAG = True\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_run_workspace_code_writer_blocks_full_file_updates_in_patch_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Tried whole-file update.",
                            "changed_files": ["src/model.py"],
                            "file_updates": [{"path": "src/model.py", "content": "FEATURE_FLAG = True\n"}],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertIn("full_file_update_not_allowed_in_patch_only:src/model.py", result["issues"])
            self.assertEqual("FEATURE_FLAG = False\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_patch_mode_allows_creating_a_brand_new_file(self):
        # Regression for trial_030: the plan needed a new helper module, but
        # patch-only mode rejected every file_update outright. The planner
        # kept proposing the module and the code writer kept refusing, so the
        # trial burned its whole LLM budget on retries without writing a line.
        # A file that does not exist yet overwrites nothing, so creating it is
        # safe even in patch-only mode.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Add a shared helper module and use it.",
                            "changed_files": ["src/helpers.py", "src/model.py"],
                            "patch_updates": [
                                {
                                    "path": "src/model.py",
                                    "find": "FEATURE_FLAG = False\n",
                                    "replace": "from src.helpers import build\n\nFEATURE_FLAG = True\n",
                                    "reason": "Use the new helper.",
                                }
                            ],
                            "file_updates": [
                                {"path": "src/helpers.py", "content": "def build():\n    return 1\n"}
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("accepted", result["status"])
            self.assertEqual("def build():\n    return 1\n", (project / "src" / "helpers.py").read_text(encoding="utf-8"))
            self.assertIn("from src.helpers import build", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_patch_mode_still_blocks_rewriting_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Rewrite an existing file wholesale.",
                            "changed_files": ["src/model.py"],
                            "file_updates": [{"path": "src/model.py", "content": "FEATURE_FLAG = True\n"}],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertIn("full_file_update_not_allowed_in_patch_only:src/model.py", result["issues"])
            self.assertEqual("FEATURE_FLAG = False\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

    def test_new_file_outside_allowed_paths_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Create a file outside the write scope.",
                            "changed_files": ["data/leak.py"],
                            "file_updates": [{"path": "data/leak.py", "content": "X = 1\n"}],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertFalse((project / "data" / "leak.py").exists())

    def test_run_workspace_code_writer_blocks_too_many_patch_updates_in_patch_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            (project / "src" / "model.py").write_text("A = 1\nB = 2\nC = 3\n", encoding="utf-8")
            self._write_handoff(root, project, edit_policy={"mode": "patch_only", "allow_full_file_updates": False})
            client = FakeWorkspaceCodeWriterClient(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Tried too many small patches.",
                            "changed_files": ["src/model.py"],
                            "patch_updates": [
                                {"path": "src/model.py", "find": "A = 1\n", "replace": "A = 10\n", "reason": "A"},
                                {"path": "src/model.py", "find": "B = 2\n", "replace": "B = 20\n", "reason": "B"},
                                {"path": "src/model.py", "find": "C = 3\n", "replace": "C = 30\n", "reason": "C"},
                            ],
                            "file_updates": [],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_workspace_code_writer("demo", "trial_002", client=client, allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertIn("too_many_patch_updates_for_budget:3", result["issues"])
            self.assertEqual("A = 1\nB = 2\nC = 3\n", (project / "src" / "model.py").read_text(encoding="utf-8"))

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

    def _write_handoff(
        self,
        root: Path,
        project: Path,
        edit_policy: dict | None = None,
        source_trial_id: str | None = None,
        allowed_write_paths: list[str] | None = None,
        predict_commands: list[str] | None = None,
        validation_commands: list[str] | None = None,
    ) -> None:
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
                    "source_trial_id": source_trial_id,
                    "code_base_trial_id": source_trial_id,
                    "context_files": ["experiments/demo/trial_002/next_experiment.md"],
                    "retrieval_context": {
                        "task": "workspace_code_writing",
                        "document_count": 1,
                        "context_pack_md_file": "experiments/demo/trial_002/context_pack_workspace_code_writing.md",
                        "retrieval_manifest_file": "experiments/demo/trial_002/retrieval_manifest_workspace_code_writing.json",
                    },
                    "data_card_summary": {
                        "task_type": "tabular_classification",
                        "target_column": "Survived",
                        "id_column": "PassengerId",
                        "include_features_first": ["Pclass", "Sex", "Age"],
                        "defer_features_first": ["Name", "Ticket"],
                        "avoid_first_trial": ["gaussian_naive_bayes_for_mixed_numeric_and_categorical_data"],
                    },
                    "edit_policy": edit_policy or {"mode": "full_file_allowed", "allow_full_file_updates": True},
                    "allowed_write_paths": allowed_write_paths or ["src/", "tests/", "train.py"],
                    "forbidden_paths": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                    "validation_commands": validation_commands or ["{python} -m pytest tests -q"],
                    "predict_commands": predict_commands or [],
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
