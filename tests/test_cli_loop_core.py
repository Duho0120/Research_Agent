import json
import io
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from kaggle_research_agent.cli import main


class CliLoopCoreTest(unittest.TestCase):
    def test_diagnose_command_creates_diagnosis_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.7, "objective": "maximize"}),
                encoding="utf-8",
            )
            (root / "competitions" / "demo").mkdir(parents=True)
            (root / "competitions" / "demo" / "state.yaml").write_text(
                "competition:\n  objective: maximize\ncurrent_state:\n  consecutive_failures: 0\n",
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["diagnose", "--competition", "demo", "--trial", "trial_001"])
            self.assertEqual(code, 0)
            self.assertTrue((trial / "diagnosis.md").exists())

    def test_request_review_command_creates_review_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.82,
                        "lb_score": 0.78,
                        "objective": "maximize",
                        "segment_errors": {"fold_3": 0.31},
                    }
                ),
                encoding="utf-8",
            )
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["request-review", "--competition", "demo", "--trial", "trial_001"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "user_review_request.md").exists())
            self.assertTrue((trial / "review_pack" / "manifest.json").exists())

    def test_decide_llm_command_records_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "decide-llm",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--reason",
                            "human_review_needed",
                        ]
                    )

            self.assertEqual(code, 0)
            log_path = root / "memory" / "demo" / "decision_log.jsonl"
            self.assertTrue(log_path.exists())
            row = json.loads(log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["decision_type"], "llm_call")

    def test_inspect_competition_command_creates_inspection_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "config", "view"]:
                    return {"returncode": 0, "stdout": "username: hidden\n", "stderr": ""}
                if args == ["kaggle", "competitions", "files", "-c", "titanic"]:
                    return {"returncode": 0, "stdout": "name,size\ntrain.csv,10KB\n", "stderr": ""}
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.integrations.kaggle_cli.run_args", side_effect=runner):
                    with redirect_stdout(io.StringIO()):
                        code = main(
                            [
                                "inspect-competition",
                                "--competition",
                                "https://www.kaggle.com/competitions/titanic",
                            ]
                        )

            self.assertEqual(code, 0)
            self.assertTrue((root / "competitions" / "titanic" / "competition_inspection.md").exists())

    def test_start_competition_command_initializes_and_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(args, cwd):
                if args == ["kaggle", "--version"]:
                    return {"returncode": 0, "stdout": "Kaggle API 1.6.17\n", "stderr": ""}
                if args == ["kaggle", "config", "view"]:
                    return {"returncode": 0, "stdout": "username: hidden\n", "stderr": ""}
                if args == ["kaggle", "competitions", "files", "-c", "titanic"]:
                    return {"returncode": 0, "stdout": "name,size\ntrain.csv,10KB\nsample_submission.csv,2KB\n", "stderr": ""}
                return {"returncode": 1, "stdout": "", "stderr": "unexpected command"}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.integrations.kaggle_cli.run_args", side_effect=runner):
                    with redirect_stdout(io.StringIO()):
                        code = main(
                            [
                                "start-competition",
                                "--competition",
                                "https://www.kaggle.com/competitions/titanic",
                                "--metric",
                                "accuracy",
                                "--objective",
                                "maximize",
                            ]
                        )

            self.assertEqual(code, 0)
            self.assertTrue((root / "competitions" / "titanic" / "overview.md").exists())
            self.assertTrue((root / "experiments" / "titanic" / "trial_001" / "plan.md").exists())

    def test_profile_data_command_writes_data_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "titanic"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("id,target,x\n1,0,10\n", encoding="utf-8")
            (data / "test.csv").write_text("id,x\n2,11\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("id,target\n2,0\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["profile-data", "--competition", "titanic"])

            self.assertEqual(code, 0)
            self.assertTrue((root / "competitions" / "titanic" / "data_profile.md").exists())
            profile = json.loads((root / "competitions" / "titanic" / "data_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["target_candidates"], ["target"])

    def test_generate_baseline_command_writes_baseline_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "titanic"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("id,target,x\n1,0,10\n2,1,11\n3,1,12\n", encoding="utf-8")
            (data / "test.csv").write_text("id,x\n4,13\n", encoding="utf-8")
            (data / "sample_submission.csv").write_text("id,target\n4,0\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["generate-baseline", "--competition", "titanic", "--trial", "trial_001"])

            self.assertEqual(code, 0)
            trial = root / "experiments" / "titanic" / "trial_001"
            self.assertTrue((trial / "baseline_pipeline.json").exists())
            self.assertTrue((trial / "baseline_train.py").exists())

    def test_record_submission_command_creates_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments" / "demo" / "trial_001").mkdir(parents=True)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "record-submission",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--version-name",
                            "demo_trial_001_baseline_v01",
                            "--submission-file",
                            "experiments/demo/trial_001/submission.csv",
                            "--cv-score",
                            "0.7",
                            "--previous-lb-score",
                            "0.6",
                            "--previous-rank",
                            "200",
                            "--submitted-lb-score",
                            "0.72",
                            "--submitted-rank",
                            "150",
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())

    def test_record_feedback_command_updates_review_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "experiments" / "demo" / "trial_001" / "review_pack"
            pack.mkdir(parents=True)
            (pack / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "status": "pending_user_feedback"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "record-feedback",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--topic",
                            "validation",
                            "--question",
                            "Is the split appropriate?",
                            "--feedback",
                            "Use group split before large model changes.",
                            "--decision",
                            "change_validation",
                            "--follow-up-action",
                            "plan validation review trial",
                        ]
                    )

            self.assertEqual(code, 0)
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "feedback_recorded")
            self.assertTrue((pack / "human_feedback.json").exists())

    def test_submit_trial_command_records_submission_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0.42\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.83, "objective": "maximize"}),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "submit-trial",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--version-name",
                            "demo_trial_001_v01",
                            "--submission-file",
                            "experiments/demo/trial_001/submission.csv",
                            "--before-score",
                            "0.80",
                            "--before-rank",
                            "120",
                            "--after-score",
                            "0.84",
                            "--after-rank",
                            "90",
                            "--objective",
                            "maximize",
                            "--notes",
                            "CLI submission record",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "submission_run.md").exists())
            self.assertTrue((trial / "BEST_MARKER.md").exists())
            self.assertTrue((root / "submissions" / "demo" / "submission_log.jsonl").exists())

    def test_prepare_submission_command_creates_manifest_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "submission.csv").write_text("id,target\n1,0.42\n", encoding="utf-8")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.83, "objective": "maximize"}),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "prepare-submission",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--version-name",
                            "demo_trial_001_v01",
                            "--submission-file",
                            "experiments/demo/trial_001/submission.csv",
                            "--objective",
                            "maximize",
                            "--notes",
                            "Approval gate",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "submit_manifest.md").exists())
            self.assertFalse((trial / "BEST_MARKER.md").exists())
            self.assertFalse((root / "submissions" / "demo" / "submission_log.jsonl").exists())

    def test_plan_next_command_creates_next_experiment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "metrics.json").write_text(json.dumps({"cv_score": 0.7, "objective": "maximize"}), encoding="utf-8")
            memory = root / "memory" / "demo"
            memory.mkdir(parents=True)
            (memory / "decision_log.jsonl").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "decision_type": "diagnosis",
                        "evidence": {"strategy_recommendation": "continue_refinement", "issues": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "plan-next",
                            "--competition",
                            "demo",
                            "--source-trial",
                            "trial_001",
                            "--next-trial",
                            "trial_002",
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "next_experiment.md").exists())

    def test_plan_improvement_command_creates_pipeline_improvement_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.8\n"
                "    lb_score: 0.79\n",
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            config_dir = root / "configs" / "demo"
            config_dir.mkdir(parents=True)
            (config_dir / "research_policy.yaml").write_text(
                "leaderboard_tracking:\n  enabled: true\n",
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.82, "lb_score": 0.78, "objective": "maximize"}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["plan-improvement", "--competition", "demo", "--trial", "trial_001"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "pipeline_improvement_plan.md").exists())
            plan = json.loads((trial / "pipeline_improvement_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["primary_axis"], "validation")

    def test_advise_models_command_creates_model_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "data_profile.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "status": "ready",
                        "task_type": "tabular",
                        "target_candidates": ["target"],
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.71, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["advise-models", "--competition", "demo", "--trial", "trial_001"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "model_candidates.md").exists())
            result = json.loads((trial / "model_candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(result["candidates"][0]["model_family"], "gradient_boosted_trees")

    def test_prepare_patch_command_creates_code_patch_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments" / "demo" / "trial_001"
            source.mkdir(parents=True)
            (source / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "lightgbm",
                            "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8},
                        },
                        "features": {"use_frequency_encoding": False, "use_missing_indicators": True},
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            next_trial = root / "experiments" / "demo" / "trial_002"
            next_trial.mkdir(parents=True)
            (next_trial / "next_experiment.md").write_text(
                "# trial_002 Next Experiment\n\n## Strategy\n\ncontrolled_refinement\n",
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "prepare-patch",
                            "--competition",
                            "demo",
                            "--source-trial",
                            "trial_001",
                            "--next-trial",
                            "trial_002",
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertTrue((next_trial / "code_patch_plan.md").exists())

    def test_validate_patch_command_writes_validation_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "protected_axes": ["validation", "model_family"],
                        "requires_user_approval": False,
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "config_changes": {"training.sampler": "balanced"},
                        "validation_commands": ["python -B -m unittest discover -s tests -v"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["validate-patch", "--competition", "demo", "--trial", "trial_002"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "patch_validation.md").exists())

    def test_apply_patch_command_writes_code_edit_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "next_trial_id": "trial_002",
                        "strategy": "controlled_refinement",
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["apply-patch", "--competition", "demo", "--trial", "trial_002"])
            self.assertEqual(code, 0)
            self.assertTrue((trial / "code_edit_result.md").exists())

    def test_prepare_handoff_command_writes_coding_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {"n_splits": 5}}),
                encoding="utf-8",
            )
            (trial / "code_patch_plan.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "next_trial_id": "trial_002",
                        "strategy": "controlled_refinement",
                        "pipeline_axis": "sampling",
                        "target_files": ["experiments/demo/trial_002/config.yaml"],
                        "implementation_steps": ["Add balanced sampler support."],
                        "validation_commands": [
                            "python -B -m kaggle_research_agent.cli validate-config --competition demo --trial trial_002"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["prepare-handoff", "--competition", "demo", "--trial", "trial_002"])
            self.assertEqual(code, 0)
            self.assertTrue((trial / "coding_handoff.json").exists())
            self.assertTrue((trial / "coding_agent_request.md").exists())

    def test_write_code_dry_run_command_creates_blocked_coding_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_coding_handoff(trial)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["write-code-dry-run", "--competition", "demo", "--trial", "trial_002"])

            self.assertEqual(code, 1)
            result = json.loads((trial / "coding_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "blocked")

    def test_validate_coding_result_command_writes_validation_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_coding_handoff(trial)
            (trial / "coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Updated config.",
                        "changed_files": ["experiments/demo/trial_002/config.yaml"],
                        "validation_results": [{"command": "python -B -m unittest discover -s tests -v", "status": "passed"}],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["validate-coding-result", "--competition", "demo", "--trial", "trial_002"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "coding_result_validation.json").exists())

    def test_run_code_writer_command_uses_mock_response_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_coding_handoff(trial)
            response_path = root / "mock_response.json"
            response_path.write_text(
                json.dumps(
                    {
                        "output_text": json.dumps(
                            {
                                "status": "completed",
                                "summary": "Updated config from mock response.",
                                "changed_files": ["experiments/demo/trial_002/config.yaml"],
                                "file_updates": [
                                    {
                                        "path": "experiments/demo/trial_002/config.yaml",
                                        "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                    }
                                ],
                                "validation_results": [
                                    {"command": "python -B -m unittest discover -s tests -v", "status": "not_run"}
                                ],
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
                            "run-code-writer",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--mock-response-file",
                            str(response_path),
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertIn("balanced", (trial / "config.yaml").read_text(encoding="utf-8"))

    def test_run_code_writer_command_can_run_validation_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_coding_handoff(
                trial,
                validation_commands=[
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/cli_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            response_path = root / "mock_response.json"
            response_path.write_text(
                json.dumps(
                    {
                        "output_text": json.dumps(
                            {
                                "status": "completed",
                                "summary": "Updated config from mock response.",
                                "changed_files": ["experiments/demo/trial_002/config.yaml"],
                                "file_updates": [
                                    {
                                        "path": "experiments/demo/trial_002/config.yaml",
                                        "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                    }
                                ],
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
                            "run-code-writer",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--mock-response-file",
                            str(response_path),
                            "--run-validation-commands",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "validation_run.json").exists())
            self.assertTrue((trial / "cli_validated.txt").exists())

    def test_run_validation_commands_cli_writes_validation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            self._write_coding_handoff(
                trial,
                validation_commands=[
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/standalone_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            (trial / "coding_result_validation.json").write_text(
                json.dumps({"competition": "demo", "trial_id": "trial_002", "status": "accepted", "issues": []}),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["run-validation-commands", "--competition", "demo", "--trial", "trial_002"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "validation_run.json").exists())
            self.assertTrue((trial / "standalone_validated.txt").exists())

    def test_run_after_validation_command_creates_local_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            (trial / "validation_run.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "trial_id": "trial_002",
                        "status": "passed",
                        "issues": [],
                        "commands": [],
                        "next_action": "run-trial",
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "run-after-validation",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--run-command",
                            "python train.py",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())
            self.assertTrue((trial / "post_validation_execution.json").exists())

    def test_run_safe_execution_chain_command_creates_local_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text("model:\n  type: lightgbm\n", encoding="utf-8")
            self._write_coding_handoff(
                trial,
                validation_commands=[
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/chain_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            response_path = root / "mock_response.json"
            response_path.write_text(
                json.dumps(
                    {
                        "output_text": json.dumps(
                            {
                                "status": "completed",
                                "summary": "Updated config from mock response.",
                                "changed_files": ["experiments/demo/trial_002/config.yaml"],
                                "file_updates": [
                                    {
                                        "path": "experiments/demo/trial_002/config.yaml",
                                        "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                    }
                                ],
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
                            "run-safe-execution-chain",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--mock-response-file",
                            str(response_path),
                            "--run-command",
                            "python train.py",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "safe_execution_chain.json").exists())
            self.assertTrue((trial / "chain_validated.txt").exists())
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def test_cycle_command_can_run_safe_execution_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_minimal_competition_state(root)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {}}), encoding="utf-8")
            self._write_coding_handoff(
                trial,
                validation_commands=[
                    (
                        "python -c \"from pathlib import Path; "
                        "Path(r'experiments/demo/trial_002/cli_cycle_validated.txt').write_text('ok', encoding='utf-8')\""
                    )
                ],
            )
            response_path = root / "mock_response.json"
            self._write_mock_code_response(response_path)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "cycle",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_002",
                            "--run-safe-chain",
                            "--mock-response-file",
                            str(response_path),
                            "--run-command",
                            "python train.py",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "safe_execution_chain.json").exists())
            self.assertTrue((root / "jobs" / "demo" / "demo_trial_002.yaml").exists())

    def test_cycle_accepts_apply_next_patch_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.7\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "lightgbm",
                            "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8},
                        },
                        "features": {
                            "use_frequency_encoding": False,
                            "use_target_encoding": False,
                            "use_interactions": False,
                            "use_missing_indicators": True,
                        },
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.71, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "cycle",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--no-job",
                            "--next-trial",
                            "trial_002",
                            "--apply-next-patch",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((root / "experiments" / "demo" / "trial_002" / "code_edit_result.md").exists())

    def test_run_auto_loop_command_executes_safe_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.7\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(json.dumps({"model": {"type": "lightgbm"}, "features": {}, "cv": {}}), encoding="utf-8")
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.71, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "run-auto-loop",
                            "--competition",
                            "demo",
                            "--start-trial",
                            "trial_001",
                            "--max-trials",
                            "1",
                            "--submit-policy",
                            "never",
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue((trial / "diagnosis.md").exists())

    def test_run_graph_cycle_command_executes_langgraph_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            (comp / "state.yaml").write_text(
                "competition:\n"
                "  objective: maximize\n"
                "current_state:\n"
                "  consecutive_failures: 0\n"
                "  best_trial:\n"
                "    trial_id: old\n"
                "    cv_score: 0.7\n",
                encoding="utf-8",
            )
            cfg = root / "configs" / "demo"
            cfg.mkdir(parents=True)
            (cfg / "allowed_space.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": ["lightgbm"],
                            "params": {
                                "learning_rate": {"min": 0.005, "max": 0.2},
                                "num_leaves": {"min": 16, "max": 256},
                                "max_depth": {"min": 3, "max": 12},
                            },
                        },
                        "features": {
                            "use_frequency_encoding": [True, False],
                            "use_target_encoding": [True, False],
                            "use_interactions": [True, False],
                            "use_missing_indicators": [True, False],
                        },
                        "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                    }
                ),
                encoding="utf-8",
            )
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "config.yaml").write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "lightgbm",
                            "params": {"learning_rate": 0.03, "num_leaves": 64, "max_depth": 8},
                        },
                        "features": {"use_missing_indicators": True},
                        "cv": {"n_splits": 5, "seed": 42},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(json.dumps({"cv_score": 0.72, "objective": "maximize"}), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["run-graph-cycle", "--competition", "demo", "--trial", "trial_001", "--no-job"])

            self.assertEqual(code, 0)
            self.assertTrue((trial / "diagnosis.md").exists())

    def _write_coding_handoff(self, trial: Path, validation_commands: list[str] | None = None) -> None:
        (trial / "coding_handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "request_id": "demo:trial_002:coding",
                    "competition": "demo",
                    "trial_id": "trial_002",
                    "status": "ready",
                    "allowed_write_files": ["experiments/demo/trial_002/config.yaml"],
                    "create_files": [],
                    "forbidden_paths": [
                        "data/",
                        "submissions/",
                        "experiments/demo/trial_002/submission.csv",
                        "experiments/demo/trial_002/metrics.json",
                    ],
                    "validation_commands": validation_commands or [
                        "python -B -m unittest discover -s tests -v",
                    ],
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
                }
            ),
            encoding="utf-8",
        )

    def _write_minimal_competition_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "state.yaml").write_text(
            "competition:\n"
            "  objective: maximize\n"
            "current_state:\n"
            "  consecutive_failures: 0\n",
            encoding="utf-8",
        )
        cfg = root / "configs" / "demo"
        cfg.mkdir(parents=True)
        (cfg / "allowed_space.yaml").write_text(
            json.dumps(
                {
                    "model": {
                        "type": ["lightgbm"],
                        "params": {
                            "learning_rate": {"min": 0.005, "max": 0.2},
                            "num_leaves": {"min": 16, "max": 256},
                            "max_depth": {"min": 3, "max": 12},
                        },
                    },
                    "features": {
                        "use_frequency_encoding": [True, False],
                        "use_target_encoding": [True, False],
                        "use_interactions": [True, False],
                        "use_missing_indicators": [True, False],
                    },
                    "cv": {"n_splits": [5], "seed": {"min": 1, "max": 9999}},
                }
            ),
            encoding="utf-8",
        )

    def _write_mock_code_response(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Updated config from mock response.",
                            "changed_files": ["experiments/demo/trial_002/config.yaml"],
                            "file_updates": [
                                {
                                    "path": "experiments/demo/trial_002/config.yaml",
                                    "content": "model:\n  type: lightgbm\ntraining:\n  sampler: balanced\n",
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    )
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()


