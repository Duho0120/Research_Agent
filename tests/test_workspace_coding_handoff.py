import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.workspace_code_writer import build_workspace_code_writer_payload
from kaggle_research_agent.workspace_coding_handoff import (
    HARNESS_INIT_TRIAL_ID,
    generate_scoring_harness,
    prepare_workspace_coding_handoff,
)


class WorkspaceCodingHandoffTest(unittest.TestCase):
    def test_code_priority_is_generic_not_titanic_specific(self):
        from kaggle_research_agent.workspace_coding_handoff import (
            _code_file_priority,
            _context_file_limit,
            _delta_code_file_priority,
        )

        self.assertEqual(0, _code_file_priority(Path("src/pipeline.py"))[0])
        self.assertEqual(0, _code_file_priority(Path("src/baseline.py"))[0])
        self.assertEqual(2, _code_file_priority(Path("src/titanic_pipeline.py"))[0])
        self.assertEqual(14000, _context_file_limit("src/pipeline.py", 2000))
        delta_plan = {
            "code_change_targets": [
                "train_step.py: replace the estimator",
                "predict_step.py: verify inference",
            ]
        }
        self.assertLess(
            _delta_code_file_priority(Path("train_step.py"), delta_plan),
            _delta_code_file_priority(Path("src/baseline.py"), delta_plan),
        )

    def test_current_workspace_file_listing_prioritizes_plan_relevant_file_over_alphabetical_tiebreak(self):
        # Regression for trial_026: the "Current Workspace Code" section (the
        # live files, as opposed to the saved base-trial snapshot) built its
        # file order with the same plain alphabetical tie-break bug fixed
        # earlier for the base-trial-snapshot section. When predict_step.py
        # and train_step.py tie on priority, predict_step.py wins
        # alphabetically and gets listed first -- if the char budget then
        # runs out, train_step.py (the file the plan actually needs) can be
        # cut off entirely even in the "expanded retry" snapshot.
        from kaggle_research_agent.workspace_coding_handoff import _allowed_readable_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "predict_step.py").write_text("def predict():\n    return 'noop'\n", encoding="utf-8")
            (root / "test_step.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
            (root / "train_step.py").write_text(
                "def train_validate():\n    return 'StandardScaler'\n", encoding="utf-8"
            )
            delta_plan = {"code_change_targets": [], "required_code_symbols": ["train_validate"]}

            files = _allowed_readable_files(
                root,
                ["predict_step.py", "test_step.py", "train_step.py"],
                max_files=10,
                delta_plan=delta_plan,
            )

            self.assertEqual("train_step.py", files[0].name)

    def test_prepare_workspace_coding_handoff_writes_scoped_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="continue_with_caution")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("ready", result["status"])
            self.assertEqual("workspace_coding_agent_request", result["handoff_type"])
            self.assertEqual(str(project), result["project_root"])
            self.assertEqual(["src/", "tests/", "train.py"], result["allowed_write_paths"])
            self.assertIn("outputs/metrics.json", result["forbidden_paths"])
            self.assertIn("outputs/submission.csv", result["forbidden_paths"])
            self.assertIn("scoring_harness.py", result["forbidden_paths"])
            self.assertNotIn("scoring_harness.py", result["allowed_write_paths"])
            self.assertEqual("patch_only", result["edit_policy"]["mode"])
            self.assertTrue(result["edit_policy"]["prefer_patch_updates"])
            self.assertFalse(result["edit_policy"]["allow_full_file_updates"])
            self.assertTrue(result["edit_policy"]["restore_base_before_patch"])
            self.assertEqual("trial_001", result["source_trial_id"])
            self.assertEqual("trial_001", result["code_base_trial_id"])
            self.assertEqual("experiments/demo/trial_001/user_view/code", result["edit_policy"]["base_code_source"])
            self.assertEqual("outputs/metrics.json", result["metrics_output_contract"]["path"])
            self.assertEqual("cv_score", result["metrics_output_contract"]["score_key"])
            self.assertIn("cv_score", result["metrics_output_contract"]["required_keys"])
            # Classic single-table competition (data card declares a target
            # column): the harness interface is not imposed on it.
            self.assertIsNone(result["scoring_interface_contract"])
            request_text = (trial / "workspace_coding_agent_request.md").read_text(encoding="utf-8")
            self.assertNotIn("## Prediction Function Interface Contract", request_text)
            self.assertFalse(result["artifact_policy"]["save_model"]["default"])
            self.assertIn("required_for_separate_predict_command", result["artifact_policy"]["save_model"]["allowed_when"])
            self.assertTrue(result["execution_constraints"]["do_not_run_training"])
            self.assertTrue(result["execution_constraints"]["do_not_submit"])
            self.assertTrue(result["execution_constraints"]["do_not_edit_data_or_outputs"])
            self.assertTrue(result["pending_human_review"])
            self.assertIn("experiments/demo/trial_002/next_experiment.md", result["context_files"])
            self.assertIn("experiments/demo/trial_002/workspace_context_snapshot.md", result["context_files"])
            self.assertIn("experiments/demo/trial_002/context_pack_workspace_code_writing.md", result["context_files"])
            self.assertIn("experiments/demo/trial_002/retrieval_manifest_workspace_code_writing.json", result["context_files"])
            self.assertEqual("workspace_code_writing", result["retrieval_context"]["task"])
            self.assertEqual("Survived", result["data_card_summary"]["target_column"])
            self.assertIn("Pclass", result["data_card_summary"]["include_features_first"])
            self.assertIn("Name", result["data_card_summary"]["defer_features_first"])
            self.assertTrue((trial / "workspace_coding_handoff.json").exists())
            self.assertTrue((trial / "context_pack_workspace_code_writing.json").exists())
            self.assertTrue((trial / "retrieval_manifest_workspace_code_writing.json").exists())
            snapshot = trial / "workspace_context_snapshot.md"
            self.assertTrue(snapshot.exists())
            snapshot_text = snapshot.read_text(encoding="utf-8")
            self.assertIn("Previous Trial Evidence", snapshot_text)
            self.assertIn("Recommended Base Trial Code Snapshot", snapshot_text)
            self.assertIn("Current Workspace Code Inventory", snapshot_text)
            self.assertIn("train.py", snapshot_text)
            self.assertIn("MODEL = 'source'", snapshot_text)
            self.assertNotIn("MODEL = 'baseline'", snapshot_text)
            self.assertNotIn("tests/README.md", snapshot_text)
            request = trial / "workspace_coding_agent_request.md"
            self.assertTrue(request.exists())
            pending_snapshot = json.loads(
                (trial / "internal" / "pending_execution_plan_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual("pending", pending_snapshot["status"])
            self.assertEqual("trial_002", pending_snapshot["trial_id"])
            self.assertEqual(
                "workspace_coding_agent_request.md:Next Experiment",
                pending_snapshot["source"],
            )
            text = request.read_text(encoding="utf-8")
            self.assertIn("Allowed External Write Paths", text)
            self.assertIn("src/", text)
            self.assertIn("Do not run training", text)
            self.assertIn("Metrics Output Contract", text)
            self.assertIn("Edit Policy", text)
            self.assertIn("patch_updates", text)
            self.assertIn("restore_base_before_patch: True", text)
            self.assertIn("base_code_source", text)
            self.assertIn("cv_score", text)
            self.assertIn("Artifact Policy", text)
            self.assertIn("Do not persist trained model", text)
            self.assertIn("RAG Context Pack", text)
            self.assertIn("Data Card Summary", text)
            self.assertIn("Survived", text)
            log = root / "memory" / "demo" / "decision_log.jsonl"
            last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual("workspace_coding_handoff", last["decision_type"])
            self.assertEqual("ready", last["decision"])

    def test_patch_only_handoff_keeps_patch_target_functions_visible_without_full_workspace_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source_code = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src" / "baseline.py"
            source_code.write_text(_long_baseline_code(), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")
                trial = root / "experiments" / "demo" / "trial_002"
                snapshot_text = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")
                handoff = json.loads((trial / "workspace_coding_handoff.json").read_text(encoding="utf-8"))
                payload = build_workspace_code_writer_payload(handoff, model="gpt-5.5")

            prompt = payload["input"][1]["content"]

            self.assertEqual("ready", result["status"])
            self.assertIn("Compacted baseline.py for continuation patching", snapshot_text)
            self.assertIn("def train()", prompt)
            self.assertIn("def predict()", prompt)
            self.assertIn("def _make_features", prompt)
            self.assertIn("def _build_pipeline", prompt)
            self.assertIn("Current Workspace Code Inventory", prompt)
            self.assertNotIn("MODEL = 'baseline'", prompt)

    def test_expanded_retry_handoff_includes_current_code_when_source_snapshot_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source_code = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src" / "model.py"
            source_code.unlink()
            (project / "src" / "extra.py").write_text("EXTRA = 'current'\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff(
                    "demo",
                    "trial_002",
                    expanded_snapshot=True,
                    retry_reason="code_writer_blocked_snapshot_context",
                )
                trial = root / "experiments" / "demo" / "trial_002"
                snapshot_text = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")
                handoff = json.loads((trial / "workspace_coding_handoff.json").read_text(encoding="utf-8"))
                payload = build_workspace_code_writer_payload(handoff, model="gpt-5.5")

        prompt = payload["input"][1]["content"]

        self.assertEqual("ready", result["status"])
        self.assertEqual("expanded_after_code_writer_blocked", result["snapshot_mode"])
        self.assertEqual("code_writer_blocked_snapshot_context", result["retry_reason"])
        self.assertIn("No saved source trial code snapshot was found", snapshot_text)
        self.assertIn("Current Workspace Code", snapshot_text)
        self.assertIn("MODEL = 'baseline'", prompt)
        self.assertIn("EXTRA = 'current'", prompt)

    def test_runtime_repair_handoff_preserves_current_workspace_and_failure_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            (project / "src" / "baseline.py").write_text("MODEL = 'failed_delta'\n", encoding="utf-8")
            failure = {
                "failed_stage": "train",
                "failure_type": "artifact_serialization",
                "log_tail": "TypeError: Object of type ufunc is not JSON serializable",
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff(
                    "demo",
                    "trial_002",
                    expanded_snapshot=True,
                    retry_reason="workspace_runtime_failure",
                    runtime_failure_context=failure,
                )
                trial = root / "experiments" / "demo" / "trial_002"
                handoff = json.loads((trial / "workspace_coding_handoff.json").read_text(encoding="utf-8"))
                payload = build_workspace_code_writer_payload(handoff, model="gpt-5.5")

        prompt = payload["input"][1]["content"]
        self.assertEqual("expanded_runtime_repair", result["snapshot_mode"])
        self.assertFalse(result["edit_policy"]["restore_base_before_patch"])
        self.assertEqual(failure, result["runtime_failure_context"])
        self.assertIn("Runtime repair mode", prompt)
        self.assertIn("Object of type ufunc is not JSON serializable", prompt)
        self.assertIn("failed_delta", prompt)
        self.assertIn("JSON serializable", prompt)

    def test_patch_only_handoff_keeps_class_based_pipeline_targets_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source_code = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src" / "baseline.py"
            source_code.write_text(_class_based_baseline_code(), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")
                trial = root / "experiments" / "demo" / "trial_002"
                snapshot_text = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")

            self.assertEqual("ready", result["status"])
            self.assertIn("class FeatureBuilder", snapshot_text)
            self.assertIn("class GroupedAgeImputer", snapshot_text)
            self.assertIn("def build_pipeline", snapshot_text)
            self.assertIn("def run_experiment", snapshot_text)
            self.assertIn("def run_prediction", snapshot_text)
            self.assertIn("Cabin", snapshot_text)

    def test_active_axis_patch_handoff_skips_code_writing_rag_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                decision_context={
                    "active_axis": "feature_engineering_title",
                    "axis_attempt_count": 1,
                    "axis_attempt_limit": 3,
                    "recommended_base_trial": "trial_001",
                },
            )
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_title",
                        "candidate": {"name": "TitleVariant"},
                        "code_change_targets": ["src/baseline.py"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.md").write_text("# Delta Patch Plan\n\n- candidate: TitleVariant\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertEqual("ready", result["status"])
            self.assertTrue(result["retrieval_context"]["skipped"])
            self.assertEqual(
                "continuation_uses_structured_memory_and_code_snapshot",
                result["retrieval_context"]["skip_reason"],
            )
            self.assertFalse(result["retrieval_context"]["policy"]["use_rag"])
            self.assertIn("experiments/demo/trial_002/workspace_context_snapshot.md", result["context_files"])
            self.assertIn("experiments/demo/trial_002/delta_plan.json", result["context_files"])
            self.assertIn("experiments/demo/trial_002/delta_plan.md", result["context_files"])
            self.assertNotIn("experiments/demo/trial_002/next_experiment.md", result["context_files"])
            self.assertNotIn("experiments/demo/trial_002/continuation_context.json", result["context_files"])
            self.assertNotIn("experiments/demo/trial_002/context_pack_workspace_code_writing.md", result["context_files"])
            self.assertFalse((trial / "context_pack_workspace_code_writing.json").exists())
            self.assertFalse((trial / "retrieval_manifest_workspace_code_writing.json").exists())

    def test_classic_single_table_competition_is_not_forced_onto_the_harness_interface(self):
        # Regression: requiring load_samples()/predict() in every competition
        # retroactively blocked the already-working single-table pipelines
        # (Titanic, bike-sharing-demand). Their predict_step.py has no such
        # functions and never needed them -- cv_score is computed inside
        # train_step.py. Only competitions the single-table flow cannot carry
        # (no usable target column) opt into the harness interface.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            card_path = root / "competitions" / "demo" / "competition_data_card.json"
            card = json.loads(card_path.read_text(encoding="utf-8"))
            card["target_column"] = "Survived"
            card_path.write_text(json.dumps(card), encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertIsNone(result["scoring_interface_contract"])

    def test_handoff_keeps_scoring_harness_locked_for_an_unrelated_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                decision_context={
                    "active_axis": "model_family",
                    "axis_attempt_count": 1,
                    "axis_attempt_limit": 3,
                    "recommended_base_trial": "trial_001",
                },
            )
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "candidate": {"name": "KNNRegressor"},
                        "code_change_targets": ["src/baseline.py"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.md").write_text("# Delta Patch Plan\n\n- candidate: KNNRegressor\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertIn("scoring_harness.py", result["forbidden_paths"])
            self.assertNotIn("scoring_harness.py", result["allowed_write_paths"])

    def test_handoff_unlocks_scoring_harness_for_the_scoring_logic_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                decision_context={
                    "active_axis": "validation_structure",
                    "axis_attempt_count": 1,
                    "axis_attempt_limit": 3,
                    "recommended_base_trial": "trial_001",
                },
            )
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        # Deliberately a synonym spelling, not the canonical
                        # "scoring_logic" -- must still normalize to it.
                        "primary_change_axis": "validation_structure",
                        "candidate": {"name": "kfold_holdout"},
                        "code_change_targets": ["scoring_harness.py"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.md").write_text("# Delta Patch Plan\n\n- candidate: kfold_holdout\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertNotIn("scoring_harness.py", result["forbidden_paths"])
            self.assertIn("scoring_harness.py", result["allowed_write_paths"])

    def test_continuation_handoff_skips_code_writing_rag_without_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("ready", result["status"])
            self.assertTrue(result["retrieval_context"]["skipped"])
            self.assertEqual(
                "continuation_uses_structured_memory_and_code_snapshot",
                result["retrieval_context"]["skip_reason"],
            )
            self.assertNotIn("experiments/demo/trial_002/context_pack_workspace_code_writing.md", result["context_files"])
            self.assertFalse((trial / "context_pack_workspace_code_writing.json").exists())

    def test_continuation_handoff_uses_rag_when_user_feedback_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                extra_context={"user_feedback": "Review prior feature experiments before changing model family."},
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertEqual("ready", result["status"])
            self.assertFalse(result["retrieval_context"]["skipped"])
            self.assertTrue(result["retrieval_context"]["policy"]["use_rag"])
            self.assertEqual("user_feedback_or_human_review_present", result["retrieval_context"]["policy"]["reason"])
            self.assertIn("experiments/demo/trial_002/context_pack_workspace_code_writing.md", result["context_files"])

    def test_delta_patch_handoff_uses_targeted_code_snapshot_for_prompt_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                decision_context={
                    "active_axis": "feature_engineering_cabin_missing_indicator",
                    "axis_attempt_count": 1,
                    "axis_attempt_limit": 3,
                    "recommended_base_trial": "trial_001",
                },
            )
            source_code = root / "experiments" / "demo" / "trial_001" / "user_view" / "code" / "src" / "baseline.py"
            source_code.write_text(_class_based_baseline_code(), encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_cabin_missing_indicator",
                        "candidate": {"name": "CabinMissing_Pclass3"},
                        "code_change_targets": ["feature construction block", "numeric feature list"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "delta_plan.md").write_text("# Delta Patch Plan\n\n- candidate: CabinMissing_Pclass3\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")
                snapshot_text = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")
                handoff = json.loads((trial / "workspace_coding_handoff.json").read_text(encoding="utf-8"))
                payload = build_workspace_code_writer_payload(handoff, model="gpt-5.5")

            prompt = payload["input"][1]["content"]

            self.assertEqual("ready", result["status"])
            self.assertLess(len(snapshot_text), 10000)
            self.assertLess(len(prompt), 13000)
            self.assertIn("class FeatureBuilder", prompt)
            self.assertIn("def build_pipeline", prompt)
            self.assertIn("def pipeline_summary", prompt)
            self.assertIn("CabinMissing_Pclass3", prompt)
            self.assertNotIn("def run_prediction", prompt)
            self.assertIn("Omitted for compact delta patch mode", prompt)

    def test_prepare_workspace_coding_handoff_blocks_when_plan_find_target_missing_from_base_code(self):
        # Regression for trial_027: the plan continued axis
        # model_hyperparameters:Ridge and assumed Ridge( already existed in
        # the base trial's code (because an earlier, unaccepted attempt at
        # that axis had it), but the actual recommended base trial
        # (trial_014-equivalent here) still used a different estimator. This
        # used to only surface after a wasted code-writing LLM call
        # ("patch_find_not_found"); it should now be caught up front.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source = root / "experiments" / "demo" / "trial_001" / "user_view" / "code"
            (source / "src").mkdir(parents=True, exist_ok=True)
            (source / "train_step.py").write_text(
                "from sklearn.ensemble import HistGradientBoostingRegressor\n\n"
                "def train():\n    return HistGradientBoostingRegressor()\n",
                encoding="utf-8",
            )
            (source / "predict_step.py").write_text("def predict():\n    return 'noop'\n", encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_hyperparameters:Ridge",
                        "code_change_targets": [
                            "File: train_step.py",
                            "Find: Ridge(",
                            "Replace Hint: Ridge(alpha=1.0, solver='cholesky')",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("plan_find_target_missing_in_base_code:train_step.py", result["blocking_issues"])
            self.assertEqual("resolve-workspace-handoff-blockers", result["next_action"])
            self.assertFalse((trial / "workspace_coding_agent_request.md").exists())

    def test_expanded_snapshot_hides_current_code_body_when_base_will_be_restored(self):
        # Regression for trial_030: on the expanded retry the snapshot dumped
        # the current workspace body labelled "Used as exact patch context",
        # but restore_base_before_patch overwrites that workspace with the
        # base trial snapshot before the patch is applied. The code writer
        # copied find text from the doomed current code (which used a
        # different local variable name) and every patch failed with
        # patch_find_not_found.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            # Base trial snapshot uses `X`; the live workspace uses `Xb`.
            source = root / "experiments" / "demo" / "trial_001" / "user_view" / "code"
            source.mkdir(parents=True, exist_ok=True)
            (source / "train_step.py").write_text(
                "def train():\n    X = build()\n    X = X.assign(hr_sin=1)\n    return X\n",
                encoding="utf-8",
            )
            (project / "train_step.py").write_text(
                "def train():\n    Xb = build()\n    Xb = Xb.assign(hr_sin=1)\n    return Xb\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002", expanded_snapshot=True)
                snapshot = (
                    root / "experiments" / "demo" / "trial_002" / "workspace_context_snapshot.md"
                ).read_text(encoding="utf-8")

            self.assertEqual("ready", result["status"])
            self.assertTrue(result["edit_policy"]["restore_base_before_patch"])
            # The soon-to-be-overwritten body must not be offered as patch source.
            self.assertNotIn("Xb", snapshot)
            self.assertIn("WILL BE OVERWRITTEN", snapshot)
            # The authoritative base code is still available to patch against.
            self.assertIn("X = X.assign(hr_sin=1)", snapshot)

    def test_coding_feedback_is_rendered_into_the_retry_prompt(self):
        # Real incident: a retry after a guardrail block got more base-code
        # context but was never told *why* it was rejected, so it reproduced
        # the exact same rejected code. coding_feedback is how the specific
        # rejected issues actually reach the prompt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff(
                    "demo",
                    "trial_002",
                    coding_feedback={
                        "changed_files": ["predict_step.py"],
                        "rejected_issues": ["local_validation_not_computed:validation_method_none"],
                    },
                )

            request_text = (
                root / "experiments" / "demo" / "trial_002" / "workspace_coding_agent_request.md"
            ).read_text(encoding="utf-8")

        self.assertEqual(
            {"changed_files": ["predict_step.py"], "rejected_issues": ["local_validation_not_computed:validation_method_none"]},
            result["coding_feedback"],
        )
        self.assertIn("## Previous Attempt Was Rejected", request_text)
        self.assertIn("local_validation_not_computed:validation_method_none", request_text)
        self.assertIn("predict_step.py", request_text)

    def test_no_coding_feedback_section_when_first_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                prepare_workspace_coding_handoff("demo", "trial_002")

            request_text = (
                root / "experiments" / "demo" / "trial_002" / "workspace_coding_agent_request.md"
            ).read_text(encoding="utf-8")

        self.assertNotIn("## Previous Attempt Was Rejected", request_text)

    def test_runtime_repair_still_shows_current_code_because_nothing_is_restored(self):
        # Runtime repair fixes a crash in code that already ran, so the base
        # snapshot is deliberately NOT restored -- there the current workspace
        # really is the authoritative patch source and must stay visible.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            (project / "train.py").write_text(
                "def train():\n    Xb = build()\n    return Xb\n", encoding="utf-8"
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff(
                    "demo",
                    "trial_002",
                    expanded_snapshot=True,
                    runtime_failure_context={"failed_stage": "train", "returncode": 1},
                )
                snapshot = (
                    root / "experiments" / "demo" / "trial_002" / "workspace_context_snapshot.md"
                ).read_text(encoding="utf-8")

            self.assertFalse(result["edit_policy"]["restore_base_before_patch"])
            self.assertIn("Xb", snapshot)

    def test_delta_patch_snapshot_prioritizes_explicit_target_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source = root / "experiments" / "demo" / "trial_001" / "user_view" / "code"
            for relative, content in {
                "src/baseline.py": "BASELINE = True\n",
                "predict_step.py": "def predict():\n    return 'ridge'\n",
                "train_step.py": "from sklearn.linear_model import Ridge\n\ndef train():\n    return Ridge()\n",
            }.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "code_change_targets": [
                            "train_step.py: replace Ridge",
                            "predict_step.py: verify inference",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                prepare_workspace_coding_handoff("demo", "trial_002")

            snapshot = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")
            self.assertIn("### train_step.py", snapshot)
            self.assertIn("### predict_step.py", snapshot)
            self.assertNotIn("### src/baseline.py", snapshot)

    def test_delta_patch_snapshot_falls_back_to_required_code_symbols_when_targets_empty(self):
        # Regression for trial_015: the planner sometimes leaves
        # code_change_targets empty even though the plan clearly needs a
        # specific file changed (it still names the function via
        # required_code_symbols). With the tight max_files=2 budget for delta
        # refinement, a plain alphabetical tie-break between predict_step.py
        # and train_step.py picked predict_step.py and silently dropped
        # train_step.py -- the file that actually needed the patch.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")
            source = root / "experiments" / "demo" / "trial_001" / "user_view" / "code"
            for relative, content in {
                "src/baseline.py": "BASELINE = True\n",
                "predict_step.py": "def predict():\n    return 'noop'\n",
                "train_step.py": (
                    "from sklearn.preprocessing import StandardScaler\n\n"
                    "def train_validate():\n    return StandardScaler()\n"
                ),
            }.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            trial = root / "experiments" / "demo" / "trial_002"
            (trial / "delta_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "preprocessing:remove_standard_scaler_for_tree_model",
                        "code_change_targets": [],
                        "required_code_symbols": ["train_validate"],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                prepare_workspace_coding_handoff("demo", "trial_002")

            snapshot = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")
            self.assertIn("### train_step.py", snapshot)
            self.assertIn("StandardScaler", snapshot)

    def test_handoff_uses_recommended_base_trial_for_code_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(
                root,
                continuation_mode="can_continue",
                decision_context={
                    "active_axis": "user_insight_model_ensemble",
                    "axis_attempt_count": 1,
                    "axis_attempt_limit": 3,
                },
            )
            trial = root / "experiments" / "demo" / "trial_002"
            continuation = json.loads((trial / "continuation_context.json").read_text(encoding="utf-8"))
            continuation["source_trial_id"] = "trial_006"
            continuation["recommended_base_trial"] = "trial_003"
            (trial / "continuation_context.json").write_text(json.dumps(continuation), encoding="utf-8")
            base_code = root / "experiments" / "demo" / "trial_003" / "user_view" / "code" / "src" / "model.py"
            base_code.parent.mkdir(parents=True, exist_ok=True)
            base_code.write_text("MODEL = 'best-submitted-base'\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")
                snapshot_text = (trial / "workspace_context_snapshot.md").read_text(encoding="utf-8")

            self.assertEqual("ready", result["status"])
            self.assertEqual("trial_006", result["source_trial_id"])
            self.assertEqual("trial_003", result["code_base_trial_id"])
            self.assertEqual("trial_003", result["recommended_base_trial"])
            self.assertEqual(
                "experiments/demo/trial_003/user_view/code",
                result["edit_policy"]["base_code_source"],
            )
            self.assertIn("best-submitted-base", snapshot_text)
            self.assertNotIn("MODEL = 'source'", snapshot_text)

    def test_prepare_workspace_coding_handoff_blocks_must_wait_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="must_wait")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            trial = root / "experiments" / "demo" / "trial_002"
            self.assertEqual("blocked", result["status"])
            self.assertIn("continuation_requires_user_feedback", result["blocking_issues"])
            self.assertTrue((trial / "workspace_coding_handoff.json").exists())
            self.assertFalse((trial / "workspace_coding_agent_request.md").exists())

    def test_prepare_workspace_coding_handoff_blocks_invalid_execution_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_next_trial(root, continuation_mode="can_continue")
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            simple_yaml.dump(
                {
                    "schema_version": "1.0",
                    "competition": "demo",
                    "platform": "external",
                    "project_root": str(root / "missing_project"),
                    "python": str(root / "missing_python.exe"),
                    "commands": {"test": ["{python} -m pytest"], "train": ["{python} train.py"]},
                    "artifacts": {"metrics": ["outputs/metrics.json"], "submission": ["outputs/submission.csv"]},
                    "write_scope": {"allowed": ["src/"], "forbidden": ["outputs/"]},
                },
                comp / "execution_profile.yaml",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = prepare_workspace_coding_handoff("demo", "trial_002")

            self.assertEqual("blocked", result["status"])
            self.assertIn("execution_profile_not_ready", result["blocking_issues"])

    def test_prepare_workspace_handoff_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            self._write_next_trial(root, continuation_mode="can_continue")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with redirect_stdout(io.StringIO()):
                    code = main(["prepare-workspace-handoff", "--competition", "demo", "--trial", "trial_002"])

            self.assertEqual(0, code)
            self.assertTrue(
                (root / "experiments" / "demo" / "trial_002" / "workspace_coding_handoff.json").exists()
            )

    def test_generate_scoring_harness_skips_when_already_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            (project / "scoring_harness.py").write_text("# already generated\n", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer") as writer:
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("already_exists", result["status"])
            writer.assert_not_called()

    def test_generate_scoring_harness_skips_llm_call_when_predict_interface_not_ready(self):
        # Real incident: harness generation ran before trial_001 had ever
        # written a compliant predict_step.py (the scaffold's own default
        # predict_step.py doesn't define load_samples()/predict() either),
        # so every attempt was doomed regardless of what the harness wrote.
        # This must be caught up front, without spending an LLM call.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            # Overwrite with a predict.py that does NOT define the interface,
            # matching the scaffold's default predict_step.py shape.
            (project / "predict.py").write_text(
                "from src.baseline import predict\n\nif __name__ == '__main__':\n    predict()\n",
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer") as writer:
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertEqual("predict_interface_not_ready", result["reason"])
            self.assertIn("predict_interface_functions_not_defined:predict.py", result["issue"])
            writer.assert_not_called()

    def test_generate_scoring_harness_embeds_predict_source_in_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["scoring_harness.py"]},
                ):
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("completed", result["status"])
            context_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "predict_script_context.md"
            self.assertTrue(context_path.exists())
            self.assertIn("def load_samples", context_path.read_text(encoding="utf-8"))
            handoff_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_handoff.json"
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertIn(
                f"experiments/demo/{HARNESS_INIT_TRIAL_ID}/predict_script_context.md", handoff["context_files"]
            )
            request_text = (
                root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_agent_request.md"
            ).read_text(encoding="utf-8")
            self.assertIn("predict_script_context.md", request_text)

    def test_generate_scoring_harness_writes_scoped_handoff_and_calls_code_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["scoring_harness.py"]},
                ) as writer:
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual(1, writer.call_count)
            self.assertEqual("demo", writer.call_args.args[0])
            self.assertEqual(HARNESS_INIT_TRIAL_ID, writer.call_args.args[1])

            handoff_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_handoff.json"
            self.assertTrue(handoff_path.exists())
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertEqual(["scoring_harness.py"], handoff["allowed_write_paths"])
            self.assertIn("train.py", handoff["forbidden_paths"])
            self.assertIn("outputs/metrics.json", handoff["forbidden_paths"])
            self.assertIn("outputs/submission.csv", handoff["forbidden_paths"])
            self.assertEqual("predict.py", handoff["scoring_interface_contract"]["target_file"])
            self.assertEqual("cv_score", handoff["metrics_output_contract"]["score_key"])

            request_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_agent_request.md"
            self.assertTrue(request_path.exists())
            request_text = request_path.read_text(encoding="utf-8")
            self.assertIn('key literally named "cv_score"', request_text)

    def test_generate_scoring_harness_threads_declared_metric_name_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            simple_yaml.dump({"metric": "R-Hit@1cm"}, root / "competitions" / "demo" / "state.yaml")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    return_value={"status": "accepted", "changed_files": ["scoring_harness.py"]},
                ):
                    generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            handoff_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_handoff.json"
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertEqual("R-Hit@1cm", handoff["declared_metric_name"])
            request_text = (
                root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_agent_request.md"
            ).read_text(encoding="utf-8")
            self.assertIn("R-Hit@1cm", request_text)

    def test_generate_scoring_harness_reports_blocked_code_writer_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    return_value={"status": "blocked", "issues": ["scoring_harness_missing_interface_call:scoring_harness.py:predict"]},
                ):
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("blocked", result["status"])

    def test_generate_scoring_harness_removes_a_rejected_harness_so_it_can_be_regenerated(self):
        # Real incident: the code writer applies file updates to disk before
        # validation runs, so a harness that failed review was left sitting in
        # the workspace. Since the only "already generated?" signal is the
        # file's existence, that permanently locked the broken harness in --
        # every later cycle short-circuited on "already_exists" and it was
        # never regenerated.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)

            def write_bad_harness(*args, **kwargs):
                (project / "scoring_harness.py").write_text("# rejected\n", encoding="utf-8")
                return {"status": "blocked", "issues": ["scoring_harness_missing_score_key:scoring_harness.py:cv_score"]}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    side_effect=write_bad_harness,
                ):
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertFalse((project / "scoring_harness.py").exists())

    def test_generate_scoring_harness_retries_once_with_corrective_feedback(self):
        # Real incident (see 포트폴리오.md): a real GPT-5 harness-generation
        # call ignored the interface-reuse instruction and built its own
        # independent CV pipeline instead of calling load_samples()/predict(),
        # reproduced twice in a row in isolated testing. The retry must carry
        # the specific rejected issues into the next handoff, the same
        # pattern already proven for the regular per-trial code writer.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            results = [
                {
                    "status": "blocked",
                    "issues": ["scoring_harness_missing_interface_call:scoring_harness.py:load_samples"],
                    "changed_files": ["scoring_harness.py"],
                },
                {"status": "accepted", "changed_files": ["scoring_harness.py"]},
            ]

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer", side_effect=results
                ) as writer:
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual(2, writer.call_count)
            handoff_path = root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_handoff.json"
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ["scoring_harness_missing_interface_call:scoring_harness.py:load_samples"],
                handoff["coding_feedback"]["rejected_issues"],
            )
            request_text = (
                root / "experiments" / "demo" / HARNESS_INIT_TRIAL_ID / "workspace_coding_agent_request.md"
            ).read_text(encoding="utf-8")
            self.assertIn("## Previous Attempt Was Rejected", request_text)

    def test_generate_scoring_harness_gives_up_after_one_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._write_project(root)
            self._write_execution_profile(root, project)
            blocked = {
                "status": "blocked",
                "issues": ["scoring_harness_missing_interface_call:scoring_harness.py:load_samples"],
                "changed_files": ["scoring_harness.py"],
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer",
                    side_effect=[blocked, blocked],
                ) as writer:
                    result = generate_scoring_harness("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("blocked", result["status"])
            self.assertEqual(2, writer.call_count)

    def _write_project(self, root: Path) -> Path:
        project = root / "external_project"
        (project / "src").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "outputs").mkdir()
        (project / "train.py").write_text("print('train')\n", encoding="utf-8")
        (project / "predict.py").write_text(
            "def load_samples(data_dir, split):\n    return []\n\n\ndef predict(sample):\n    return sample\n",
            encoding="utf-8",
        )
        (project / "src" / "model.py").write_text("MODEL = 'baseline'\n", encoding="utf-8")
        (project / "tests" / "test_model.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
        (project / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
        (project / "outputs" / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
        python = root / "python.exe"
        python.write_text("fake python", encoding="utf-8")
        return project

    def _write_execution_profile(self, root: Path, project: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": str(root / "python.exe"),
                "commands": {
                    "test": ["{python} -m pytest tests -q"],
                    "train": ["{python} train.py"],
                    "predict": ["{python} predict.py"],
                },
                "artifacts": {"metrics": ["outputs/metrics.json"], "submission": ["outputs/submission.csv"]},
                "write_scope": {
                    "allowed": ["src/", "tests/", "train.py"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            },
            comp / "execution_profile.yaml",
        )
        (comp / "competition_data_card.json").write_text(
            json.dumps(
                {
                    "task_type": "tabular_classification",
                    "target_column": "Survived",
                    "id_column": "PassengerId",
                    "submission_prediction_column": "Survived",
                    "train_file": "train.csv",
                    "test_file": "test.csv",
                    "sample_submission_file": "gender_submission.csv",
                    "baseline_recommendation": {
                        "include_features_first": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
                        "defer_features_first": ["Name", "Ticket", "Cabin"],
                        "exclude_columns": ["PassengerId", "Survived"],
                        "preferred_model_families": ["logistic_regression_or_linear_classifier_for_binary_classification"],
                        "avoid_first_trial": ["gaussian_naive_bayes_for_mixed_numeric_and_categorical_data"],
                    },
                }
            ),
            encoding="utf-8",
        )

    def _write_next_trial(
        self,
        root: Path,
        *,
        continuation_mode: str,
        decision_context: dict | None = None,
        extra_context: dict | None = None,
    ) -> None:
        source = root / "experiments" / "demo" / "trial_001"
        (source / "internal").mkdir(parents=True, exist_ok=True)
        (source / "user_view").mkdir(parents=True, exist_ok=True)
        (source / "user_view" / "code" / "src").mkdir(parents=True, exist_ok=True)
        (source / "user_view" / "code" / "src" / "model.py").write_text("MODEL = 'source'\n", encoding="utf-8")
        (source / "internal" / "pipeline_structure.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "stages": [
                        {
                            "id": "data_split_cv",
                            "name": "Data Split / CV Strategy",
                            "included": True,
                            "code_locations": ["train.py"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (source / "user_view" / "02_pipeline_structure.ko.md").write_text(
            "# trial_001 파이프라인 구조\n\n## Data Split / CV Strategy\n",
            encoding="utf-8",
        )
        trial = root / "experiments" / "demo" / "trial_002"
        trial.mkdir(parents=True)
        (trial / "next_experiment.md").write_text(
            "# trial_002 Next Experiment\n\nTry a controlled feature cleanup.\n",
            encoding="utf-8",
        )
        context = {
            "competition": "demo",
            "source_trial_id": "trial_001",
            "next_trial_id": "trial_002",
            "continuation_mode": continuation_mode,
            "pending_human_review": continuation_mode == "continue_with_caution",
            "review_source_trial": "trial_001" if continuation_mode == "continue_with_caution" else None,
            "allowed_topics": ["controlled_refinement"],
            "blocked_topics": [],
            "decision_context": decision_context or {},
        }
        if extra_context:
            context.update(extra_context)
        (trial / "continuation_context.json").write_text(
            json.dumps(context),
            encoding="utf-8",
        )


def _long_baseline_code() -> str:
    filler = "\n".join(f"# filler line {index}" for index in range(500))
    return f"""from __future__ import annotations

import pandas as pd

CONSTANT = 42


def load_config():
    return {{}}


def check_required_data():
    return None


def train():
    features = _make_features(pd.DataFrame())
    model = _build_pipeline(features)
    return model


def predict():
    features = _make_features(pd.DataFrame())
    model = _build_pipeline(features)
    return model


def noisy_helper_before_target():
{_indent(filler)}


def _make_features(df):
    df = df.copy()
    df["Title"] = "Unknown"
    return df


def _single_holdout_split(X, y):
    return X, X, y, y


def _build_pipeline(X):
    return ("pipeline", X)


def noisy_helper_after_target():
{_indent(filler)}
"""


def _class_based_baseline_code() -> str:
    filler = "\n".join(f"# filler line {index}" for index in range(500))
    return f"""from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = ["Pclass", "Age"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title"]
DEFERRED_FEATURES = ["Name", "Ticket", "Cabin"]


def extract_title(value):
    return "Mr"


class FeatureBuilder:
    def transform(self, frame):
        frame = frame.copy()
        frame["Title"] = frame["Name"].map(extract_title)
        return frame


def noisy_helper_before_target():
{_indent(filler)}


class GroupedAgeImputer:
    def transform(self, frame):
        return frame


def build_pipeline():
    return ["features", "model"]


def pipeline_summary():
    return {{"features": {{"deferred": DEFERRED_FEATURES}}}}


def run_experiment():
    return build_pipeline()


def run_prediction():
    return build_pipeline()


def noisy_helper_after_target():
{_indent(filler)}
"""


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines())


if __name__ == "__main__":
    unittest.main()


class DataLoaderGenerationTest(unittest.TestCase):
    """The loader is a competition-level asset: generated once, locked from
    trials, and accepted only if RUNNING it produces usable samples. Static
    checks were bypassed repeatedly, including by a loader that walked the
    sample directory and still returned rows carrying nothing but ids."""

    def _setup(self, root):
        project = root / "external_project"
        (project / "src").mkdir(parents=True)
        (project / "outputs").mkdir()
        (project / "predict.py").write_text("def predict(s):\n    return s\n", encoding="utf-8")
        (root / "python.exe").write_text("py", encoding="utf-8")
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "dacon",
                "project_root": str(project),
                "python": str(root / "python.exe"),
                "commands": {"test": ["{python} t.py"], "train": ["{python} tr.py"], "predict": ["{python} predict.py"]},
                "artifacts": {"metrics": ["outputs/metrics.json"], "submission": ["outputs/submission.csv"]},
                "write_scope": {"allowed": ["src/", "predict.py"], "forbidden": ["data/"]},
                "submission_mode": "manual_external",
            },
            comp / "execution_profile.yaml",
        )
        (comp / "competition_data_card.json").write_text(json.dumps({"task_type": "unknown"}), encoding="utf-8")
        return project

    def test_generated_loader_is_kept_only_when_it_runs_and_returns_features(self):
        from kaggle_research_agent.workspace_coding_handoff import generate_data_loader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._setup(root)

            def accept(*args, **kwargs):
                (project / "data_loader.py").write_text("# generated\n", encoding="utf-8")
                return {"status": "accepted", "changed_files": ["data_loader.py"]}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer", side_effect=accept):
                    with patch(
                        "kaggle_research_agent.workspace_coding_handoff.run_sample_loading_probe",
                        return_value={"train": {"count": 5, "head_ids": ["A"], "feature_keys": ["x"]},
                                      "test": {"count": 2, "head_ids": ["B"], "feature_keys": ["x"]},
                                      "deterministic": True},
                    ):
                        result = generate_data_loader("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("completed", result["status"])
            self.assertTrue((project / "data_loader.py").exists())

    def test_loader_that_returns_ids_without_features_is_rejected_and_removed(self):
        from kaggle_research_agent.workspace_coding_handoff import generate_data_loader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._setup(root)

            def accept(*args, **kwargs):
                (project / "data_loader.py").write_text("# id-only loader\n", encoding="utf-8")
                return {"status": "accepted", "changed_files": ["data_loader.py"]}

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer", side_effect=accept):
                    with patch(
                        "kaggle_research_agent.workspace_coding_handoff.run_sample_loading_probe",
                        return_value={"train": {"count": 10000, "head_ids": ["A"], "feature_keys": []},
                                      "test": {"count": 10000, "head_ids": ["B"], "feature_keys": []},
                                      "deterministic": True},
                    ):
                        result = generate_data_loader("demo", model="gpt-5", provider="openai", allow_api=True)

            self.assertEqual("blocked", result["status"])
            # Must not linger: its existence is the "already generated" signal.
            self.assertFalse((project / "data_loader.py").exists())

    def test_existing_loader_is_not_regenerated(self):
        from kaggle_research_agent.workspace_coding_handoff import generate_data_loader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self._setup(root)
            (project / "data_loader.py").write_text("# already there\n", encoding="utf-8")
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch("kaggle_research_agent.workspace_coding_handoff.run_workspace_code_writer") as writer:
                    result = generate_data_loader("demo", model="gpt-5", provider="openai", allow_api=True)
            self.assertEqual("already_exists", result["status"])
            writer.assert_not_called()
