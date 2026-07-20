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
from kaggle_research_agent.workspace_coding_handoff import prepare_workspace_coding_handoff


class WorkspaceCodingHandoffTest(unittest.TestCase):
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
            self.assertEqual("patch_only", result["edit_policy"]["mode"])
            self.assertTrue(result["edit_policy"]["prefer_patch_updates"])
            self.assertFalse(result["edit_policy"]["allow_full_file_updates"])
            self.assertTrue(result["edit_policy"]["restore_base_before_patch"])
            self.assertEqual("trial_001", result["source_trial_id"])
            self.assertEqual("trial_001", result["code_base_trial_id"])
            self.assertEqual("experiments/demo/trial_001/internal/code_snapshot", result["edit_policy"]["base_code_source"])
            self.assertEqual("outputs/metrics.json", result["metrics_output_contract"]["path"])
            self.assertEqual("cv_score", result["metrics_output_contract"]["score_key"])
            self.assertIn("cv_score", result["metrics_output_contract"]["required_keys"])
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

    def _write_project(self, root: Path) -> Path:
        project = root / "external_project"
        (project / "src").mkdir(parents=True)
        (project / "tests").mkdir()
        (project / "outputs").mkdir()
        (project / "train.py").write_text("print('train')\n", encoding="utf-8")
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
