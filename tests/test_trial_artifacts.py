import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent import simple_yaml
from research_agent.trial_artifacts import (
    _extract_pipeline_details,
    _extract_pipeline_facts,
    _pipeline_submission_columns,
    organize_trial_artifacts,
    reconcile_trial_execution_metadata,
    trial_artifact_exists,
)
from research_agent.trial_user_view import _plan_note_lines, render_user_plan, render_user_scores


class TrialArtifactsTest(unittest.TestCase):
    def test_user_scores_only_marks_submission_best(self):
        without_submission = render_user_scores(
            {
                "trial_id": "trial_001",
                "status": "completed",
                "metric": "rmsle",
                "objective": "minimize",
                "local_score": 1.0,
                "is_best_local": True,
                "is_best_lb": False,
            }
        )
        with_submission_best = render_user_scores(
            {
                "trial_id": "trial_002",
                "status": "completed",
                "metric": "rmsle",
                "objective": "minimize",
                "local_score": 0.9,
                "lb_score": 0.8,
                "is_best_local": False,
                "is_best_lb": True,
            }
        )

        self.assertIn("제출 점수 기록 후 판단", without_submission)
        self.assertNotIn("local best", without_submission)
        self.assertIn("| Best 표시 | Best |", with_submission_best)

    class _UnavailableTranslationClient:
        """Simulates the translation API being unavailable, forcing the
        Korean-labeled template fallback so existing content assertions in
        this test (written against that template) stay meaningful, without
        this test making a real network call."""

        def create_response(self, payload):
            raise RuntimeError("translation API unavailable in test")

    class _SummaryClient:
        def create_response(self, payload):
            return {
                "id": "summary-test",
                "model": payload["model"],
                "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
                "output_text": json.dumps(
                    {
                        "progress_message": "trial_001 실행과 결과 기록이 완료되었습니다.",
                        "log_summary": ["테스트, 학습, 예측 명령이 완료되었습니다."],
                        "what_changed": "첫 baseline 파이프라인을 구성했습니다.",
                        "diff_from_previous": "이전 trial이 없어 비교 대상은 없습니다.",
                        "human_review_message": "현재 즉시 요청할 사용자 피드백은 없습니다.",
                    },
                    ensure_ascii=False,
                ),
            }

    def test_plan_note_lines_preserve_pipeline_blueprint_literals(self):
        lines = _plan_note_lines(
            {
                "plan_type": "initial_pipeline_plan",
                "pipeline_blueprint": [
                    "Train File: train.csv",
                    "Test File: test.csv",
                    "Submission Columns: PassengerId",
                    "Submission Columns: Survived",
                    "Method: train_test_split",
                ],
            }
        )

        joined = "\n".join(lines)
        self.assertIn("Pipeline blueprint: Test File: test.csv", joined)
        self.assertIn("Pipeline blueprint: Submission Columns: PassengerId", joined)
        self.assertNotIn("test_step.py", joined)
        self.assertNotIn("predict_step.py", joined)

    def test_delta_patch_user_plan_uses_continuation_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "demo_experiment_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_007",
                        "primary_change_axis": "feature_engineering_cabin_missing_indicator",
                        "candidate": {
                            "name": "CabinMissing_Pclass3",
                            "description": "Add one binary feature for missing Cabin and Pclass 3.",
                            "implementation_hint": "Create CabinMissing_Pclass3 before preprocessing.",
                        },
                        "keep_unchanged": ["train/validation split", "model"],
                        "change_details": ["Add only CabinMissing_Pclass3."],
                        "success_criteria": ["local accuracy > 0.8547"],
                    }
                ),
                encoding="utf-8",
            )

            text = render_user_plan(
                out_dir,
                {
                    "competition": "demo",
                    "trial_id": "trial_013",
                    "metric": "accuracy",
                    "objective": "maximize",
                    "plan_title": "Add CabinMissing_Pclass3",
                },
            )

            self.assertIn("trial_007", text)
            self.assertIn("feature_engineering_cabin_missing_indicator", text)
            self.assertIn("CabinMissing_Pclass3", text)
            self.assertNotIn("첫 회차", text)
            self.assertNotIn("first trial", text.lower())

    def test_execution_metadata_reconciliation_uses_executed_model_and_plan_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_002"
            trial.mkdir(parents=True)
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "trial_id": "trial_001",
                        "cv_score": 0.81,
                        "model": "LogisticRegression",
                    }
                ),
                encoding="utf-8",
            )
            structure = {
                "stages": [
                    {
                        "id": "model_definition",
                        "structured_details": {
                            "estimator": "VotingClassifier",
                            "parameters": {"voting": "soft"},
                        },
                    }
                ],
                "consistency_issues": [],
            }
            plan = {
                "source_trial_id": "trial_001",
                "primary_change_axis": "model_ensemble",
            }

            with patch("research_agent.paths.ROOT", root):
                with patch(
                    "research_agent.trial_artifacts.build_trial_summary",
                    return_value={"trial_id": "trial_002"},
                ):
                    with patch(
                        "research_agent.trial_artifacts.build_pipeline_structure",
                        return_value=structure,
                    ):
                        with patch(
                            "research_agent.trial_artifacts.resolve_trial_plan",
                            return_value=plan,
                        ):
                            with patch(
                                "research_agent.trial_artifacts.write_executed_trial_facts",
                                return_value={"model": "VotingClassifier"},
                            ):
                                result = reconcile_trial_execution_metadata("demo", "trial_002")

            repaired = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("corrected", result["status"])
            self.assertEqual("VotingClassifier", repaired["model"])
            self.assertEqual("trial_002", repaired["trial_id"])
            self.assertEqual("trial_001", repaired["source_trial_id"])
            self.assertEqual("model_ensemble", repaired["change_axis"])
            self.assertTrue((trial / "internal" / "execution_consistency.json").exists())

    def test_organize_trial_artifacts_writes_readme_and_moves_debug_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            project = root / "workspace"
            project.mkdir()
            (project / "train.py").write_text(
                "\n".join(
                    [
                        "import pandas as pd",
                        "from sklearn.impute import SimpleImputer",
                        "from sklearn.linear_model import LogisticRegression",
                        "from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score",
                        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
                        'NUMERIC_FEATURES = ["Age", "Fare"]',
                        'CATEGORICAL_FEATURES = ["Sex", "Embarked"]',
                        'FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["FamilySize", "IsAlone", "Title"]',
                        'df = pd.read_csv("data/train.csv")',
                        'cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)',
                        'imputer = SimpleImputer(strategy="median")',
                        'scaler = StandardScaler()',
                        'encoder = OneHotEncoder(handle_unknown="ignore")',
                        'model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)',
                        'train_test_split(df, test_size=0.2, random_state=42)',
                        'model.fit(df, df)',
                    ]
                ),
                encoding="utf-8",
            )
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            simple_yaml.dump({"project_root": str(project)}, comp / "execution_profile.yaml")
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.8,
                        "metric": "accuracy",
                        "objective": "maximize",
                        "validation_method": "stratified_5_fold_cv",
                        "fold_scores": [0.7, 0.8, 0.9],
                        "features": ["Age", "Fare", "Sex", "Embarked", "FamilySize", "IsAlone", "Title"],
                        "model_type": "LogisticRegression",
                        "random_state": 42,
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(
                json.dumps({"status": "collected", "score_source": "cv_score"}),
                encoding="utf-8",
            )
            (trial / "workspace_run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "command_results": [{"log_path": "experiments/demo/trial_001/workspace_logs/train.log"}],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["train.py"]}),
                encoding="utf-8",
            )
            (trial / "next_experiment.md").write_text("- title: Baseline\n", encoding="utf-8")
            (trial / "demo_experiment_plan.json").write_text(
                json.dumps(
                    {
                        "plan_type": "initial_pipeline_plan",
                        "title": "First-cycle baseline",
                        "objective": "Build a first-cycle Titanic baseline and create metrics.json and submission.csv.",
                        "pipeline_blueprint": [
                            "Numeric: Age, Fare",
                            "Categorical: Sex, Embarked",
                            "Method: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
                            "Model: LogisticRegression(max_iter=1000, random_state=42)",
                            "Excluded: Name, Ticket, Cabin",
                            "Persist Trained Model: false",
                        ],
                        "implementation_notes": [
                            "Do not use GaussianNB, raw Name/Ticket/Cabin one-hot encoding, ensembling, leaderboard submission, or human review.",
                        ],
                        "success_criteria": [
                            "Validation accuracy is recorded under an accuracy key.",
                            "submission.csv contains PassengerId and Survived with the same row count as test.csv.",
                            "pipeline_summary.json records feature, preprocessing, model, split, and seed choices.",
                        ],
                        "expected_outputs": [
                            "metrics.json",
                            "submission.csv",
                            "pipeline_summary.json",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_api_request.json").write_text("{}", encoding="utf-8")
            (trial / "workspace_coding_api_response.json").write_text("{}", encoding="utf-8")

            with patch("research_agent.paths.project_root", return_value=root):
                result = organize_trial_artifacts(
                    "demo",
                    "trial_001",
                    low_cost_user_summary=True,
                    allow_api=True,
                    low_cost_summary_client=self._SummaryClient(),
                    plan_translation_client=self._UnavailableTranslationClient(),
                )

            self.assertEqual("demo", result["competition"])
            self.assertEqual(0.8, result["local_score"])
            self.assertTrue((trial / "README.md").exists())
            self.assertTrue((trial / "internal" / "artifact_manifest.json").exists())
            self.assertFalse((trial / "workspace_coding_api_request.json").exists())
            self.assertFalse((trial / "workspace_coding_api_response.json").exists())
            self.assertTrue((trial / "debug" / "workspace_coding_api_request.json").exists())
            self.assertTrue((trial / "debug" / "workspace_coding_api_response.json").exists())
            self.assertFalse((trial / "workspace_run.json").exists())
            self.assertTrue((trial / "internal" / "workspace_run.json").exists())
            self.assertTrue((trial / "internal" / "pipeline_structure.json").exists())
            self.assertTrue(trial_artifact_exists(trial, "workspace_run.json"))
            self.assertTrue((trial / "user_view" / "01_plan.ko.md").exists())
            self.assertTrue((trial / "user_view" / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((trial / "user_view" / "03_scores.ko.md").exists())
            self.assertEqual(
                {"01_plan.ko.md", "02_pipeline_structure.ko.md", "03_scores.ko.md"},
                {path.name for path in (trial / "user_view").iterdir()},
            )
            self.assertTrue((trial / "internal" / "low_cost_user_summary_card.json").exists())
            self.assertEqual("ready", result["low_cost_user_summary"]["status"])
            token_usage = (root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8")
            self.assertIn('"call_type": "user_summary_card"', token_usage)
            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            self.assertEqual("1.0", structure["schema_version"])
            self.assertTrue(any(stage["id"] == "data_split_cv" for stage in structure["stages"]))
            data_split = next(stage for stage in structure["stages"] if stage["id"] == "data_split_cv")
            preprocessing = next(stage for stage in structure["stages"] if stage["id"] == "preprocessing")
            model = next(stage for stage in structure["stages"] if stage["id"] == "model_definition")
            features = next(stage for stage in structure["stages"] if stage["id"] == "feature_representation")
            augmentation = next(stage for stage in structure["stages"] if stage["id"] == "data_augmentation")
            self.assertTrue(any("StratifiedKFold" in item for item in data_split["actual_applied"]))
            self.assertTrue(any("StandardScaler" in item for item in preprocessing["actual_applied"]))
            self.assertTrue(any("LogisticRegression" in item for item in model["actual_applied"]))
            self.assertEqual("stratified_5_fold_cv", data_split["structured_details"]["method"])
            self.assertTrue(
                any(
                    step["operation"] == "StandardScaler"
                    for step in preprocessing["structured_details"]["steps"]
                )
            )
            self.assertTrue(
                any(
                    item["name"] == "FamilySize"
                    for item in features["structured_details"]["derived_features"]
                )
            )
            self.assertFalse(augmentation["included"])
            browse = root / "runs" / "demo" / "trial_001"
            self.assertTrue((browse / "01_plan.ko.md").exists())
            self.assertTrue((browse / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((browse / "03_scores.ko.md").exists())
            self.assertEqual(
                {"01_plan.ko.md", "02_pipeline_structure.ko.md", "03_scores.ko.md"},
                {path.name for path in browse.iterdir()},
            )
            self.assertTrue((root / "runs" / "demo" / "README.ko.md").exists())
            user_plan = (browse / "01_plan.ko.md").read_text(encoding="utf-8")
            user_structure = (browse / "02_pipeline_structure.ko.md").read_text(encoding="utf-8")
            self.assertIn("## 목적", user_plan)
            self.assertNotIn("평가: `cv_score`", user_plan)
            self.assertIn("재현 가능한 baseline을 만들고 제출 파일 형식을 검증합니다", user_plan)
            self.assertIn("## 왜 하는가", user_plan)
            self.assertIn("검증 정확도가 `accuracy` 계열 키로 기록됩니다.", user_plan)
            self.assertIn("`outputs/pipeline_summary.json`", user_plan)
            self.assertNotIn("Do not use", user_plan)
            self.assertNotIn("Validation accuracy is recorded", user_plan)
            self.assertIn("파이프라인 구조", user_structure)
            self.assertIn("## 단계별 실행 구조", user_structure)
            self.assertIn("전처리", user_structure)
            self.assertIn("검증 분리 / CV", user_structure)
            self.assertIn("LogisticRegression", user_structure)
            self.assertIn("테스트 추론과 제출 파일 생성", user_structure)
            self.assertIn("관련 코드", user_structure)
            self.assertNotIn("다음 개선축", user_structure)

    def test_pipeline_structure_extracts_random_forest_features_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            project = root / "workspace"
            (project / "src").mkdir(parents=True)
            (project / "workspace_config.json").write_text(
                json.dumps(
                    {
                        "target_column": "Survived",
                        "id_column": "PassengerId",
                        "required_data_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (project / "src" / "baseline.py").write_text(
                "\n".join(
                    [
                        "import pandas as pd",
                        "import joblib",
                        "from pathlib import Path",
                        "from sklearn.ensemble import RandomForestClassifier",
                        "from sklearn.impute import SimpleImputer",
                        "from sklearn.model_selection import train_test_split",
                        "from sklearn.preprocessing import OneHotEncoder",
                        "ROOT = Path(__file__).resolve().parents[1]",
                        "DATA_DIR = ROOT / 'data'",
                        "OUTPUT_DIR = ROOT / 'outputs'",
                        "MODEL_PATH = ROOT / 'src' / 'titanic_random_forest_pipeline.joblib'",
                        "TARGET_COLUMN = 'Survived'",
                        "ID_COLUMN = 'PassengerId'",
                        "RANDOM_STATE = 42",
                        "NUMERIC_FEATURES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'is_alone']",
                        "CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Title']",
                        "FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES",
                        "train_df = pd.read_csv(DATA_DIR / 'train.csv')",
                        "test_df = pd.read_csv(DATA_DIR / 'test.csv')",
                        "train_df['family_size'] = train_df['SibSp'] + train_df['Parch'] + 1",
                        "train_df['is_alone'] = (train_df['family_size'] == 1).astype(int)",
                        "train_df['Title'] = train_df['Name'].map(extract_title)",
                        "SimpleImputer(strategy='median')",
                        "SimpleImputer(strategy='most_frequent')",
                        "OneHotEncoder(handle_unknown='ignore')",
                        "train_test_split(train_df, train_df[TARGET_COLUMN], test_size=0.2, stratify=train_df[TARGET_COLUMN], random_state=RANDOM_STATE)",
                        "model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)",
                        "model.fit(train_df, train_df[TARGET_COLUMN])",
                        "joblib.dump({'model': model}, MODEL_PATH)",
                        "submission = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: model.predict(test_df).astype(int)})",
                        "submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)",
                        "def extract_title(name): return 'Mr'",
                    ]
                ),
                encoding="utf-8",
            )
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            simple_yaml.dump({"project_root": str(project)}, comp / "execution_profile.yaml")
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.8268,
                        "validation_accuracy": 0.8268,
                        "metric": "accuracy",
                        "objective": "maximize",
                        "model": "RandomForestClassifier",
                        "n_estimators": 300,
                        "max_depth": 6,
                        "min_samples_leaf": 2,
                        "random_state": 42,
                        "feature_columns": [
                            "Pclass",
                            "Age",
                            "SibSp",
                            "Parch",
                            "Fare",
                            "family_size",
                            "is_alone",
                            "Sex",
                            "Embarked",
                            "Title",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(json.dumps({"status": "collected"}), encoding="utf-8")
            (trial / "workspace_run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["src/baseline.py"]}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                organize_trial_artifacts("demo", "trial_001")

            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            by_id = {stage["id"]: stage for stage in structure["stages"]}
            rendered = "\n".join(
                item
                for stage in by_id.values()
                for item in stage.get("actual_applied", [])
            )
            self.assertIn("Survived", rendered)
            self.assertIn("PassengerId", rendered)
            self.assertIn("RandomForestClassifier", rendered)
            self.assertIn("n_estimators=300", rendered)
            self.assertIn("random_state=42", rendered)
            self.assertIn("family_size", rendered)
            self.assertIn("is_alone", rendered)
            self.assertIn("src/titanic_random_forest_pipeline.joblib", rendered)
            self.assertNotIn("`id`, `target`", rendered)
            preprocessing = by_id["preprocessing"]["structured_details"]
            features = by_id["feature_representation"]["structured_details"]
            split = by_id["data_split_cv"]["structured_details"]
            model = by_id["model_definition"]["structured_details"]
            self.assertEqual(
                ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Name"],
                preprocessing["raw_feature_columns"],
            )
            self.assertTrue(any(step["operation"] == "OneHotEncoder" for step in preprocessing["steps"]))
            self.assertTrue(any(item["name"] == "family_size" for item in features["derived_features"]))
            self.assertEqual("holdout_train_test_split", split["method"])
            self.assertEqual("0.2", split["test_size"])
            self.assertEqual("RandomForestClassifier", model["estimator"])
            self.assertEqual("300", model["parameters"]["n_estimators"])

    def test_pipeline_structure_extracts_regression_time_split_and_imported_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "bike" / "trial_001"
            trial.mkdir(parents=True)
            project = root / "workspace"
            (project / "src").mkdir(parents=True)
            (project / "workspace_config.json").write_text(
                json.dumps(
                    {
                        "target_column": "count",
                        "id_column": "datetime",
                        "metric": "rmsle",
                        "objective": "minimize",
                        "required_data_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (project / "src" / "features.py").write_text(
                "\n".join(
                    [
                        "BASE_FEATURES = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']",
                        "DT_FEATURES = ['dt_year', 'dt_month', 'dt_dayofweek', 'dt_hour']",
                        "FEATURE_COLUMNS = BASE_FEATURES + DT_FEATURES",
                    ]
                ),
                encoding="utf-8",
            )
            (project / "train_step.py").write_text(
                "\n".join(
                    [
                        "import pandas as pd",
                        "from joblib import dump",
                        "from sklearn.linear_model import Ridge",
                        "from sklearn.impute import SimpleImputer",
                        "from sklearn.preprocessing import StandardScaler",
                        "df = pd.read_csv('data/train.csv')",
                        "df = df.sort_values('datetime').reset_index(drop=True)",
                        "split_at = max(1, int(len(df) * 0.8))",
                        "X_train, X_valid = X.iloc[:split_at], X.iloc[split_at:]",
                        "# fallback when the dataset is too small",
                        "model = Ridge(alpha=1.0, fit_intercept=True)",
                        "SimpleImputer(strategy='median')",
                        "StandardScaler()",
                        "model.fit(X_train, y_train)",
                        "dump(model, outputs / 'model.joblib')",
                    ]
                ),
                encoding="utf-8",
            )
            comp = root / "competitions" / "bike"
            comp.mkdir(parents=True)
            simple_yaml.dump({"project_root": str(project)}, comp / "execution_profile.yaml")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 1.0086, "metric": "rmsle", "objective": "minimize"}),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(json.dumps({"status": "collected"}), encoding="utf-8")
            (trial / "workspace_run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["train_step.py", "src/features.py"]}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                organize_trial_artifacts("bike", "trial_001")

            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            by_id = {stage["id"]: stage for stage in structure["stages"]}
            self.assertEqual("Ridge", by_id["model_definition"]["structured_details"]["estimator"])
            self.assertEqual("1.0", by_id["model_definition"]["structured_details"]["parameters"]["alpha"])
            self.assertEqual("time_ordered_holdout", by_id["data_split_cv"]["structured_details"]["method"])
            self.assertEqual(0.2, by_id["data_split_cv"]["structured_details"]["test_size"])
            self.assertEqual(
                [
                    "season",
                    "holiday",
                    "workingday",
                    "weather",
                    "temp",
                    "atemp",
                    "humidity",
                    "windspeed",
                    "dt_year",
                    "dt_month",
                    "dt_dayofweek",
                    "dt_hour",
                ],
                by_id["feature_representation"]["structured_details"]["final_feature_columns"],
            )
            self.assertFalse(by_id["dataset_dataloader"]["included"])
            self.assertEqual("joblib.dump", by_id["model_checkpoint"]["structured_details"]["writer"])
            self.assertEqual(
                "outputs/model.joblib",
                by_id["model_checkpoint"]["structured_details"]["checkpoint_path"],
            )

    def test_pipeline_structure_prefers_runtime_feature_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            project = root / "workspace"
            (project / "src").mkdir(parents=True)
            (project / "workspace_config.json").write_text(
                json.dumps(
                    {
                        "target_column": "Survived",
                        "id_column": "PassengerId",
                        "required_data_files": ["train.csv", "test.csv"],
                    }
                ),
                encoding="utf-8",
            )
            (project / "src" / "baseline.py").write_text(
                "\n".join(
                    [
                        "import pickle",
                        "from pathlib import Path",
                        "from sklearn.impute import SimpleImputer",
                        "from sklearn.linear_model import LogisticRegression",
                        "from sklearn.model_selection import train_test_split",
                        "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
                        "OUTPUTS_DIR = Path('outputs')",
                        "MODEL_PATH = OUTPUTS_DIR / 'model.pkl'",
                        "NUMERIC_FEATURE_CANDIDATES = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']",
                        "CATEGORICAL_FEATURE_CANDIDATES = ['Sex', 'Embarked']",
                        "SimpleImputer(strategy='median')",
                        "SimpleImputer(strategy='most_frequent')",
                        "StandardScaler()",
                        "OneHotEncoder(handle_unknown='ignore')",
                        "train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)",
                        "model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)",
                        "pickle.dump({'pipeline': model}, open(MODEL_PATH, 'wb'))",
                    ]
                ),
                encoding="utf-8",
            )
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            simple_yaml.dump({"project_root": str(project)}, comp / "execution_profile.yaml")
            (trial / "metrics.json").write_text(
                json.dumps(
                    {
                        "cv_score": 0.8,
                        "validation_accuracy": 0.8,
                        "metric": "accuracy",
                        "objective": "maximize",
                        "model_type": "logistic_regression",
                        "features": [
                            "Pclass",
                            "Age",
                            "SibSp",
                            "Parch",
                            "Fare",
                            "FamilySize",
                            "IsAlone",
                            "Sex",
                            "Embarked",
                        ],
                        "numeric_features": [
                            "Pclass",
                            "Age",
                            "SibSp",
                            "Parch",
                            "Fare",
                            "FamilySize",
                            "IsAlone",
                        ],
                        "categorical_features": ["Sex", "Embarked"],
                        "preprocessing": {
                            "numeric_imputation": "median",
                            "categorical_imputation": "most_frequent",
                            "categorical_encoding": "one_hot_handle_unknown_ignore",
                            "numeric_scaling": "standard_scaler",
                        },
                        "split_strategy": "deterministic_stratified_split",
                        "random_state": 42,
                    }
                ),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(json.dumps({"status": "collected"}), encoding="utf-8")
            (trial / "workspace_run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["src/baseline.py"]}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                organize_trial_artifacts("demo", "trial_001")

            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            by_id = {stage["id"]: stage for stage in structure["stages"]}
            preprocessing = by_id["preprocessing"]["structured_details"]
            split = by_id["data_split_cv"]["structured_details"]
            checkpoint = by_id["model_checkpoint"]["structured_details"]
            user_structure = (root / "runs" / "demo" / "trial_001" / "02_pipeline_structure.ko.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"],
                preprocessing["numeric_features"],
            )
            self.assertEqual(["Sex", "Embarked"], preprocessing["categorical_features"])
            self.assertTrue(any(step["columns"] for step in preprocessing["steps"]))
            self.assertEqual("deterministic_stratified_split", split["split_strategy"])
            self.assertEqual("pickle.dump", checkpoint["writer"])
            self.assertEqual("outputs/model.pkl", checkpoint["checkpoint_path"])
            self.assertIn("Pclass", user_structure)
            self.assertIn("Sex", user_structure)
            self.assertIn("pickle", json.dumps(checkpoint))

    def test_pipeline_submission_columns_reads_nested_submission_dict(self):
        # pipeline_summary.json writers nest the schema under
        # {"submission": {"columns": [...]}}; a bare top-level
        # "submission_columns" key must still work for callers that already
        # flattened it.
        self.assertEqual(
            ["id", "x", "y", "z"],
            _pipeline_submission_columns({"submission": {"columns": ["id", "x", "y", "z"]}}),
        )
        self.assertEqual(
            ["id", "target"],
            _pipeline_submission_columns({"submission_columns": ["id", "target"]}),
        )
        self.assertIsNone(_pipeline_submission_columns({}))
        self.assertIsNone(_pipeline_submission_columns({"submission": {}}))

    def test_extract_pipeline_details_uses_real_multi_column_submission_schema(self):
        # Regression test: this pipeline_summary shape (nested "submission":
        # {"columns": [...]}) previously missed the lookup entirely and fell
        # back to a hardcoded "target" prediction column even though the
        # real submission wrote id,x,y,z -- which fed a false "trial emitted
        # id,target" claim into later planning.
        summary = {
            "project_root": "/does/not/matter",
            "metrics": {},
        }
        with patch(
            "research_agent.trial_artifacts._workspace_pipeline_summary",
            return_value={"submission": {"columns": ["id", "x", "y", "z"]}},
        ):
            details = _extract_pipeline_details(summary, code_files=[], code_text="")

        self.assertEqual("x,y,z", details["test_inference_output"]["prediction_column"])
        self.assertEqual("x,y,z", details["data_load"]["target_column"])

    def test_extract_pipeline_facts_does_not_fabricate_a_target_column(self):
        # Regression test for the fabricated "id,target" claim: with no
        # explicit target_column anywhere and a real multi-column submission
        # schema, the data_load fact must describe the real columns, not
        # assert a "target" column that was never detected.
        summary = {"project_root": "/does/not/matter", "metrics": {}}
        with patch(
            "research_agent.trial_artifacts._workspace_pipeline_summary",
            return_value={"submission": {"columns": ["id", "x", "y", "z"]}},
        ):
            facts = _extract_pipeline_facts(summary, code_files=[], code_text="")

        joined = "\n".join(facts["data_load"])
        self.assertIn("학습 타깃은 `x,y,z`", joined)
        self.assertNotIn("학습 타깃은 `target`", joined)

    def test_organize_trial_artifacts_prefers_internal_code_snapshot_over_current_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            internal = trial / "internal"
            snapshot = internal / "code_snapshot" / "src"
            snapshot.mkdir(parents=True)
            (snapshot / "baseline.py").write_text(
                "from sklearn.ensemble import RandomForestClassifier\n"
                "FEATURES = ['Pclass', 'Sex']\n"
                "model = RandomForestClassifier(n_estimators=200, random_state=42)\n",
                encoding="utf-8",
            )
            (internal / "code_snapshot_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "snapshot_dir": "internal/code_snapshot",
                        "files": [{"path": "src/baseline.py"}],
                    }
                ),
                encoding="utf-8",
            )
            project = root / "workspace"
            (project / "src").mkdir(parents=True)
            (project / "src" / "baseline.py").write_text(
                "from sklearn.linear_model import LogisticRegression\n"
                "FEATURES = ['CabinKnown']\n"
                "model = LogisticRegression(random_state=42)\n",
                encoding="utf-8",
            )
            comp = root / "competitions" / "demo"
            comp.mkdir(parents=True)
            simple_yaml.dump({"project_root": str(project)}, comp / "execution_profile.yaml")
            (trial / "metrics.json").write_text(
                json.dumps({"cv_score": 0.8, "metric": "accuracy", "objective": "maximize"}),
                encoding="utf-8",
            )
            (trial / "metrics_collection.json").write_text(json.dumps({"status": "collected"}), encoding="utf-8")
            (trial / "workspace_run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps({"changed_files": ["src/baseline.py"]}),
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                organize_trial_artifacts("demo", "trial_001")

            self.assertFalse((root / "runs" / "demo" / "trial_001" / "code").exists())
            structure = json.loads((trial / "internal" / "pipeline_structure.json").read_text(encoding="utf-8"))
            structure_text = json.dumps(structure)
            self.assertIn("RandomForestClassifier", structure_text)
            self.assertNotIn("LogisticRegression", structure_text)


if __name__ == "__main__":
    unittest.main()
