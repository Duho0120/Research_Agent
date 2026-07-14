import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.trial_artifacts import organize_trial_artifacts, trial_artifact_exists


class TrialArtifactsTest(unittest.TestCase):
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
            (trial / "workspace_coding_api_request.json").write_text("{}", encoding="utf-8")
            (trial / "workspace_coding_api_response.json").write_text("{}", encoding="utf-8")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = organize_trial_artifacts("demo", "trial_001")

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
            self.assertTrue((trial / "user_view" / "README.ko.md").exists())
            self.assertTrue((trial / "user_view" / "01_plan.ko.md").exists())
            self.assertTrue((trial / "user_view" / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((trial / "user_view" / "03_code_pipeline.ko.md").exists())
            self.assertTrue((trial / "user_view" / "04_result.ko.md").exists())
            self.assertTrue((trial / "user_view" / "05_submission.ko.md").exists())
            user_readme = (trial / "user_view" / "README.ko.md").read_text(encoding="utf-8")
            self.assertIn("사용자가 바로 확인할 핵심 산출물", user_readme)
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
            self.assertTrue((browse / "README.ko.md").exists())
            self.assertTrue((browse / "01_plan.ko.md").exists())
            self.assertTrue((browse / "02_pipeline_structure.ko.md").exists())
            self.assertTrue((browse / "03_code_pipeline.ko.md").exists())
            self.assertTrue((browse / "04_result.ko.md").exists())
            self.assertTrue((browse / "05_submission.ko.md").exists())
            self.assertTrue((browse / "06_paths.ko.md").exists())
            self.assertTrue((root / "runs" / "demo" / "README.ko.md").exists())
            user_plan = (browse / "01_plan.ko.md").read_text(encoding="utf-8")
            user_structure = (browse / "02_pipeline_structure.ko.md").read_text(encoding="utf-8")
            self.assertIn("이번 실험의 핵심 선택", user_plan)
            self.assertNotIn("평가: `cv_score`", user_plan)
            self.assertIn("| 단계 | 실제 적용 내용 | 확인 포인트 |", user_structure)
            self.assertIn("## 단계별 체크", user_structure)
            self.assertIn("StratifiedKFold", user_structure)
            self.assertIn("StandardScaler", user_structure)
            self.assertIn("LogisticRegression", user_structure)
            self.assertIn("파생변수", user_structure)
            self.assertNotIn("다음 개선축", user_structure)
            browse_paths = (browse / "06_paths.ko.md").read_text(encoding="utf-8")
            self.assertIn("원본 trial 기록", browse_paths)

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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
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

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
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


if __name__ == "__main__":
    unittest.main()
