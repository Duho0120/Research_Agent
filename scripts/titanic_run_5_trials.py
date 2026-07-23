"""Deprecated Titanic-only manual experiment script.

This helper is kept for reproducing archived prototype trials only. New agent
experiments should run through generic_workspace_auto_loop.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "demo_workspaces" / "titanic"
DATA_DIR = WORKSPACE / "data"
RUN_DIR = WORKSPACE / "manual_trials"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "gender_submission.csv"
ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"
RANDOM_SEED = 42
VALIDATION_SIZE = 0.2


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    change_axis: str
    feature_mode: str
    model_name: str
    model: Any


TRIALS = [
    TrialSpec(
        trial_id="trial_001",
        change_axis="baseline_logistic_regression",
        feature_mode="baseline",
        model_name="LogisticRegression",
        model=LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED),
    ),
    TrialSpec(
        trial_id="trial_002",
        change_axis="feature_engineering_family_structure",
        feature_mode="family",
        model_name="LogisticRegression",
        model=LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED),
    ),
    TrialSpec(
        trial_id="trial_003",
        change_axis="feature_engineering_name_title",
        feature_mode="title",
        model_name="LogisticRegression",
        model=LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED),
    ),
    TrialSpec(
        trial_id="trial_004",
        change_axis="model_family_random_forest",
        feature_mode="title_cabin",
        model_name="RandomForestClassifier",
        model=RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    ),
    TrialSpec(
        trial_id="trial_005",
        change_axis="model_family_gradient_boosting",
        feature_mode="title_cabin",
        model_name="GradientBoostingClassifier",
        model=GradientBoostingClassifier(random_state=RANDOM_SEED),
    ),
]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_title(title: str) -> str:
    common = {"Mr", "Miss", "Mrs", "Master"}
    if title in common:
        return title
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return "Rare"


def add_features(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    enriched = df.copy()
    if mode in {"family", "title", "title_cabin"}:
        family_size = enriched["SibSp"].fillna(0) + enriched["Parch"].fillna(0) + 1
        enriched["FamilySize"] = family_size
        enriched["IsAlone"] = (family_size == 1).astype(int)
    if mode in {"title", "title_cabin"}:
        titles = enriched["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Rare")
        enriched["Title"] = titles.map(normalize_title)
    if mode == "title_cabin":
        enriched["CabinKnown"] = enriched["Cabin"].notna().astype(int)
        enriched["Deck"] = enriched["Cabin"].fillna("U").astype(str).str[0]
        enriched["FarePerPerson"] = enriched["Fare"] / enriched.get("FamilySize", 1).replace(0, 1)
    return enriched


def feature_columns(mode: str) -> tuple[list[str], list[str]]:
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked"]
    if mode in {"family", "title", "title_cabin"}:
        numeric += ["FamilySize", "IsAlone"]
    if mode in {"title", "title_cabin"}:
        categorical += ["Title"]
    if mode == "title_cabin":
        numeric += ["CabinKnown", "FarePerPerson"]
        categorical += ["Deck"]
    return numeric, categorical


def build_pipeline(spec: TrialSpec) -> Pipeline:
    numeric, categorical = feature_columns(spec.feature_mode)
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", spec.model)])


def run_trial(spec: TrialSpec, submit: bool, poll_seconds: int) -> dict[str, Any]:
    train_df = add_features(pd.read_csv(TRAIN_PATH), spec.feature_mode)
    test_df = add_features(pd.read_csv(TEST_PATH), spec.feature_mode)
    sample = pd.read_csv(SAMPLE_PATH)
    numeric, categorical = feature_columns(spec.feature_mode)
    features = numeric + categorical

    x_train, x_valid, y_train, y_valid = train_test_split(
        train_df[features],
        train_df[TARGET_COLUMN].astype(int),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_df[TARGET_COLUMN].astype(int),
    )

    pipeline = build_pipeline(spec)
    pipeline.fit(x_train, y_train)
    valid_pred = pipeline.predict(x_valid).astype(int)
    local_score = float(accuracy_score(y_valid, valid_pred))

    full_pipeline = build_pipeline(spec)
    full_pipeline.fit(train_df[features], train_df[TARGET_COLUMN].astype(int))
    test_pred = full_pipeline.predict(test_df[features]).astype(int)

    trial_dir = RUN_DIR / spec.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    submission_path = trial_dir / "submission.csv"
    submission = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: test_pred})
    submission = submission[list(sample.columns)]
    submission.to_csv(submission_path, index=False)

    metrics = {
        "trial_id": spec.trial_id,
        "local_score": local_score,
        "metric": "accuracy",
        "objective": "maximize",
        "change_axis": spec.change_axis,
        "feature_mode": spec.feature_mode,
        "model": spec.model_name,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "submission_file": str(submission_path),
        "kaggle_submitted": False,
        "kaggle_returncode": None,
        "kaggle_lb_score": None,
        "kaggle_rank": None,
    }

    if submit:
        message = f"titanic {spec.trial_id} local CV {local_score:.6f} axis {spec.change_axis}"
        command = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "titanic",
            "-f",
            str(submission_path),
            "-m",
            message,
        ]
        print("Submitting:", " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True)
        metrics["kaggle_submitted"] = completed.returncode == 0
        metrics["kaggle_returncode"] = completed.returncode
        if poll_seconds > 0:
            time.sleep(poll_seconds)
            subprocess.run(["kaggle", "competitions", "submissions", "-c", "titanic"], cwd=ROOT, text=True)

    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_user_artifacts(trial_dir, spec, metrics, numeric, categorical)
    return metrics


def write_user_artifacts(
    trial_dir: Path,
    spec: TrialSpec,
    metrics: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
) -> None:
    user_dir = trial_dir / "user_view"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "01_plan.ko.md").write_text(render_user_plan(spec), encoding="utf-8")
    (user_dir / "02_pipeline_structure.ko.md").write_text(
        render_user_pipeline(spec, metrics, numeric, categorical),
        encoding="utf-8",
    )
    (user_dir / "03_scores.ko.md").write_text(render_user_scores(metrics), encoding="utf-8")


def render_user_plan(spec: TrialSpec) -> str:
    purpose = {
        "trial_001": "첫 회차 기준선으로 사용할 재현 가능한 baseline을 만들고 제출 파일 형식을 검증합니다.",
        "trial_002": "trial_001 기준선에서 가족 구조 피처만 추가해 성능 변화가 있는지 확인합니다.",
        "trial_003": "trial_002까지의 구조 위에 이름에서 추출한 Title 피처를 추가해 성능 개선 여부를 확인합니다.",
        "trial_004": "동일한 피처 세트에서 모델 family를 RandomForest로 바꿔 비선형 모델 효과를 확인합니다.",
        "trial_005": "동일한 피처 세트에서 GradientBoosting을 적용해 boosting 계열 모델 효과를 확인합니다.",
    }[spec.trial_id]
    rationale = {
        "trial_001": "복잡한 피처나 모델 탐색 전에 안정적인 전처리, 검증 split, 제출 형식을 먼저 고정해야 이후 trial과 공정하게 비교할 수 있습니다.",
        "trial_002": "Titanic에서는 동승 가족 수와 단독 탑승 여부가 생존 확률과 관련될 수 있으므로, 모델은 그대로 두고 피처만 바꿔 효과를 분리합니다.",
        "trial_003": "이름의 호칭은 성별/나이 외의 사회적 정보와 가족 역할을 일부 담을 수 있어, family 피처 다음의 자연스러운 피처 확장입니다.",
        "trial_004": "피처가 늘어난 뒤에는 선형 모델이 포착하지 못한 상호작용을 tree 계열 모델이 잡는지 비교할 수 있습니다.",
        "trial_005": "RandomForest와 다른 boosting 방식의 일반화 성능을 비교해 모델 family 선택 근거를 확보합니다.",
    }[spec.trial_id]
    base = {
        "trial_001": "-",
        "trial_002": "trial_001",
        "trial_003": "trial_002",
        "trial_004": "trial_003",
        "trial_005": "trial_003",
    }[spec.trial_id]
    changed = {
        "baseline": "기본 수치형/범주형 피처와 LogisticRegression baseline 구성",
        "family": "`FamilySize`, `IsAlone` 추가",
        "title": "`FamilySize`, `IsAlone`, `Title` 추가",
        "title_cabin": "`Title`, `CabinKnown`, `Deck`, `FarePerPerson`까지 사용",
    }[spec.feature_mode]
    return "\n".join(
        [
            f"# {spec.trial_id} 실험 계획",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            "| 대회 | `titanic` |",
            "| 평가 지표 | `accuracy` |",
            "| 목표 방향 | `maximize` |",
            f"| 기준 trial | {base} |",
            f"| 개선축 | {spec.change_axis} |",
            f"| 모델 | {spec.model_name} |",
            "",
            "## 목적",
            "",
            f"- {purpose}",
            "",
            "## 왜 하는가",
            "",
            f"- {rationale}",
            "",
            "## 이번에 바꾸는 것",
            "",
            f"- {changed}",
            "",
            "## 그대로 두는 것",
            "",
            "- 같은 train/validation split, 같은 seed, 같은 accuracy 지표를 사용합니다.",
            "- 제출 형식은 `PassengerId`, `Survived` 두 컬럼으로 유지합니다.",
            "",
            "## 다음에 볼 것",
            "",
            "- `02_pipeline_structure.ko.md`: 데이터부터 제출 파일까지의 실행 흐름",
            "- `03_scores.ko.md`: 로컬 점수와 Kaggle 제출 점수",
            "",
        ]
    )


def render_user_pipeline(
    spec: TrialSpec,
    metrics: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
) -> str:
    return "\n".join(
        [
            f"# {spec.trial_id} 파이프라인 구조",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| workspace | `{WORKSPACE}` |",
            "| 평가 지표 | `accuracy` |",
            f"| 로컬 점수 | `{metrics['local_score']}` |",
            f"| 모델 | `{spec.model_name}` |",
            f"| 제출 파일 | `{metrics['submission_file']}` |",
            "",
            "## 실행 흐름",
            "",
            "1. 데이터 로드",
            "2. trial별 파생 피처 생성",
            "3. 수치형 결측치 대체와 스케일링",
            "4. 범주형 결측치 대체와 one-hot 인코딩",
            "5. stratified holdout 검증 분리",
            "6. 모델 학습",
            "7. 로컬 accuracy 계산",
            "8. 전체 train 재학습 후 test 예측",
            "9. `submission.csv` 생성",
            "",
            "## 핵심 구성",
            "",
            f"- 수치형 피처: {inline_list(numeric)}",
            f"- 범주형 피처: {inline_list(categorical)}",
            "- 검증 방식: `train_test_split(test_size=0.2, stratify=Survived, random_state=42)`",
            "- 제출 형식: `PassengerId`, `Survived`",
            "",
        ]
    )


def render_user_scores(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {metrics['trial_id']} 점수",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            "| 상태 | completed |",
            "| 지표 | accuracy |",
            "| 목표 방향 | maximize |",
            f"| 로컬 점수 | {metrics['local_score']} |",
            f"| 제출 상태 | {'기록됨' if metrics.get('kaggle_submitted') else '미제출'} |",
            f"| 제출 LB 점수 | {metrics.get('kaggle_lb_score') or '-'} |",
            f"| 제출 순위 | {metrics.get('kaggle_rank') or '-'} |",
            f"| 제출 파일 | {metrics['submission_file']} |",
            "",
        ]
    )


def inline_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "-"


def wait_for_leaderboard_input(metrics: dict[str, Any]) -> None:
    print("\nKaggle submission was sent for", metrics["trial_id"])
    print("Check the Kaggle leaderboard/submissions result, then enter the public LB score.")
    print("Leave blank only if the score is not available yet.")
    lb_score = input("Public LB score: ").strip()
    rank = input("Rank (optional): ").strip()
    notes = input("Notes (optional): ").strip()
    if lb_score:
        try:
            metrics["kaggle_lb_score"] = float(lb_score)
        except ValueError:
            metrics["kaggle_lb_score"] = lb_score
    if rank:
        try:
            metrics["kaggle_rank"] = int(rank)
        except ValueError:
            metrics["kaggle_rank"] = rank
    if notes:
        metrics["kaggle_notes"] = notes
    trial_dir = Path(metrics["submission_file"]).parent
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, Any]]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial_id",
                "local_score",
                "change_axis",
                "feature_mode",
                "model",
                "submission_file",
                "kaggle_submitted",
                "kaggle_returncode",
                "kaggle_lb_score",
                "kaggle_rank",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(f"\nSummary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Titanic manual trials 001-005 and optionally submit each to Kaggle.")
    parser.add_argument("--submit", action="store_true", help="Submit each generated submission.csv with Kaggle CLI.")
    parser.add_argument("--wait-for-lb", action="store_true", help="Pause after each submission and require LB score entry before continuing.")
    parser.add_argument("--start", default="trial_001", choices=[trial.trial_id for trial in TRIALS])
    parser.add_argument("--end", default="trial_005", choices=[trial.trial_id for trial in TRIALS])
    parser.add_argument("--poll-seconds", type=int, default=0, help="Seconds to wait before printing Kaggle submissions list after each submit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial_ids = [trial.trial_id for trial in TRIALS]
    start_index = trial_ids.index(args.start)
    end_index = trial_ids.index(args.end)
    if start_index > end_index:
        raise SystemExit("--start must be before or equal to --end")

    rows = []
    for spec in TRIALS[start_index : end_index + 1]:
        print(f"\n=== {spec.trial_id}: {spec.change_axis} ===", flush=True)
        metrics = run_trial(spec, submit=args.submit, poll_seconds=args.poll_seconds)
        if args.submit and args.wait_for_lb:
            wait_for_leaderboard_input(metrics)
        rows.append(metrics)
        print(f"local_score={metrics['local_score']:.12f}")
        print(f"submission={metrics['submission_file']}")
        if not args.submit:
            print(
                "submit_command="
                f'kaggle competitions submit -c titanic -f "{metrics["submission_file"]}" '
                f'-m "titanic {spec.trial_id} local CV {metrics["local_score"]:.6f}"'
            )

    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
