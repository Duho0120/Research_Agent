from __future__ import annotations

import json
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from . import paths, simple_yaml
from .paths import competition_dir, trial_dir
from .store import read_text, write_text


DEBUG_FILE_NAMES = {
    "demo_plan_api_request.json",
    "demo_plan_api_response.json",
    "workspace_coding_api_request.json",
    "workspace_coding_api_response.json",
    "workspace_coding_agent_request.md",
}

INTERNAL_FILE_NAMES = {
    "demo_context.json",
    "demo_cycle_record.json",
    "demo_experiment_plan.json",
    "demo_one_cycle.json",
    "metrics_collection.json",
    "workspace_coding_handoff.json",
    "workspace_coding_result.json",
    "workspace_coding_result_validation.json",
    "workspace_run.json",
}


def trial_artifact_path(out_dir: Path, name: str) -> Path:
    root_path = out_dir / name
    if root_path.exists():
        return root_path
    return out_dir / "internal" / name


def trial_artifact_exists(out_dir: Path, name: str) -> bool:
    return trial_artifact_path(out_dir, name).exists()


def read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    return _read_json(trial_artifact_path(out_dir, name))


def organize_trial_artifacts(competition: str, trial_id: str) -> dict[str, Any]:
    """Write a compact human-facing README and tuck obvious debug artifacts away."""
    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_trial_summary(competition, trial_id)
    pipeline_structure = build_pipeline_structure(out_dir, summary)
    moved_debug_files = _move_debug_files(out_dir)
    moved_internal_files = _move_internal_files(out_dir)
    summary["moved_debug_files"] = moved_debug_files
    summary["moved_internal_files"] = moved_internal_files
    summary["pipeline_structure_file"] = "internal/pipeline_structure.json"
    internal_dir = out_dir / "internal"
    write_text(
        internal_dir / "pipeline_structure.json",
        json.dumps(pipeline_structure, ensure_ascii=False, indent=2) + "\n",
    )
    summary["user_view_files"] = _write_user_view(out_dir, summary)
    summary["browse_view_files"] = _write_browse_view(competition, trial_id, out_dir, summary)

    write_text(out_dir / "README.md", render_trial_readme(summary))
    write_text(internal_dir / "artifact_manifest.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_trial_summary(competition: str, trial_id: str) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    metrics = _read_json(out_dir / "metrics.json")
    demo_cycle = read_trial_json(out_dir, "demo_one_cycle.json")
    after_coding = read_trial_json(out_dir, "workspace_after_coding_cycle.json")
    demo_record = read_trial_json(out_dir, "demo_cycle_record.json")
    workspace_run = read_trial_json(out_dir, "workspace_run.json")
    code_result = read_trial_json(out_dir, "workspace_coding_result.json")
    metrics_collection = read_trial_json(out_dir, "metrics_collection.json")
    profile = _load_execution_profile(competition)

    status = (
        demo_cycle.get("status")
        or after_coding.get("status")
        or demo_record.get("status")
        or metrics_collection.get("status")
        or "unknown"
    )
    score = metrics.get("cv_score")
    metric = metrics.get("metric") or metrics_collection.get("metric")
    objective = metrics.get("objective") or metrics_collection.get("objective")
    changed_files = demo_record.get("changed_files") or code_result.get("changed_files") or []
    log_paths = demo_record.get("log_paths") or _workspace_log_paths(workspace_run)

    return {
        "competition": competition,
        "trial_id": trial_id,
        "status": status,
        "metric": metric,
        "objective": objective,
        "local_score": score,
        "score_source": metrics_collection.get("score_source") or demo_record.get("score_source"),
        "plan_title": demo_record.get("plan_title") or _read_plan_title(out_dir),
        "changed_files": changed_files,
        "key_files": _key_files(out_dir),
        "log_paths": log_paths,
        "project_root": profile.get("project_root"),
        "workspace_status": workspace_run.get("status"),
        "metrics_collection_status": metrics_collection.get("status"),
    }


def build_pipeline_structure(out_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    code_files = _project_code_files(summary)
    code_text = "\n".join(text for _, text in code_files)
    stage_context = _stage_context(summary, code_files, code_text)
    stages = [
        _stage(
            "imports_setup",
            "Imports & Setup",
            True,
            "모든 실행 파이프라인은 라이브러리 import와 경로/config 초기화가 필요합니다.",
            stage_context,
            role="실험 실행에 필요한 라이브러리, 경로, 설정값을 준비합니다.",
            inputs=["workspace_config.json", "Execution Profile", "Python runtime"],
            outputs=["reproducible runtime context"],
            checks=["필요 패키지가 import 되는지", "입출력 경로가 workspace 내부인지"],
        ),
        _stage(
            "data_load",
            "Data Load",
            True,
            "대회 데이터 또는 사용자 제공 데이터를 읽어야 하므로 항상 포함합니다.",
            stage_context,
            role="원본 train/test 데이터를 읽고 이후 단계에 전달합니다.",
            inputs=["data/train.csv", "data/test.csv", "workspace data directory"],
            outputs=["raw train/test dataframe or dataset objects"],
            checks=["파일 존재 여부", "shape", "column list", "target/id column 확인"],
        ),
        _stage(
            "preprocessing",
            "Preprocessing",
            True,
            "결측치, 범주형/수치형 변환, 스케일링 등 최소 전처리 전략을 명시해야 합니다.",
            stage_context,
            role="원본 데이터를 모델 입력 가능한 형태로 정리합니다.",
            inputs=["raw features"],
            outputs=["clean model-ready features"],
            checks=["결측치 처리 방식", "categorical/numeric feature 분리", "train/test transform 일관성"],
        ),
        _stage(
            "data_split_cv",
            "Data Split / CV Strategy",
            True,
            "검증 점수의 의미를 판단하려면 split 또는 CV 전략이 독립적으로 보여야 합니다.",
            stage_context,
            role="학습/검증 분리 또는 CV 전략을 정의합니다.",
            inputs=["training data", "target"],
            outputs=["train/validation split or fold assignments"],
            checks=["stratify/group/time split 필요 여부", "random seed", "metric과 split의 적합성"],
        ),
        _stage(
            "feature_representation",
            "Feature / Representation Construction",
            _contains_any(code_text, ["feature", "token", "embedding", "representation", "derive", "engineer"]),
            "원본 feature를 그대로 쓰지 않고 파생변수, 토큰, 임베딩 등 표현을 만들 때 포함합니다.",
            stage_context,
            role="모델이 더 잘 학습할 수 있도록 입력 표현을 구성합니다.",
            inputs=["preprocessed features"],
            outputs=["derived features or representations"],
            checks=["새 feature의 의미", "leakage 가능성", "train/test 동일 적용"],
            conditional=True,
        ),
        _stage(
            "data_augmentation",
            "Data Augmentation",
            _contains_any(code_text, ["augment", "augmentation", "transform(", "randomcrop", "flip", "colorjitter"]),
            "이미지, 시퀀스, 딥러닝 학습처럼 무작위 변형이 필요한 경우에만 포함합니다.",
            stage_context,
            role="학습 데이터에 변형을 적용해 일반화를 돕습니다.",
            inputs=["training samples"],
            outputs=["augmented training samples"],
            checks=["검증/테스트에는 augmentation 미적용", "label 보존 여부", "seed/reproducibility"],
            conditional=True,
        ),
        _stage(
            "dataset_dataloader",
            "Dataset / DataLoader",
            _contains_any(code_text, ["dataloader", "dataset", "__getitem__", "torch.utils.data", "tf.data"]),
            "배치 기반 프레임워크를 사용할 때 포함합니다. tabular sklearn baseline에서는 보통 제외됩니다.",
            stage_context,
            role="배치 단위 학습을 위한 데이터 접근 골격을 제공합니다.",
            inputs=["preprocessed samples"],
            outputs=["batch iterator"],
            checks=["Dataset 내부에 전처리/증강 로직을 숨기지 않았는지", "batch shape", "shuffle 설정"],
            conditional=True,
        ),
        _stage(
            "model_definition",
            "Model Definition",
            True,
            "모든 실험은 어떤 모델을 학습하는지 명확히 정의해야 합니다.",
            stage_context,
            role="모델 구조 또는 모델 family를 정의합니다.",
            inputs=["model-ready features"],
            outputs=["untrained model or estimator"],
            checks=["모델 family", "주요 hyperparameter", "pretrained/fine-tuning 여부"],
        ),
        _stage(
            "loss_objective",
            "Loss Function / Objective",
            True,
            "딥러닝 loss 또는 sklearn 내부 objective와 대회 metric의 관계를 분리해 기록합니다.",
            stage_context,
            role="학습 최적화 목표와 평가 metric의 관계를 명시합니다.",
            inputs=["model predictions", "target"],
            outputs=["optimization objective"],
            checks=["loss와 metric이 어긋나지 않는지", "class imbalance 처리 필요 여부"],
        ),
        _stage(
            "training",
            "Training",
            True,
            "실험 파이프라인은 모델 학습 단계를 반드시 포함합니다.",
            stage_context,
            role="모델을 학습하고 재현 가능한 학습 결과를 생성합니다.",
            inputs=["train split", "model", "objective"],
            outputs=["trained model"],
            checks=["seed", "fit 호출", "학습 시간", "학습 실패 로그"],
        ),
        _stage(
            "training_curve",
            "Training Curve",
            _contains_any(code_text, ["epoch", "history", "loss_curve", "train_loss", "val_loss"]),
            "epoch 기반 반복 학습일 때 강력히 권장됩니다. 단순 sklearn baseline에서는 보통 제외됩니다.",
            stage_context,
            role="학습 과정의 수렴/과적합 여부를 확인합니다.",
            inputs=["per-epoch training logs"],
            outputs=["curve data or plot"],
            checks=["train/validation divergence", "early stopping 근거"],
            conditional=True,
        ),
        _stage(
            "evaluation",
            "Evaluation",
            True,
            "로컬 검증 점수 없이는 다음 개선 방향을 판단할 수 없습니다.",
            stage_context,
            role="대회 metric 또는 로컬 proxy metric을 계산합니다.",
            inputs=["validation predictions", "validation target"],
            outputs=["metrics.json", "cv_score"],
            checks=["metric key", "objective direction", "score reproducibility"],
        ),
        _stage(
            "model_checkpoint",
            "Model Save / Checkpoint",
            _contains_any(code_text, ["joblib", "pickle", "torch.save", "save_model", ".save("]),
            "다음 trial 재사용, 롤백, 제출 재현이 필요하면 포함합니다.",
            stage_context,
            role="학습된 모델 또는 checkpoint를 저장합니다.",
            inputs=["trained model"],
            outputs=["model artifact"],
            checks=["저장 위치가 outputs/ 또는 허용된 artifact 경로인지", "재로드 가능성"],
            conditional=True,
        ),
        _stage(
            "test_inference_output",
            "Test Inference / Output",
            True,
            "대회 제출 또는 결과 검토를 위해 test 추론과 output 생성 단계가 필요합니다.",
            stage_context,
            role="test set 예측과 제출/출력 파일을 생성합니다.",
            inputs=["test data", "trained model"],
            outputs=["submission.csv or prediction artifact"],
            checks=["id column 보존", "submission format", "row count"],
        ),
    ]
    return {
        "schema_version": "1.0",
        "competition": summary["competition"],
        "trial_id": summary["trial_id"],
        "source": "rule_based_trial_artifact_analysis",
        "metric": summary.get("metric"),
        "objective": summary.get("objective"),
        "project_root": summary.get("project_root"),
        "notes": [
            "실행 코드는 .py 파일을 기준으로 유지합니다.",
            "이 구조 명세는 노트북처럼 위에서 아래로 읽히도록 생성된 사용자/에이전트 공용 컨텍스트입니다.",
            "다음 trial에서는 stages[].included, code_locations, improvement_handles를 참고해 수정 축을 정합니다.",
        ],
        "stages": stages,
    }


def _stage_context(summary: dict[str, Any], code_files: list[tuple[str, str]], code_text: str) -> dict[str, Any]:
    return {
        "summary": summary,
        "code_files": code_files,
        "code_text": code_text,
    }


def _stage(
    stage_id: str,
    name: str,
    included: bool,
    reason: str,
    context: dict[str, Any],
    *,
    role: str,
    inputs: list[str],
    outputs: list[str],
    checks: list[str],
    conditional: bool = False,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "name": name,
        "included": bool(included),
        "required": not conditional,
        "reason": reason if included else f"이번 trial에서는 제외 또는 미탐지: {reason}",
        "role": role,
        "code_locations": _code_locations_for_stage(stage_id, context["code_files"]),
        "inputs": inputs,
        "outputs": outputs,
        "checks": checks,
        "improvement_handles": _improvement_handles(stage_id),
    }


def _project_code_files(summary: dict[str, Any]) -> list[tuple[str, str]]:
    project = summary.get("project_root")
    if not project:
        return []
    root = Path(str(project))
    if not root.is_dir():
        return []
    result: list[tuple[str, str]] = []
    for item in summary.get("changed_files", []):
        relative = _safe_relative_file(item)
        if relative is None:
            continue
        path = root / Path(*relative.parts)
        if not path.is_file() or not _is_text_code_file(path):
            continue
        result.append((relative.as_posix(), read_text(path, default="")[:12000]))
    return result


def _code_locations_for_stage(stage_id: str, code_files: list[tuple[str, str]]) -> list[str]:
    keywords = {
        "imports_setup": ["import ", "from "],
        "data_load": ["read_csv", "read_parquet", "load_data", "data_dir"],
        "preprocessing": ["preprocess", "imputer", "onehot", "scaler", "columntransformer", "pipeline("],
        "data_split_cv": ["train_test_split", "stratified", "kfold", "cross_val", "fold"],
        "feature_representation": ["feature", "token", "embedding", "representation", "derive"],
        "data_augmentation": ["augment", "transform(", "randomcrop", "flip", "colorjitter"],
        "dataset_dataloader": ["dataset", "dataloader", "__getitem__", "torch.utils.data", "tf.data"],
        "model_definition": ["model", "classifier", "regressor", "logisticregression", "randomforest", "xgb", "lgbm"],
        "loss_objective": ["loss", "criterion", "objective", "metric", "accuracy", "roc_auc", "rmse"],
        "training": [".fit(", "fit(", "train", "trainer"],
        "training_curve": ["epoch", "history", "loss_curve", "train_loss", "val_loss"],
        "evaluation": ["accuracy_score", "roc_auc", "f1_score", "mean_squared_error", "cv_score", "metrics"],
        "model_checkpoint": ["joblib", "pickle", "torch.save", "save_model", ".save("],
        "test_inference_output": ["predict", "submission", "to_csv", "id_column"],
    }
    needles = keywords.get(stage_id, [])
    locations: list[str] = []
    for relative, text in code_files:
        lowered = text.lower()
        if any(needle.lower() in lowered for needle in needles):
            locations.append(relative)
    if not locations:
        fallback = {
            "imports_setup": ["train_step.py", "predict_step.py", "test_step.py"],
            "data_load": ["src/", "train_step.py"],
            "preprocessing": ["src/"],
            "data_split_cv": ["train_step.py"],
            "model_definition": ["src/", "train_step.py"],
            "loss_objective": ["train_step.py"],
            "training": ["train_step.py"],
            "evaluation": ["train_step.py"],
            "test_inference_output": ["predict_step.py"],
        }.get(stage_id, [])
        locations.extend(item for item in fallback if _changed_file_matches(code_files, item))
    return _unique(locations)


def _changed_file_matches(code_files: list[tuple[str, str]], candidate: str) -> bool:
    if candidate.endswith("/"):
        return any(relative.startswith(candidate) for relative, _ in code_files)
    return any(relative == candidate for relative, _ in code_files)


def _improvement_handles(stage_id: str) -> list[str]:
    mapping = {
        "data_load": ["data quality", "schema validation"],
        "preprocessing": ["missing value policy", "categorical encoding", "scaling", "leakage prevention"],
        "data_split_cv": ["validation strategy", "CV configuration", "seed stability"],
        "feature_representation": ["feature engineering", "representation learning"],
        "data_augmentation": ["augmentation policy", "generalization"],
        "dataset_dataloader": ["batching", "sampling", "data pipeline speed"],
        "model_definition": ["model family", "architecture", "pretraining"],
        "loss_objective": ["loss/metric alignment", "class imbalance"],
        "training": ["hyperparameters", "optimizer", "runtime"],
        "training_curve": ["overfitting diagnosis", "early stopping"],
        "evaluation": ["metric reliability", "error analysis"],
        "model_checkpoint": ["rollback", "reproducibility"],
        "test_inference_output": ["submission format", "post-processing"],
    }
    return mapping.get(stage_id, [])


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _is_text_code_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
    }


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def render_trial_readme(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['competition']} / {summary['trial_id']}",
        "",
        "## Summary",
        "",
        f"- status: {summary.get('status')}",
        f"- metric: {summary.get('metric')}",
        f"- objective: {summary.get('objective')}",
        f"- local_score: {summary.get('local_score')}",
        f"- score_source: {summary.get('score_source')}",
        f"- plan_title: {summary.get('plan_title')}",
        "",
        "## Key Files",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("key_files", []) or ["None"])
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- {item}" for item in summary.get("changed_files", []) or ["None"])
    lines.extend(["", "## Logs", ""])
    lines.extend(f"- {item}" for item in summary.get("log_paths", []) or ["None"])
    lines.extend(["", "## User View", ""])
    user_view_files = summary.get("user_view_files", [])
    if user_view_files:
        lines.extend(f"- user_view/{item}" for item in user_view_files)
    else:
        lines.append("- Not generated")
    browse_view_files = summary.get("browse_view_files", [])
    lines.extend(["", "## Folder Browsing View", ""])
    if browse_view_files:
        lines.append(f"- runs/{summary['competition']}/{summary['trial_id']}/")
        lines.extend(f"  - {item}" for item in browse_view_files)
    else:
        lines.append("- Not generated")
    moved = summary.get("moved_debug_files", [])
    lines.extend(["", "## Debug Artifacts", ""])
    if moved:
        lines.extend(f"- moved to debug/{Path(item).name}" for item in moved)
    else:
        lines.append("- None moved")
    internal = summary.get("moved_internal_files", [])
    lines.extend(["", "## Internal Artifacts", ""])
    if internal:
        lines.extend(f"- moved to internal/{Path(item).name}" for item in internal)
    else:
        lines.append("- None moved")
    lines.append("")
    return "\n".join(lines)


def _move_debug_files(out_dir: Path) -> list[str]:
    debug_dir = out_dir / "debug"
    moved: list[str] = []
    for name in sorted(DEBUG_FILE_NAMES):
        source = out_dir / name
        if not source.is_file():
            continue
        destination = debug_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        source.replace(destination)
        moved.append(name)
    return moved


def _move_internal_files(out_dir: Path) -> list[str]:
    internal_dir = out_dir / "internal"
    moved: list[str] = []
    for name in sorted(INTERNAL_FILE_NAMES):
        source = out_dir / name
        if not source.is_file():
            continue
        destination = internal_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        source.replace(destination)
        moved.append(name)
    return moved


def _write_user_view(out_dir: Path, summary: dict[str, Any]) -> list[str]:
    user_dir = out_dir / "user_view"
    _reset_user_view_dir(out_dir, user_dir)
    copied_files = _copy_user_code_files(user_dir, summary)
    files = {
        "README.ko.md": _render_user_readme(summary),
        "01_plan.ko.md": _render_user_plan(out_dir, summary),
        "02_pipeline_structure.ko.md": _render_user_pipeline_structure(out_dir, summary),
        "03_code_pipeline.ko.md": _render_user_code_pipeline(summary, copied_files),
        "04_result.ko.md": _render_user_result(summary),
    }
    for name, content in files.items():
        write_text(user_dir / name, content)
    return [*files, *[f"code/{item}" for item in copied_files]]


def _reset_user_view_dir(out_dir: Path, user_dir: Path) -> None:
    if user_dir.exists():
        if user_dir.parent.resolve() != out_dir.resolve():
            raise ValueError(f"Refusing to reset unexpected user view directory: {user_dir}")
        shutil.rmtree(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)


def _render_user_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {summary['competition']} / {summary['trial_id']} 사용자용 보기",
            "",
            "이 폴더는 사용자가 바로 확인할 만한 파일만 모아 둔 공간입니다.",
            "내부 JSON, API 요청/응답, 디버그 로그는 상위 폴더의 `internal/` 또는 `debug/`에 보관됩니다.",
            "",
            "## 바로 볼 파일",
            "",
            "- `01_plan.ko.md`: 이번 실험에서 무엇을 하려고 했는지",
            "- `02_pipeline_structure.ko.md`: 노트북처럼 읽는 단계별 파이프라인 구조",
            "- `03_code_pipeline.ko.md`: 어떤 코드 파일이 만들어졌는지",
            "- `04_result.ko.md`: 실행 결과와 점수",
            "- `code/`: 이번 실험에서 생성 또는 수정된 코드 복사본",
            "",
            "## 한눈에 보기",
            "",
            f"- 상태: {summary.get('status')}",
            f"- 평가 지표: {summary.get('metric')}",
            f"- 목표 방향: {summary.get('objective')}",
            f"- 로컬 점수: {summary.get('local_score')}",
            f"- 실험 제목: {summary.get('plan_title')}",
            "",
        ]
    )


def _render_user_plan(out_dir: Path, summary: dict[str, Any]) -> str:
    objective = _read_markdown_section(out_dir / "demo_experiment_plan.md", "Objective")
    rationale = _read_markdown_section(out_dir / "demo_experiment_plan.md", "Rationale")
    notes = _read_markdown_section(out_dir / "demo_experiment_plan.md", "Implementation Notes")
    lines = [
        f"# {summary['trial_id']} 실험 계획",
        "",
        "## 목적",
        "",
        f"- 이번 실험은 `{summary['competition']}`에서 먼저 끝까지 실행되는 기준선 파이프라인을 만드는 단계입니다.",
        f"- 평가 지표는 `{summary.get('metric')}`이고, 목표는 `{summary.get('objective')}`입니다.",
        f"- 실험 제목: {summary.get('plan_title')}",
        "",
        "## 계획 요약",
        "",
        "- 데이터를 읽고 전처리합니다.",
        "- 하나의 단순한 모델 또는 기준선 모델을 학습합니다.",
        "- 로컬 검증 점수를 계산합니다.",
        "- 예측 결과와 실행 로그를 남깁니다.",
        "",
    ]
    if objective:
        lines.extend(["## 원문 계획의 목적", "", objective, ""])
    if rationale:
        lines.extend(["## 원문 판단 근거", "", rationale, ""])
    if notes:
        lines.extend(["## 구현 메모", "", notes, ""])
    lines.extend(
        [
            "## 다음에 확인할 것",
            "",
            "- `02_pipeline_structure.ko.md`에서 단계별 파이프라인 구조를 확인합니다.",
            "- `03_code_pipeline.ko.md`에서 실제 코드 구성을 확인합니다.",
            "- `04_result.ko.md`에서 실행 결과와 점수를 확인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_user_pipeline_structure(out_dir: Path, summary: dict[str, Any]) -> str:
    structure = _read_json(out_dir / "internal" / "pipeline_structure.json")
    stages = structure.get("stages", []) if isinstance(structure, dict) else []
    lines = [
        f"# {summary['trial_id']} 파이프라인 구조 명세",
        "",
        "이 문서는 `.ipynb`를 별도로 만들지 않고도 노트북처럼 위에서 아래로 파이프라인을 읽을 수 있게 만든 구조 명세입니다.",
        "실제 실행과 수정은 `.py` 파일 기준으로 유지하고, 이 문서는 사용자 이해와 다음 trial의 에이전트 컨텍스트로 함께 사용됩니다.",
        "",
        "## 요약",
        "",
        f"- competition: {summary['competition']}",
        f"- metric: {summary.get('metric')}",
        f"- objective: {summary.get('objective')}",
        f"- project_root: {summary.get('project_root')}",
        f"- machine-readable source: `{summary.get('pipeline_structure_file')}`",
        "",
        "## 단계별 구조",
        "",
    ]
    for index, stage in enumerate(stages, 1):
        included = "포함" if stage.get("included") else "제외/미탐지"
        required = "필수" if stage.get("required") else "조건부"
        lines.extend(
            [
                f"### {index}. {stage.get('name')}",
                "",
                f"- 상태: {included}",
                f"- 구분: {required}",
                f"- 역할: {stage.get('role')}",
                f"- 근거: {stage.get('reason')}",
                "- 코드 위치:",
            ]
        )
        lines.extend(f"  - `{item}`" for item in stage.get("code_locations", []) or ["미탐지"])
        lines.append("- 입력:")
        lines.extend(f"  - {item}" for item in stage.get("inputs", []) or ["미정"])
        lines.append("- 출력:")
        lines.extend(f"  - {item}" for item in stage.get("outputs", []) or ["미정"])
        lines.append("- 확인 포인트:")
        lines.extend(f"  - {item}" for item in stage.get("checks", []) or ["미정"])
        handles = stage.get("improvement_handles", [])
        if handles:
            lines.append("- 다음 trial 개선 축:")
            lines.extend(f"  - {item}" for item in handles)
        lines.append("")
    return "\n".join(lines)


def _render_user_code_pipeline(summary: dict[str, Any], copied_files: list[str]) -> str:
    project_root = summary.get("project_root") or "unknown"
    lines = [
        f"# {summary['trial_id']} 코드 파이프라인",
        "",
        "## 코드 위치",
        "",
        f"- 원본 workspace: `{project_root}`",
        "- 사용자 확인용 코드 복사본: `code/`",
        "",
        "## 이번 실험에서 변경된 파일",
        "",
    ]
    for item in summary.get("changed_files", []) or ["None"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## 복사된 코드 파일", ""])
    if copied_files:
        lines.extend(f"- `code/{item}`" for item in copied_files)
    else:
        lines.append("- 복사된 코드 파일이 없습니다. 상위 trial README의 Changed Files를 확인하세요.")
    lines.extend(
        [
            "",
            "## 읽는 순서",
            "",
            "1. `train_step.py`: 학습과 검증 점수 생성",
            "2. `predict_step.py`: 예측 및 제출 형식 파일 생성",
            "3. `test_step.py`: 실행 전후 기본 검증",
            "4. `src/` 아래 파일: 재사용되는 전처리/모델 파이프라인",
            "",
        ]
    )
    return "\n".join(lines)


def _render_user_result(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['trial_id']} 실행 결과",
        "",
        "## 결과 요약",
        "",
        f"- 상태: {summary.get('status')}",
        f"- workspace 실행 상태: {summary.get('workspace_status')}",
        f"- metrics 수집 상태: {summary.get('metrics_collection_status')}",
        f"- 평가 지표: {summary.get('metric')}",
        f"- 목표 방향: {summary.get('objective')}",
        f"- 로컬 점수: {summary.get('local_score')}",
        f"- 점수 출처: {summary.get('score_source')}",
        "",
        "## 실행 로그",
        "",
    ]
    lines.extend(f"- `{item}`" for item in summary.get("log_paths", []) or ["None"])
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 이번 trial은 코드 작성, 로컬 실행, metric 수집까지 한 번 이어지는지 확인하는 1회 루프입니다.",
            "- 점수가 높고 낮은지보다, 계획에서 코드 작성과 실행 결과 기록까지 끊기지 않았는지가 이번 단계의 핵심입니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _copy_user_code_files(user_dir: Path, summary: dict[str, Any]) -> list[str]:
    project_root = summary.get("project_root")
    if not project_root:
        return []
    root = Path(str(project_root))
    if not root.is_dir():
        return []
    code_dir = user_dir / "code"
    _reset_user_code_dir(user_dir, code_dir)
    copied: list[str] = []
    for item in summary.get("changed_files", []):
        relative = _safe_relative_file(item)
        if relative is None:
            continue
        source = root / Path(*relative.parts)
        if not source.is_file():
            continue
        destination = code_dir / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    return copied


def _reset_user_code_dir(user_dir: Path, code_dir: Path) -> None:
    if code_dir.exists() and code_dir.parent.resolve() == user_dir.resolve():
        shutil.rmtree(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)


def _safe_relative_file(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return path


def _read_markdown_section(path: Path, heading: str) -> str:
    text = read_text(path, default="")
    if not text:
        return ""
    marker = f"## {heading}"
    if marker not in text:
        return ""
    rest = text.split(marker, 1)[1]
    section = rest.split("\n## ", 1)[0].strip()
    return section


def _load_execution_profile(competition: str) -> dict[str, Any]:
    profile = simple_yaml.load(competition_dir(competition) / "execution_profile.yaml", default={})
    return profile if isinstance(profile, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_plan_title(out_dir: Path) -> str | None:
    text = (out_dir / "next_experiment.md").read_text(encoding="utf-8") if (out_dir / "next_experiment.md").exists() else ""
    for line in text.splitlines():
        if line.startswith("- title:"):
            return line.split(":", 1)[1].strip()
    return None


def _workspace_log_paths(workspace_run: dict[str, Any]) -> list[str]:
    return [
        item.get("log_path")
        for item in workspace_run.get("command_results", [])
        if isinstance(item, dict) and item.get("log_path")
    ]


def _key_files(out_dir: Path) -> list[str]:
    candidates = [
        "next_experiment.md",
        "workspace_coding_result.md",
        "workspace_run.md",
        "metrics_collection.md",
        "demo_cycle_record.md",
        "workspace_result_cycle.md",
        "workspace_after_coding_cycle.md",
        "metrics.json",
    ]
    return [name for name in candidates if (out_dir / name).exists()]


def _write_browse_view(competition: str, trial_id: str, out_dir: Path, summary: dict[str, Any]) -> list[str]:
    """Mirror the compact user view into runs/ for direct folder browsing."""
    source = out_dir / "user_view"
    browse_root = paths.project_root() / "runs" / competition
    browse_dir = browse_root / trial_id
    _reset_browse_dir(browse_root, browse_dir)
    copied: list[str] = []
    if source.is_dir():
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
            destination = browse_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            copied.append(relative.as_posix())
    paths_file = browse_dir / "05_paths.ko.md"
    write_text(paths_file, _render_browse_paths(competition, trial_id, out_dir, browse_dir, summary))
    if "05_paths.ko.md" not in copied:
        copied.append("05_paths.ko.md")
    _write_browse_index(competition, browse_root)
    return sorted(copied)


def _reset_browse_dir(browse_root: Path, browse_dir: Path) -> None:
    browse_root.mkdir(parents=True, exist_ok=True)
    if browse_dir.exists():
        if browse_dir.parent.resolve() != browse_root.resolve():
            raise ValueError(f"Refusing to reset unexpected browse directory: {browse_dir}")
        shutil.rmtree(browse_dir)
    browse_dir.mkdir(parents=True, exist_ok=True)


def _write_browse_index(competition: str, browse_root: Path) -> None:
    trials = sorted(path.name for path in browse_root.iterdir() if path.is_dir())
    lines = [
        f"# {competition} 실험 보기",
        "",
        "이 폴더는 SFTP나 파일 탐색기로 바로 확인하기 위한 사용자용 보기 폴더입니다.",
        "에이전트 내부 원본 기록은 `experiments/`와 `memory/`에 계속 보존됩니다.",
        "",
        "## Trial 목록",
        "",
    ]
    lines.extend(f"- `{trial}/`" for trial in trials)
    lines.append("")
    write_text(browse_root / "README.ko.md", "\n".join(lines))


def _render_browse_paths(
    competition: str,
    trial_id: str,
    out_dir: Path,
    browse_dir: Path,
    summary: dict[str, Any],
) -> str:
    project = summary.get("project_root") or "unknown"
    logs = summary.get("log_paths", []) or []
    lines = [
        f"# {competition} / {trial_id} 경로 안내",
        "",
        "## 이 폴더의 역할",
        "",
        "이 폴더는 사람이 빠르게 확인하기 위한 복사본입니다.",
        "실행 코드 원본, 실험 원본 기록, 장기 memory는 아래 위치에 따로 남아 있습니다.",
        "",
        "## 주요 위치",
        "",
        f"- 보기용 폴더: `{browse_dir}`",
        f"- 실험 원본 기록: `{out_dir}`",
        f"- 실행 workspace: `{project}`",
        f"- 장기 memory: `{paths.project_root() / 'memory' / competition}`",
        "",
        "## 추천 확인 순서",
        "",
        "1. `01_plan.ko.md`: 이번 실험 계획",
        "2. `02_pipeline_structure.ko.md`: 단계별 파이프라인 구조",
        "3. `03_code_pipeline.ko.md`: 코드 파일 구성",
        "4. `04_result.ko.md`: 실행 결과와 점수",
        "5. `code/`: 이번 trial에서 생성 또는 수정된 코드 복사본",
        "",
        "## 실행 로그",
        "",
    ]
    lines.extend(f"- `{item}`" for item in logs) if logs else lines.append("- 기록된 로그 경로가 없습니다.")
    lines.append("")
    return "\n".join(lines)
