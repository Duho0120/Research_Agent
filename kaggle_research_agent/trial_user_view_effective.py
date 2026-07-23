from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_effective_user_plan(out_dir: Path, summary: dict[str, Any]) -> str:
    plan = _read_trial_json(out_dir, "demo_experiment_plan.json")
    delta = _read_json(out_dir / "delta_plan.json")
    source = plan.get("source_trial_id") or delta.get("source_trial_id")
    axis = plan.get("primary_change_axis") or delta.get("primary_change_axis") or "baseline"
    delta_candidate = delta.get("candidate") if isinstance(delta.get("candidate"), dict) else {}
    title = plan.get("plan_title") or summary.get("plan_title") or delta_candidate.get("description") or delta_candidate.get("name") or "-"
    lines = [
        f"# {summary['trial_id']} 실험 계획",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 계획 유형 | `{'후속 개선 계획' if source else '초기 전체 계획'}` |",
        f"| 계획명 | {_display(title)} |",
        f"| 기준 trial | `{source or '-'}` |",
        f"| 개선축 | `{axis}` |",
        f"| 평가 지표 | `{_display(summary.get('metric'))}` |",
        f"| 목표 방향 | `{_display(summary.get('objective'))}` |",
        "",
    ]
    objective = _items(plan.get("objective"))
    if source and objective:
        lines.extend(["## 목적", "", *[f"- {item}" for item in objective], ""])
    elif not source:
        lines.extend(
            [
                "## 목적",
                "",
                "- 첫 회차 기준선으로 사용할 재현 가능한 baseline을 만들고 제출 파일 형식을 검증합니다.",
                *[f"- 원본 목표: {item}" for item in objective],
                "",
            ]
        )
    rationale = _items(plan.get("rationale"))
    lines.extend(["## 왜 하는가", ""])
    if source:
        lines.append("- 기준 trial의 검증 조건을 유지하면서 이번 개선축의 효과만 분리해 확인합니다.")
    else:
        lines.append("- 복잡한 탐색 전에 데이터 처리, 검증 방식과 제출 형식을 고정해 후속 실험의 비교 기준을 만듭니다.")
    lines.extend(f"- 근거: {item}" for item in rationale)
    lines.append("")
    if source:
        candidate = delta.get("candidate") if isinstance(delta.get("candidate"), dict) else {}
        lines.extend(["## 이번 회차 변경", "", f"- 후보: `{candidate.get('name') or title}`"])
        if candidate.get("description"):
            lines.append(f"- 설명: {candidate['description']}")
        lines.extend(f"- {item}" for item in _items(delta.get("change_details") or plan.get("change_details")))
        lines.extend(["", "## 그대로 유지", ""])
        lines.extend(
            f"- {item}"
            for item in _items(delta.get("keep_unchanged") or plan.get("keep_unchanged"))
            or ["기준 trial의 나머지 파이프라인"]
        )
        affected = _items(delta.get("affected_stages") or plan.get("affected_stages"))
        if affected:
            lines.extend(["", "## 영향받는 단계", "", *[f"- `{item}`" for item in affected]])
    else:
        blueprint = _items(plan.get("pipeline_blueprint"))
        lines.extend(["## 초기 파이프라인 설계", ""])
        lines.extend(f"- {item}" for item in blueprint or ["구조화된 pipeline blueprint가 없습니다."])
    success = _items(delta.get("success_criteria") or plan.get("success_criteria"))
    if success:
        lines.extend(["", "## 성공 기준", "", *[f"- {_success_text(item)}" for item in success]])
    lines.append("")
    return "\n".join(lines)


def render_effective_pipeline_structure(out_dir: Path, summary: dict[str, Any]) -> str:
    structure = _read_json(out_dir / "internal" / "pipeline_structure.json")
    delta = _read_json(out_dir / "internal" / "pipeline_delta_applied.json")
    stages = [stage for stage in structure.get("stages", []) if isinstance(stage, dict) and stage.get("included")]
    source_trial_id = delta.get("source_trial_id")
    changed = {
        str(item.get("stage_id"))
        for item in delta.get("changes", [])
        if isinstance(item, dict) and item.get("stage_id")
    } if source_trial_id else set()
    model_stage = next((stage for stage in stages if stage.get("id") == "model_definition"), {})
    lines = [
        f"# {summary['trial_id']} 파이프라인 구조",
        "",
        "실제 실행 코드와 실행 결과를 기준으로 재구성한 현재 유효 파이프라인입니다.",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 기준 trial | `{_display(source_trial_id)}` |",
        f"| 평가 지표 | `{_display(structure.get('metric') or summary.get('metric'))}` |",
        f"| 로컬 점수 | `{_display(summary.get('local_score'))}` |",
        f"| 모델 | `{_model_label(model_stage)}` |",
        f"| 제출 파일 | `{_display(summary.get('submission_file') or 'outputs/submission.csv')}` |",
        "",
        "## 단계별 실행 구조",
        "",
    ]
    issues = _items(structure.get("consistency_issues"))
    if issues:
        lines.extend(["## 일관성 경고", "", *[f"- `{item}`" for item in issues], ""])
    for index, stage in enumerate(stages, 1):
        stage_id = str(stage.get("id") or "unknown")
        marker = " · 이번 회차 변경" if stage_id in changed else ""
        lines.extend([f"### {index}. {_stage_name(stage_id)}{marker}", "", f"- 역할: {_stage_role(stage_id)}"])
        actual = [item for item in _items(stage.get("actual_applied")) if _readable_text(item)]
        if actual:
            lines.append("- 실제 적용:")
            lines.extend(f"  - {item}" for item in actual)
        details = stage.get("structured_details") if isinstance(stage.get("structured_details"), dict) else {}
        detail_lines = _stage_details(stage_id, details)
        if detail_lines:
            lines.append("- 주요 설정:")
            lines.extend(f"  - {item}" for item in detail_lines)
        inputs = _items(stage.get("inputs"))
        outputs = _items(stage.get("outputs"))
        if inputs:
            lines.append("- 입력: " + ", ".join(f"`{item}`" for item in inputs))
        if outputs:
            lines.append("- 출력: " + ", ".join(f"`{item}`" for item in outputs))
        locations = _items(stage.get("code_locations"))
        if locations:
            lines.append("- 관련 코드: " + ", ".join(f"`{item}`" for item in locations))
        lines.append("")
    lines.extend(
        [
            "## 재현 명령",
            "",
            "```bat",
            "python test_step.py",
            "python train_step.py",
            "python predict_step.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_name(stage_id: str) -> str:
    return {
        "imports_setup": "준비",
        "data_load": "데이터 로드",
        "preprocessing": "전처리",
        "data_split_cv": "검증 분리 / CV",
        "feature_representation": "피처 구성",
        "data_augmentation": "데이터 증강",
        "dataset_dataloader": "Dataset / DataLoader",
        "model_definition": "모델 정의",
        "loss_objective": "손실함수와 평가 기준",
        "training": "학습",
        "training_curve": "학습 곡선",
        "evaluation": "로컬 평가",
        "model_checkpoint": "모델 저장",
        "test_inference_output": "테스트 추론과 제출 파일 생성",
    }.get(stage_id, stage_id)


def _stage_role(stage_id: str) -> str:
    return {
        "imports_setup": "실행에 필요한 라이브러리, 경로와 설정값을 준비합니다.",
        "data_load": "학습 및 테스트 데이터를 읽고 목표·ID 컬럼을 확인합니다.",
        "preprocessing": "결측치 처리, 스케일링과 범주형 인코딩을 적용합니다.",
        "data_split_cv": "재현 가능한 로컬 검증 데이터를 구성합니다.",
        "feature_representation": "원본 컬럼에서 모델 입력 피처와 파생 피처를 구성합니다.",
        "model_definition": "사용할 모델과 주요 하이퍼파라미터를 정의합니다.",
        "loss_objective": "학습 목적과 평가 지표의 관계를 명시합니다.",
        "training": "학습 데이터로 모델을 적합합니다.",
        "evaluation": "검증 데이터에서 로컬 점수를 계산합니다.",
        "model_checkpoint": "필요한 경우 학습 모델을 저장합니다.",
        "test_inference_output": "테스트 예측과 제출 파일을 생성합니다.",
    }.get(stage_id, "해당 파이프라인 단계를 실행합니다.")


def _model_label(stage: dict[str, Any]) -> str:
    details = stage.get("structured_details") if isinstance(stage.get("structured_details"), dict) else {}
    estimator = str(details.get("estimator") or "-")
    members = details.get("members") if isinstance(details.get("members"), list) else []
    names = [str(item.get("estimator")) for item in members if isinstance(item, dict) and item.get("estimator")]
    return f"{estimator}({', '.join(names)})" if names else estimator


def _stage_details(stage_id: str, details: dict[str, Any]) -> list[str]:
    if not details:
        return []
    if stage_id == "preprocessing":
        lines = []
        if details.get("numeric_features"):
            lines.append("수치형 피처: " + ", ".join(f"`{item}`" for item in details["numeric_features"]))
        if details.get("categorical_features"):
            lines.append("범주형 피처: " + ", ".join(f"`{item}`" for item in details["categorical_features"]))
        for step in details.get("steps", []):
            if isinstance(step, dict):
                params = ", ".join(f"{key}={value}" for key, value in (step.get("parameters") or {}).items())
                lines.append(f"{step.get('operation')}({params}) → {step.get('applies_to')}")
        return lines
    if stage_id == "feature_representation":
        lines = []
        for feature in details.get("derived_features", []):
            if isinstance(feature, dict):
                lines.append(f"파생 피처 `{feature.get('name')}`: {feature.get('formula') or feature.get('purpose')}")
        if details.get("final_feature_columns"):
            lines.append("최종 피처: " + ", ".join(f"`{item}`" for item in details["final_feature_columns"]))
        return lines
    if stage_id == "model_definition":
        lines = [f"estimator: `{details.get('estimator')}`"]
        params = details.get("parameters") or {}
        if params:
            lines.append("파라미터: " + ", ".join(f"`{key}={value}`" for key, value in params.items()))
        for member in details.get("members", []):
            if isinstance(member, dict):
                member_params = ", ".join(f"{key}={value}" for key, value in (member.get("parameters") or {}).items())
                lines.append(f"구성 모델: `{member.get('estimator')}` ({member_params or '기본값'})")
        return lines
    labels = {
        "method": "방식",
        "split_strategy": "분할 전략",
        "test_size": "검증 비율",
        "random_state": "seed",
        "stratify": "stratify",
        "train_rows": "학습 행",
        "validation_rows": "검증 행",
        "cv_score": "CV 점수",
        "validation_accuracy": "검증 정확도",
        "submission_path": "제출 경로",
        "id_column": "ID 컬럼",
        "prediction_column": "예측 컬럼",
    }
    return [f"{labels.get(key, key)}: `{value}`" for key, value in details.items() if key in labels and value is not None]


def _read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    return _read_json(out_dir / name) or _read_json(out_dir / "internal" / name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[str]:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            result.extend(f"{key}: {child}" for key, child in item.items())
        elif str(item).strip():
            result.append(str(item).strip())
    return result


def _display(value: Any) -> str:
    return "-" if value is None or value == "" else str(value)


def _readable_text(value: str) -> bool:
    return "?" not in value and not any(marker in value for marker in ["紐", "媛", "瑜", "寃"])


def _success_text(value: str) -> str:
    lowered = value.lower()
    if "validation accuracy" in lowered and "accuracy key" in lowered:
        return "검증 정확도가 `accuracy` 계열 키로 기록됩니다."
    if "submission.csv" in lowered and "passengerid" in lowered and "survived" in lowered:
        return "`submission.csv`에 `PassengerId`, `Survived` 컬럼과 테스트 데이터와 같은 행 수가 기록됩니다."
    if "pipeline_summary.json" in lowered:
        return "`outputs/pipeline_summary.json`에 피처, 전처리, 모델, 분할과 seed가 기록됩니다."
    return value
