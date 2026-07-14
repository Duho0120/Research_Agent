from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .store import write_text


def render_user_view_files(out_dir: Path, summary: dict[str, Any], copied_files: list[str]) -> dict[str, str]:
    return {
        "README.ko.md": render_user_readme(summary),
        "01_plan.ko.md": render_user_plan(out_dir, summary),
        "02_pipeline_structure.ko.md": render_user_pipeline_structure(out_dir, summary),
        "03_code_pipeline.ko.md": render_user_code_pipeline(summary, copied_files),
        "04_result.ko.md": render_user_result(summary),
    }


def render_user_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {summary['competition']} / {summary['trial_id']} 실험 보기",
            "",
            "사용자가 바로 확인할 핵심 산출물만 모은 폴더입니다.",
            "상세 JSON, API 요청/응답, 디버그 로그는 원본 trial 폴더의 `internal/`, `debug/`에 보관됩니다.",
            "",
            "## 먼저 볼 파일",
            "",
            "| 파일 | 내용 |",
            "|---|---|",
            "| `01_plan.ko.md` | 실험 목적과 핵심 선택 |",
            "| `02_pipeline_structure.ko.md` | 단계별 실제 적용 내용 |",
            "| `03_code_pipeline.ko.md` | 생성/수정된 코드 파일 |",
            "| `04_result.ko.md` | 실행 상태와 점수 |",
            "| `code/` | 이번 trial 코드 복사본 |",
            "",
            "## 한눈에 보기",
            "",
            f"- 상태: {_display_value(summary.get('status'))}",
            f"- 지표: {_display_value(summary.get('metric'))} / {_display_value(summary.get('objective'))}",
            f"- 로컬 점수: {_display_value(summary.get('local_score'))}",
            f"- 실험 제목: {_display_value(summary.get('plan_title'))}",
            "",
        ]
    )


def render_user_plan(out_dir: Path, summary: dict[str, Any]) -> str:
    choices = _planned_choice_lines(out_dir)
    objective = _compact_block(_read_markdown_section(out_dir / "demo_experiment_plan.md", "Objective"))
    rationale = _compact_block(_read_markdown_section(out_dir / "demo_experiment_plan.md", "Rationale"))
    notes = _compact_block(_read_markdown_section(out_dir / "demo_experiment_plan.md", "Implementation Notes"))
    lines = [
        f"# {summary['trial_id']} 실험 계획 요약",
        "",
        "## 목표",
        "",
        f"- 대회/주제: `{summary['competition']}`",
        f"- 평가 지표: `{_display_value(summary.get('metric'))}`",
        f"- 목표 방향: `{_display_value(summary.get('objective'))}`",
        f"- 계획명: {_display_value(summary.get('plan_title'))}",
        "",
        "## 이번 실험의 핵심 선택",
        "",
    ]
    lines.extend(f"- {item}" for item in choices or ["계획서에서 핵심 선택을 추출하지 못했습니다."])
    if objective:
        lines.extend(["", "## LLM 계획 요약", "", f"- {objective}"])
    if rationale:
        lines.extend(["", "## 선택 근거", "", f"- {rationale}"])
    if notes:
        lines.extend(["", "## 구현 메모", "", f"- {notes}"])
    lines.extend(
        [
            "",
            "## 다음 확인",
            "",
            "- `02_pipeline_structure.ko.md`: 이번 계획이 실제 코드에 어떻게 적용됐는지 확인",
            "- `03_code_pipeline.ko.md`: 생성/수정된 코드 파일 확인",
            "- `04_result.ko.md`: 실행 결과와 점수 확인",
            "",
        ]
    )
    return "\n".join(lines)


def render_user_pipeline_structure(out_dir: Path, summary: dict[str, Any]) -> str:
    structure = _read_json(out_dir / "internal" / "pipeline_structure.json")
    stages = structure.get("stages", []) if isinstance(structure, dict) else []
    included_stages = [stage for stage in stages if stage.get("included")]
    lines = [
        f"# {summary['trial_id']} 파이프라인 요약",
        "",
        "실행 기준은 `.py` 코드입니다. 이 문서는 사람이 빠르게 확인하기 위한 요약입니다.",
        "",
        "## 전체 요약",
        "",
        "| 단계 | 실제 적용 내용 | 확인 포인트 |",
        "|---|---|---|",
    ]
    for stage in included_stages:
        lines.append(
            "| "
            + " | ".join(
                [
                    _stage_display_name(stage),
                    _join_short(stage.get("actual_applied", []), limit=2),
                    _join_short(stage.get("checks", []), limit=1),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 단계별 체크", ""])
    for index, stage in enumerate(included_stages, 1):
        detail_lines = _stage_detail_lines(stage)
        lines.extend(
            [
                f"### {index}. {_stage_display_name(stage)}",
                "",
                f"- 역할: {_display_value(stage.get('role'))}",
                f"- 적용: {_join_short(stage.get('actual_applied', []), limit=4)}",
            ]
        )
        lines.extend(f"- {item}" for item in detail_lines)
        locations = stage.get("code_locations", [])
        if locations:
            lines.append(f"- 코드: {_join_code_locations(locations, limit=3)}")
        lines.append("")

    lines.extend(
        [
            "## 내부 원본",
            "",
            f"- 구조화 JSON: `{summary.get('pipeline_structure_file')}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_user_code_pipeline(summary: dict[str, Any], copied_files: list[str]) -> str:
    project_root = summary.get("project_root") or "unknown"
    changed_files = summary.get("changed_files", []) or []
    lines = [
        f"# {summary['trial_id']} 코드 구성",
        "",
        "## 위치",
        "",
        f"- 실행 workspace: `{project_root}`",
        "- 보기용 코드 복사본: `code/`",
        "",
        "## 변경 파일",
        "",
    ]
    lines.extend(f"- `{item}`" for item in changed_files or ["None"])
    lines.extend(["", "## 복사된 코드", ""])
    if copied_files:
        lines.extend(f"- `code/{item}`" for item in copied_files)
    else:
        lines.append("- 복사된 코드 파일이 없습니다.")
    lines.extend(
        [
            "",
            "## 읽는 순서",
            "",
            "1. `train_step.py`: 학습과 검증",
            "2. `predict_step.py`: 예측과 제출 파일 생성",
            "3. `test_step.py`: 기본 실행 검증",
            "4. `src/`: 재사용되는 전처리/모델 로직",
            "",
        ]
    )
    return "\n".join(lines)


def render_user_result(summary: dict[str, Any]) -> str:
    logs = summary.get("log_paths", []) or []
    lines = [
        f"# {summary['trial_id']} 실행 결과",
        "",
        "## 결과 카드",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| trial 상태 | {_display_value(summary.get('status'))} |",
        f"| workspace 실행 | {_display_value(summary.get('workspace_status'))} |",
        f"| metrics 수집 | {_display_value(summary.get('metrics_collection_status'))} |",
        f"| 지표 | {_display_value(summary.get('metric'))} |",
        f"| 목표 방향 | {_display_value(summary.get('objective'))} |",
        f"| 로컬 점수 | {_display_value(summary.get('local_score'))} |",
        f"| 점수 출처 | {_display_value(summary.get('score_source'))} |",
        "",
        "## 실행 로그",
        "",
    ]
    if logs:
        lines.extend(f"- `{item}`" for item in logs)
    else:
        lines.append("- 기록된 로그 경로가 없습니다.")
    lines.extend(
        [
            "",
            "## 의미",
            "",
            "- 이번 trial은 계획 생성, 코드 작성, 로컬 실행, metric 수집까지 1회 루프가 연결되는지 확인한 결과입니다.",
            "- 성능 판단이나 축 전환은 별도 평가 단계에서 다룹니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_browse_index(competition: str, browse_root: Path) -> None:
    trials = sorted(path.name for path in browse_root.iterdir() if path.is_dir())
    lines = [
        f"# {competition} 실험 보기",
        "",
        "`runs/`는 사용자가 파일 탐색기나 SFTP에서 바로 보기 위한 요약 폴더입니다.",
        "원본 기록과 내부 JSON은 `experiments/`, `memory/`에 계속 보관됩니다.",
        "",
        "## Trial 목록",
        "",
    ]
    lines.extend(f"- `{trial}/`" for trial in trials)
    lines.append("")
    write_text(browse_root / "README.ko.md", "\n".join(lines))


def render_browse_paths(
    competition: str,
    trial_id: str,
    out_dir: Path,
    browse_dir: Path,
    summary: dict[str, Any],
) -> str:
    logs = summary.get("log_paths", []) or []
    lines = [
        f"# {competition} / {trial_id} 경로 안내",
        "",
        "## 주요 위치",
        "",
        f"- 보기용 폴더: `{browse_dir}`",
        f"- 원본 trial 기록: `{out_dir}`",
        f"- 실행 workspace: `{summary.get('project_root') or 'unknown'}`",
        f"- 장기 memory: `{paths.project_root() / 'memory' / competition}`",
        "",
        "## 추천 확인 순서",
        "",
        "1. `01_plan.ko.md`: 실험 계획 요약",
        "2. `02_pipeline_structure.ko.md`: 단계별 실제 적용 내용",
        "3. `03_code_pipeline.ko.md`: 코드 파일 구성",
        "4. `04_result.ko.md`: 실행 결과와 점수",
        "5. `code/`: 이번 trial 코드 복사본",
        "",
        "## 실행 로그",
        "",
    ]
    if logs:
        lines.extend(f"- `{item}`" for item in logs)
    else:
        lines.append("- 기록된 로그 경로가 없습니다.")
    lines.append("")
    return "\n".join(lines)


def _planned_choice_lines(out_dir: Path) -> list[str]:
    notes = _read_markdown_section(out_dir / "demo_experiment_plan.md", "Implementation Notes")
    lines: list[str] = []
    for raw_line in notes.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        lowered = item.lower()
        if any(
            keyword in lowered
            for keyword in [
                "implement one end-to-end",
                "implement a single pipeline",
                "create basic",
                "use core",
                "handle missing",
                "encode categorical",
                "train a single",
                "train one",
                "single deterministic",
                "deterministic train/validation",
                "use a deterministic",
                "report validation",
                "write outputs/metrics",
                "write outputs/submission",
                "make predict_step",
                "make test_step",
                "keep changes",
                "do not add",
            ]
        ):
            lines.append(_shorten(item, 160))
    return lines


def _stage_display_name(stage: dict[str, Any]) -> str:
    names = {
        "imports_setup": "준비",
        "data_load": "데이터 로드",
        "preprocessing": "전처리",
        "data_split_cv": "검증 분리",
        "feature_representation": "피처",
        "data_augmentation": "증강",
        "dataset_dataloader": "Dataset/DataLoader",
        "model_definition": "모델",
        "loss_objective": "손실/목표",
        "training": "학습",
        "training_curve": "학습 곡선",
        "evaluation": "평가",
        "model_checkpoint": "저장",
        "test_inference_output": "추론/출력",
    }
    return names.get(str(stage.get("id")), str(stage.get("name") or stage.get("id") or "unknown"))


def _stage_detail_lines(stage: dict[str, Any]) -> list[str]:
    details = stage.get("structured_details")
    if not isinstance(details, dict) or not details:
        return []
    stage_id = str(stage.get("id"))
    if stage_id == "data_load":
        parts = []
        if details.get("target_column"):
            parts.append(f"타깃 `{details['target_column']}`")
        if details.get("id_column"):
            parts.append(f"ID `{details['id_column']}`")
        files = details.get("required_files") or []
        if files:
            parts.append(f"필수 파일 {_inline_list(files, 4)}")
        return [f"세부: {', '.join(parts)}"] if parts else []
    if stage_id == "preprocessing":
        lines = []
        raw = details.get("raw_feature_columns") or []
        numeric = details.get("numeric_features") or []
        categorical = details.get("categorical_features") or []
        if raw:
            lines.append(f"입력 컬럼: {_inline_list(raw, 10)}")
        if numeric or categorical:
            lines.append(f"피처 그룹: 수치형 {_inline_list(numeric, 8)} / 범주형 {_inline_list(categorical, 8)}")
        steps = _format_preprocessing_steps(details.get("steps") or [])
        if steps:
            lines.append(f"처리 방식: {steps}")
        return lines
    if stage_id == "feature_representation":
        lines = []
        derived = _format_derived_features(details.get("derived_features") or [])
        if derived:
            lines.append(f"파생변수: {derived}")
        final_features = details.get("final_feature_columns") or []
        if final_features:
            lines.append(f"최종 피처: {_inline_list(final_features, 12)}")
        return lines
    if stage_id == "data_split_cv":
        parts = []
        for key, label in [
            ("method", "방식"),
            ("split_strategy", "전략"),
            ("test_size", "검증 비율"),
            ("n_splits", "fold"),
            ("random_state", "seed"),
        ]:
            if details.get(key) is not None:
                parts.append(f"{label} `{details[key]}`")
        if details.get("stratify"):
            parts.append("stratify 사용")
        return [f"세부: {', '.join(parts)}"] if parts else []
    if stage_id == "model_definition":
        estimator = details.get("estimator")
        params = _format_params(details.get("parameters") or {})
        if estimator and params:
            return [f"모델 설정: `{estimator}` ({params})"]
        if estimator:
            return [f"모델 설정: `{estimator}`"]
        return []
    if stage_id == "loss_objective":
        metric = details.get("metric")
        objective = details.get("objective")
        if metric:
            return [f"평가 기준: `{metric}` / `{objective or '-'}`"]
        return []
    if stage_id == "training":
        parts = []
        if details.get("train_rows") is not None:
            parts.append(f"학습 rows `{details['train_rows']}`")
        if details.get("validation_rows") is not None:
            parts.append(f"검증 rows `{details['validation_rows']}`")
        if details.get("checkpoint_path"):
            parts.append(f"저장 `{details['checkpoint_path']}`")
        return [f"세부: {', '.join(parts)}"] if parts else []
    if stage_id == "evaluation":
        parts = []
        for key in ["cv_score", "validation_accuracy"]:
            if details.get(key) is not None:
                parts.append(f"{key} `{details[key]}`")
        return [f"점수: {', '.join(parts)}"] if parts else []
    if stage_id == "model_checkpoint" and details.get("checkpoint_path"):
        return [f"저장 위치: `{details['checkpoint_path']}`"]
    if stage_id == "test_inference_output":
        parts = []
        if details.get("submission_path"):
            parts.append(f"파일 `{details['submission_path']}`")
        if details.get("id_column") and details.get("prediction_column"):
            parts.append(f"컬럼 `{details['id_column']}`, `{details['prediction_column']}`")
        post = details.get("postprocessing") or []
        if post:
            parts.append(f"후처리 {_inline_list(post, 3)}")
        return [f"출력: {', '.join(parts)}"] if parts else []
    return []


def _format_preprocessing_steps(steps: list[Any]) -> str:
    formatted: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        operation = step.get("operation")
        applies_to = step.get("applies_to")
        params = _format_params(step.get("parameters") or {})
        suffix = f"({params})" if params else ""
        formatted.append(f"`{operation}{suffix}` -> {applies_to}")
    return "; ".join(formatted)


def _format_derived_features(features: list[Any]) -> str:
    formatted: list[str] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        name = feature.get("name")
        formula = feature.get("formula")
        if name and formula:
            formatted.append(f"`{name} = {formula}`")
        elif name:
            formatted.append(f"`{name}`")
    return "; ".join(formatted)


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _inline_list(items: Iterable[Any], limit: int) -> str:
    values = [f"`{item}`" for item in items if item is not None and item != ""]
    if not values:
        return "`-`"
    shown = values[:limit]
    if len(values) > limit:
        shown.append(f"외 {len(values) - limit}개")
    return ", ".join(shown)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_markdown_section(path: Path, heading: str) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    target = f"## {heading}".strip().lower()
    collected: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_section:
                break
            in_section = stripped.lower() == target
            continue
        if in_section:
            collected.append(line)
    return "\n".join(collected).strip()


def _join_short(items: Iterable[Any], limit: int = 2) -> str:
    values = [_shorten(str(item), 95) for item in items if item]
    if not values:
        return "-"
    shown = values[:limit]
    if len(values) > limit:
        shown.append(f"외 {len(values) - limit}개")
    return "<br>".join(shown)


def _join_code_locations(items: Iterable[Any], limit: int = 3) -> str:
    values = [f"`{str(item)}`" for item in items if item]
    if not values:
        return "-"
    shown = values[:limit]
    if len(values) > limit:
        shown.append(f"외 {len(values) - limit}개")
    return ", ".join(shown)


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _compact_block(text: str, limit: int = 240) -> str:
    if not text:
        return ""
    compact = " ".join(line.strip(" -") for line in text.splitlines() if line.strip())
    return _shorten(compact, limit)


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."
