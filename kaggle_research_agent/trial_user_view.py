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
    choices = _implemented_choice_lines(out_dir)
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
    lines.extend(f"- {item}" for item in choices or ["아직 코드에서 핵심 선택을 추출하지 못했습니다."])
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
            "- `02_pipeline_structure.ko.md`: 실제 적용된 split, 전처리, 피처, 모델",
            "- `03_code_pipeline.ko.md`: 생성/수정된 코드 파일",
            "- `04_result.ko.md`: 실행 결과와 점수",
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
        lines.extend(
            [
                f"### {index}. {_stage_display_name(stage)}",
                "",
                f"- 역할: {_display_value(stage.get('role'))}",
                f"- 적용: {_join_short(stage.get('actual_applied', []), limit=4)}",
            ]
        )
        locations = stage.get("code_locations", [])
        if locations:
            lines.append(f"- 코드: {_join_code_locations(locations, limit=3)}")
        handles = stage.get("improvement_handles", [])
        if handles:
            lines.append(f"- 다음 개선축: {_join_short(handles, limit=3)}")
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


def _implemented_choice_lines(out_dir: Path) -> list[str]:
    structure = _read_json(out_dir / "internal" / "pipeline_structure.json")
    stages = structure.get("stages", []) if isinstance(structure, dict) else []
    focus_ids = {
        "data_split_cv": "검증",
        "preprocessing": "전처리",
        "feature_representation": "피처",
        "model_definition": "모델",
        "evaluation": "평가",
        "test_inference_output": "출력",
    }
    lines: list[str] = []
    for stage in stages:
        label = focus_ids.get(stage.get("id"))
        actual = stage.get("actual_applied") or []
        if label and actual:
            lines.append(f"{label}: {_shorten(actual[0], 130)}")
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
        "loss_function": "손실/목표",
        "training": "학습",
        "training_curve": "학습 곡선",
        "evaluation": "평가",
        "model_save_checkpoint": "저장",
        "test_inference_output": "추론/출력",
    }
    return names.get(str(stage.get("id")), str(stage.get("name") or stage.get("id") or "unknown"))


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
