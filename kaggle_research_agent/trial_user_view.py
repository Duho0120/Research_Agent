from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .plan_translation import render_plan_ko
from .store import write_text
from .trial_user_view_effective import render_effective_pipeline_structure


def render_user_view_files(
    out_dir: Path,
    summary: dict[str, Any],
    copied_files: list[str],
    *,
    allow_api: bool = False,
    plan_translation_client: Any | None = None,
) -> dict[str, str]:
    return {
        "01_plan.ko.md": render_plan_ko(
            out_dir, summary, allow_api=allow_api, client=plan_translation_client
        ),
        "02_pipeline_structure.ko.md": render_effective_pipeline_structure(out_dir, summary),
        "03_scores.ko.md": render_user_scores(summary),
    }


def write_proposed_plan_preview(
    competition: str,
    trial_id: str,
    plan: dict[str, Any],
    *,
    metric: str | None = None,
    objective: str | None = None,
    allow_api: bool = False,
) -> Path:
    """Write a pre-execution Korean plan preview from the proposed plan.

    Unlike render_effective_user_plan (which waits for resolve_trial_plan's
    post-execution facts), this renders directly from the plan the planner just
    produced, so the dashboard has a Korean document even before code
    writing/execution happens. It is overwritten with the effective (executed)
    version once the trial actually runs.
    """
    out_dir = paths.trial_dir(competition, trial_id)
    summary = {
        "competition": competition,
        "trial_id": trial_id,
        "metric": metric or "-",
        "objective": objective or plan.get("objective") or "-",
        "plan_title": plan.get("plan_title"),
    }
    content = render_plan_ko(out_dir, summary, plan=plan, allow_api=allow_api)
    preview_path = out_dir / "user_view" / "01_plan.ko.md"
    write_text(preview_path, content)
    return preview_path


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
            "| `05_submission.ko.md` | 제출 준비 상태와 다음 조치 |",
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


def _render_plan_detail_sections(
    plan: dict[str, Any],
    *,
    objective: str,
    rationale: str,
    summary: dict[str, Any],
) -> list[str]:
    if not isinstance(plan, dict) or not plan:
        return ["", "## 목적", "", "- 계획서에서 목적과 실험축을 추출하지 못했습니다."]
    if _is_continuation_plan(plan):
        return _render_continuation_plan_sections(plan, objective=objective, rationale=rationale, summary=summary)
    return _render_initial_plan_sections(plan, objective=objective, rationale=rationale, summary=summary)


def _is_continuation_plan(plan: dict[str, Any]) -> bool:
    plan_type = str(plan.get("plan_type") or "").strip()
    if plan_type in {"continuation_delta_plan", "delta_patch"}:
        return True
    return bool(plan.get("source_trial_id") and plan.get("primary_change_axis"))


def _render_initial_plan_sections(
    plan: dict[str, Any],
    *,
    objective: str,
    rationale: str,
    summary: dict[str, Any],
) -> list[str]:
    blueprint = _parse_pipeline_blueprint(plan.get("pipeline_blueprint"))
    purpose = _koreanize_plan_summary(objective, kind="objective") or "재현 가능한 첫 baseline을 만들고 로컬 실행/결과 기록 루프가 연결되는지 확인합니다."
    hypothesis = _initial_hypothesis(blueprint, rationale)
    evidence = _initial_key_evidence(blueprint)
    exclusions = _initial_intentional_exclusions(plan, blueprint)
    success = _success_criteria_lines(plan.get("success_criteria"))
    outputs = _expected_output_lines(plan.get("expected_outputs"))
    lines = [
        "",
        "## 목적",
        "",
        f"- {purpose}",
        "",
        "## 기준 상태",
        "",
        "- 이전 trial이 없는 첫 회차입니다.",
        "- 현재 best나 비교 기준이 없으므로 이번 trial을 이후 실험의 baseline으로 사용합니다.",
        "",
        "## 핵심 가설",
        "",
        f"- {hypothesis}",
        "",
        "## 실험축",
        "",
        "- 축: baseline 구축",
        "- 이번 회차에서는 성능 최적화보다 재현 가능한 기준점과 제출 형식 검증을 우선합니다.",
    ]
    if evidence:
        lines.extend(["- 핵심 근거: " + "; ".join(evidence)])
    if exclusions:
        lines.extend(["", "## 의도적으로 하지 않는 것", ""])
        lines.extend(f"- {item}" for item in exclusions)
    if success:
        lines.extend(["", "## 성공 기준", ""])
        lines.extend(f"- {item}" for item in success)
    if outputs:
        lines.extend(["", "## 주요 산출물", ""])
        lines.extend(f"- {item}" for item in outputs)
    lines.extend(
        [
            "",
            "## 다음 판단",
            "",
            "- 이번 trial이 정상 완료되면 baseline으로 고정합니다.",
            "- 다음 trial에서는 한 번에 하나의 개선축만 바꾸고, 상세 구현 차이는 `02_pipeline_structure.ko.md`와 내부 JSON을 기준으로 비교합니다.",
        ]
    )
    return lines


def _render_continuation_plan_sections(
    plan: dict[str, Any],
    *,
    objective: str,
    rationale: str,
    summary: dict[str, Any],
) -> list[str]:
    axis = str(plan.get("primary_change_axis") or "").strip()
    source = str(plan.get("source_trial_id") or "").strip()
    purpose = _koreanize_plan_summary(objective, kind="objective") or "이전 trial을 기준으로 하나의 개선축만 바꿔 성능 변화를 검증합니다."
    promoted = _implementation_note_buckets(plan.get("implementation_notes"))
    raw_change_details = _normalize_plan_items(plan.get("change_details")) or promoted["change_details"]
    raw_keep_unchanged = _normalize_plan_items(plan.get("keep_unchanged")) or promoted["keep_unchanged"]
    candidate = plan.get("candidate") if isinstance(plan.get("candidate"), dict) else {}
    candidate_name = str(candidate.get("name") or "").strip()
    candidate_description = str(candidate.get("description") or "").strip()
    candidate_hint = str(candidate.get("implementation_hint") or "").strip()
    candidate_lines: list[str] = []
    if candidate_name:
        candidate_text = f"Candidate `{candidate_name}`"
        if candidate_description:
            candidate_text += f": {candidate_description}"
        candidate_lines.append(candidate_text)
    if candidate_hint:
        candidate_lines.append(f"Implementation hint: {candidate_hint}")
    if candidate_lines:
        raw_change_details = [*candidate_lines, *raw_change_details]
    if str(plan.get("plan_type") or "").strip() == "delta_patch" and candidate_name:
        purpose = (
            f"기준 trial `{source or summary.get('recommended_base_trial') or 'unknown'}`에서 "
            f"`{axis or 'single_change_axis'}` 축의 후보 `{candidate_name}`만 추가해 점수 변화를 검증합니다."
        )
        if candidate_description:
            rationale = (
                f"`{candidate_name}` 후보가 기존 best 대비 도움이 되는지 확인합니다. "
                f"후보 설명: {candidate_description}"
            )
    change_details = [_koreanize_continuation_item(item) for item in raw_change_details[:8]]
    keep_unchanged = [_koreanize_continuation_item(item) for item in raw_keep_unchanged[:8]]
    success = _success_criteria_lines(plan.get("success_criteria"))
    failure = [_koreanize_continuation_item(item) for item in _normalize_plan_items(plan.get("failure_decision"))[:5]]
    lines = [
        "",
        "## 목적",
        "",
        f"- {purpose}",
        "",
        "## 기준 상태",
        "",
    ]
    if source:
        lines.append(f"- 기준 trial: `{source}`")
    else:
        lines.append("- 기준 trial을 찾지 못했습니다.")
    lines.extend(
        [
            "- 이전 trial의 결과와 decision card를 기준으로 이번 변경을 평가합니다.",
            "",
            "## 핵심 가설",
            "",
            f"- {_koreanize_continuation_rationale(rationale, axis=axis, source=source)}",
            "",
            "## 실험축",
            "",
            f"- 축: {_display_value(axis)}",
            "- 이번 회차에서는 이 축 하나만 바꾸고 나머지는 가능한 한 유지합니다.",
        ]
    )
    if keep_unchanged:
        lines.extend(["", "## 유지할 것", ""])
        lines.extend(f"- {item}" for item in keep_unchanged)
    if change_details:
        lines.extend(["", "## 바꿀 것", ""])
        lines.extend(f"- {item}" for item in change_details)
    exclusions = _continuation_intentional_exclusions(plan)
    if exclusions:
        lines.extend(["", "## 의도적으로 하지 않는 것", ""])
        lines.extend(f"- {item}" for item in exclusions)
    if success:
        lines.extend(["", "## 성공 기준", ""])
        lines.extend(f"- {item}" for item in success)
    if failure:
        lines.extend(["", "## 실패 시 판단 기준", ""])
        lines.extend(f"- {item}" for item in failure)
    lines.extend(
        [
            "",
            "## 다음 판단",
            "",
            "- 로컬 점수와 decision card를 기준으로 변경축을 유지, 보류, 기각할지 판단합니다.",
            "- 상세 파라미터와 실제 코드 차이는 `02_pipeline_structure.ko.md`에서 확인합니다.",
        ]
    )
    return lines


def _render_plan_list_section(title: str, value: Any, *, limit: int, preserve_raw: bool = False) -> list[str]:
    items = (_normalize_plan_items_raw(value) if preserve_raw else _normalize_plan_items(value))[:limit]
    if not items:
        return []
    lines = ["", f"## {title}", ""]
    lines.extend(f"- {item}" for item in items)
    return lines


def _initial_hypothesis(blueprint: dict[str, list[str]], rationale: str) -> str:
    model = _first_blueprint_value(blueprint, "type") or _first_blueprint_value(blueprint, "family")
    numeric = blueprint.get("numeric_features", [])
    categorical = blueprint.get("categorical_features", [])
    if not model and "logistic regression" in rationale.lower():
        model = "LogisticRegression"
    if model and numeric and categorical:
        return (
            f"표 형식 이진 분류 문제에서 기본 결측치 처리와 범주형 인코딩을 적용한 단일 `{model}` 모델로 "
            "다음 실험의 비교 기준이 되는 안정적인 baseline을 만들 수 있다고 봅니다."
        )
    if rationale:
        return _shorten(rationale.replace("\n", " "), 240)
    return "복잡한 개선을 시작하기 전에, 단순하고 재현 가능한 baseline이 필요하다고 봅니다."


def _initial_key_evidence(blueprint: dict[str, list[str]]) -> list[str]:
    evidence: list[str] = []
    method = _first_blueprint_value(blueprint, "method")
    valid = _first_blueprint_value(blueprint, "validation_fraction") or _first_blueprint_value(blueprint, "validation_size")
    stratify = _first_blueprint_value(blueprint, "stratify")
    model = _first_blueprint_value(blueprint, "type") or _first_blueprint_value(blueprint, "family")
    numeric_count = len(blueprint.get("numeric_features", []))
    categorical_count = len(blueprint.get("categorical_features", []))
    if method:
        detail = method
        if valid:
            detail += f"(valid={valid})"
        if stratify:
            detail += f", stratify={stratify}"
        evidence.append(f"로컬 검증은 `{detail}`로 둡니다")
    if model:
        evidence.append(f"모델은 빠르게 검증 가능한 `{model}`을 사용합니다")
    if numeric_count or categorical_count:
        evidence.append(f"입력은 수치형 {numeric_count}개, 범주형 {categorical_count}개 중심으로 제한합니다")
    return evidence


def _initial_intentional_exclusions(plan: dict[str, Any], blueprint: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    excluded = [item for item in blueprint.get("excluded", []) if item]
    if excluded:
        lines.append(f"`{', '.join(excluded)}` 컬럼은 이번 회차에서 학습 입력으로 직접 사용하지 않습니다.")
    notes = _normalize_plan_items_raw(plan.get("implementation_notes"))
    for item in notes:
        lowered = item.lower()
        if "do not add" in lowered and "leaderboard" in lowered:
            lines.append("Leaderboard 제출, Human Review, 앙상블, 다중 모델 탐색은 이번 회차 범위에서 제외합니다.")
        elif "do not use gaussian naive bayes" in lowered:
            lines.append("혼합형 tabular baseline에 부적절할 수 있는 Gaussian Naive Bayes나 고카디널리티 원문 컬럼의 직접 one-hot 사용은 피합니다.")
        elif "do not use" in lowered:
            lines.append(_koreanize_exclusion_note(item))
    persist = _first_blueprint_value(blueprint, "persist_trained_model").lower()
    if persist == "false":
        lines.append("학습된 모델 파일은 기본 저장하지 않고, metrics/submission/code snapshot/pipeline summary를 주요 기록으로 남깁니다.")
    return _unique_lines(lines)


def _koreanize_exclusion_note(text: str) -> str:
    raw = str(text).strip()
    lowered = raw.lower()
    excluded: list[str] = []
    if "gaussiannb" in lowered or "gaussian naive bayes" in lowered:
        excluded.append("GaussianNB")
    if "name" in lowered or "ticket" in lowered or "cabin" in lowered:
        excluded.append("원문 `Name`/`Ticket`/`Cabin` 컬럼의 직접 one-hot 인코딩")
    if "ensemble" in lowered:
        excluded.append("앙상블")
    if "leaderboard" in lowered:
        excluded.append("Leaderboard 제출")
    if "human review" in lowered:
        excluded.append("Human Review")
    if excluded:
        return f"다음 항목은 이번 회차에서 사용하지 않습니다: {', '.join(excluded)}."
    return _shorten(raw, 180)


def _continuation_intentional_exclusions(plan: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    notes = _normalize_plan_items_raw(plan.get("implementation_notes"))
    for item in notes:
        lowered = item.lower()
        if "do not" in lowered or "avoid" in lowered:
            lines.append(_koreanize_continuation_item(item))
    if not lines:
        lines.append("선택한 개선축 밖의 전처리, split, 모델 구조는 가능한 한 유지합니다.")
    return _unique_lines(lines)


def _success_criteria_lines(value: Any) -> list[str]:
    result: list[str] = []
    for item in _normalize_plan_items_raw(value):
        result.append(_koreanize_success_criterion(item))
    return _unique_lines(result)[:8]


def _expected_output_lines(value: Any) -> list[str]:
    result = [_koreanize_expected_output(item) for item in _normalize_plan_items_raw(value)]
    return _unique_lines(result)[:6]


def _koreanize_success_criterion(text: str) -> str:
    lowered = str(text).lower()
    if "validation accuracy is recorded" in lowered and "accuracy" in lowered:
        return "검증 정확도가 `accuracy` 계열 키로 기록됩니다."
    if "local validation" in lowered and "accuracy" in lowered and "record" in lowered:
        return "로컬 검증 정확도와 사용한 split 정보가 함께 기록됩니다."
    if "train_step.py completes" in lowered:
        return "`train_step.py`가 넓은 스키마 탐색 없이 명시된 데이터 프로필 기준으로 완료됩니다."
    if "metrics.json" in lowered and ("accuracy" in lowered or "score" in lowered):
        return "`outputs/metrics.json`에 로컬 검증 점수와 split 정보가 기록됩니다."
    if "submission.csv" in lowered and "passengerid" in lowered and "survived" in lowered:
        return "`outputs/submission.csv`가 계획에 명시된 ID와 예측 컬럼 형식으로 생성됩니다."
    if "row count" in lowered and "test.csv" in lowered:
        return "제출 파일의 행 수와 ID 순서가 `test.csv`와 일치합니다."
    if "binary" in lowered or "0/1" in lowered:
        return "예측값은 결측치 없는 0/1 정수 값으로 생성됩니다."
    if "pipeline_summary" in lowered:
        return "`outputs/pipeline_summary.json`에 피처, 전처리, 모델, split, seed가 기록됩니다."
    if "pipeline summary" in lowered:
        return "파이프라인 요약에 기준 trial과 이번 변경축이 기록됩니다."
    if "code_snapshot" in lowered:
        return "`outputs/code_snapshot.json`에 이번 trial의 코드 스냅샷이 기록됩니다."
    if "test_step.py passes" in lowered:
        return "`test_step.py` 검증이 생성된 metrics/submission 산출물을 기준으로 통과합니다."
    if "no trained model" in lowered or "model artifact" in lowered:
        return "별도 모델 파일은 저장하지 않습니다."
    return _shorten(str(text), 180)


def _koreanize_continuation_rationale(text: str, *, axis: str, source: str) -> str:
    if not text:
        return "선택한 개선축 하나가 기준 trial 대비 로컬 지표를 개선하거나, 최소한 다음 판단에 필요한 근거를 제공합니다."
    lowered = text.lower()
    if "change exactly one axis" in lowered or "model family" in lowered:
        base = f"`{source}`" if source else "기준 trial"
        selected = f"`{axis}`" if axis else "선택한 개선축"
        return f"{base}의 파이프라인을 기준으로 두고, 이번 회차에서는 {selected} 하나만 바꿔 성능 변화를 확인합니다."
    return _shorten(text.replace("\n", " "), 220)


def _koreanize_continuation_item(text: str) -> str:
    raw = str(text).strip()
    lowered = raw.lower()
    if lowered.startswith("data files:"):
        return "데이터 파일은 기존 train/test 사용 방식을 유지합니다."
    if lowered.startswith("target/id/output columns:"):
        return "타깃, ID, 출력 컬럼은 기준 trial의 구성을 유지합니다."
    if lowered.startswith("features:"):
        return "입력 피처 목록은 기준 trial과 동일하게 유지합니다."
    if lowered.startswith("deferred/excluded:"):
        return "보류/제외 컬럼은 기준 trial과 동일하게 유지합니다."
    if lowered.startswith("from:"):
        return "변경 전: " + raw.split(":", 1)[1].strip()
    if lowered.startswith("to:"):
        return "변경 후: " + raw.split(":", 1)[1].strip()
    if lowered.startswith("no leaderboard"):
        return "Leaderboard 제출과 앙상블은 이번 회차에서 제외합니다."
    if lowered.startswith("검증 분리:") and "preserve" in lowered:
        return "검증 분리 방식과 seed는 기준 trial의 구현을 유지합니다."
    if "modify only the estimator" in lowered:
        return "모델 estimator 구성만 수정하고, 새 피처나 전처리 변경은 추가하지 않습니다."
    if "do not persist" in lowered:
        return "모델 파일은 기본 저장하지 않고, 꼭 필요한 경우에만 최소 artifact로 저장합니다."
    if "validation_accuracy is not greater" in lowered:
        return "`validation_accuracy`가 기준 점수를 넘지 못하면 이번 변경 후보를 기각하고 기준 trial을 유지합니다."
    return _shorten(raw, 180)


def _unique_lines(lines: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        item = str(line or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _parse_pipeline_blueprint(value: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "numeric_features": [],
        "categorical_features": [],
        "numeric_processing": [],
        "categorical_processing": [],
    }
    for raw in _normalize_plan_items_raw(value):
        if ":" not in raw:
            result.setdefault("notes", []).append(raw)
            continue
        key, body = raw.split(":", 1)
        normalized = _normalize_plan_key(key)
        item = body.strip()
        if not item:
            continue
        if normalized == "numeric":
            target = "numeric_processing" if _looks_like_processing_step(item) else "numeric_features"
            result[target].append(item)
        elif normalized == "categorical":
            target = "categorical_processing" if _looks_like_processing_step(item) else "categorical_features"
            result[target].append(item)
        else:
            result.setdefault(normalized, []).append(item)
    return result


def _looks_like_processing_step(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["imputer", "scaler", "encoder", "onehot", "one-hot", "strategy=", "handle_unknown"])


def _first_blueprint_value(blueprint: dict[str, list[str]], key: str) -> str:
    values = blueprint.get(key, [])
    return values[0] if values else ""


def _label_value(label: str, value: Any) -> str:
    text = str(value or "").strip()
    return f"`{label}`={text}" if text else ""


def _join_display_parts(parts: Iterable[Any]) -> str:
    return "<br>".join(str(part).strip() for part in parts if str(part or "").strip())


def render_user_pipeline_structure(out_dir: Path, summary: dict[str, Any]) -> str:
    structure = _read_json(out_dir / "internal" / "pipeline_structure.json")
    workspace_run = _read_json(out_dir / "internal" / "workspace_run.json")
    stages = structure.get("stages", []) if isinstance(structure, dict) else []
    included_stages = [stage for stage in stages if stage.get("included")]
    lines = [
        f"# {summary['trial_id']} 재현 파이프라인 구조도",
        "",
        "실행 기준은 `.py` 코드입니다. 이 문서는 노트북을 위에서 아래로 읽듯이 이번 trial의 실행 흐름과 재현 조건을 확인하기 위한 구조도입니다.",
        "",
        "## 재현 목표",
        "",
        f"- 같은 데이터와 같은 설정으로 로컬 `{_display_value(summary.get('metric'))}` 점수를 다시 계산합니다.",
        "- `outputs/metrics.json`과 `outputs/submission.csv`를 다시 생성할 수 있어야 합니다.",
        "- 세부 구현의 원본은 `code/`에 복사된 `.py` 파일과 내부 `pipeline_structure.json`입니다.",
        "",
        "## 기준 정보",
        "",
        f"- workspace: `{_display_value(summary.get('project_root'))}`",
        f"- 평가 지표: `{_display_value(summary.get('metric'))}`",
        f"- 목표 방향: `{_display_value(summary.get('objective'))}`",
        f"- 로컬 점수: `{_display_value(summary.get('local_score'))}`",
        "",
        "## 실행 명령 순서",
        "",
    ]
    command_lines = _workspace_command_lines(workspace_run)
    if command_lines:
        lines.extend(f"{index}. `{command}`" for index, command in enumerate(command_lines, 1))
    else:
        lines.extend(
            [
                "1. `python test_step.py`",
                "2. `python train_step.py`",
                "3. `python predict_step.py`",
            ]
        )
    lines.extend(["", "## 노트북형 실행 흐름", ""])
    for index, stage in enumerate(included_stages, 1):
        actual = _stage_items(stage.get("actual_applied"))
        details = _stage_detail_lines(stage)
        checks = _stage_items(stage.get("checks"))
        lines.extend(
            [
                f"### {index}. {_notebook_stage_title(stage)}",
                "",
                f"- 목적: {_display_value(stage.get('role'))}",
                f"- 입력: {_stage_input_summary(stage, summary)}",
            ]
        )
        if actual:
            lines.append("- 처리:")
            lines.extend(f"  - {item}" for item in actual[:8])
            if len(actual) > 8:
                lines.append("  - 나머지 세부 처리 항목은 내부 JSON을 확인합니다.")
        else:
            lines.append("- 처리: 내부 구조화 정보에서 명시된 처리 내용이 없습니다.")
        output = _stage_output_summary(stage)
        if output:
            lines.append(f"- 출력: {output}")
        if details:
            lines.append(f"- 핵심값: {'; '.join(_strip_detail_prefix(item) for item in details)}")
        if checks:
            lines.append(f"- 재현 체크: {_join_short(checks, limit=4, overflow_label='추가 체크는 내부 JSON 참조')}")
        locations = stage.get("code_locations", [])
        if locations:
            lines.append(f"- 코드: {_join_code_locations(locations, limit=3)}")
        lines.append("")

    lines.extend(
        [
            "## 내부 원본",
            "",
            f"- 구조화 JSON: `{summary.get('pipeline_structure_file')}`",
            "- 사용자용 설명은 위 구조도를 보고, 에이전트/재실행용 세부 데이터는 내부 JSON을 기준으로 확인합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _notebook_stage_title(stage: dict[str, Any]) -> str:
    names = {
        "imports_setup": "Setup / 준비",
        "data_load": "Data Load / 데이터 로드",
        "preprocessing": "Preprocessing / 전처리",
        "feature_representation": "Feature Construction / 피처 구성",
        "data_split_cv": "Data Split / 검증 분리",
        "data_augmentation": "Data Augmentation / 증강",
        "dataset_dataloader": "Dataset & DataLoader / 배치 구성",
        "model_definition": "Model Definition / 모델 정의",
        "loss_objective": "Loss & Metric / 손실·평가 기준",
        "training": "Training / 학습",
        "training_curve": "Training Curve / 학습 곡선",
        "evaluation": "Evaluation / 평가",
        "model_checkpoint": "Checkpoint / 모델 저장",
        "test_inference_output": "Test Inference & Output / 추론·출력",
    }
    stage_id = str(stage.get("id") or "")
    return names.get(stage_id, f"{_stage_display_name(stage)} / {stage_id or 'unknown'}")


def _workspace_command_lines(workspace_run: dict[str, Any]) -> list[str]:
    results = workspace_run.get("command_results") if isinstance(workspace_run, dict) else []
    commands: list[str] = []
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            command = str(result.get("command") or "").strip()
            if command:
                commands.append(_compact_command(command))
    if commands:
        return commands
    command_groups = workspace_run.get("commands") if isinstance(workspace_run, dict) else {}
    if not isinstance(command_groups, dict):
        return []
    for key in ["test", "train", "predict"]:
        values = command_groups.get(key) or []
        if isinstance(values, str):
            values = [values]
        for command in values:
            if str(command or "").strip():
                commands.append(_compact_command(str(command)))
    return commands


def _compact_command(command: str) -> str:
    return (
        command.replace("C:\\Users\\ASUS\\anaconda3\\python.exe", "python")
        .replace(str(paths.project_root()) + "\\", "")
        .replace(str(paths.project_root()) + "/", "")
    )


def _stage_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _stage_input_summary(stage: dict[str, Any], summary: dict[str, Any]) -> str:
    details = stage.get("structured_details") if isinstance(stage.get("structured_details"), dict) else {}
    stage_id = str(stage.get("id") or "")
    if stage_id == "imports_setup":
        return "workspace 경로, 설정 파일, 필요한 Python 라이브러리"
    if stage_id == "data_load":
        files = details.get("required_files") or []
        target = details.get("target_column")
        id_column = details.get("id_column")
        parts = []
        if files:
            parts.append(f"데이터 파일 {_inline_list(files, 6)}")
        if target:
            parts.append(f"타깃 `{target}`")
        if id_column:
            parts.append(f"ID `{id_column}`")
        return ", ".join(parts) if parts else "원본 train/test 데이터"
    if stage_id == "preprocessing":
        raw = details.get("raw_feature_columns") or []
        return f"원본 피처 {_inline_list(raw, 12)}" if raw else "로드된 학습 데이터"
    if stage_id == "feature_representation":
        raw = details.get("raw_feature_columns") or details.get("final_feature_columns") or []
        return f"전처리 대상 피처 {_inline_list(raw, 12)}" if raw else "전처리된 피처"
    if stage_id == "data_split_cv":
        return "타깃이 포함된 학습 데이터와 split 설정"
    if stage_id in {"model_definition", "loss_objective"}:
        return "전처리 파이프라인과 평가 설정"
    if stage_id == "training":
        return "학습 split, 검증 split, 모델 정의"
    if stage_id == "evaluation":
        return "검증 데이터 예측값과 정답"
    if stage_id == "test_inference_output":
        id_column = details.get("id_column") or summary.get("id_column")
        return f"test 데이터와 ID `{id_column}`" if id_column else "test 데이터와 학습된 파이프라인"
    return "이전 단계의 출력"


def _stage_output_summary(stage: dict[str, Any]) -> str:
    details = stage.get("structured_details") if isinstance(stage.get("structured_details"), dict) else {}
    stage_id = str(stage.get("id") or "")
    if stage_id == "imports_setup":
        return "재현 실행에 필요한 경로 상수와 설정값"
    if stage_id == "data_load":
        return "학습용 DataFrame, 테스트용 DataFrame, 타깃/ID 컬럼 확인 결과"
    if stage_id == "preprocessing":
        return "결측치 처리, 스케일링, 인코딩이 포함된 전처리 파이프라인"
    if stage_id == "feature_representation":
        final_features = details.get("final_feature_columns") or []
        return f"최종 모델 입력 피처 {_inline_list(final_features, 12)}" if final_features else "모델 입력 feature set"
    if stage_id == "data_split_cv":
        return "학습 split과 검증 split"
    if stage_id == "model_definition":
        estimator = details.get("estimator")
        return f"`{estimator}` 모델 객체" if estimator else "모델 객체"
    if stage_id == "loss_objective":
        metric = details.get("metric")
        return f"`{metric}` 기준의 검증 점수" if metric else "평가 기준"
    if stage_id == "training":
        return "fit이 완료된 모델/파이프라인"
    if stage_id == "evaluation":
        score_parts = []
        for key in ["cv_score", "validation_accuracy"]:
            if details.get(key) is not None:
                score_parts.append(f"{key} `{details[key]}`")
        return ", ".join(score_parts) if score_parts else "검증 metric 기록"
    if stage_id == "model_checkpoint":
        checkpoint = details.get("checkpoint_path")
        return f"모델 checkpoint `{checkpoint}`" if checkpoint else "선택적 모델 checkpoint"
    if stage_id == "test_inference_output":
        submission = details.get("submission_path")
        return f"제출/추론 파일 `{submission}`" if submission else "test 예측 파일"
    return "다음 단계로 전달되는 중간 산출물"


def _strip_detail_prefix(text: str) -> str:
    for prefix in ["세부: ", "점수: ", "출력: ", "모델 설정: ", "평가 기준: ", "저장 위치: "]:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


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
        f"| 제출 준비 | {_display_value(summary.get('submission_prepare_status'))} |",
        f"| 제출 LB 점수 | {_display_value(summary.get('submitted_lb_score'))} |",
        f"| 현재 best 제출 | {_display_value(summary.get('is_best_submission'))} |",
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


def render_user_submission(summary: dict[str, Any]) -> str:
    manifest = summary.get("submit_manifest") if isinstance(summary.get("submit_manifest"), dict) else {}
    submission_run = summary.get("submission_run") if isinstance(summary.get("submission_run"), dict) else {}
    submission_result = summary.get("submission_result") if isinstance(summary.get("submission_result"), dict) else {}
    lines = [
        f"# {summary['trial_id']} 제출 준비",
        "",
        "## 제출 준비 상태",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 상태 | {_display_value(manifest.get('status') or summary.get('submission_prepare_status'))} |",
        f"| 제출 파일 | {_display_value(manifest.get('submission_file') or summary.get('submission_file'))} |",
        f"| 로컬 점수 | {_display_value(manifest.get('cv_score') or summary.get('local_score'))} |",
        f"| 목표 방향 | {_display_value(manifest.get('objective') or summary.get('objective'))} |",
        f"| 사용자 승인 필요 | {_display_value(manifest.get('requires_user_approval'))} |",
        f"| 승인됨 | {_display_value(manifest.get('approved'))} |",
        f"| 다음 단계 | {_display_value(manifest.get('next_step'))} |",
        "",
        "## 확인해야 할 것",
        "",
    ]
    checks = manifest.get("checks") if isinstance(manifest.get("checks"), list) else []
    if checks:
        lines.extend(f"- {item}" for item in checks)
    elif manifest:
        lines.append("- 제출 파일과 metrics 파일이 존재합니다.")
    else:
        lines.append("- 아직 제출 준비 manifest가 없습니다. 로컬 실행과 metrics 수집을 먼저 완료해야 합니다.")
    lines.extend(
        [
            "",
            "## 사용 방법",
            "",
            "- 실제 제출은 사용자가 승인한 뒤에만 진행합니다.",
            "- Kaggle/DACON 웹에서 수동 제출했다면 LB 점수와 rank를 `record-submission`으로 기록합니다.",
            "- 자동 제출을 붙일 경우에도 이 manifest를 먼저 확인한 뒤 진행합니다.",
            "",
        ]
    )
    if submission_run or submission_result:
        lines.extend(
            [
                "## 제출/기록 결과",
                "",
                f"- 상태: {_display_value(submission_run.get('status'))}",
                f"- 제출 후 점수: {_display_value(submission_result.get('submitted_lb_score') or submission_run.get('submitted_lb_score'))}",
                f"- 제출 후 rank: {_display_value(submission_result.get('submitted_rank') or submission_run.get('submitted_rank'))}",
                f"- 현재 best 제출: {_display_value(submission_result.get('is_best'))}",
                f"- 기록 메모: {_display_value(submission_result.get('notes'))}",
                "",
            ]
        )
    return "\n".join(lines)


def render_user_decision(summary: dict[str, Any]) -> str:
    card = summary.get("decision_card") if isinstance(summary.get("decision_card"), dict) else {}
    constraints = card.get("planner_constraints") if isinstance(card.get("planner_constraints"), list) else []
    rejected = card.get("rejected_axes") if isinstance(card.get("rejected_axes"), list) else summary.get("rejected_axes", [])
    lines = [
        f"# {summary['trial_id']} 다음 실험 판단",
        "",
        "## 핵심 판단",
        "",
        f"- decision: {_display_value(card.get('decision') or summary.get('trial_decision'))}",
        f"- change_axis: {_display_value(card.get('change_axis'))}",
        f"- source_trial_id: {_display_value(card.get('source_trial_id'))}",
        f"- recommended_base_trial: {_display_value(card.get('recommended_base_trial') or summary.get('recommended_base_trial'))}",
        "",
        "## 점수 비교",
        "",
        f"- local_score: {_display_value(card.get('local_score') or summary.get('local_score'))}",
        f"- previous_local_score: {_display_value(card.get('previous_local_score'))}",
        f"- local_status: {_display_value(card.get('local_status'))}",
        f"- local_delta: {_display_value(card.get('local_delta'))}",
        f"- lb_score: {_display_value(card.get('lb_score') or summary.get('submitted_lb_score'))}",
        f"- previous_lb_score: {_display_value(card.get('previous_lb_score'))}",
        f"- lb_status: {_display_value(card.get('lb_status'))}",
        f"- lb_delta: {_display_value(card.get('lb_delta'))}",
        "",
        "## 다음 계획 제약",
        "",
    ]
    lines.extend(f"- {item}" for item in constraints or ["아직 판단 카드가 없습니다."])
    lines.extend(["", "## 기각/보류된 개선축", ""])
    lines.extend(f"- {item}" for item in rejected or ["None"])
    lines.extend(["", "## 메모", "", _display_value(card.get("next_guidance")), ""])
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
        "5. `05_submission.ko.md`: 제출 준비 상태와 다음 조치",
        "6. `code/`: 이번 trial 코드 복사본",
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


def _read_plan_json(out_dir: Path) -> dict[str, Any]:
    for path in [out_dir / "internal" / "demo_experiment_plan.json", out_dir / "demo_experiment_plan.json"]:
        data = _read_json(path)
        if data:
            return data
    return {}


def _planned_choice_lines(out_dir: Path, plan: dict[str, Any] | None = None) -> list[str]:
    plan_items = _plan_note_lines(plan or {})
    if plan_items:
        return [_shorten(item, 180) for item in plan_items[:8]]
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


def _plan_note_lines(plan: dict[str, Any]) -> list[str]:
    if not isinstance(plan, dict):
        return []
    plan_type = plan.get("plan_type")
    lines: list[str] = []
    if _is_continuation_plan(plan):
        axis = str(plan.get("primary_change_axis") or "").strip()
        if axis:
            lines.append(f"Primary change axis: {axis}")
        candidate = plan.get("candidate") if isinstance(plan.get("candidate"), dict) else {}
        candidate_name = str(candidate.get("name") or "").strip()
        candidate_description = str(candidate.get("description") or "").strip()
        if candidate_name:
            lines.append(f"Candidate: {candidate_name}")
        if candidate_description:
            lines.append(f"Candidate description: {candidate_description}")
        lines.extend(f"Keep unchanged: {item}" for item in _normalize_plan_items_raw(plan.get("keep_unchanged"))[:6])
        lines.extend(f"Change detail: {item}" for item in _normalize_plan_items_raw(plan.get("change_details"))[:6])
        lines.extend(f"Code target: {item}" for item in _normalize_plan_items_raw(plan.get("code_change_targets"))[:4])
        lines.extend(f"Success criterion: {item}" for item in _normalize_plan_items_raw(plan.get("success_criteria"))[:4])
        lines.extend(f"Failure decision: {item}" for item in _normalize_plan_items_raw(plan.get("failure_decision"))[:3])
    else:
        lines.extend(f"Pipeline blueprint: {item}" for item in _normalize_plan_items_raw(plan.get("pipeline_blueprint"))[:8])
        lines.extend(f"Code target: {item}" for item in _normalize_plan_items_raw(plan.get("code_change_targets"))[:4])
        lines.extend(f"Success criterion: {item}" for item in _normalize_plan_items_raw(plan.get("success_criteria"))[:4])
    notes = _normalize_plan_items(plan.get("implementation_notes"))
    lines.extend(item for item in notes if item)
    return [item for item in lines if item][:12]


def _implementation_note_buckets(value: Any) -> dict[str, list[str]]:
    buckets = {"keep_unchanged": [], "change_details": [], "code_change_targets": []}
    prefixes = {
        "keep unchanged": "keep_unchanged",
        "change details": "change_details",
        "code change targets": "code_change_targets",
    }
    for item in _normalize_plan_items_raw(value):
        label, separator, body = item.partition(":")
        if not separator:
            continue
        bucket = prefixes.get(label.strip().casefold())
        if bucket and body.strip():
            buckets[bucket].append(body.strip())
    return buckets


def _normalize_plan_items_raw(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    items = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in items:
        result.extend(_flatten_plan_item_raw(_coerce_plan_item(item)))
    return [item for item in result if item]


def _flatten_plan_item_raw(item: Any, *, label: str | None = None) -> list[str]:
    if isinstance(item, dict):
        lines: list[str] = []
        for key, value in item.items():
            lines.extend(_flatten_plan_item_raw(value, label=str(key).replace("_", " ")))
        return lines
    if isinstance(item, list):
        lines: list[str] = []
        for value in item:
            lines.extend(_flatten_plan_item_raw(value, label=label))
        return lines
    text = str(item).strip()
    if not text:
        return []
    return [f"{label}: {text}" if label else text]


def _normalize_plan_items(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    items = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in items:
        result.extend(_flatten_plan_item(_coerce_plan_item(item)))
    return [item for item in result if item]


def _coerce_plan_item(item: Any) -> Any:
    if not isinstance(item, str):
        return item
    stripped = item.strip()
    if not stripped or stripped[0] not in "[{":
        return item
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return item


def _flatten_plan_item(item: Any, *, label: str | None = None) -> list[str]:
    if isinstance(item, dict):
        lines: list[str] = []
        for key, value in item.items():
            lines.extend(_flatten_plan_item(value, label=_plan_label(str(key))))
        return lines
    if isinstance(item, list):
        lines = []
        for value in item:
            lines.extend(_flatten_plan_item(value, label=label))
        return lines
    text = str(item).strip()
    if not text:
        return []
    return [_format_plan_line(label, text)]


def _plan_label(key: str) -> str:
    normalized = _normalize_plan_key(key)
    labels = {
        "applied_data_assumptions": "데이터 가정",
        "applied_data": "데이터 적용",
        "data": "데이터",
        "split": "검증 분리",
        "validation_split": "검증 분리",
        "preprocessing": "전처리",
        "model": "모델",
        "commands": "실행 명령",
        "prediction_output": "예측/제출 출력",
        "testing": "테스트",
        "artifact_policy": "산출물 정책",
        "workspace_constraints": "작업 범위",
        "write_scope": "수정 범위",
        "metrics": "지표 파일",
        "metrics_artifact": "지표 파일",
        "submission": "제출 파일",
        "submission_artifact": "제출 파일",
        "code_snapshot": "코드 스냅샷",
        "pipeline_summary": "파이프라인 요약",
        "model_artifact": "모델 파일",
    }
    return labels.get(normalized, key.replace("_", " "))


def _normalize_plan_key(key: str) -> str:
    return "_".join(key.strip().lower().replace("-", " ").replace("/", " ").split())


def _format_plan_line(label: str | None, text: str) -> str:
    if label is None and ":" in text:
        prefix, body = text.split(":", 1)
        if len(prefix.strip()) <= 40:
            label = _plan_label(prefix.strip())
            text = body.strip()
    translated = _koreanize_plan_text(label, text)
    return f"{label}: {translated}" if label else translated


def _koreanize_plan_text(label: str | None, text: str) -> str:
    lowered = text.lower()
    key = _normalize_plan_key(label or "")
    if key in {"데이터_가정", "데이터_적용", "데이터", "applied_data", "applied_data_assumptions"} or "data" in key:
        if "titanic" in lowered:
            return "Titanic 표 형식 데이터를 사용하고, 원본 `data/` 폴더는 수정하지 않습니다."
        return _shorten(text, 180)
    if key in {"검증_분리", "validation_split", "split"} or "validation" in key or "split" in key:
        if "strat" in lowered:
            return "타깃 분포를 유지하는 고정 holdout 검증 분리를 사용합니다."
        if "holdout" in lowered or "train/validation" in lowered:
            return "고정 train/validation holdout 분리로 로컬 검증 점수를 계산합니다."
        return _shorten(text, 180)
    if key in {"전처리", "preprocessing"} or "preprocess" in key:
        parts = []
        if "median" in lowered:
            parts.append("수치형 결측치는 median으로 대체")
        if "most frequent" in lowered or "most_frequent" in lowered:
            parts.append("범주형 결측치는 most_frequent로 대체")
        if "one-hot" in lowered or "onehot" in lowered:
            parts.append("범주형은 one-hot 인코딩")
        if "leakage" in lowered or "training fold" in lowered:
            parts.append("전처리는 학습 fold에만 fit")
        return ", ".join(parts) + "." if parts else _shorten(text, 180)
    if key in {"모델", "model"} or "model" in key:
        if "logistic" in lowered:
            return "빠르고 재현 가능한 단일 `LogisticRegression` baseline을 사용합니다."
        if "randomforest" in lowered or "random forest" in lowered:
            return "단일 `RandomForestClassifier` baseline을 후보로 사용합니다."
        return _shorten(text, 180)
    if key in {"예측_제출_출력", "prediction_output", "제출_파일", "submission", "submission_artifact"} or "prediction" in key or "submission" in key:
        return "`predict_step.py`에서 전체 학습 데이터로 재학습한 뒤 `outputs/submission.csv`를 생성합니다."
    if key in {"산출물_정책", "artifact_policy", "모델_파일", "model_artifact"} or "artifact" in key:
        if "do not persist" in lowered or "no trained model" in lowered or "no model" in lowered:
            return "모델 파일은 기본 저장하지 않고, metrics/submission/code snapshot/pipeline summary를 주요 기록으로 남깁니다."
        return _shorten(text, 180)
    if key in {"작업_범위", "수정_범위", "workspace_constraints", "write_scope"} or "scope" in key or "constraints" in key:
        return "Execution Profile의 허용 경로 안에서만 코드를 수정하고, `data/`와 결과 산출물은 직접 수정하지 않습니다."
    if key in {"테스트", "testing"} or "test" in key:
        return "`test_step.py`로 import, 학습/예측 실행, metrics/submission 생성 여부를 검증합니다."
    if key in {"실행_명령", "commands", "execution_commands"} or "command" in key:
        return "`test_step.py` 검증 후 `train_step.py`로 학습하고, `predict_step.py`로 제출 파일을 생성합니다."
    if key in {"지표_파일", "metrics", "metrics_artifact"} or "metrics" in key:
        return "`outputs/metrics.json`에 검증 점수, split, 행 수, 모델/전처리 요약을 기록합니다."
    if key in {"코드_스냅샷", "code_snapshot"}:
        return "이번 trial의 주요 코드 파일을 스냅샷으로 남겨 다음 회차가 참고할 수 있게 합니다."
    if key in {"파이프라인_요약", "pipeline_summary"}:
        return "데이터 가정, split, 전처리, 모델, 검증 metric, 제출 생성 방식을 요약합니다."
    return _shorten(text, 180)


def _koreanize_expected_output(text: str) -> str:
    lowered = str(text).lower()
    if "metrics.json" in lowered:
        return "`outputs/metrics.json`: 로컬 검증 점수와 파이프라인 메타데이터를 기록합니다."
    if "submission.csv" in lowered:
        return "`outputs/submission.csv`: 계획에 명시된 ID와 예측 컬럼 형식의 파일을 생성합니다."
    if "pipeline summary" in lowered or "pipeline_summary.json" in lowered:
        return "파이프라인 요약: 데이터 가정, split, 전처리, 모델, 출력 형식을 기록합니다."
    if "code snapshot" in lowered:
        return "코드 스냅샷: 이번 trial의 주요 코드 파일을 보관합니다."
    if "no persisted" in lowered or "trained model" in lowered or "model artifact" in lowered:
        return "모델 파일: 기본 저장하지 않고, 필요 사유가 있을 때만 별도 정책에 따라 저장합니다."
    return _shorten(str(text), 180)


def _koreanize_plan_summary(text: str, *, kind: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if kind == "objective":
        if "previous best/local score" in lowered or "previous best" in lowered or "previous local" in lowered:
            return "기준 trial의 로컬 검증 점수를 기준으로, 이번 변경이 실제로 성능을 개선하는지 확인합니다."
        if "baseline" in lowered and "metrics.json" in lowered and "submission.csv" in lowered:
            return "재현 가능한 첫 baseline 파이프라인을 만들고, `metrics.json`과 `submission.csv`까지 생성되는지 검증합니다."
        if "baseline" in lowered:
            return "첫 실험으로 빠르게 실행 가능한 baseline 파이프라인을 구성합니다."
    if kind == "rationale":
        parts = []
        if "first trial" in lowered or "no recent" in lowered or "no best" in lowered:
            parts.append("이전 실험이 없는 첫 회차이므로 안정적인 baseline을 우선 구성합니다")
        if "accuracy" in lowered:
            parts.append("평가 지표는 accuracy이므로 로컬 검증 점수를 기록합니다")
        if "execution profile" in lowered:
            parts.append("Execution Profile의 허용 경로와 실행 명령을 기준으로 작업합니다")
        if parts:
            return "; ".join(parts) + "."
    return _shorten(text, 240)


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


def _join_short(items: Iterable[Any], limit: int = 2, *, overflow_label: str | None = None) -> str:
    values = [_shorten(str(item), 120) for item in items if item]
    if not values:
        return "-"
    shown = values[:limit]
    if len(values) > limit and overflow_label:
        shown.append(overflow_label)
    return "<br>".join(shown)


def _join_code_locations(items: Iterable[Any], limit: int = 3) -> str:
    values = [f"`{str(item)}`" for item in items if item]
    if not values:
        return "-"
    shown = values[:limit]
    if len(values) > limit:
        shown.append("전체 목록은 `03_code_pipeline.ko.md` 참조")
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


def render_user_plan(out_dir: Path, summary: dict[str, Any]) -> str:
    plan = _read_plan_json(out_dir)
    objective = _compact_block(_read_markdown_section(out_dir / "demo_experiment_plan.md", "Objective"))
    rationale = _compact_block(_read_markdown_section(out_dir / "demo_experiment_plan.md", "Rationale"))
    title = summary.get("plan_title") or plan.get("title") or "-"
    axis = plan.get("primary_change_axis") or summary.get("change_axis") or summary.get("active_axis") or "baseline"
    source = plan.get("source_trial_id") or summary.get("recommended_base_trial") or "-"
    change_details = _normalize_plan_items(plan.get("change_details"))[:5]
    keep_unchanged = _normalize_plan_items(plan.get("keep_unchanged"))[:5]
    success = _success_criteria_lines(plan.get("success_criteria"))[:5]
    purpose_text = _user_plan_purpose(plan, objective, source, axis)
    rationale_text = _user_plan_rationale(plan, rationale, source, axis)

    lines = [
        f"# {summary['trial_id']} 실험 계획",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 대회 | `{summary['competition']}` |",
        f"| 계획명 | {_display_value(title)} |",
        f"| 평가 지표 | `{_display_value(summary.get('metric'))}` |",
        f"| 목표 방향 | `{_display_value(summary.get('objective'))}` |",
        f"| 기준 trial | {_display_value(source)} |",
        f"| 개선축 | {_display_value(axis)} |",
        "",
        "## 목적",
        "",
        f"- {purpose_text}",
        "",
        "## 왜 하는가",
        "",
        f"- {rationale_text}",
        "",
    ]
    if change_details:
        lines.extend(["## 이번에 바꾸는 것", ""])
        lines.extend(f"- {_shorten(_koreanize_continuation_item(item), 180)}" for item in change_details)
        lines.append("")
    if keep_unchanged:
        lines.extend(["## 그대로 두는 것", ""])
        lines.extend(f"- {_shorten(_koreanize_continuation_item(item), 180)}" for item in keep_unchanged)
        lines.append("")
    if success:
        lines.extend(["## 성공 기준", ""])
        lines.extend(f"- {_shorten(item, 180)}" for item in success)
        lines.append("")
    lines.extend(
        [
            "## 다음에 볼 것",
            "",
            "- `02_pipeline_structure.ko.md`: 데이터부터 제출 파일까지의 실행 흐름",
            "- `03_scores.ko.md`: 로컬 점수와 Kaggle 제출 점수",
            "",
        ]
    )
    return "\n".join(lines)


def render_user_scores(summary: dict[str, Any]) -> str:
    lb_score = summary.get("submitted_lb_score")
    if lb_score is None:
        lb_score = summary.get("lb_score")
    rank = summary.get("submitted_rank")
    if rank is None:
        rank = summary.get("rank")
    submitted = "기록됨" if lb_score is not None or rank is not None else "미기록"
    best_label = "Best" if summary.get("is_best_lb") else ("제출 점수 기록 후 판단" if lb_score is None else "-")
    return "\n".join(
        [
            f"# {summary['trial_id']} 점수",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| 상태 | {_display_value(summary.get('status'))} |",
            f"| 지표 | {_display_value(summary.get('metric'))} |",
            f"| 목표 방향 | {_display_value(summary.get('objective'))} |",
            f"| 로컬 점수 | {_display_value(summary.get('local_score'))} |",
            f"| 제출 상태 | {submitted} |",
            f"| 제출 LB 점수 | {_display_value(lb_score)} |",
            f"| 제출 순위 | {_display_value(rank)} |",
            f"| Best 표시 | {_display_value(best_label)} |",
            f"| 제출 파일 | {_display_value(summary.get('submission_file'))} |",
            "",
        ]
    )


def _user_plan_purpose(plan: dict[str, Any], objective: str, source: str, axis: str) -> str:
    plan_type = str(plan.get("plan_type") or "")
    if "initial" in plan_type or axis == "baseline":
        return "첫 회차 기준선으로 사용할 재현 가능한 baseline을 만들고 제출 파일 형식을 검증합니다."
    base = source if source and source != "-" else "이전 best trial"
    return f"`{base}`를 기준으로 `{axis}` 개선축 하나만 바꿔 성능 변화를 확인합니다."


def _user_plan_rationale(plan: dict[str, Any], rationale: str, source: str, axis: str) -> str:
    plan_type = str(plan.get("plan_type") or "")
    if "initial" in plan_type or axis == "baseline":
        return "복잡한 피처나 모델 탐색 전에 안정적인 전처리, 검증 split, 제출 형식을 먼저 고정해야 이후 trial과 공정하게 비교할 수 있습니다."
    base = source if source and source != "-" else "기준 trial"
    return f"`{base}`와 비교 가능하도록 split, 지표, 출력 형식은 유지하고 `{axis}` 변경의 효과만 분리해서 봅니다."
