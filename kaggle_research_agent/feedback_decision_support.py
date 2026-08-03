from __future__ import annotations

import re
from typing import Any


AXIS_INTERACTION_TYPES = {
    "validation": "domain_question",
    "preprocessing": "domain_question",
    "feature_engineering": "domain_question",
    "augmentation": "domain_question",
    "sampling": "constraint_confirmation",
    "loss_metric_alignment": "autonomous_action",
    "hyperparameter": "autonomous_action",
    "training_recipe": "autonomous_action",
    "model_architecture": "constraint_confirmation",
    "model_family": "autonomous_action",
    "pretraining_strategy": "constraint_confirmation",
    "post_processing": "constraint_confirmation",
    "ensemble_submission": "autonomous_action",
    "compute_backend": "external_cost_approval",
    "error_analysis": "domain_question",
    "human_review": "blocking_human_review",
}

INTERACTION_LABELS = {
    "autonomous_action": "에이전트 자율 진행",
    "domain_question": "도메인 정보 질문",
    "constraint_confirmation": "가치·운영 기준 확인",
    "external_cost_approval": "외부 계산 자원 승인",
    "blocking_human_review": "사람 검토 전 실행 차단",
}

AXIS_COPY = {
    "validation": {
        "title": "로컬 검증이 실제 제출 성능을 충분히 설명하지 못하고 있습니다",
        "interpretation": "현재 검증 분할이 실제 테스트 데이터의 그룹 또는 시간 관계를 재현하지 못할 가능성이 있습니다.",
        "recommendation": "모델은 유지하고 검증 분할부터 재점검하는 것을 권장합니다.",
        "why": "그룹 관계나 실제 예측 순서는 데이터 통계만으로 확정할 수 없습니다.",
        "question": "같은 그룹으로 묶어야 하는 열이나 실제 예측 순서를 나타내는 시간 열이 있나요?",
        "options": (
            ("group_split", "그룹 열이 있음", "그룹이 train과 validation에 섞이지 않도록 검증합니다.", "change_validation"),
            ("time_split", "시간 순서가 중요함", "과거 데이터로 학습하고 미래 데이터로 검증합니다.", "change_validation"),
            ("repeated_cv", "둘 다 없음", "반복 교차검증으로 분할 우연성을 줄입니다.", "change_validation"),
        ),
    },
    "preprocessing": {
        "title": "전처리 방식이 특정 데이터 구간의 특성을 지울 수 있습니다",
        "interpretation": "결측이나 이상치가 특정 수집 과정과 연결되어 있다면 하나의 전역 처리 규칙은 적합하지 않을 수 있습니다.",
        "recommendation": "결측 여부를 보존하고 그룹별 처리와 전역 처리를 비교하는 것을 권장합니다.",
        "why": "값이 비어 있거나 달라진 실제 원인은 수집 맥락을 아는 사용자가 확인해야 합니다.",
        "question": "문제가 된 값은 특정 집단이나 수집 과정에서 구조적으로 발생하나요?",
        "options": (
            ("group_specific", "특정 집단에서 발생", "집단별 전처리와 결측 여부 피처를 비교합니다.", "consider_in_next_plan"),
            ("random", "대체로 무작위", "전역 전처리를 유지하고 안정성을 재검증합니다.", "consider_in_next_plan"),
            ("unknown", "원인을 모름", "데이터를 바꾸지 않고 두 후보를 비교합니다.", "inspect_more_samples"),
        ),
    },
    "sampling": {
        "title": "놓침과 과잉 경고 중 어느 오류를 더 감수할지 결정해야 합니다",
        "interpretation": "이 선택은 모델 기술보다 실제 업무에서 발생하는 오류 비용에 좌우됩니다.",
        "recommendation": "더 큰 피해를 만드는 오류를 기준으로 sampling 목표를 정해야 합니다.",
        "why": "오류 한 건의 실제 비용은 실험 점수에 포함되어 있지 않습니다.",
        "question": "중요 사례를 놓치는 것과 잘못 경고하는 것 중 어느 쪽의 피해가 더 큰가요?",
        "options": (
            ("recall", "놓침이 더 위험", "recall을 우선하고 더 많은 오경보를 허용합니다.", "consider_in_next_plan"),
            ("precision", "오경보가 더 부담", "precision을 우선하고 일부 놓침을 허용합니다.", "consider_in_next_plan"),
            ("balanced", "비용이 비슷함", "두 지표의 균형점을 사용합니다.", "consider_in_next_plan"),
        ),
    },
    "post_processing": {
        "title": "판정 기준에 따라 놓침과 오경보의 비율이 달라집니다",
        "interpretation": "threshold의 정답은 점수만이 아니라 실제 오류 비용으로 결정됩니다.",
        "recommendation": "업무상 더 위험한 오류를 확인한 뒤 threshold를 선택해야 합니다.",
        "why": "놓침과 오경보의 실제 위험도는 사용자가 정해야 합니다.",
        "question": "실제 사용에서 양성을 놓치는 것과 잘못 양성으로 경고하는 것 중 어느 쪽이 더 위험한가요?",
        "options": (
            ("recall", "놓침이 더 위험", "낮은 threshold로 recall을 우선합니다.", "consider_in_next_plan"),
            ("precision", "오경보가 더 위험", "높은 threshold로 precision을 우선합니다.", "consider_in_next_plan"),
            ("balanced", "비용이 비슷함", "균형 지점의 threshold를 사용합니다.", "consider_in_next_plan"),
        ),
    },
    "compute_backend": {
        "title": "현재 환경의 시간 또는 메모리 한계가 예상됩니다",
        "interpretation": "이 결정은 연구축이 아니라 외부 계산 자원의 비용과 사용 권한에 관한 승인입니다.",
        "recommendation": "실행 연결이 완성되기 전까지는 추가 비용을 사용하지 않고 로컬 경량화를 기본값으로 둡니다.",
        "why": "에이전트는 유료 GPU나 외부 인프라를 사용자 승인 없이 사용할 수 없습니다.",
        "question": "추가 비용이 발생할 수 있는 외부 계산 환경 사용을 검토할까요?",
        "options": (
            ("local_only", "현재 환경만 사용", "모델과 탐색 범위를 줄이는 제약을 다음 계획에 기록합니다.", "record_constraint_only"),
            ("estimate_first", "비용 추정 후 결정", "실행하지 않고 예상 자원과 비용 산출 요청만 기록합니다.", "record_constraint_only"),
            ("external_allowed", "외부 환경 사용 허용", "승인 의사만 기록합니다. 현재 버전에서는 외부 실행이 자동 시작되지 않습니다.", "record_constraint_only"),
        ),
        "execution_supported": False,
    },
    "error_analysis": {
        "title": "같은 유형의 오류가 반복되고 있습니다",
        "interpretation": "이 사례가 데이터 문제인지 실제 예외인지 확인해야 다음 변경축을 올바르게 고를 수 있습니다.",
        "recommendation": "대표 오류 사례를 확인한 뒤 데이터 정제 또는 도메인 규칙 적용 여부를 결정합니다.",
        "why": "현실적인 예외와 잘못된 데이터는 통계만으로 구분하기 어렵습니다.",
        "question": "반복 오류는 수정해야 할 데이터 문제인가요, 별도 규칙이 필요한 실제 예외인가요?",
        "options": (
            ("data_issue", "데이터 문제", "원본 값을 정제하거나 제외한 뒤 재검증합니다.", "consider_in_next_plan"),
            ("real_exception", "실제 예외", "예외를 설명하는 피처나 규칙을 시험합니다.", "consider_in_next_plan"),
            ("inspect", "판단하기 어려움", "대표 사례를 더 수집하고 현재 모델을 유지합니다.", "inspect_more_samples"),
        ),
    },
    "human_review": {
        "title": "정답 기준이 모호해 사람의 확인 없이 계속하기 어렵습니다",
        "interpretation": "모델 성능보다 라벨 또는 업무 기준을 먼저 정리해야 하는 차단 상황입니다.",
        "recommendation": "대표 모호 사례의 판정 기준을 정한 뒤 라벨을 다시 점검합니다.",
        "why": "정답의 업무적 정의는 모델이나 통계가 대신 결정할 수 없습니다.",
        "question": "모호한 경계 사례를 어떤 기준으로 처리해야 하나요?",
        "options": (
            ("conservative", "보수적 기준", "확실한 사례만 양성으로 처리합니다.", "consider_in_next_plan"),
            ("sensitive", "민감한 기준", "의심 사례도 양성에 포함해 놓침을 줄입니다.", "consider_in_next_plan"),
            ("exclude", "판단 보류", "모호 사례를 학습에서 제외하고 별도 검토합니다.", "inspect_more_samples"),
        ),
    },
}


def build_decision_support(diagnosis: dict[str, Any]) -> dict[str, Any]:
    existing = diagnosis.get("decision_support")
    if isinstance(existing, dict) and existing.get("schema_version"):
        return existing

    axis = _infer_axis(diagnosis)
    interaction_type = AXIS_INTERACTION_TYPES.get(axis, "domain_question")
    copy = _specialized_copy(diagnosis, axis, AXIS_COPY.get(axis) or _generic_copy(diagnosis))
    options = [
        {
            "value": value,
            "label": label,
            "impact": impact,
            "follow_up_action": follow_up,
        }
        for value, label, impact, follow_up in copy.get("options", ())
    ]
    problem = _problem_text(diagnosis)
    evidence = _score_evidence(diagnosis)
    execution_supported = bool(copy.get("execution_supported", True))
    default = str(copy.get("default") or _default_action(axis, interaction_type))
    return {
        "schema_version": "1.0",
        "axis": axis,
        "interaction_type": interaction_type,
        "interaction_label": INTERACTION_LABELS[interaction_type],
        "title": copy["title"],
        "problem": problem,
        "evidence_snapshot": evidence,
        "interpretation": copy["interpretation"],
        "recommendation": copy["recommendation"],
        "why_user_needed": copy["why"],
        "question": copy["question"],
        "options": options,
        "default_if_no_response": default,
        "execution_supported": execution_supported,
        "execution_note": (
            ""
            if execution_supported
            else "현재 버전은 답변을 연구 제약으로 기록하지만 외부 계산 환경을 자동 실행하지 않습니다."
        ),
    }


def pending_action_payload(decision_support: dict[str, Any], *, context_files: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "title": decision_support["title"],
        "topic": decision_support["axis"],
        "question": decision_support["question"],
        "problem": decision_support["problem"],
        "evidence_snapshot": decision_support["evidence_snapshot"],
        "interpretation": decision_support["interpretation"],
        "recommendation": decision_support["recommendation"],
        "why_user_needed": decision_support["why_user_needed"],
        "options": decision_support["options"],
        "default_if_no_response": decision_support["default_if_no_response"],
        "interaction_type": decision_support["interaction_type"],
        "interaction_label": decision_support["interaction_label"],
        "execution_supported": decision_support["execution_supported"],
        "execution_note": decision_support["execution_note"],
        "context_files": context_files,
        "questions": [
            {
                "id": "decision",
                "label": decision_support["question"],
                "answer_type": "choice",
                "choices": [
                    {"value": item["value"], "label": item["label"]}
                    for item in decision_support["options"]
                ],
                "required": True,
            }
        ],
    }


def _infer_axis(diagnosis: dict[str, Any]) -> str:
    for key in ("change_axis", "primary_axis", "active_axis"):
        value = str(diagnosis.get(key) or "").strip()
        if value in AXIS_INTERACTION_TYPES:
            return value
    text = " ".join(
        [
            *[str(item) for item in diagnosis.get("issues", [])],
            *[str(item) for item in diagnosis.get("user_questions", [])],
        ]
    ).casefold()
    if any(term in text for term in ("out of memory", "gpu", "runtime", "compute", "resource")):
        return "compute_backend"
    if any(term in text for term in ("cv/lb", "validation", "leakage", "split")):
        return "validation"
    if any(term in text for term in ("segment", "concentrated", "error cluster")):
        return "error_analysis"
    if any(term in text for term in ("label", "ambiguous", "boundary")):
        return "human_review"
    return "human_review"


def _specialized_copy(
    diagnosis: dict[str, Any],
    axis: str,
    base_copy: dict[str, Any],
) -> dict[str, Any]:
    issue_text = " ".join(str(item) for item in diagnosis.get("issues", [])).casefold()
    if "leakage" in issue_text or diagnosis.get("leakage_warning"):
        features = _leakage_features(diagnosis)
        feature_text = ", ".join(f"`{item}`" for item in features) if features else "의심 피처 또는 분할 정보"
        return {
            "title": "예측 시점에 사용할 수 없는 정보가 학습에 포함됐을 가능성이 있습니다",
            "interpretation": (
                f"{feature_text}가 정답 발생 이후에 만들어지는 정보라면 로컬 점수는 실제보다 과도하게 높아집니다. "
                "현재 결과를 성능 개선으로 간주하기 전에 정보 생성 시점을 확인해야 합니다."
            ),
            "recommendation": (
                "의심 정보를 일단 격리한 뒤 동일한 검증 조건으로 다시 실행하고, "
                "예측 시점에 사용 가능하다는 근거가 확인된 경우에만 복원합니다."
            ),
            "why": "피처가 언제 생성되고 실제 예측 시점에 존재하는지는 데이터 값만으로 확정할 수 없습니다.",
            "question": f"{feature_text}는 실제 예측 시점에도 확실히 사용할 수 있는 정보인가요?",
            "options": (
                (
                    "available_at_prediction",
                    "예측 시점에 사용 가능",
                    "생성 시점 근거를 기록하고 시간·그룹 누수가 없는 검증으로 다시 확인합니다.",
                    "change_validation",
                ),
                (
                    "post_prediction",
                    "예측 이후 생성됨",
                    "해당 정보를 피처에서 제외하고 로컬 검증과 제출 파일을 다시 생성합니다.",
                    "exclude_leakage_and_revalidate",
                ),
                (
                    "unknown_quarantine",
                    "사용 가능 여부를 모름",
                    "의심 정보를 격리한 안전한 파이프라인으로 재검증하고 데이터 정의를 추가 조사합니다.",
                    "exclude_leakage_and_revalidate",
                ),
            ),
            "default": "답변이 없으면 의심 정보를 피처에서 제외하고 동일 조건으로 다시 검증합니다.",
        }
    if axis == "error_analysis" and _segment_error_rows(diagnosis):
        segment_text = _segment_summary_text(diagnosis)
        return {
            **base_copy,
            "title": "특정 데이터 구간에서 같은 오류가 반복되고 있습니다",
            "interpretation": (
                f"{segment_text} 전체 점수만 바꾸는 모델 교체보다 이 구간이 데이터 오류인지 "
                "실제 예외인지 먼저 구분해야 다음 개선축을 올바르게 선택할 수 있습니다."
            ),
            "recommendation": (
                "오류율이 높은 구간과 대표 사례를 확인한 뒤 데이터 정제, 전용 피처, "
                "별도 규칙 중 하나를 다음 실험 계획에 반영합니다."
            ),
            "why": "통계만으로는 잘못 기록된 데이터와 현실에서 드물게 발생하는 정상 예외를 구분하기 어렵습니다.",
            "question": "표시된 고오류 구간은 수정해야 할 데이터 문제인가요, 모델이 학습해야 할 실제 예외인가요?",
            "default": "답변이 없으면 데이터를 바꾸지 않고 대표 오류 사례를 더 수집한 뒤 다시 요청합니다.",
        }
    return dict(base_copy)


def _problem_text(diagnosis: dict[str, Any]) -> str:
    issue_text = " ".join(str(item) for item in diagnosis.get("issues", [])).casefold()
    if "leakage" in issue_text or diagnosis.get("leakage_warning"):
        features = _leakage_features(diagnosis)
        feature_text = ", ".join(features) if features else "하나 이상의 피처 또는 데이터 분할"
        return (
            f"{feature_text}에 leakage 경고가 감지됐습니다. "
            "이 정보가 실제 예측 시점보다 나중에 생성된다면 현재 로컬 점수를 신뢰할 수 없습니다."
        )
    if "cv/lb" in issue_text:
        local_delta = _delta(diagnosis.get("cv_score"), diagnosis.get("best_cv_before"))
        submit_delta = _delta(diagnosis.get("lb_score"), diagnosis.get("best_lb_before"))
        return (
            f"로컬 점수는 이전 최고 대비 {_signed(local_delta)} 변했지만 "
            f"제출 점수는 {_signed(submit_delta)} 변해 두 평가의 방향이 일치하지 않습니다."
        )
    if "concentrated" in issue_text or "segment" in issue_text:
        if _segment_error_rows(diagnosis):
            return f"{_segment_summary_text(diagnosis)} 이 고오류 구간이 여러 trial에서 반복됐습니다."
        return "특정 세그먼트에 오류가 집중됐지만 세그먼트 이름과 오류율 근거가 아직 충분하지 않습니다."
    issues = [str(item).strip() for item in diagnosis.get("issues", []) if str(item).strip()]
    if issues:
        return " ".join(issues[:3])
    return "에이전트가 다음 연구 결정을 확정하기에 필요한 정보가 부족합니다."


def _score_evidence(diagnosis: dict[str, Any]) -> list[dict[str, str]]:
    fields = [
        ("cv_score", "현재 로컬 점수", "내부 검증 데이터에서 측정한 점수입니다."),
        ("best_cv_before", "이전 최고 로컬 점수", "이번 trial 이전의 최고 내부 검증 점수입니다."),
        ("lb_score", "현재 제출 점수", "대회 서버의 숨겨진 정답으로 측정한 점수입니다."),
        ("best_lb_before", "이전 최고 제출 점수", "이번 trial 이전의 최고 제출 점수입니다."),
        ("consecutive_failures", "연속 비개선", "점수가 개선되지 않은 연속 trial 수입니다."),
    ]
    result = []
    for key, label, meaning in fields:
        value = diagnosis.get(key)
        if value is None:
            continue
        if key == "consecutive_failures" and int(value or 0) == 0:
            continue
        rendered = f"{value:.5f}" if isinstance(value, float) else str(value)
        result.append({"label": label, "value": rendered, "meaning": meaning})
    local_delta = _delta(diagnosis.get("cv_score"), diagnosis.get("best_cv_before"))
    if local_delta is not None:
        result.append(
            {
                "label": "로컬 점수 변화",
                "value": _signed(local_delta),
                "meaning": "현재 로컬 점수에서 이전 최고 로컬 점수를 뺀 값입니다.",
            }
        )
    submit_delta = _delta(diagnosis.get("lb_score"), diagnosis.get("best_lb_before"))
    if submit_delta is not None:
        result.append(
            {
                "label": "제출 점수 변화",
                "value": _signed(submit_delta),
                "meaning": "현재 제출 점수에서 이전 최고 제출 점수를 뺀 값입니다.",
            }
        )
    for segment, error_rate, sample_count in _segment_error_rows(diagnosis)[:3]:
        count_text = f", 표본 {sample_count}개" if sample_count is not None else ""
        result.append(
            {
                "label": f"고오류 구간: {segment}",
                "value": _rate_text(error_rate),
                "meaning": f"이 구간에서 관측된 오류율{count_text}입니다.",
            }
        )
    overall_error_rate = _number(diagnosis.get("overall_error_rate"))
    if overall_error_rate is not None:
        result.append(
            {
                "label": "전체 오류율",
                "value": _rate_text(overall_error_rate),
                "meaning": "전체 검증 데이터에서 관측된 오류율입니다.",
            }
        )
    features = _leakage_features(diagnosis)
    if features:
        result.append(
            {
                "label": "leakage 의심 정보",
                "value": ", ".join(features),
                "meaning": "실제 예측 시점에 사용 가능한지 확인해야 하는 피처 또는 데이터 정보입니다.",
            }
        )
    cases = [item for item in diagnosis.get("representative_error_cases", []) if isinstance(item, dict)]
    if cases:
        identifiers = [
            str(item.get("sample_id") or item.get("case_id") or item.get("id") or f"case_{index}")
            for index, item in enumerate(cases[:3], start=1)
        ]
        result.append(
            {
                "label": "대표 오류 사례",
                "value": ", ".join(identifiers),
                "meaning": "사용자가 원본 데이터 또는 review pack에서 확인할 수 있는 사례입니다.",
            }
        )
    return result


def _segment_error_rows(diagnosis: dict[str, Any]) -> list[tuple[str, float, int | None]]:
    raw = diagnosis.get("segment_errors")
    rows: list[tuple[str, float, int | None]] = []
    if isinstance(raw, dict):
        for segment, value in raw.items():
            if isinstance(value, dict):
                rate = _number(value.get("error_rate"), value.get("rate"))
                count = _integer(value.get("sample_count"), value.get("count"), value.get("n"))
            else:
                rate = _number(value)
                count = None
            if rate is not None:
                rows.append((str(segment), rate, count))
    elif isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            segment = str(item.get("segment") or item.get("group") or item.get("name") or f"segment_{index}")
            rate = _number(item.get("error_rate"), item.get("rate"))
            count = _integer(item.get("sample_count"), item.get("count"), item.get("n"))
            if rate is not None:
                rows.append((segment, rate, count))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def _segment_summary_text(diagnosis: dict[str, Any]) -> str:
    rows = _segment_error_rows(diagnosis)
    if not rows:
        return ""
    segment, error_rate, sample_count = rows[0]
    count_text = f", 표본 {sample_count}개" if sample_count is not None else ""
    overall = _number(diagnosis.get("overall_error_rate"))
    comparison = f"이며 전체 오류율 {_rate_text(overall)}보다 높습니다." if overall is not None else "입니다."
    return f"`{segment}` 구간의 오류율은 {_rate_text(error_rate)}{count_text}{comparison}"


def _leakage_features(diagnosis: dict[str, Any]) -> list[str]:
    raw = diagnosis.get("leakage_features") or diagnosis.get("suspected_leakage_features") or []
    if isinstance(raw, str):
        features = [raw]
    elif isinstance(raw, list):
        features = [str(item) for item in raw if str(item).strip()]
    else:
        features = []
    if features:
        return list(dict.fromkeys(features))
    issue_text = " ".join(str(item) for item in diagnosis.get("issues", []))
    matches = re.findall(r"(?:suspected feature|feature)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", issue_text, re.I)
    return list(dict.fromkeys(matches))


def _delta(current: Any, previous: Any) -> float | None:
    current_number = _number(current)
    previous_number = _number(previous)
    if current_number is None or previous_number is None:
        return None
    return current_number - previous_number


def _signed(value: float | None) -> str:
    return "-" if value is None else f"{value:+.5f}"


def _rate_text(value: float) -> str:
    return f"{value * 100:.1f}%" if 0.0 <= value <= 1.0 else f"{value:.5f}"


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _integer(*values: Any) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _generic_copy(diagnosis: dict[str, Any]) -> dict[str, Any]:
    candidates = [str(item) for item in diagnosis.get("improvement_candidates", []) if str(item).strip()]
    recommendation = candidates[0] if candidates else "현재 베스트를 유지하고 낮은 위험의 후보를 한 번 더 검증합니다."
    questions = [str(item) for item in diagnosis.get("user_questions", []) if str(item).strip()]
    question = questions[0] if questions else "에이전트의 추천 방향으로 다음 실험을 진행해도 될까요?"
    return {
        "title": "다음 연구 판단에 사용자 정보가 필요합니다",
        "interpretation": "여러 기술 후보가 가능하지만 사용자의 업무 기준 없이 우선순위를 확정하기 어렵습니다.",
        "recommendation": recommendation,
        "why": "실험 기록에 없는 도메인 기준이나 허용 가능한 위험을 확인해야 합니다.",
        "question": question,
        "options": (
            ("accept", "추천안 적용", "에이전트 추천을 다음 계획에 반영합니다.", "consider_in_next_plan"),
            ("inspect", "근거 더 확인", "대표 사례와 근거를 추가 수집합니다.", "inspect_more_samples"),
            ("hold", "현재 상태 유지", "현재 베스트를 유지하고 해당 축을 보류합니다.", "consider_in_next_plan"),
        ),
    }


def _default_action(axis: str, interaction_type: str) -> str:
    if axis == "compute_backend":
        return "답변이 없으면 외부 비용을 사용하지 않고 로컬 경량화를 다음 계획의 제약으로 기록합니다."
    if interaction_type == "blocking_human_review":
        return "답변 전까지 모호한 사례를 변경하지 않고 관련 실행을 보류합니다."
    return "답변이 없으면 현재 베스트를 유지하고 가장 낮은 위험의 후보를 한 번 검증합니다."
