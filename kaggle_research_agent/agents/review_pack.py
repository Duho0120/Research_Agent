from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import trial_dir
from ..feedback_decision_support import build_decision_support
from ..store import now_iso, write_text


def prepare_review_pack(
    competition: str,
    trial_id: str,
    diagnosis: dict[str, Any],
    *,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    pack_dir = out_dir / "review_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = cases or _cases_from_diagnosis(diagnosis)
    decision_support = build_decision_support(diagnosis)
    manifest = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "created_at": now_iso(),
        "review_type": _review_type(diagnosis),
        "priority": "blocking" if diagnosis.get("strategy_recommendation") == "strategy_escalation" else "high",
        "status": "pending_user_feedback",
        "source_files": ["metrics.json", "evaluation.md", "diagnosis.md"],
        "questions_file": "questions.ko.md",
        "cases_file": "cases.jsonl",
        "metrics_file": "metrics_snapshot.json",
        "decision_support_file": "decision_support.json",
    }
    metrics_snapshot = {
        "competition": competition,
        "trial_id": trial_id,
        "objective": diagnosis.get("objective"),
        "cv_score": diagnosis.get("cv_score"),
        "lb_score": diagnosis.get("lb_score"),
        "best_cv_before": diagnosis.get("best_cv_before"),
        "diagnosis": {
            "needs_user_review": diagnosis.get("needs_user_review"),
            "strategy_recommendation": diagnosis.get("strategy_recommendation"),
            "issues": diagnosis.get("issues", []),
        },
    }

    write_text(pack_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(pack_dir / "decision_support.json", json.dumps(decision_support, ensure_ascii=False, indent=2) + "\n")
    write_text(pack_dir / "summary.ko.md", _render_summary(diagnosis, decision_support))
    write_text(pack_dir / "questions.ko.md", _render_questions(decision_support))
    write_text(pack_dir / "metrics_snapshot.json", json.dumps(metrics_snapshot, ensure_ascii=False, indent=2) + "\n")
    write_text(
        pack_dir / "cases.jsonl",
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in selected_cases),
    )
    return {
        "pack_dir": str(pack_dir.as_posix()),
        "manifest": manifest,
        "case_count": len(selected_cases),
        "decision_support": decision_support,
    }


def _cases_from_diagnosis(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    representative = [
        item
        for item in diagnosis.get("representative_error_cases", [])
        if isinstance(item, dict)
    ]
    if representative:
        return [
            {
                "case_id": str(
                    item.get("case_id")
                    or item.get("sample_id")
                    or item.get("id")
                    or f"case_{index:03d}"
                ),
                "source": str(item.get("source") or "metrics.representative_error_cases"),
                "sample_id": item.get("sample_id") or item.get("id"),
                "fold": item.get("fold"),
                "group": item.get("group") or item.get("segment"),
                "true_label": item.get("true_label"),
                "pred_label": item.get("pred_label"),
                "confidence": item.get("confidence"),
                "error_type": item.get("error_type") or "representative_error",
                "reason_for_review": item.get("reason") or item.get("reason_for_review"),
                "artifacts": list(item.get("artifacts") or []),
                "question_refs": ["Q1"],
            }
            for index, item in enumerate(representative[:10], start=1)
        ]
    issues = diagnosis.get("issues", [])
    if not issues:
        return []
    return [
        {
            "case_id": f"case_{index:03d}",
            "source": "diagnosis",
            "sample_id": None,
            "fold": None,
            "group": None,
            "true_label": None,
            "pred_label": None,
            "confidence": None,
            "error_type": "diagnosis_issue",
            "reason_for_review": issue,
            "artifacts": [],
            "question_refs": ["Q1"],
        }
        for index, issue in enumerate(issues, start=1)
    ]


def _review_type(diagnosis: dict[str, Any]) -> str:
    issues = " ".join(diagnosis.get("issues", [])).casefold()
    if "cv/lb" in issues or "leakage" in issues:
        return "validation_question"
    if diagnosis.get("strategy_recommendation") == "strategy_escalation":
        return "strategy_shift_question"
    return "data_quality_question"


def _render_summary(diagnosis: dict[str, Any], decision_support: dict[str, Any]) -> str:
    lines = [
        "# 피드백 요청 요약",
        "",
        f"## {decision_support['title']}",
        "",
        decision_support["problem"],
        "",
        "## 핵심 근거",
        "",
    ]
    lines.extend(
        f"- {item['label']}: {item['value']} - {item['meaning']}"
        for item in decision_support["evidence_snapshot"]
    )
    if not decision_support["evidence_snapshot"]:
        lines.append("- 정량 근거가 부족해 진단 이슈와 산출물을 기준으로 판단했습니다.")
    lines.extend(
        [
            "",
            "## 에이전트 해석",
            "",
            decision_support["interpretation"],
            "",
            "## 에이전트 추천",
            "",
            decision_support["recommendation"],
            "",
            "## 왜 사용자에게 묻는가",
            "",
            decision_support["why_user_needed"],
        ]
    )
    if decision_support.get("execution_note"):
        lines.extend(["", "## 실행 범위", "", decision_support["execution_note"]])
    lines.append("")
    return "\n".join(lines)


def _render_questions(decision_support: dict[str, Any]) -> str:
    lines = [
        "# 피드백 요청",
        "",
        f"## Q1. {decision_support['question']}",
        "",
        "### 선택지와 반영 결과",
        "",
    ]
    for option in decision_support["options"]:
        lines.append(f"- **{option['label']}**: {option['impact']}")
    lines.extend(
        [
            "",
            "### 답변이 없을 때",
            "",
            decision_support["default_if_no_response"],
            "",
        ]
    )
    return "\n".join(lines)
