from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import trial_dir
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
    write_text(pack_dir / "summary.ko.md", _render_summary(diagnosis))
    write_text(pack_dir / "questions.ko.md", _render_questions(diagnosis))
    write_text(pack_dir / "metrics_snapshot.json", json.dumps(metrics_snapshot, ensure_ascii=False, indent=2) + "\n")
    write_text(
        pack_dir / "cases.jsonl",
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in selected_cases),
    )
    return {"pack_dir": str(pack_dir.as_posix()), "manifest": manifest, "case_count": len(selected_cases)}


def _cases_from_diagnosis(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
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


def _render_summary(diagnosis: dict[str, Any]) -> str:
    lines = [
        "# Review Summary",
        "",
        "## 왜 확인이 필요한가",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis.get("issues", []) or ["진단상 큰 이슈는 없지만 확인 요청이 생성되었습니다."])
    lines.extend(["", "## 에이전트 판단", ""])
    lines.append(f"- needs_user_review: {diagnosis.get('needs_user_review')}")
    lines.append(f"- strategy_recommendation: {diagnosis.get('strategy_recommendation')}")
    lines.extend(["", "## 다음 결정 후보", "", "- continue", "- change_validation", "- change_feature", "- strategy_shift"])
    lines.append("")
    return "\n".join(lines)


def _render_questions(diagnosis: dict[str, Any]) -> str:
    questions = diagnosis.get("user_questions", []) or ["이 trial의 다음 action을 계속 진행해도 될까요?"]
    lines = ["# Review Questions", ""]
    for index, question in enumerate(questions[:3], start=1):
        lines.extend([f"## Q{index}. {question}", "", "- 판단 옵션:", "  - continue", "  - change_validation", "  - change_feature", "  - inspect_more_samples", ""])
    return "\n".join(lines)
