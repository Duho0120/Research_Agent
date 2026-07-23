from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Protocol

from .agents.code_writer_adapter import OpenAIResponsesClient
from .agents.memory import log_token_usage
from .paths import project_root


MAX_DOCUMENTS = 16
MAX_CHARS_PER_DOCUMENT = 8_000
MAX_CONTEXT_CHARS = 60_000


class ResponseClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def answer_experiment_question(
    competition: str,
    trial_id: str | None,
    question: str,
    *,
    client: ResponseClient | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    evidence = collect_experiment_evidence(competition, trial_id)
    if not evidence:
        return {
            "answer": "현재 선택된 실험에서 답변 근거가 될 문서를 찾지 못했습니다.",
            "mode": "no_evidence",
            "sources": [],
            "warning": None,
        }

    warning = None
    if use_llm and (client is not None or os.environ.get("OPENAI_API_KEY")):
        try:
            model = os.environ.get("RESEARCH_AGENT_CHAT_MODEL", "gpt-5.6-luna")
            active_client = client or OpenAIResponsesClient()
            response = active_client.create_response(_question_payload(model, competition, trial_id, question, evidence))
            answer = _extract_output_text(response).strip()
            if not answer:
                raise ValueError("LLM response did not contain text.")
            usage = response.get("usage")
            if isinstance(usage, dict):
                log_token_usage(
                    competition,
                    trial_id,
                    provider="openai",
                    model=str(response.get("model") or model),
                    call_type="experiment_question",
                    usage=usage,
                    request_id=str(response.get("id") or "") or None,
                )
            return {
                "answer": answer,
                "mode": "low_cost_llm",
                "sources": [item[0] for item in evidence],
                "warning": None,
            }
        except Exception as error:  # The CLI must remain usable during API/quota outages.
            warning = f"저비용 LLM 호출 실패: {error}"

    answer, selected_sources = _local_evidence_answer(question, evidence)
    return {
        "answer": answer,
        "mode": "local_evidence",
        "sources": selected_sources,
        "warning": warning,
    }


def collect_experiment_evidence(competition: str, trial_id: str | None) -> list[tuple[str, str]]:
    root = project_root()
    candidates: list[Path] = []
    manual_root = root / "demo_workspaces" / competition / "manual_trials"
    manual_summary = _manual_score_summary(root, manual_root)
    for base in [root / "runs" / competition, root / "experiments" / competition]:
        if trial_id:
            candidates.extend(_document_files(base / trial_id))
        candidates.extend(_document_files(base, recursive=False))
    if trial_id:
        candidates.extend(_document_files(manual_root / trial_id))
    candidates.extend(sorted(manual_root.glob("trial_*/metrics.json")))
    for user_view in sorted(manual_root.glob("trial_*/user_view")):
        candidates.extend(_document_files(user_view))
    candidates.extend(_document_files(root / "memory" / competition, recursive=False))

    seen: set[Path] = set()
    evidence: list[tuple[str, str]] = list(manual_summary)
    total_chars = sum(len(content) for _, content in evidence)
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not content:
            continue
        content = content[:MAX_CHARS_PER_DOCUMENT]
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        content = content[:remaining]
        evidence.append((str(path.relative_to(root)), content))
        total_chars += len(content)
        if len(evidence) >= MAX_DOCUMENTS:
            break
    return evidence


def _manual_score_summary(root: Path, manual_root: Path) -> list[tuple[str, str]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(manual_root.glob("trial_*/metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            row["_path"] = str(path.relative_to(root))
            rows.append(row)
    if not rows:
        return []

    lines = [
        "# Trial score summary",
        "",
        "| trial | local_score | kaggle_submitted | kaggle_lb_score | kaggle_ref | change_axis | model | source |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {local} | {submitted} | {lb} | {ref} | {axis} | {model} | {source} |".format(
                trial=row.get("trial_id") or "-",
                local=_display_field(row.get("local_score") or row.get("cv_score")),
                submitted=_display_field(row.get("kaggle_submitted")),
                lb=_display_field(row.get("kaggle_lb_score") or row.get("lb_score")),
                ref=_display_field(row.get("kaggle_ref")),
                axis=_display_field(row.get("change_axis")),
                model=_display_field(row.get("model")),
                source=_display_field(row.get("_path")),
            )
        )
    return [("demo_workspaces/{}/manual_trials/SCORE_SUMMARY".format(manual_root.parent.name), "\n".join(lines))]


def _display_field(value: Any) -> str:
    return "-" if value is None or value == "" else str(value)


def _document_files(base: Path, *, recursive: bool = True) -> list[Path]:
    if not base.exists():
        return []
    patterns = ("*.md", "*.json", "*.jsonl", "*.csv")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(base.rglob(pattern) if recursive else base.glob(pattern))
    priorities = {"01_plan.ko.md": 0, "02_pipeline_structure.ko.md": 1, "03_scores.ko.md": 2, "metrics.json": 3}
    return sorted(paths, key=lambda item: (priorities.get(item.name, 20), str(item)))


def _question_payload(
    model: str,
    competition: str,
    trial_id: str | None,
    question: str,
    evidence: list[tuple[str, str]],
) -> dict[str, Any]:
    context = "\n\n".join(f"[SOURCE: {path}]\n{text}" for path, text in evidence)
    prompt = (
        "선택된 실험의 문서 근거만 사용해 한국어로 간결하게 답하세요. "
        "근거에 없는 사실은 추측하지 말고 '문서에서 확인되지 않습니다'라고 밝히세요. "
        "답변 끝에 사용한 파일 경로를 '근거:'로 짧게 적으세요.\n\n"
        f"실험: {competition}\nTrial: {trial_id or '-'}\n질문: {question}\n\n{context}"
    )
    return {"model": model, "input": prompt, "max_output_tokens": 700}


def _extract_output_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(str(text))
    return "\n".join(parts)


def _local_evidence_answer(question: str, evidence: list[tuple[str, str]]) -> tuple[str, list[str]]:
    terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_.]+|[가-힣]{2,}", question)}
    scored: list[tuple[int, str, str]] = []
    for path, content in evidence:
        for line in content.splitlines():
            line = line.strip()
            if not line or len(line) > 500:
                continue
            score = sum(1 for term in terms if term in line.lower())
            score += 2 if any(key in line.lower() for key in ("local_score", "lb_score", "점수", "목적", "개선축")) else 0
            if score:
                scored.append((score, path, line))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = scored[:6]
    if not selected:
        return "문서에서 질문과 직접 연결되는 근거를 확인하지 못했습니다.", []
    lines = ["API를 사용하지 않고 로컬 문서에서 확인한 관련 내용입니다:"]
    sources: list[str] = []
    for _, path, line in selected:
        lines.append(f"- {line}")
        if path not in sources:
            sources.append(path)
    lines.append("근거: " + ", ".join(sources))
    return "\n".join(lines), sources
