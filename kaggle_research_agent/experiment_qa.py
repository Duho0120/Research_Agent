from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from .agents.code_writer_adapter import OpenAIResponsesClient
from .agents.memory import log_token_usage
from .experiment_qa_retrieval import retrieve_experiment_evidence
from .interaction_contract import interaction_contract
from .paths import project_root
from .policies import resolve_model_for_call


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
    interaction = interaction_contract("experiment_question")
    evidence = collect_experiment_evidence(competition, trial_id, question)
    if not evidence:
        return {
            "answer": "현재 선택된 실험에서 답변 근거가 될 문서를 찾지 못했습니다.",
            "mode": "no_evidence",
            "mode_label": "근거 없음",
            "sources": [],
            "warning": None,
            "interaction": interaction,
        }

    warning = None
    demo_mode = _truthy_env("RESEARCH_AGENT_CHAT_DEMO_MODE")
    if not demo_mode and use_llm and (client is not None or os.environ.get("OPENAI_API_KEY")):
        try:
            model_selection = resolve_model_for_call(
                "experiment_question",
                model_env_var="RESEARCH_AGENT_CHAT_MODEL",
            )
            model = str(model_selection.get("model"))
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
                "mode_label": "저비용 LLM · 근거 검색",
                "sources": [item[0] for item in evidence],
                "warning": None,
                "interaction": interaction,
            }
        except Exception as error:  # The CLI must remain usable during API/quota outages.
            warning = f"저비용 LLM 호출 실패: {error}"

    answer, selected_sources = _local_rag_answer(question, evidence)
    return {
        "answer": answer,
        "mode": "demo_local_rag" if demo_mode else "local_evidence",
        "mode_label": "DEMO · 로컬 근거 모드" if demo_mode else "로컬 근거 모드",
        "sources": selected_sources,
        "warning": warning,
        "interaction": interaction,
    }


def collect_experiment_evidence(
    competition: str,
    trial_id: str | None,
    question: str = "전체 trial 점수와 실험 기록",
) -> list[tuple[str, str]]:
    return retrieve_experiment_evidence(project_root(), competition, trial_id, question)


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
    configured_limit = int(os.environ.get("RESEARCH_AGENT_CHAT_MAX_OUTPUT_TOKENS", "1000"))
    return {"model": model, "input": prompt, "max_output_tokens": max(300, min(configured_limit, 2000))}


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


def _local_rag_answer(question: str, evidence: list[tuple[str, str]]) -> tuple[str, list[str]]:
    structured = next((content for source, content in evidence if source == "sqlite:trial_summary"), None)
    if structured and _is_score_question(question):
        try:
            payload = json.loads(structured)
        except json.JSONDecodeError:
            payload = {}
        trials = payload.get("trials") if isinstance(payload, dict) else None
        if isinstance(trials, list) and trials:
            return _render_structured_score_answer(trials), ["sqlite:trial_summary"]
    return _local_evidence_answer(question, evidence)


def _render_structured_score_answer(trials: list[dict[str, Any]]) -> str:
    rows = [row for row in trials if isinstance(row, dict)]
    lines = [
        "| trial | 로컬 점수 | 제출 점수 | 베스트 |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {local} | {lb} | {best} |".format(
                trial=row.get("trial_id") or "-",
                local=_display_score(row.get("local_score")),
                lb=_display_score(row.get("lb_score")),
                best="Best" if row.get("is_best_lb") else "-",
            )
        )
    scored = [row for row in rows if isinstance(row.get("lb_score"), (int, float))]
    if scored:
        objective = str(scored[0].get("objective") or "maximize").lower()
        best = (min if objective == "minimize" else max)(scored, key=lambda row: float(row["lb_score"]))
        lines.extend(
            [
                "",
                f"제출 점수 기준 베스트는 {best.get('trial_id')}이며 점수는 {_display_score(best.get('lb_score'))}입니다.",
            ]
        )
    lines.extend(["", "근거: sqlite:trial_summary"])
    return "\n".join(lines)


def _display_score(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.5f}"


def _is_score_question(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in ("점수", "score", "베스트", "best", "lb", "리더보드"))


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}
