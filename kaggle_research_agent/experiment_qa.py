from __future__ import annotations

import os
import re
from typing import Any, Protocol

from .agents.code_writer_adapter import OpenAIResponsesClient
from .agents.memory import log_token_usage
from .experiment_qa_retrieval import retrieve_experiment_evidence
from .paths import project_root


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
    evidence = collect_experiment_evidence(competition, trial_id, question)
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
