from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .agents.code_writer_adapter import create_llm_client, provider_log_name
from .agents.memory import log_token_usage
from .policies import load_policy, resolve_model_for_call
from .store import write_text


class SummaryClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def generate_low_cost_user_summary_card(
    competition: str,
    trial_id: str,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    allow_api: bool = False,
    client: SummaryClient | None = None,
) -> dict[str, Any]:
    """Create an optional human-facing summary card using the low-cost model.

    This artifact is never required for the research loop. If the API is not
    available or the response is malformed, the failure is recorded but the
    trial remains valid.
    """

    model_policy = load_policy("model_policy")
    model_selection = resolve_model_for_call(
        "user_summary_card",
        policy=model_policy,
        model_env_var="RESEARCH_AGENT_SUMMARY_MODEL",
    )
    provider = str(model_selection.get("provider") or "openai")
    model = str(model_selection.get("model"))
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "skipped",
        "competition": competition,
        "trial_id": trial_id,
        "provider": provider,
        "model": model,
        "call_type": "user_summary_card",
        "next_action": "none",
    }
    if not allow_api and client is None:
        result["reason"] = "allow_api_false"
        return result

    request = {
        "model": model,
        "input": [
            {"role": "developer", "content": _summary_card_prompt()},
            {"role": "user", "content": json.dumps(_summary_card_input(out_dir, summary), ensure_ascii=False)},
        ],
        "max_output_tokens": 700,
    }
    try:
        active_client = client or create_llm_client(provider)
        response = active_client.create_response(request)
        summary_data = _parse_summary_response(response)
    except Exception as exc:  # pragma: no cover - exact API failure shape is provider dependent.
        result.update(
            {
                "status": "failed",
                "reason": "low_cost_summary_generation_failed",
                "error": str(exc),
                "next_action": "ignore-user-summary-card-or-retry",
            }
        )
        return result

    usage = response.get("usage") if isinstance(response, dict) else None
    token_usage = None
    if isinstance(usage, dict):
        token_usage = log_token_usage(
            competition,
            trial_id,
            provider=provider_log_name(provider),
            model=model,
            call_type="user_summary_card",
            usage=usage,
            request_id=response.get("id"),
        )

    result.update(
        {
            "status": "ready",
            "summary": summary_data,
            "token_usage": token_usage,
            "next_action": "show-user-summary-card",
        }
    )
    return result


def write_low_cost_user_summary_artifacts(out_dir: Path, result: dict[str, Any]) -> dict[str, str]:
    internal_path = out_dir / "internal" / "low_cost_user_summary_card.json"
    write_text(internal_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    files = {"internal/low_cost_user_summary_card.json": str(internal_path)}
    if result.get("status") == "ready":
        user_path = out_dir / "user_view" / "00_summary_card.ko.md"
        write_text(user_path, render_summary_card(result))
        files["user_view/00_summary_card.ko.md"] = str(user_path)
    return files


def render_summary_card(result: dict[str, Any]) -> str:
    data = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        f"# {result.get('trial_id')} 한눈에 보기",
        "",
        f"- 진행 상태: {_display(data.get('progress_message'))}",
        f"- 이번 변경: {_display(data.get('what_changed'))}",
        f"- 이전 대비: {_display(data.get('diff_from_previous'))}",
        f"- 사용자 확인: {_display(data.get('human_review_message'))}",
        "",
        "## 실행 요약",
        "",
    ]
    logs = data.get("log_summary") if isinstance(data.get("log_summary"), list) else []
    if logs:
        lines.extend(f"- {_display(item)}" for item in logs)
    else:
        lines.append("- 실행 요약이 없습니다.")
    lines.extend(
        [
            "",
            "## 참고",
            "",
            "- 이 문서는 저비용 LLM이 사용자 확인용으로 생성한 요약입니다.",
            "- 연구 판단에는 내부 JSON, decision card, trial memory card를 사용합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_card_input(out_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    decision_card = summary.get("decision_card") if isinstance(summary.get("decision_card"), dict) else {}
    trial_memory_card = _read_trial_json(out_dir, "trial_memory_card.json")
    return {
        "competition": summary.get("competition"),
        "trial_id": summary.get("trial_id"),
        "status": summary.get("status"),
        "metric": summary.get("metric"),
        "objective": summary.get("objective"),
        "local_score": summary.get("local_score"),
        "score_source": summary.get("score_source"),
        "workspace_status": summary.get("workspace_status"),
        "metrics_collection_status": summary.get("metrics_collection_status"),
        "submission_prepare_status": summary.get("submission_prepare_status"),
        "plan_title": summary.get("plan_title"),
        "changed_files": summary.get("changed_files", []),
        "decision_card": _small_decision_card(decision_card),
        "trial_memory_card": _small_memory_card(trial_memory_card),
        "log_status": _log_status(summary),
    }


def _summary_card_prompt() -> str:
    return """
다음 trial 구조화 자료를 읽고 사용자 확인용 요약 카드만 작성하세요.
연구 전략을 새로 세우지 말고, 코드 변경을 제안하지 말고, 입력에 있는 사실만 요약하세요.
한국어로 간결하게 작성하세요.

반드시 JSON만 출력하세요. 스키마:
{
  "progress_message": "한 줄 진행 메시지",
  "log_summary": ["실행 로그 요약 bullet 1", "bullet 2"],
  "what_changed": "이번 trial에서 실제로 바뀐 점 한 문단",
  "diff_from_previous": "이전 trial 대비 차이점 한 문단",
  "human_review_message": "사람 검토가 필요하면 요청 문구, 아니면 '현재 즉시 요청할 사용자 피드백은 없습니다.'"
}
""".strip()


def _parse_summary_response(response: dict[str, Any]) -> dict[str, Any]:
    text = _extract_text(response).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Low-cost summary response JSON is not an object.")
    return {
        "progress_message": str(data.get("progress_message") or "").strip(),
        "log_summary": [str(item).strip() for item in data.get("log_summary", []) if str(item).strip()],
        "what_changed": str(data.get("what_changed") or "").strip(),
        "diff_from_previous": str(data.get("diff_from_previous") or "").strip(),
        "human_review_message": str(data.get("human_review_message") or "").strip()
        or "현재 즉시 요청할 사용자 피드백은 없습니다.",
    }


def _extract_text(response: dict[str, Any]) -> str:
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


def _small_decision_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_trial_id",
        "change_axis",
        "candidate_label",
        "local_score",
        "previous_local_score",
        "best_local_score",
        "local_delta",
        "local_status",
        "decision",
        "active_axis",
        "axis_attempt_count",
        "axis_attempt_limit",
        "recommended_base_trial",
        "next_guidance",
    ]
    return {key: card.get(key) for key in keys if key in card}


def _small_memory_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "plan_type",
        "source_trial_id",
        "plan_title",
        "change_axis",
        "candidate_label",
        "change_details",
        "kept_unchanged",
        "local_score",
        "local_status",
        "decision",
        "recommended_base_trial",
    ]
    return {key: card.get(key) for key in keys if key in card}


def _log_status(summary: dict[str, Any]) -> dict[str, Any]:
    logs = summary.get("log_paths") if isinstance(summary.get("log_paths"), list) else []
    return {
        "log_count": len(logs),
        "workspace_status": summary.get("workspace_status"),
        "metrics_collection_status": summary.get("metrics_collection_status"),
    }


def _read_trial_json(out_dir: Path, name: str) -> dict[str, Any]:
    for path in (out_dir / name, out_dir / "internal" / name):
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _display(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "-"
