from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from .agents.code_writer_adapter import create_llm_client, provider_log_name
from .agents.memory import log_token_usage
from .execution_facts import resolve_trial_plan
from .policies import load_policy, resolve_model_for_call
from .store import read_text, write_text
from .trial_user_view_effective import render_effective_user_plan

_TRANSLATION_FIELDS = [
    "plan_type",
    "source_trial_id",
    "primary_change_axis",
    "plan_title",
    "objective",
    "rationale",
    "keep_unchanged",
    "change_details",
    "candidate",
    "success_criteria",
    "failure_decision",
]


class TranslationClient(Protocol):
    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def render_plan_ko(
    out_dir: Path,
    summary: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
    allow_api: bool = False,
    client: TranslationClient | None = None,
) -> str:
    """Render 01_plan.ko.md using a real Korean translation from the low-cost model.

    Falls back to the Korean-labeled template (render_effective_user_plan) if
    the API is unavailable or the call fails/returns nothing usable -- this
    artifact is never required for the research loop, so a translation
    failure must not block planning or trial finalization.

    This is called twice per trial in the common case: once as a
    pre-execution preview right after planning (prepare_workspace_trial_plan
    -> write_proposed_plan_preview), and once more after execution finishes
    (organize_trial_artifacts), by which point the plan usually has not
    changed at all. A cache keyed on the exact translation payload skips the
    second LLM call whenever that is true, and only re-translates when the
    plan actually changed (e.g. a user insight forced a replan).
    """
    competition = str(summary["competition"])
    trial_id = str(summary["trial_id"])
    if plan is None:
        plan = resolve_trial_plan(competition, trial_id)

    def fallback() -> str:
        return render_effective_user_plan(out_dir, summary, plan=plan)

    payload = _translation_payload(trial_id, plan)
    fingerprint = _payload_fingerprint(payload)
    cached = _read_cached_translation(out_dir, fingerprint)
    if cached is not None:
        return cached

    if not allow_api and client is None:
        return fallback()

    model_policy = load_policy("model_policy")
    model_selection = resolve_model_for_call(
        "plan_translation",
        policy=model_policy,
        model_env_var="RESEARCH_AGENT_PLAN_TRANSLATION_MODEL",
    )
    provider = str(model_selection.get("provider") or "openai")
    model = str(model_selection.get("model"))
    request = _build_request(model, payload)
    try:
        active_client = client or create_llm_client(provider)
        response = active_client.create_response(request)
        text = _extract_output_text(response).strip()
        if not text:
            raise ValueError("Plan translation response did not contain text.")
    except Exception:
        return fallback()

    usage = response.get("usage")
    if isinstance(usage, dict):
        log_token_usage(
            competition,
            trial_id,
            provider=provider_log_name(provider),
            model=model,
            call_type="plan_translation",
            usage=usage,
            request_id=response.get("id"),
        )
    _write_cached_translation(out_dir, fingerprint, text)
    return text


def _write_cached_translation(out_dir: Path, fingerprint: str, text: str) -> None:
    cache_path = _translation_cache_path(out_dir)
    write_text(cache_path, json.dumps({"fingerprint": fingerprint, "text": text}, ensure_ascii=False))


def _translation_payload(trial_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    """The exact subset of the plan that gets sent to the translator.

    Used both to build the request and, hashed, as the cache key -- so two
    plans are considered "the same for translation purposes" exactly when
    they would produce the same prompt, regardless of unrelated bookkeeping
    fields (timestamps, internal ids, etc.) that live elsewhere in the plan.
    """
    compact: dict[str, Any] = {"trial_id": trial_id}
    compact.update(
        {key: plan.get(key) for key in _TRANSLATION_FIELDS if plan.get(key) not in (None, "", [], {})}
    )
    return compact


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _translation_cache_path(out_dir: Path) -> Path:
    # Deliberately outside user_view/: organize_trial_artifacts wipes that
    # whole directory (shutil.rmtree) right before its final render call, so
    # anything cached there would already be gone by the time this is read.
    return out_dir / "internal" / "plan_translation_cache.json"


def _read_cached_translation(out_dir: Path, fingerprint: str) -> str | None:
    cache_path = _translation_cache_path(out_dir)
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(read_text(cache_path))
    except (ValueError, TypeError):
        return None
    if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
        return None
    text = cached.get("text")
    return text if isinstance(text, str) and text else None


def _build_request(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    compact = payload
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "You translate an ML experiment plan document into Korean for a "
                    "non-expert user-facing dashboard. Return only Markdown, no JSON, no code fences."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "다음 실험 계획 JSON을 한국어 마크다운 문서로 번역/재구성하세요.",
                        "이 계획은 `trial_id` 필드에 명시된 trial의 계획입니다. "
                        "문서 제목과 본문에서 절대 다른 trial 번호로 착각하지 마세요.",
                        "규칙:",
                        "- 모델명, 라이브러리 클래스명(예: HistGradientBoostingRegressor, StandardScaler), "
                        "파일명(예: train_step.py), 컬럼/피처 이름, 코드 식별자, 수식, 숫자 파라미터는 "
                        "번역하지 말고 원문 그대로 유지하세요.",
                        "- 설명 문장(rationale, 목적, 왜 하는가, 성공/실패 기준 같은 서술)만 자연스러운 한국어로 옮기세요.",
                        "- 마크다운 헤더는 다음을 사용하세요: # {trial_id} 실험 계획, ## 목적, ## 왜 하는가, "
                        "## 그대로 유지, ## 이번 회차 변경, ## 성공 기준, ## 실패 시 판단",
                        "- 표(| 항목 | 값 |)로 계획 유형/계획명/기준 trial/개선축을 먼저 보여주세요.",
                        "- 불필요한 사설 없이 문서만 출력하세요.",
                        "",
                        "## 원본 계획 JSON",
                        "",
                        json.dumps(compact, ensure_ascii=False, indent=2),
                    ]
                ),
            },
        ],
        "max_output_tokens": 1600,
    }


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
