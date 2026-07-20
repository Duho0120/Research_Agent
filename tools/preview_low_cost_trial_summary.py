from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_research_agent.agents.code_writer_adapter import OpenAIResponsesClient
from kaggle_research_agent.policies import load_policy


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Preview a low-cost LLM summary card for an existing trial without changing project files."
    )
    parser.add_argument("--competition", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to experiments/<competition>/<trial>/low_cost_summary_preview.json.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    trial_dir = root / "experiments" / args.competition / args.trial
    if not trial_dir.exists():
        raise SystemExit(f"Trial directory not found: {trial_dir}")

    payload_input = build_payload_input(root, args.competition, trial_dir)
    model_policy = load_policy("model_policy")
    low_cost = model_policy.get("low_cost", {})
    model = str(low_cost.get("model") or model_policy.get("fallback", {}).get("model") or "gpt-5.6-luna")

    print("=== BEFORE: rule/user-view source snippets ===")
    print_source_preview(payload_input)
    print()
    print("=== LOW-COST SUMMARY TEST ===")
    print(f"model: {model}")
    print(f"trial: {args.competition}/{args.trial}")
    print("mode: API call" if args.allow_api else "mode: dry-run only")

    request = {
        "model": model,
        "input": [
            {"role": "developer", "content": build_prompt()},
            {"role": "user", "content": json.dumps(payload_input, ensure_ascii=False)},
        ],
        "max_output_tokens": args.max_output_tokens,
    }

    if not args.allow_api:
        approx_chars = len(json.dumps(payload_input, ensure_ascii=False))
        print()
        print("API를 호출하지 않았습니다. 실제 테스트하려면 --allow-api를 붙여 실행하세요.")
        print(f"approx_input_chars: {approx_chars}")
        print("output_schema: progress_message, log_summary, what_changed, diff_from_previous, human_review_message")
        return 0

    response = OpenAIResponsesClient(timeout_seconds=120).create_response(request)
    summary_text = extract_text(response)
    output_path = Path(args.output) if args.output else trial_dir / "low_cost_summary_preview.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "competition": args.competition,
        "trial": args.trial,
        "model": response.get("model"),
        "usage": response.get("usage", {}),
        "summary_text": summary_text,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("=== AFTER: low-cost LLM summary card ===")
    print(summary_text)
    print()
    print("=== USAGE ===")
    print(json.dumps(response.get("usage", {}), ensure_ascii=False, indent=2))
    print()
    print(f"saved: {output_path}")
    return 0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_payload_input(root: Path, competition: str, trial_dir: Path) -> dict[str, str]:
    memory_cards = root / "memory" / competition / "trial_memory_cards.jsonl"
    previous_cards = ""
    if memory_cards.exists():
        previous_cards = "\n".join(memory_cards.read_text(encoding="utf-8").splitlines()[-3:])

    return {
        "plan": read_limited(trial_dir / "user_view" / "01_plan.ko.md", 2600),
        "pipeline": read_limited(trial_dir / "user_view" / "02_pipeline_structure.ko.md", 3600),
        "result": read_limited(trial_dir / "user_view" / "04_result.ko.md", 1600),
        "recent_memory_cards": previous_cards[:3500],
        "events": read_tail_limited(trial_dir / "node_events.jsonl", 2500),
    }


def build_prompt() -> str:
    return """
다음 trial 자료를 읽고, 사용자에게 보여줄 저비용 LLM 요약 카드만 작성하세요.
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


def read_limited(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")[:limit]


def read_tail_limited(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[-limit:]


def print_source_preview(payload_input: dict[str, str]) -> None:
    for key in ("plan", "pipeline", "result"):
        value = payload_input.get(key, "").strip()
        preview = value[:700].replace("\r\n", "\n")
        print(f"\n[{key}]")
        print(preview or "(missing)")


def extract_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if output_text:
        return str(output_text)
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


if __name__ == "__main__":
    raise SystemExit(main())
