from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, TextIO

from . import paths
from .demo_one_cycle import run_demo_one_cycle
from .store import read_text, write_text
from .trial_artifacts import trial_artifact_exists
from .workspace_preparer import prepare_workspace


InputFn = Callable[[str], str]
PrepareWorkspaceFn = Callable[..., dict[str, Any]]
RunCycleFn = Callable[..., dict[str, Any]]

DEMO_API_PROVIDER = "openai"
DEMO_API_MODEL = "gpt-5.5"
MIN_READABLE_PAGE_CHARS = 800


def run_demo_guide(
    *,
    input_fn: InputFn = input,
    output: TextIO | None = None,
    prepare_workspace_fn: PrepareWorkspaceFn = prepare_workspace,
    run_cycle_fn: RunCycleFn = run_demo_one_cycle,
) -> int:
    stream = output or sys.stdout
    _print_header(stream)
    experiments = list_demo_experiments()
    _print_experiment_list(experiments, stream)
    _print_menu(stream)
    choice = input_fn("select: ").strip()
    if choice == "1":
        return _start_new_demo_experiment(
            input_fn=input_fn,
            output=stream,
            prepare_workspace_fn=prepare_workspace_fn,
            run_cycle_fn=run_cycle_fn,
        )
    if choice == "2":
        return _run_existing_demo_cycle(
            experiments,
            input_fn=input_fn,
            output=stream,
            run_cycle_fn=run_cycle_fn,
        )
    if choice == "3":
        return _show_existing_experiment(experiments, input_fn=input_fn, output=stream)
    if choice == "4":
        print("종료합니다.", file=stream)
        return 0
    print("알 수 없는 선택입니다. 1, 2, 3, 4 중에서 선택해주세요.", file=stream)
    return 1


def list_demo_experiments() -> list[dict[str, Any]]:
    competitions_root = paths.project_root() / "competitions"
    if not competitions_root.exists():
        return []
    rows = []
    for competition_dir in sorted(competitions_root.iterdir(), key=lambda item: item.name.casefold()):
        if not competition_dir.is_dir():
            continue
        rows.append(summarize_demo_experiment(competition_dir.name))
    return rows


def summarize_demo_experiment(competition: str) -> dict[str, Any]:
    source_record = _read_json(paths.competition_dir(competition) / "workspace_source.json")
    state = _read_json(paths.trial_dir(competition, "trial_001") / "agent_status.json")
    status = state.get("status") or source_record.get("status") or "initialized"
    source_path = source_record.get("source_path")
    required_files = source_record.get("required_data_files") or []
    data_status = _check_demo_data_files(source_path, required_files)
    if required_files and data_status["missing_files"]:
        status = "needs_data"
    latest_trial = _latest_trial_id(competition)
    return {
        "competition": competition,
        "status": status,
        "source_path": source_path,
        "required_data_files": required_files,
        "data_status": data_status,
        "latest_trial": latest_trial,
        "latest_agent_status": state,
    }


def render_demo_experiment_status(summary: dict[str, Any]) -> str:
    data_status = summary.get("data_status", {})
    lines = [
        f"실험 현황: {summary['competition']}",
        "",
        f"status: {summary.get('status')}",
        f"workspace: {summary.get('source_path') or '-'}",
        f"latest_trial: {summary.get('latest_trial') or '-'}",
        "",
        "data:",
    ]
    required = summary.get("required_data_files") or []
    if required:
        found = set(data_status.get("found_files", []))
        for name in required:
            lines.append(f"- {name}: {'found' if name in found else 'missing'}")
    else:
        lines.append("- 필요한 데이터 파일이 선언되지 않았습니다.")
    artifacts = _demo_artifact_status(summary["competition"], summary.get("latest_trial") or "trial_001")
    lines.extend(["", "latest_outputs:"])
    if artifacts:
        lines.extend(f"- {item['path']}: {item['status']}" for item in artifacts)
    else:
        lines.append("- 아직 데모 산출물이 없습니다.")
    return "\n".join(lines)


def inspect_competition_url(url: str) -> dict[str, Any]:
    url = url.strip()
    if not url:
        return {"status": "skipped", "url": None, "inferred": {}}
    inferred = _infer_from_url(url)
    try:
        raw = _fetch_url_text(url)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        return {
            "status": "unreadable",
            "url": url,
            "inferred": inferred,
            "reason": f"fetch_failed:{error.__class__.__name__}",
            "excerpt": "",
        }
    text = _html_to_text(raw)
    title = _extract_title(raw)
    if title and not inferred.get("topic"):
        inferred["topic"] = title
    if len(text) < MIN_READABLE_PAGE_CHARS:
        return {
            "status": "unreadable",
            "url": url,
            "inferred": inferred,
            "reason": "page_text_too_short_or_dynamic",
            "excerpt": text[:1200],
        }
    return {
        "status": "readable",
        "url": url,
        "inferred": inferred,
        "reason": "fetched",
        "excerpt": text[:4000],
    }


def _start_new_demo_experiment(
    *,
    input_fn: InputFn,
    output: TextIO,
    prepare_workspace_fn: PrepareWorkspaceFn,
    run_cycle_fn: RunCycleFn,
) -> int:
    print("", file=output)
    print("새 실험을 시작하려면 대회 링크 또는 기본 정보를 입력해주세요.", file=output)
    print("링크 내용을 자동으로 확인하지 못하면 문제/데이터/제출 정보를 직접 요청합니다.", file=output)
    print("", file=output)
    competition_url = _prompt_optional(input_fn, "competition_url")
    url_info = inspect_competition_url(competition_url)
    _print_url_status(url_info, output)

    inferred = url_info.get("inferred", {})
    competition = _prompt_with_default(input_fn, "competition", inferred.get("competition", ""))
    platform = _prompt_choice(
        input_fn,
        "platform",
        ["kaggle", "dacon", "external", "local_research"],
        default=inferred.get("platform") or "kaggle",
    )
    topic = _prompt_with_default(input_fn, "topic", inferred.get("topic", ""))
    metric = _prompt_required(input_fn, "metric")
    objective = _prompt_choice(input_fn, "objective", ["maximize", "minimize"], default="maximize")
    target_column = _prompt_required(input_fn, "target_column")
    id_column = _prompt_optional(input_fn, "id_column")
    required_data_files = _parse_csv_list(_prompt_required(input_fn, "required_data_files"))

    source_materials = _prompt_source_materials(input_fn, output, url_info=url_info)
    _write_demo_competition_materials(
        competition,
        competition_url=competition_url,
        topic=topic,
        metric=metric,
        objective=objective,
        target_column=target_column,
        id_column=id_column,
        required_data_files=required_data_files,
        url_info=url_info,
        source_materials=source_materials,
    )

    result = prepare_workspace_fn(
        competition,
        topic=topic,
        platform=platform,
        metric=metric,
        objective=objective,
        create_workspace=True,
        target_column=target_column,
        id_column=id_column or None,
        required_data_files=required_data_files,
    )
    workspace = result.get("source_path") or str(paths.project_root() / "demo_workspaces" / competition)
    print("", file=output)
    print("workspace가 생성되었습니다.", file=output)
    print("", file=output)
    print("데이터 파일을 아래 폴더에 넣어주세요:", file=output)
    print(str(Path(workspace) / "data"), file=output)
    print("", file=output)
    print("필요한 파일:", file=output)
    for name in required_data_files:
        print(f"- {name}", file=output)
    print("", file=output)
    input_fn("파일을 넣은 뒤 Enter를 누르면 1회 실험 사이클을 시작합니다. 현재는 데모용으로 1회 실험만 진행됩니다.")

    refreshed = prepare_workspace_fn(
        competition,
        topic=topic,
        platform=platform,
        metric=metric,
        objective=objective,
        create_workspace=True,
        target_column=target_column,
        id_column=id_column or None,
        required_data_files=required_data_files,
    )
    if refreshed.get("status") == "needs_data":
        missing = refreshed.get("data_check", {}).get("missing_files", [])
        print("", file=output)
        print("아직 필요한 데이터 파일이 없습니다.", file=output)
        for name in missing:
            print(f"- {name}", file=output)
        print("데이터를 넣은 뒤 demo-guide를 다시 실행하거나 기존 실험 사이클 실행을 선택해주세요.", file=output)
        return 1
    return _run_cycle_prompt(
        competition,
        input_fn=input_fn,
        output=output,
        run_cycle_fn=run_cycle_fn,
    )


def _run_existing_demo_cycle(
    experiments: list[dict[str, Any]],
    *,
    input_fn: InputFn,
    output: TextIO,
    run_cycle_fn: RunCycleFn,
) -> int:
    selected = _select_experiment(experiments, input_fn=input_fn, output=output)
    if selected is None:
        return 1
    return _run_cycle_prompt(
        selected["competition"],
        input_fn=input_fn,
        output=output,
        run_cycle_fn=run_cycle_fn,
    )


def _show_existing_experiment(
    experiments: list[dict[str, Any]],
    *,
    input_fn: InputFn,
    output: TextIO,
) -> int:
    selected = _select_experiment(experiments, input_fn=input_fn, output=output)
    if selected is None:
        return 1
    print("", file=output)
    print(render_demo_experiment_status(summarize_demo_experiment(selected["competition"])), file=output)
    return 0


def _run_cycle_prompt(
    competition: str,
    *,
    input_fn: InputFn,
    output: TextIO,
    run_cycle_fn: RunCycleFn,
) -> int:
    print("", file=output)
    print(f"데모 LLM provider: {DEMO_API_PROVIDER}", file=output)
    print(f"데모 LLM model   : {DEMO_API_MODEL}", file=output)
    allow_api_answer = input_fn("실제 OpenAI API를 호출할까요? 비용이 발생할 수 있습니다. [y/N]: ").strip().lower()
    allow_api = allow_api_answer in {"y", "yes"}
    if not allow_api:
        print("API 호출 전에 데모 실행을 취소합니다.", file=output)
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되어 있지 않습니다. 설정 후 새 터미널에서 demo-guide를 다시 실행해주세요.", file=output)
        return 1
    print("", file=output)
    print(f"선택한 실험: {competition}", file=output)
    print("1회 실험 사이클을 시작합니다. 현재는 데모용으로 1회 실험만 진행됩니다.", file=output)
    result = run_cycle_fn(
        competition,
        "trial_001",
        provider=DEMO_API_PROVIDER,
        model=DEMO_API_MODEL,
        allow_api=True,
        run_now=True,
        show_progress=True,
        trial_llm_calls=0,
        strategy_calls_today=0,
    )
    print("", file=output)
    print(f"Demo one cycle: {competition} trial_001 status={result.get('status')}", file=output)
    return 0 if result.get("status") in {"planned", "completed"} else 1


def _prompt_source_materials(input_fn: InputFn, output: TextIO, *, url_info: dict[str, Any]) -> dict[str, str]:
    print("", file=output)
    if url_info.get("status") == "readable":
        print("링크 내용을 일부 읽었습니다. 추가 요약이 있으면 한 번만 입력하고, 없으면 Enter를 누르세요.", file=output)
    elif url_info.get("url"):
        print("링크 내용을 자동으로 충분히 확인하지 못했습니다.", file=output)
        print("필요하면 기본 LLM에게 링크 내용을 요약하게 한 뒤, 핵심만 한 번에 붙여넣으세요. 없으면 Enter로 건너뜁니다.", file=output)
    else:
        print("링크가 없으면 선택적으로 대회/데이터/평가 방식 요약을 한 번에 입력할 수 있습니다. 없으면 Enter로 건너뜁니다.", file=output)
    source_summary = _prompt_optional(input_fn, "source_summary_optional")
    return _source_materials_from_summary(source_summary)


def _source_materials_from_summary(source_summary: str) -> dict[str, str]:
    return {
        "source_summary": source_summary,
        "problem_description": source_summary,
        "data_description": "",
        "submission_format": "",
        "evaluation_rule": "",
        "constraints_or_rules": "",
        "notes": "",
    }


def _print_url_status(url_info: dict[str, Any], output: TextIO) -> None:
    if url_info.get("status") == "skipped":
        return
    print("", file=output)
    if url_info.get("status") == "readable":
        print("대회 링크 내용을 일부 읽었습니다.", file=output)
    else:
        print("대회 링크 내용을 자동으로 충분히 확인하지 못했습니다.", file=output)
        print(f"reason: {url_info.get('reason')}", file=output)
    inferred = url_info.get("inferred", {})
    if inferred:
        print("링크에서 추정한 기본값:", file=output)
        for key in ("competition", "platform", "topic"):
            if inferred.get(key):
                print(f"- {key}: {inferred[key]}", file=output)


def _write_demo_competition_materials(
    competition: str,
    *,
    competition_url: str,
    topic: str,
    metric: str,
    objective: str,
    target_column: str,
    id_column: str,
    required_data_files: list[str],
    url_info: dict[str, Any],
    source_materials: dict[str, str],
) -> None:
    root = paths.competition_dir(competition)
    record = {
        "competition": competition,
        "competition_url": competition_url,
        "topic": topic,
        "metric": metric,
        "objective": objective,
        "target_column": target_column,
        "id_column": id_column,
        "required_data_files": required_data_files,
        "url_read_status": url_info,
        "source_materials": source_materials,
    }
    write_text(root / "source_materials.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    write_text(root / "source_materials.md", _render_source_materials(record))
    write_text(
        root / "overview.md",
        "\n".join(
            [
                f"# {competition}",
                "",
                f"- topic: {topic}",
                f"- competition_url: {competition_url or '-'}",
                "",
                "## Problem Description",
                "",
                source_materials.get("problem_description") or url_info.get("excerpt") or "Not provided.",
                "",
                "## Constraints / Rules",
                "",
                source_materials.get("constraints_or_rules") or "Not provided.",
                "",
            ]
        ),
    )
    write_text(
        root / "data_notes.md",
        "\n".join(
            [
                "# Data Notes",
                "",
                f"- target_column: {target_column}",
                f"- id_column: {id_column or '-'}",
                "- required_data_files:",
                *[f"  - {name}" for name in required_data_files],
                "",
                "## Data Description",
                "",
                source_materials.get("data_description") or "Not provided.",
                "",
                "## Submission Format",
                "",
                source_materials.get("submission_format") or "Not provided.",
                "",
            ]
        ),
    )
    write_text(
        root / "metric.md",
        "\n".join(
            [
                "# Metric",
                "",
                f"- name: {metric}",
                f"- objective: {objective}",
                "",
                "## Evaluation Rule",
                "",
                source_materials.get("evaluation_rule") or "Not provided.",
                "",
                "## Notes",
                "",
                source_materials.get("notes") or "Not provided.",
                "",
            ]
        ),
    )


def _render_source_materials(record: dict[str, Any]) -> str:
    materials = record["source_materials"]
    lines = [
        f"# {record['competition']} Source Materials",
        "",
        f"- competition_url: {record.get('competition_url') or '-'}",
        f"- topic: {record.get('topic')}",
        f"- metric: {record.get('metric')}",
        f"- objective: {record.get('objective')}",
        "",
        "## URL Read Status",
        "",
        f"- status: {record.get('url_read_status', {}).get('status')}",
        f"- reason: {record.get('url_read_status', {}).get('reason')}",
        "",
    ]
    for key, title in [
        ("source_summary", "Source Summary"),
        ("problem_description", "Problem Description"),
        ("data_description", "Data Description"),
        ("submission_format", "Submission Format"),
        ("evaluation_rule", "Evaluation Rule"),
        ("constraints_or_rules", "Constraints / Rules"),
        ("notes", "Notes"),
    ]:
        lines.extend(["", f"## {title}", "", materials.get(key) or "Not provided."])
    return "\n".join(lines) + "\n"


def _select_experiment(
    experiments: list[dict[str, Any]],
    *,
    input_fn: InputFn,
    output: TextIO,
) -> dict[str, Any] | None:
    if not experiments:
        print("등록된 실험이 없습니다. 먼저 새 실험을 시작해주세요.", file=output)
        return None
    print("", file=output)
    print("실험을 선택해주세요:", file=output)
    for index, row in enumerate(experiments, start=1):
        print(f"{index}. {row['competition']:<16} status: {row['status']}", file=output)
    raw = input_fn("select: ").strip()
    try:
        selected_index = int(raw)
    except ValueError:
        print("숫자로 선택해주세요.", file=output)
        return None
    if not 1 <= selected_index <= len(experiments):
        print("목록에 있는 번호를 선택해주세요.", file=output)
        return None
    return experiments[selected_index - 1]


def _print_header(output: TextIO) -> None:
    print("Autonomous ML Research Agent", file=output)
    print("", file=output)


def _print_experiment_list(experiments: list[dict[str, Any]], output: TextIO) -> None:
    print("현재 등록된 실험:", file=output)
    if not experiments:
        print("- 아직 등록된 실험이 없습니다.", file=output)
        return
    for index, row in enumerate(experiments, start=1):
        print(f"{index}. {row['competition']:<16} status: {row['status']}", file=output)


def _print_menu(output: TextIO) -> None:
    print("", file=output)
    print("무엇을 하시겠습니까?", file=output)
    print("1. 새 실험 시작", file=output)
    print("2. 기존 실험 사이클 실행", file=output)
    print("3. 해당 실험 현황보기", file=output)
    print("4. 종료", file=output)


def _prompt_required(input_fn: InputFn, label: str) -> str:
    while True:
        value = input_fn(f"{label}: ").strip()
        if value:
            return value


def _prompt_optional(input_fn: InputFn, label: str) -> str:
    return input_fn(f"{label}: ").strip()


def _prompt_with_default(input_fn: InputFn, label: str, default: str) -> str:
    if default:
        value = input_fn(f"{label} [{default}]: ").strip()
        return value or default
    return _prompt_required(input_fn, label)


def _prompt_choice(input_fn: InputFn, label: str, choices: list[str], *, default: str) -> str:
    while True:
        value = input_fn(f"{label} [{default}]: ").strip() or default
        if value in choices:
            return value


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(read_text(path, "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _check_demo_data_files(source_path: str | None, required_files: list[str]) -> dict[str, Any]:
    if not source_path:
        return {"found_files": [], "missing_files": required_files}
    data_dir = Path(source_path) / "data"
    found = [name for name in required_files if (data_dir / name).exists()]
    missing = [name for name in required_files if name not in found]
    return {"found_files": found, "missing_files": missing}


def _latest_trial_id(competition: str) -> str | None:
    root = paths.experiment_dir(competition)
    if not root.exists():
        return None
    trials = [item.name for item in root.iterdir() if item.is_dir()]
    return sorted(trials)[-1] if trials else None


def _demo_artifact_status(competition: str, trial_id: str) -> list[dict[str, str]]:
    trial = paths.trial_dir(competition, trial_id)
    source = summarize_source_path(competition)
    workspace = Path(source) if source else None
    candidates = [
        ("experiments", trial / "demo_one_cycle.md"),
        ("experiments", trial / "next_experiment.md"),
        ("experiments", trial / "workspace_coding_result.json"),
        ("experiments", trial / "workspace_run.json"),
    ]
    if workspace:
        candidates.extend(
            [
                ("workspace", workspace / "outputs" / "metrics.json"),
                ("workspace", workspace / "outputs" / "submission.csv"),
            ]
        )
    artifacts = []
    for scope, path in candidates:
        found = path.exists()
        if scope == "experiments":
            found = trial_artifact_exists(trial, path.name)
        artifacts.append(
            {
                "path": f"{scope}/{path.name}" if scope == "workspace" else path.as_posix(),
                "status": "found" if found else "missing",
            }
        )
    return artifacts


def summarize_source_path(competition: str) -> str | None:
    source_record = _read_json(paths.competition_dir(competition) / "workspace_source.json")
    source_path = source_record.get("source_path")
    return str(source_path) if source_path else None


def _infer_from_url(url: str) -> dict[str, str]:
    parsed = urllib.parse.urlparse(url)
    inferred: dict[str, str] = {}
    host = parsed.netloc.lower()
    if "kaggle.com" in host:
        inferred["platform"] = "kaggle"
    path_parts = [part for part in parsed.path.split("/") if part]
    if "competitions" in path_parts:
        index = path_parts.index("competitions")
        if len(path_parts) > index + 1:
            slug = path_parts[index + 1]
            inferred["competition"] = slug.replace("-", "_")
            inferred["topic"] = slug.replace("-", " ").title()
    return inferred


def _fetch_url_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAgentDemo/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        content_type = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(content_type, errors="replace")


def _extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()


def _html_to_text(raw_html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
