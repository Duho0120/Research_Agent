from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


MAX_EVIDENCE_ITEMS = 7
MAX_CONTEXT_CHARS = 10_000
MAX_CHUNK_CHARS = 1_800

_STRUCTURED_TERMS = {
    "점수",
    "score",
    "로컬",
    "local",
    "cv",
    "제출",
    "submit",
    "submission",
    "lb",
    "리더보드",
    "베스트",
    "best",
    "상태",
    "status",
    "순위",
    "rank",
}
_DOCUMENT_TERMS = {
    "계획",
    "plan",
    "파이프라인",
    "pipeline",
    "코드",
    "code",
    "모델",
    "model",
    "인사이트",
    "insight",
    "개선축",
    "axis",
    "오류",
    "error",
    "실패",
    "failure",
    "이유",
    "why",
    "반영",
    "applied",
    "실행",
    "execution",
}
_EXCLUDED_NAMES = {
    "document_index.jsonl",
    "document_index.json",
    "document_index.md",
    "document_index_manifest.json",
    "graph_rag_manifest.json",
    "graph_rag_manifest.md",
    "token_usage.jsonl",
}
_EXCLUDED_PREFIXES = ("retrieval_manifest_", "context_pack_")
_SYNONYMS = {
    "제출점수": ("제출", "점수", "lb_score", "kaggle_lb_score", "submit", "submission"),
    "로컬점수": ("로컬", "점수", "local_score", "cv_score", "local"),
    "베스트": ("best", "is_best_lb"),
    "앙상블": ("ensemble", "votingclassifier", "stackingclassifier", "model_ensemble"),
    "모델": ("model", "estimator"),
    "개선축": ("axis", "change_axis", "active_axis"),
    "실행": ("execution", "executed", "runtime"),
    "계획": ("plan", "delta_plan"),
    "파이프라인": ("pipeline", "pipeline_structure"),
}


def retrieve_experiment_evidence(
    root: Path,
    competition: str,
    trial_id: str | None,
    question: str,
) -> list[tuple[str, str]]:
    mode = classify_question(question)
    structured = _structured_evidence(root, competition, trial_id, question)
    if mode == "structured" and structured:
        return _fit_budget(structured)

    documents = _rank_document_chunks(root, competition, trial_id, question, mode=mode)
    return _fit_budget([*structured, *documents])


def classify_question(question: str) -> str:
    normalized = _normalize(question)
    has_structured = any(term in normalized for term in _STRUCTURED_TERMS)
    has_document = any(term in normalized for term in _DOCUMENT_TERMS)
    if has_structured and not has_document:
        return "structured"
    if has_structured and has_document:
        return "hybrid"
    return "document"


def _structured_evidence(
    root: Path,
    competition: str,
    trial_id: str | None,
    question: str,
) -> list[tuple[str, str, float]]:
    rows = _structured_trial_rows(root, competition)
    requested = _requested_trials(question)
    asks_for_all = _asks_for_all_trials(question)
    if requested:
        rows = [row for row in rows if str(row.get("trial_id") or "").lower() in requested]
    elif trial_id and not asks_for_all:
        selected = [row for row in rows if str(row.get("trial_id") or "").lower() == trial_id.lower()]
        if selected:
            rows = selected
    if not rows:
        return []
    payload = {
        "source_priority": "실제 점수와 실행 기록을 계획 문서보다 우선합니다.",
        "competition": competition,
        "trials": rows,
    }
    return [("sqlite:trial_summary", json.dumps(payload, ensure_ascii=False, indent=2), 100.0)]


def _structured_trial_rows(root: Path, competition: str) -> list[dict[str, Any]]:
    rows_by_trial = _sqlite_trial_rows(root, competition)
    for row in _manual_metric_rows(root, competition):
        trial = str(row.get("trial_id") or "")
        if not trial:
            continue
        current = rows_by_trial.setdefault(trial, {"trial_id": trial})
        for key, value in row.items():
            if value is not None and current.get(key) in (None, "", "-"):
                current[key] = value
    return [rows_by_trial[key] for key in sorted(rows_by_trial)]


def _sqlite_trial_rows(root: Path, competition: str) -> dict[str, dict[str, Any]]:
    db_path = root / "memory" / "research_agent.sqlite3"
    if not db_path.is_file():
        return {}
    query = """
        SELECT
            t.trial_id,
            t.status,
            t.source_trial_id,
            t.plan_type,
            t.plan_summary,
            t.primary_change_axis,
            s.metric,
            s.objective,
            s.local_score,
            s.lb_score,
            s.is_best_lb,
            d.decision,
            d.change_axis,
            d.active_axis,
            d.axis_attempt_count,
            d.axis_attempt_limit
        FROM trials t
        LEFT JOIN trial_scores s
          ON s.competition_id = t.competition_id AND s.trial_id = t.trial_id
        LEFT JOIN trial_decisions d
          ON d.competition_id = t.competition_id AND d.trial_id = t.trial_id
        WHERE t.competition_id = ?
        ORDER BY t.trial_id
    """
    try:
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (competition,)).fetchall()
        connection.close()
    except (sqlite3.Error, OSError):
        return {}
    return {str(row["trial_id"]): _compact_row(dict(row)) for row in rows}


def _manual_metric_rows(root: Path, competition: str) -> list[dict[str, Any]]:
    manual_root = root / "demo_workspaces" / competition / "manual_trials"
    rows: list[dict[str, Any]] = []
    for path in sorted(manual_root.glob("trial_*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(
            _compact_row(
                {
                    "trial_id": payload.get("trial_id") or path.parent.name,
                    "status": payload.get("status"),
                    "metric": payload.get("metric"),
                    "local_score": payload.get("local_score", payload.get("cv_score")),
                    "lb_score": payload.get("kaggle_lb_score", payload.get("lb_score")),
                    "change_axis": payload.get("change_axis"),
                    "model": payload.get("model"),
                    "submission_status": "submitted" if payload.get("kaggle_submitted") else None,
                    "kaggle_ref": payload.get("kaggle_ref"),
                }
            )
        )
    return rows


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}


def _rank_document_chunks(
    root: Path,
    competition: str,
    trial_id: str | None,
    question: str,
    *,
    mode: str,
) -> list[tuple[str, str, float]]:
    query_terms = _query_terms(question)
    if not query_terms:
        return []
    requested = _requested_trials(question)
    if not requested and trial_id and not _asks_for_all_trials(question):
        requested = {trial_id.lower()}

    documents: list[tuple[str, str, list[str], float]] = []
    seen_content: set[str] = set()
    for path in _candidate_files(root, competition, requested, include_code=_asks_about_code(question)):
        source = str(path.relative_to(root))
        for chunk_index, content in enumerate(_chunks(path), start=1):
            fingerprint = _content_fingerprint(content)
            if fingerprint in seen_content:
                continue
            seen_content.add(fingerprint)
            searchable = f"{source} {content}"
            documents.append(
                (
                    f"{source}#chunk-{chunk_index}",
                    content,
                    _index_terms(searchable),
                    _source_boost(source, mode),
                )
            )
    if not documents:
        return []

    document_frequency: Counter[str] = Counter()
    for _, _, document_terms, _ in documents:
        document_frequency.update(set(document_terms))
    average_length = sum(len(item[2]) for item in documents) / len(documents)

    scored: list[tuple[str, str, float]] = []
    for source, content, document_terms, boost in documents:
        frequencies = Counter(document_terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(document_terms) / max(average_length, 1)
            )
            score += inverse_frequency * (frequency * 2.5 / denominator)
        if score > 0:
            scored.append((source, content, score + boost))
    scored.sort(key=lambda item: (-item[2], item[0]))
    return scored[: MAX_EVIDENCE_ITEMS - 1]


def _candidate_files(
    root: Path,
    competition: str,
    requested_trials: set[str],
    *,
    include_code: bool,
) -> list[Path]:
    roots = [
        root / "experiments" / competition,
        root / "runs" / competition,
        root / "memory" / competition,
        root / "demo_workspaces" / competition / "manual_trials",
    ]
    if include_code:
        roots.append(root / "demo_workspaces" / competition)

    paths: list[Path] = []
    patterns = ("*.md", "*.json", "*.jsonl", "*.py")
    for base in roots:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if not path.is_file() or _excluded(path):
                    continue
                relative = str(path.relative_to(root)).lower()
                path_trial = _trial_from_path(relative)
                if requested_trials and path_trial and path_trial not in requested_trials:
                    continue
                paths.append(path)
    return sorted(set(paths), key=lambda path: (_path_priority(path), str(path)))


def _excluded(path: Path) -> bool:
    name = path.name.lower()
    if name in _EXCLUDED_NAMES or name.startswith(_EXCLUDED_PREFIXES):
        return True
    lowered_parts = {part.lower() for part in path.parts}
    return bool({"__pycache__", "internal", "workspace_logs"} & lowered_parts)


def _chunks(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return []
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            result.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
        while len(current) > MAX_CHUNK_CHARS:
            result.append(current[:MAX_CHUNK_CHARS])
            current = current[MAX_CHUNK_CHARS:]
    if current:
        result.append(current)
    return result


def _fit_budget(items: list[tuple[str, str, float]]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    total_chars = 0
    for source, content, _ in items:
        if len(selected) >= MAX_EVIDENCE_ITEMS:
            break
        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        clipped = content[:remaining]
        if clipped:
            selected.append((source, clipped))
            total_chars += len(clipped)
    return selected


def _query_terms(text: str) -> list[str]:
    normalized = _normalize(text)
    tokens = _index_terms(normalized)
    if "제출 점수" in normalized:
        tokens.append("제출점수")
    if "로컬 점수" in normalized:
        tokens.append("로컬점수")
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, ()))
    return list(dict.fromkeys(expanded))


def _index_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"trial_\d+|[a-z][a-z0-9_]+|[가-힣]{2,}|\d+\.\d+", _normalize(text))
        if len(token) >= 2
    ]


def _normalize(text: str) -> str:
    return text.lower().replace("\\", "/")


def _requested_trials(question: str) -> set[str]:
    return set(re.findall(r"trial_\d+", question.lower()))


def _asks_for_all_trials(question: str) -> bool:
    normalized = _normalize(question)
    return any(term in normalized for term in ("각 실험", "각 trial", "전체", "지금까지", "모든 trial"))


def _asks_about_code(question: str) -> bool:
    normalized = _normalize(question)
    return any(term in normalized for term in ("코드", "code", "실제 실행", "실행 코드", "모델"))


def _trial_from_path(relative_path: str) -> str | None:
    matched = re.search(r"trial_\d+", relative_path)
    return matched.group(0) if matched else None


def _content_fingerprint(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _path_priority(path: Path) -> int:
    name = path.name.lower()
    if name in {"execution_facts.json", "trial_memory_card.md", "metrics.json", "03_scores.ko.md"}:
        return 0
    if name in {"user_insights.jsonl", "02_pipeline_structure.ko.md", "pipeline_structure.json"}:
        return 1
    if name in {"delta_plan.json", "01_plan.ko.md", "plan.md"}:
        return 2
    return 10


def _source_boost(source: str, mode: str) -> float:
    lowered = source.lower()
    boost = 0.0
    if any(name in lowered for name in ("execution_facts", "trial_memory_card", "metrics.json", "03_scores")):
        boost += 3.0
    if any(name in lowered for name in ("user_insights", "pipeline_structure")):
        boost += 2.0
    if mode == "document" and any(name in lowered for name in ("delta_plan", "01_plan", "plan.md")):
        boost += 1.0
    return boost
