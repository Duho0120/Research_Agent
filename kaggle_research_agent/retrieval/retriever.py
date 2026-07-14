from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import paths
from ..paths import competition_memory_dir
from .index_builder import build_document_index


def retrieve_documents(competition: str, query: str, *, limit: int = 5, rebuild_if_missing: bool = True) -> dict[str, Any]:
    index_path = competition_memory_dir(competition) / "document_index.jsonl"
    if rebuild_if_missing and not index_path.exists():
        build_document_index(competition)
    docs = _load_index(index_path)
    terms = _terms(query)
    scored = []
    for doc in docs:
        text = _search_text(doc)
        score = _score(text, terms)
        if score > 0:
            scored.append({**doc, "score": score})
    scored.sort(key=lambda item: (-int(item["score"]), str(item["source_path"])))
    return {
        "competition": competition,
        "query": query,
        "limit": limit,
        "index_file": f"memory/{competition}/document_index.jsonl",
        "result_count": min(len(scored), limit),
        "results": scored[:limit],
    }


def load_document_text(source_path: str, *, max_chars: int = 4000) -> str:
    path = paths.project_root() / source_path
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return text[:max_chars]


def _load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _search_text(doc: dict[str, Any]) -> str:
    parts = [
        doc.get("source_path", ""),
        doc.get("source_kind", ""),
        doc.get("title", ""),
        doc.get("text_preview", ""),
        doc.get("trial_id", "") or "",
    ]
    return " ".join(str(part).lower() for part in parts)


def _terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[\w가-힣]+", query) if len(term) >= 2]


def _score(text: str, terms: list[str]) -> int:
    if not terms:
        return 0
    score = 0
    for term in terms:
        occurrences = text.count(term)
        if occurrences:
            score += 2 + occurrences
    return score
