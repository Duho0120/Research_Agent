from __future__ import annotations

import json
import re
from typing import Any

from ..policies import load_policy
from ..paths import trial_dir
from ..store import now_iso, write_text
from .index_builder import build_document_index
from .retriever import load_document_text, retrieve_documents


TASK_QUERIES = {
    "experiment_planning": (
        "competition overview metric data profile source materials previous trial plan metrics result "
        "pipeline structure failed issues user feedback"
    ),
    "workspace_code_writing": (
        "next experiment current pipeline code structure allowed files previous trial pipeline structure "
        "metrics result implementation notes validation"
    ),
}


def build_context_pack(
    competition: str,
    trial_id: str,
    *,
    task: str,
    query: str | None = None,
    limit: int = 8,
    max_chars_per_document: int = 1800,
) -> dict[str, Any]:
    normalized_task = _normalize_task(task)
    budget = _task_budget(normalized_task)
    effective_limit = min(limit, int(budget.get("max_documents") or limit))
    effective_max_chars = min(max_chars_per_document, int(budget.get("max_chars_per_document") or max_chars_per_document))
    effective_query = query or TASK_QUERIES.get(normalized_task, normalized_task.replace("_", " "))
    build_document_index(competition)
    retrieval = retrieve_documents(competition, effective_query, limit=max(limit * 3, effective_limit), rebuild_if_missing=False)
    documents = []
    for item in _rank_for_task(retrieval["results"], budget)[:effective_limit]:
        text = load_document_text(str(item["source_path"]), max_chars=effective_max_chars)
        documents.append(
            {
                "document_id": item["document_id"],
                "source_path": item["source_path"],
                "source_kind": item["source_kind"],
                "trial_id": item.get("trial_id"),
                "score": item["score"],
                "title": item.get("title"),
                "content_type": item.get("content_type"),
                "text": text,
            }
        )

    out_dir = trial_dir(competition, trial_id)
    context_pack_file = f"context_pack_{normalized_task}.json"
    context_pack_md_file = f"context_pack_{normalized_task}.md"
    manifest_file = f"retrieval_manifest_{normalized_task}.json"
    context_pack = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "task": normalized_task,
        "built_at": now_iso(),
        "query": effective_query,
        "budget": {
            "max_documents": effective_limit,
            "max_chars_per_document": effective_max_chars,
            "prompt_chars_per_document": budget.get("prompt_chars_per_document"),
        },
        "document_count": len(documents),
        "documents": documents,
        "retrieval_manifest_file": f"experiments/{competition}/{trial_id}/{manifest_file}",
    }
    manifest = {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "task": normalized_task,
        "built_at": context_pack["built_at"],
        "query": effective_query,
        "budget": context_pack["budget"],
        "index_file": f"memory/{competition}/document_index.jsonl",
        "context_pack_file": f"experiments/{competition}/{trial_id}/{context_pack_file}",
        "context_pack_md_file": f"experiments/{competition}/{trial_id}/{context_pack_md_file}",
        "retrieved_documents": [
            {
                "document_id": doc["document_id"],
                "source_path": doc["source_path"],
                "source_kind": doc["source_kind"],
                "trial_id": doc["trial_id"],
                "score": doc["score"],
            }
            for doc in documents
        ],
    }
    write_text(out_dir / context_pack_file, json.dumps(context_pack, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / context_pack_md_file, render_context_pack(context_pack))
    return {
        "task": normalized_task,
        "query": effective_query,
        "document_count": len(documents),
        "budget": context_pack["budget"],
        "context_pack_file": f"experiments/{competition}/{trial_id}/{context_pack_file}",
        "context_pack_md_file": f"experiments/{competition}/{trial_id}/{context_pack_md_file}",
        "retrieval_manifest_file": f"experiments/{competition}/{trial_id}/{manifest_file}",
        "documents": documents,
    }


def render_context_pack(context_pack: dict[str, Any]) -> str:
    lines = [
        f"# Context Pack - {context_pack['task']}",
        "",
        f"- competition: {context_pack['competition']}",
        f"- trial_id: {context_pack['trial_id']}",
        f"- query: {context_pack['query']}",
        f"- document_count: {context_pack['document_count']}",
        f"- retrieval_manifest_file: `{context_pack['retrieval_manifest_file']}`",
        "",
        "## Retrieved Evidence",
        "",
    ]
    for index, doc in enumerate(context_pack.get("documents", []), start=1):
        lines.extend(
            [
                f"### {index}. {doc.get('source_kind')} - {doc.get('source_path')}",
                "",
                f"- score: {doc.get('score')}",
                f"- trial_id: {doc.get('trial_id') or '-'}",
                "",
                "```text",
                str(doc.get("text", "")).rstrip(),
                "```",
                "",
            ]
        )
    if not context_pack.get("documents"):
        lines.append("- No evidence documents retrieved.")
    return "\n".join(lines)


def context_pack_prompt_summary(context_pack: dict[str, Any], *, max_chars_per_document: int = 1200) -> str:
    budget = context_pack.get("budget") if isinstance(context_pack.get("budget"), dict) else {}
    if budget.get("prompt_chars_per_document") is not None:
        try:
            max_chars_per_document = min(max_chars_per_document, int(budget["prompt_chars_per_document"]))
        except (TypeError, ValueError):
            pass
    lines = [
        f"RAG context pack task: {context_pack.get('task')}",
        f"Query: {context_pack.get('query')}",
        f"Documents: {context_pack.get('document_count')}",
        "",
    ]
    for doc in context_pack.get("documents", []):
        text = str(doc.get("text", ""))[:max_chars_per_document]
        lines.extend(
            [
                f"## {doc.get('source_kind')} | {doc.get('source_path')} | score={doc.get('score')}",
                text,
                "",
            ]
        )
    return "\n".join(lines).strip()


def _normalize_task(task: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", task.strip().lower()).strip("_")
    return value or "general"


def _task_budget(task: str) -> dict[str, Any]:
    policy = load_policy("rag_policy")
    tasks = policy.get("tasks", {}) if isinstance(policy.get("tasks"), dict) else {}
    budget = tasks.get(task, {}) if isinstance(tasks.get(task), dict) else {}
    return {
        "max_documents": budget.get("max_documents", 6),
        "max_chars_per_document": budget.get("max_chars_per_document", 1200),
        "prompt_chars_per_document": budget.get("prompt_chars_per_document", 800),
        "preferred_source_kinds": list(budget.get("preferred_source_kinds", [])),
        "demote_source_kinds": list(budget.get("demote_source_kinds", [])),
    }


def _rank_for_task(results: list[dict[str, Any]], budget: dict[str, Any]) -> list[dict[str, Any]]:
    preferred = {str(item) for item in budget.get("preferred_source_kinds", [])}
    demoted = {str(item) for item in budget.get("demote_source_kinds", [])}

    def key(item: dict[str, Any]) -> tuple[int, int, str]:
        kind = str(item.get("source_kind") or "")
        priority = 0
        if kind in preferred:
            priority -= 100
        if kind in demoted:
            priority += 100
        if kind in {"decision_card", "trial_memory_card"}:
            priority -= 200
        return (priority, -int(item.get("score") or 0), str(item.get("source_path") or ""))

    return sorted(results, key=key)
