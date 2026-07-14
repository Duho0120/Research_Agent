from __future__ import annotations

import json
import re
from typing import Any

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
    effective_query = query or TASK_QUERIES.get(normalized_task, normalized_task.replace("_", " "))
    build_document_index(competition)
    retrieval = retrieve_documents(competition, effective_query, limit=limit, rebuild_if_missing=False)
    documents = []
    for item in retrieval["results"]:
        text = load_document_text(str(item["source_path"]), max_chars=max_chars_per_document)
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
