from __future__ import annotations

import json
from typing import Any

from ..paths import competition_memory_dir
from ..store import now_iso, write_text
from .document_registry import RetrievalDocument, collect_retrieval_documents


def build_document_index(competition: str) -> dict[str, Any]:
    docs = collect_retrieval_documents(competition)
    out_dir = competition_memory_dir(competition)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "document_index.jsonl"
    manifest_path = out_dir / "document_index_manifest.json"
    summary_path = out_dir / "document_index.md"

    index_path.write_text(
        "".join(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n" for doc in docs),
        encoding="utf-8",
    )
    by_kind: dict[str, int] = {}
    for doc in docs:
        by_kind[doc.source_kind] = by_kind.get(doc.source_kind, 0) + 1
    manifest = {
        "schema_version": "1.0",
        "competition": competition,
        "built_at": now_iso(),
        "document_count": len(docs),
        "by_source_kind": dict(sorted(by_kind.items())),
        "index_file": f"memory/{competition}/document_index.jsonl",
        "summary_file": f"memory/{competition}/document_index.md",
    }
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(summary_path, render_document_index_summary(competition, docs, manifest))
    return manifest


def render_document_index_summary(
    competition: str,
    docs: list[RetrievalDocument],
    manifest: dict[str, Any],
) -> str:
    lines = [
        f"# Retrieval Document Index - {competition}",
        "",
        "## Summary",
        "",
        f"- built_at: {manifest['built_at']}",
        f"- document_count: {manifest['document_count']}",
        f"- index_file: `{manifest['index_file']}`",
        "",
        "## Source Kinds",
        "",
    ]
    for kind, count in manifest["by_source_kind"].items():
        lines.append(f"- {kind}: {count}")
    lines.extend(["", "## Documents", ""])
    for doc in docs:
        trial = doc.trial_id or "-"
        lines.append(f"- `{doc.source_path}` | kind={doc.source_kind} | trial={trial}")
    lines.append("")
    return "\n".join(lines)
