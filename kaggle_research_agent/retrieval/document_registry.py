from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .. import paths
from ..paths import competition_dir, competition_memory_dir, experiment_dir


COMPETITION_FILES = {
    "overview.md": "competition_overview",
    "metric.md": "competition_metric",
    "data_notes.md": "data_notes",
    "source_materials.md": "source_materials",
    "source_materials.json": "source_materials",
    "data_profile.md": "data_profile",
    "data_profile.json": "data_profile",
    "execution_profile.yaml": "execution_profile",
    "workspace_inventory.md": "workspace_inventory",
    "workspace_inventory.json": "workspace_inventory",
    "state.yaml": "competition_state",
}

MEMORY_FILES = {
    "research_notes.md": "research_notes",
    "rules.md": "research_rules",
    "trial_index.jsonl": "trial_index",
    "demo_trial_index.jsonl": "trial_index",
    "decision_log.jsonl": "decision_log",
    "user_feedback.jsonl": "user_feedback",
    "token_usage.jsonl": "token_usage",
    "best_trial.json": "best_trial",
}

TRIAL_FILES = {
    "plan.md": "trial_plan",
    "demo_experiment_plan.md": "trial_plan",
    "next_experiment.md": "next_experiment",
    "metrics.json": "trial_metrics",
    "evaluation.md": "trial_evaluation",
    "diagnosis.md": "trial_diagnosis",
    "demo_cycle_record.md": "trial_record",
    "metrics_collection.md": "metrics_collection",
    "workspace_context_snapshot.md": "workspace_context_snapshot",
    "workspace_coding_result.md": "coding_result",
    "workspace_run.md": "workspace_run",
    "graph_state.json": "graph_state",
    "node_events.jsonl": "graph_events",
}

TRIAL_NESTED_FILES = {
    "internal/pipeline_structure.json": "pipeline_structure",
    "internal/artifact_manifest.json": "artifact_manifest",
    "internal/demo_experiment_plan.json": "trial_plan",
    "internal/demo_cycle_record.json": "trial_record",
    "internal/metrics_collection.json": "metrics_collection",
    "internal/workspace_run.json": "workspace_run",
    "user_view/README.ko.md": "user_summary",
    "user_view/01_plan.ko.md": "user_plan",
    "user_view/02_pipeline_structure.ko.md": "user_pipeline_structure",
    "user_view/03_code_pipeline.ko.md": "user_code_pipeline",
    "user_view/04_result.ko.md": "user_result",
}


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    competition: str
    trial_id: str | None
    source_path: str
    source_kind: str
    title: str
    content_type: str
    size_bytes: int
    modified_at: str
    sha256: str
    text_preview: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "competition": self.competition,
            "trial_id": self.trial_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "title": self.title,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "sha256": self.sha256,
            "text_preview": self.text_preview,
        }


def collect_retrieval_documents(competition: str) -> list[RetrievalDocument]:
    docs: list[RetrievalDocument] = []
    docs.extend(_collect_named_files(competition, competition_dir(competition), COMPETITION_FILES, trial_id=None))
    docs.extend(_collect_named_files(competition, competition_memory_dir(competition), MEMORY_FILES, trial_id=None))
    docs.extend(_collect_trial_documents(competition))
    return sorted(docs, key=lambda item: (item.trial_id or "", item.source_kind, item.source_path))


def _collect_trial_documents(competition: str) -> Iterable[RetrievalDocument]:
    root = experiment_dir(competition)
    if not root.exists():
        return []
    docs: list[RetrievalDocument] = []
    for trial_path in sorted(path for path in root.iterdir() if path.is_dir()):
        trial_id = trial_path.name
        docs.extend(_collect_named_files(competition, trial_path, TRIAL_FILES, trial_id=trial_id))
        docs.extend(_collect_named_files(competition, trial_path, TRIAL_NESTED_FILES, trial_id=trial_id))
    return docs


def _collect_named_files(
    competition: str,
    base: Path,
    names: dict[str, str],
    *,
    trial_id: str | None,
) -> list[RetrievalDocument]:
    docs: list[RetrievalDocument] = []
    for name, kind in names.items():
        path = base / name
        if path.is_file():
            docs.append(_document_from_path(competition, path, kind, trial_id=trial_id))
    return docs


def _document_from_path(competition: str, path: Path, source_kind: str, *, trial_id: str | None) -> RetrievalDocument:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rel = _relative_path(path)
    text = raw.decode("utf-8-sig", errors="replace")
    return RetrievalDocument(
        document_id=f"{competition}:{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:16]}",
        competition=competition,
        trial_id=trial_id,
        source_path=rel,
        source_kind=source_kind,
        title=_title_from_path(path, source_kind, trial_id),
        content_type=_content_type(path),
        size_bytes=len(raw),
        modified_at=_modified_at(path),
        sha256=digest,
        text_preview=_preview(text),
    )


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(paths.project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _title_from_path(path: Path, source_kind: str, trial_id: str | None) -> str:
    prefix = f"{trial_id} " if trial_id else ""
    return f"{prefix}{source_kind}: {path.name}"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".jsonl":
        return "application/jsonl"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    if suffix in {".yaml", ".yml"}:
        return "text/yaml"
    return "text/plain"


def _modified_at(path: Path) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _preview(text: str, limit: int = 600) -> str:
    compact = " ".join(text.split())
    return compact[:limit]
