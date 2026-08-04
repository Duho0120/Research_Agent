from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .paths import trial_dir
from .store import now_iso, read_text, write_text


PENDING_SNAPSHOT_NAME = "pending_execution_plan_snapshot.json"
EXECUTION_SNAPSHOT_NAME = "execution_plan_snapshot.json"


def capture_pending_execution_plan(
    competition: str,
    trial_id: str,
    *,
    request_text: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Capture the plan contained in the latest code-writer request.

    The pending snapshot may be replaced while code writing is retried. Once a
    coding result is accepted, ``finalize_execution_plan_snapshot`` freezes it.
    """

    out_dir = trial_dir(competition, trial_id)
    plan_text = extract_plan_from_coding_request(request_text)
    if not plan_text:
        plan_text = read_text(out_dir / "next_experiment.md", default="").strip()
    snapshot = _build_snapshot(
        competition,
        trial_id,
        plan_text=plan_text,
        request_id=request_id,
        status="pending",
    )
    write_text(
        out_dir / "internal" / PENDING_SNAPSHOT_NAME,
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
    )
    write_text(out_dir / "internal" / "pending_execution_plan_snapshot.md", _render_snapshot(snapshot))
    return snapshot


def ensure_pending_execution_plan_snapshot(
    competition: str,
    trial_id: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    request_text = _read_coding_request(out_dir)
    plan_text = extract_plan_from_coding_request(request_text)
    if not plan_text:
        plan_text = read_text(out_dir / "next_experiment.md", default="").strip()
    pending = _read_json(out_dir / "internal" / PENDING_SNAPSHOT_NAME)
    # A cached pending snapshot is only trustworthy if it still matches the plan
    # currently being sent to the code writer. If the plan was force-replanned
    # (e.g. after a stale continuation_context.json fix, or a duplicate-candidate
    # replan) between coding attempts, the old pending snapshot must not survive
    # -- reusing it would later get frozen as the "fact" for a trial that
    # actually ran a different plan, corrupting axis/candidate history for every
    # later trial's planning.
    if pending and (not plan_text or pending.get("plan_markdown") == plan_text):
        return pending
    return capture_pending_execution_plan(
        competition,
        trial_id,
        request_text=request_text,
        request_id=request_id,
    )


def finalize_execution_plan_snapshot(competition: str, trial_id: str) -> dict[str, Any]:
    """Freeze the accepted code-writer plan without overwriting an existing fact."""

    out_dir = trial_dir(competition, trial_id)
    final_path = out_dir / "internal" / EXECUTION_SNAPSHOT_NAME
    existing = _read_json(final_path)
    if existing:
        return existing

    pending = ensure_pending_execution_plan_snapshot(competition, trial_id)
    snapshot = dict(pending)
    snapshot["status"] = "finalized"
    snapshot["finalized_at"] = now_iso()
    write_text(final_path, json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "internal" / "execution_plan_snapshot.md", _render_snapshot(snapshot))
    return snapshot


def load_execution_plan_snapshot(competition: str, trial_id: str) -> dict[str, Any]:
    return _read_json(trial_dir(competition, trial_id) / "internal" / EXECUTION_SNAPSHOT_NAME)


def backfill_execution_plan_snapshot(competition: str, trial_id: str) -> dict[str, Any]:
    """Recover a finalized snapshot from a preserved historical coding request."""

    out_dir = trial_dir(competition, trial_id)
    existing = load_execution_plan_snapshot(competition, trial_id)
    if existing:
        return existing
    request_text = _read_coding_request(out_dir)
    if not request_text:
        return {}
    capture_pending_execution_plan(
        competition,
        trial_id,
        request_text=request_text,
        request_id=f"{competition}:{trial_id}:workspace-coding",
    )
    return finalize_execution_plan_snapshot(competition, trial_id)


def extract_plan_from_coding_request(request_text: str) -> str:
    marker = re.search(r"(?m)^## Next Experiment\s*$", request_text)
    if not marker:
        return ""
    return request_text[marker.end() :].strip()


def parse_execution_plan_markdown(plan_text: str, *, trial_id: str) -> dict[str, Any]:
    metadata = _metadata(plan_text)
    sections = _sections(plan_text)
    source_trial_id = metadata.get("source_trial_id") or metadata.get("base_source_trial")
    axis = metadata.get("active_axis") or _section_scalar(sections, "primary change axis")
    candidate_name = metadata.get("candidate")
    candidate: dict[str, Any] = {}
    if candidate_name:
        candidate["name"] = candidate_name
    if metadata.get("description"):
        candidate["description"] = metadata["description"]
    if metadata.get("implementation_hint"):
        candidate["implementation_hint"] = metadata["implementation_hint"]

    plan = {
        "trial_id": trial_id,
        "source_trial_id": source_trial_id,
        "plan_type": metadata.get("plan_type")
        or ("continuation_delta_plan" if source_trial_id else "initial_pipeline_plan"),
        "plan_title": metadata.get("title") or candidate_name,
        "primary_change_axis": axis,
        "candidate": candidate,
        "change_details": _section_list(sections, "change details"),
        "keep_unchanged": _section_list(sections, "keep unchanged"),
        "affected_stages": _section_list(sections, "affected stages"),
        "required_code_symbols": _section_list(sections, "required code symbols"),
        "expected_metadata_changes": _section_list(sections, "expected metadata changes"),
        "code_change_targets": _section_list(sections, "code change targets"),
        "success_criteria": _section_list(sections, "success criteria"),
        "failure_decision": _section_list(sections, "failure decision"),
        "expected_outputs": _section_list(sections, "expected outputs"),
    }
    return {key: value for key, value in plan.items() if value not in (None, "", [], {})}


def _build_snapshot(
    competition: str,
    trial_id: str,
    *,
    plan_text: str,
    request_id: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "competition": competition,
        "trial_id": trial_id,
        "status": status,
        "captured_at": now_iso(),
        "request_id": request_id,
        "source": "workspace_coding_agent_request.md:Next Experiment",
        "plan": parse_execution_plan_markdown(plan_text, trial_id=trial_id),
        "plan_markdown": plan_text,
    }


def _metadata(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*-\s*([^:]+):\s*(.*?)\s*$", line)
        if not match:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", match.group(1).strip().lower()).strip("_")
        values[key] = match.group(2).strip()
    return values


def _sections(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            current = match.group(1).strip().lower()
            result.setdefault(current, [])
        elif current is not None:
            result[current].append(line)
    return result


def _section_scalar(sections: dict[str, list[str]], name: str) -> str | None:
    for line in sections.get(name, []):
        value = line.strip()
        if value and not value.startswith("- "):
            return value
    values = _section_list(sections, name)
    return values[0] if values else None


def _section_list(sections: dict[str, list[str]], name: str) -> list[str]:
    values = []
    for line in sections.get(name, []):
        value = line.strip()
        if value.startswith("- "):
            value = value[2:].strip()
        if value and value.lower() != "none":
            values.append(value)
    return values


def _read_coding_request(out_dir: Path) -> str:
    for path in [
        out_dir / "workspace_coding_agent_request.md",
        out_dir / "debug" / "workspace_coding_agent_request.md",
    ]:
        text = read_text(path, default="").strip()
        if text:
            return text
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _render_snapshot(snapshot: dict[str, Any]) -> str:
    plan = snapshot.get("plan") if isinstance(snapshot.get("plan"), dict) else {}
    lines = [
        f"# {snapshot.get('trial_id')} Execution Plan Snapshot",
        "",
        f"- status: {snapshot.get('status')}",
        f"- source: {snapshot.get('source')}",
        f"- request_id: {snapshot.get('request_id')}",
        f"- source_trial_id: {plan.get('source_trial_id')}",
        f"- primary_change_axis: {plan.get('primary_change_axis')}",
        f"- plan_title: {plan.get('plan_title')}",
        "",
        "## Plan Delivered To Code Writer",
        "",
        str(snapshot.get("plan_markdown") or "").strip(),
        "",
    ]
    return "\n".join(lines)
