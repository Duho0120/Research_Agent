from __future__ import annotations

from typing import Any

from .competition_inspector import inspect_competition
from .data_onboarding import profile_competition_data
from .integrations import kaggle_cli
from .paths import competition_dir, competition_memory_dir
from .store import init_project, write_text
from .agents.research_planner import propose_plan


def start_competition(
    competition_reference: str,
    metric: str = "unknown",
    objective: str = "maximize",
    trial_id: str = "trial_001",
    runner: kaggle_cli.CommandRunner | None = None,
) -> dict[str, Any]:
    inspection = inspect_competition(competition_reference, runner=runner)
    slug = inspection["competition_slug"]
    result: dict[str, Any] = {
        "competition_slug": slug,
        "status": inspection["status"],
        "inspection": inspection,
        "trial_id": trial_id,
        "plan": None,
        "steps": ["inspected"],
    }
    if inspection["status"] != "ready":
        result["steps"].append("blocked_before_init")
        return result

    init_project(slug, metric=metric, objective=objective)
    result["steps"].append("initialized")
    _write_onboarding_docs(slug, inspection, metric, objective)
    result["steps"].append("documented")
    data_profile = profile_competition_data(slug)
    result["data_profile"] = {
        "status": data_profile["status"],
        "task_type": data_profile["task_type"],
        "target_candidates": data_profile["target_candidates"],
    }
    result["steps"].append("data_profiled")
    plan = propose_plan(slug, trial_id)
    result["plan"] = {"trial_id": plan["trial_id"], "hypothesis": plan["hypothesis"]}
    result["steps"].append("planned")
    return result


def _write_onboarding_docs(slug: str, inspection: dict[str, Any], metric: str, objective: str) -> None:
    files = inspection.get("files", [])
    competition_path = competition_dir(slug)
    memory_path = competition_memory_dir(slug)
    write_text(competition_path / "overview.md", _render_overview(inspection, metric, objective))
    write_text(competition_path / "data_notes.md", _render_data_notes(inspection))
    write_text(memory_path / "research_notes.md", _render_research_notes(inspection, files))


def _render_overview(inspection: dict[str, Any], metric: str, objective: str) -> str:
    lines = [
        f"# {inspection['competition_slug']}",
        "",
        f"- competition_url: {inspection['competition_url']}",
        f"- metric: {metric}",
        f"- objective: {objective}",
        f"- inspection_status: {inspection['status']}",
        "",
        "## Available Files",
        "",
    ]
    for item in inspection.get("files", []) or []:
        suffix = f" ({item['size']})" if item.get("size") else ""
        lines.append(f"- {item['name']}{suffix}")
    if not inspection.get("files"):
        lines.append("- No files discovered yet.")
    lines.extend(["", "## Initial Direction", "", "Build a reliable baseline and verify the submission format before leaderboard use.", ""])
    return "\n".join(lines)


def _render_data_notes(inspection: dict[str, Any]) -> str:
    names = [item["name"] for item in inspection.get("files", [])]
    lines = ["# Data Notes", "", "## Files", ""]
    lines.extend(f"- {name}" for name in names or ["No files discovered yet."])
    lines.extend(["", "## Submission Format", ""])
    if any("samplesubmission" in "".join(character for character in name.casefold() if character.isalnum()) for name in names):
        lines.append("- Use sample_submission file as the source of required submission columns.")
    else:
        lines.append("- Sample submission file not detected in the current listing.")
    lines.extend(["", "## Follow-up", "", "- Download and inspect train/test schemas before model-specific feature planning.", ""])
    return "\n".join(lines)


def _render_research_notes(inspection: dict[str, Any], files: list[dict[str, Any]]) -> str:
    lines = [
        "# Research Notes",
        "",
        f"- Onboarded from: {inspection['competition_url']}",
        f"- Inspection status: {inspection['status']}",
        f"- File count: {len(files)}",
        "",
        "## Immediate Plan",
        "",
        "- Download data.",
        "- Inspect schema and target column.",
        "- Create baseline training and submission pipeline.",
        "- Use prepare-submission before any real leaderboard submit.",
        "",
    ]
    return "\n".join(lines)
