from __future__ import annotations

import json
from typing import Any

from . import paths
from .integrations import kaggle_cli
from .store import now_iso, write_text


def inspect_competition(
    competition_reference: str,
    runner: kaggle_cli.CommandRunner | None = None,
) -> dict[str, Any]:
    slug = kaggle_cli.normalize_competition_slug(competition_reference)
    root = paths.project_root()
    command_runner = runner or kaggle_cli.run_args

    cli_check = kaggle_cli.check_cli_available(root, runner=command_runner)
    auth_check = kaggle_cli.check_cli_auth(root, runner=command_runner) if cli_check["ok"] else None
    files_result = None
    files: list[dict[str, Any]] = []
    status = "ready"
    reason = ""

    if not cli_check["ok"]:
        status = "cli_unavailable"
        reason = cli_check["stderr"] or cli_check["stdout"]
    elif auth_check and not auth_check["ok"]:
        status = "auth_failed"
        reason = auth_check["stderr"] or auth_check["stdout"]
    else:
        files_result = kaggle_cli.fetch_competition_files(slug, cwd=root, runner=command_runner)
        if files_result["ok"]:
            files = kaggle_cli.parse_competition_files(files_result["stdout"])
        else:
            status = "files_unavailable"
            reason = files_result["stderr"] or files_result["stdout"]

    result = {
        "competition_reference": competition_reference,
        "competition_slug": slug,
        "competition_url": f"https://www.kaggle.com/competitions/{slug}",
        "status": status,
        "reason": reason,
        "inspected_at": now_iso(),
        "files": files,
        "cli_check": cli_check,
        "auth_check": auth_check,
        "files_result": files_result,
        "next_steps": _next_steps(status, files),
    }
    out_dir = paths.competition_dir(slug)
    write_text(out_dir / "competition_inspection.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "competition_inspection.md", render_competition_inspection(result))
    return result


def render_competition_inspection(result: dict[str, Any]) -> str:
    lines = [
        f"# Competition Inspection: {result['competition_slug']}",
        "",
        f"- status: {result['status']}",
        f"- competition_url: {result['competition_url']}",
        f"- inspected_at: {result['inspected_at']}",
    ]
    if result.get("reason"):
        lines.extend([f"- reason: {result['reason']}"])
    lines.extend(["", "## Files", ""])
    for item in result["files"] or []:
        suffix = f" ({item['size']})" if item.get("size") else ""
        lines.append(f"- {item['name']}{suffix}")
    if not result["files"]:
        lines.append("- No file listing available.")
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in result["next_steps"])
    lines.append("")
    return "\n".join(lines)


def _next_steps(status: str, files: list[dict[str, Any]]) -> list[str]:
    if status == "cli_unavailable":
        return ["Install or expose the Kaggle CLI before downloading competition data."]
    if status == "auth_failed":
        return ["Check C:\\Users\\ASUS\\.kaggle\\kaggle.json and Kaggle API authentication."]
    if status == "files_unavailable":
        return ["Accept the competition rules on Kaggle if required, then retry inspection."]
    names = {item["name"] for item in files}
    steps = ["Download competition data with the Kaggle CLI.", "Inspect train/test/sample submission schemas."]
    if any("samplesubmission" in "".join(character for character in name.casefold() if character.isalnum()) for name in names):
        steps.append("Use the sample submission file to validate submission.csv format.")
    return steps
