from __future__ import annotations

from pathlib import Path
from typing import Any

from .state_db_sync import sync_competition_state_db


def sync_trial_state_after_finish(
    competition: str,
    trial_id: str,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Best-effort SQLite refresh after a trial reaches a terminal point.

    The file system remains the source of truth. SQLite is a query/cache layer
    for CLI, UI, and messenger status views, so sync failures should be visible
    without failing the research cycle itself.
    """

    try:
        summary = sync_competition_state_db(competition, db_path=db_path)
    except Exception as error:  # pragma: no cover - defensive by design.
        return {
            "status": "failed",
            "competition": competition,
            "trial_id": trial_id,
            "error": str(error),
        }
    return {
        "status": "synced",
        "competition": competition,
        "trial_id": trial_id,
        "trial_count": summary.get("trial_count", 0),
        "artifact_count": summary.get("artifact_count", 0),
        "token_usage_count": summary.get("token_usage_count", 0),
        "submission_count": summary.get("submission_count", 0),
    }
