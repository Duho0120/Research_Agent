from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from . import paths, simple_yaml
from .state_db import (
    default_db_path,
    initialize_state_db,
    record_token_usage,
    state_db_connection,
    upsert_competition,
    upsert_submission,
    upsert_trial,
    upsert_trial_artifact,
    upsert_trial_decision,
    upsert_trial_score,
)


USER_ARTIFACTS = {
    "01_plan.ko.md": "plan_ko",
    "02_pipeline_structure.ko.md": "pipeline_structure_ko",
    "03_scores.ko.md": "scores_ko",
    "04_result.ko.md": "scores_ko",
}

INTERNAL_ARTIFACTS = {
    "metrics.json": "metrics_json",
    "graph_state.json": "graph_state",
    "node_events.jsonl": "node_events",
    "internal/pipeline_structure.json": "pipeline_structure_json",
    "internal/executed_trial_facts.json": "executed_trial_facts_json",
    "internal/decision_card.json": "decision_card_json",
    "internal/trial_memory_card.json": "trial_memory_card_json",
    "internal/code_snapshot_manifest.json": "code_snapshot_manifest",
    "workspace_context_snapshot.md": "workspace_context_snapshot",
}


def sync_state_db(
    competition: str | None = None,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    target_db = initialize_state_db(db_path)
    competitions = [competition] if competition else _discover_competitions()
    result = {
        "status": "completed",
        "db_path": str(target_db),
        "competitions": [],
        "competition_count": 0,
        "trial_count": 0,
        "artifact_count": 0,
        "token_usage_count": 0,
        "submission_count": 0,
        "removed_stale_rows": 0,
    }
    for competition_id in competitions:
        summary = sync_competition_state_db(competition_id, db_path=target_db)
        result["competitions"].append(summary)
        result["trial_count"] += summary["trial_count"]
        result["artifact_count"] += summary["artifact_count"]
        result["token_usage_count"] += summary["token_usage_count"]
        result["submission_count"] += summary["submission_count"]
        result["removed_stale_rows"] += summary["removed_stale_rows"]
    result["competition_count"] = len(result["competitions"])
    return result


def sync_competition_state_db(competition: str, *, db_path: Path | None = None) -> dict[str, Any]:
    target_db = db_path or default_db_path()
    competition_record = _competition_record(competition)
    upsert_competition(competition_record, target_db)
    removed_stale_rows = _clear_competition_synced_rows(competition, target_db)

    summary = {
        "competition": competition,
        "trial_count": 0,
        "artifact_count": 0,
        "token_usage_count": 0,
        "submission_count": 0,
        "removed_stale_rows": removed_stale_rows,
    }

    trial_sources = _trial_sources(competition)
    valid_trial_ids = set(trial_sources)
    for trial_id, source in trial_sources.items():
        trial_summary = _sync_trial(
            competition,
            trial_id,
            source["trial_path"],
            target_db,
            competition_record,
            manual_path=source.get("manual_path"),
        )
        summary["trial_count"] += 1
        summary["artifact_count"] += trial_summary["artifact_count"]
        summary["submission_count"] += trial_summary["submission_count"]

    summary["token_usage_count"] = _sync_token_usage(competition, target_db)
    _sync_submission_log(competition, target_db, valid_trial_ids)
    _refresh_best_score_flags(competition, target_db, competition_record)
    summary["submission_count"] = _count_competition_rows(target_db, "submissions", competition)
    return summary


def _clear_competition_synced_rows(competition: str, db_path: Path) -> int:
    """Remove cache rows that are rebuilt from the file system during sync.

    The file system remains the source of truth. Pending actions are preserved
    because they represent live user/approval work, not derived trial metadata.
    """

    total = 0
    tables = [
        "token_usage",
        "submissions",
        "trial_artifacts",
        "trial_decisions",
        "trial_scores",
        "trials",
    ]
    with state_db_connection(db_path) as connection:
        for table in tables:
            cursor = connection.execute(f"DELETE FROM {table} WHERE competition_id = ?", [competition])
            total += max(cursor.rowcount or 0, 0)
    return total


def _sync_trial(
    competition: str,
    trial_id: str,
    trial_path: Path,
    db_path: Path,
    competition_record: dict[str, Any],
    *,
    manual_path: Path | None = None,
) -> dict[str, int]:
    plan = _read_trial_json(trial_path, "demo_experiment_plan.json")
    if not plan:
        plan = _read_trial_json(trial_path, "next_experiment.json")
    executed_facts = _read_trial_json(trial_path, "executed_trial_facts.json")
    decision = _read_trial_json(trial_path, "decision_card.json")
    metrics = _read_json(trial_path / "metrics.json")
    manual_metrics = _read_json((manual_path or trial_path) / "metrics.json") if manual_path else {}
    score_metrics = {**metrics, **manual_metrics}
    result_cycle = _read_trial_json(trial_path, "workspace_result_cycle.json")
    demo_cycle = _read_trial_json(trial_path, "demo_one_cycle.json")
    graph_cycle = _read_trial_json(trial_path, "demo_graph_cycle.json")
    workspace_run = _read_json(trial_path / "workspace_run.json")
    submission = _read_json(trial_path / "submission_run.json")
    submit_manifest = _read_json(trial_path / "submit_manifest.json")

    status = _trial_status(
        plan=plan,
        result_cycle=result_cycle,
        demo_cycle=demo_cycle,
        graph_cycle=graph_cycle,
        workspace_run=workspace_run,
        metrics=score_metrics or metrics,
    )
    if manual_metrics and status in {"blocked", "discovered"}:
        status = "completed"
    source_trial_id = executed_facts.get("source_trial_id") or plan.get("source_trial_id") or decision.get("source_trial_id")
    primary_axis = _trial_primary_axis(
        trial_path,
        plan=plan,
        decision=decision,
        manual_metrics=manual_metrics,
        executed_facts=executed_facts,
    )
    recommended_base = decision.get("recommended_base_trial") or plan.get("recommended_base_trial")
    upsert_trial(
        {
            "competition_id": competition,
            "trial_id": trial_id,
            "status": status,
            "source_trial_id": source_trial_id,
            "recommended_base_trial": recommended_base,
            "plan_type": executed_facts.get("plan_type") or plan.get("plan_type") or decision.get("plan_type"),
            "plan_summary": _trial_plan_summary(plan),
            "primary_change_axis": primary_axis,
        },
        db_path,
    )

    if score_metrics or decision:
        upsert_trial_score(
            {
                "competition_id": competition,
                "trial_id": trial_id,
                "metric": score_metrics.get("metric") or competition_record.get("metric"),
                "objective": score_metrics.get("objective") or competition_record.get("objective"),
                "local_score": score_metrics.get("cv_score")
                or score_metrics.get("validation_accuracy")
                or score_metrics.get("local_score"),
                "lb_score": decision.get("lb_score")
                or _submission_score(submission)
                or score_metrics.get("kaggle_lb_score")
                or score_metrics.get("lb_score"),
                "local_status": decision.get("local_status"),
                "lb_status": decision.get("lb_status"),
                "is_best_local": trial_id == recommended_base
                and (decision.get("decision") in {"accept", "provisional_accept", "baseline_established"}),
                "is_best_lb": bool(submission.get("is_best") or submit_manifest.get("is_best_submission")),
            },
            db_path,
        )

    if decision:
        upsert_trial_decision(
            {
                "competition_id": competition,
                "trial_id": trial_id,
                **decision,
            },
            db_path,
        )

    artifact_count = _sync_trial_artifacts(competition, trial_id, trial_path, db_path)
    if manual_path and manual_path != trial_path:
        artifact_count += _sync_trial_artifacts(competition, trial_id, manual_path, db_path)
    submission_count = _sync_trial_submission(competition, trial_id, submit_manifest, submission, score_metrics, db_path)
    return {"artifact_count": artifact_count, "submission_count": submission_count}


def _sync_trial_artifacts(competition: str, trial_id: str, trial_path: Path, db_path: Path) -> int:
    count = 0
    for relative, artifact_type in INTERNAL_ARTIFACTS.items():
        path = trial_path / relative
        if path.exists():
            upsert_trial_artifact(
                {
                    "competition_id": competition,
                    "trial_id": trial_id,
                    "artifact_type": artifact_type,
                    "path": _project_relative(path),
                    "is_user_facing": False,
                },
                db_path,
            )
            count += 1

    code_snapshot = trial_path / "internal" / "code_snapshot"
    if code_snapshot.is_dir():
        upsert_trial_artifact(
            {
                "competition_id": competition,
                "trial_id": trial_id,
                "artifact_type": "code_snapshot",
                "path": _project_relative(code_snapshot),
                "is_user_facing": False,
            },
            db_path,
        )
        count += 1

    run_dir = _user_view_dir(competition, trial_id, trial_path)
    for name, artifact_type in USER_ARTIFACTS.items():
        path = run_dir / name
        if path.exists():
            upsert_trial_artifact(
                {
                    "competition_id": competition,
                    "trial_id": trial_id,
                    "artifact_type": artifact_type,
                    "path": _project_relative(path),
                    "is_user_facing": True,
                },
                db_path,
            )
            count += 1
    return count


def _user_view_dir(competition: str, trial_id: str, trial_path: Path) -> Path:
    run_dir = paths.project_root() / "runs" / competition / trial_id
    if run_dir.is_dir():
        return run_dir
    return trial_path / "user_view"


def _sync_trial_submission(
    competition: str,
    trial_id: str,
    submit_manifest: dict[str, Any],
    submission: dict[str, Any],
    metrics: dict[str, Any],
    db_path: Path,
) -> int:
    record = submit_manifest or submission or _manual_submission_record(metrics)
    if not record:
        return 0
    submission_file = record.get("submission_file") or submission.get("submission_file") or ""
    upsert_submission(
        {
            "competition_id": competition,
            "trial_id": trial_id,
            "platform": record.get("platform") or ("kaggle" if metrics.get("kaggle_submitted") else "unknown"),
            "submission_file": submission_file,
            "status": record.get("status") or submission.get("status"),
            "lb_score": _submission_score(submission) or record.get("submitted_lb_score") or metrics.get("kaggle_lb_score"),
            "rank": submission.get("submitted_rank") or record.get("submitted_rank") or metrics.get("kaggle_rank"),
            "submitted_at": submission.get("submitted_at") or record.get("submitted_at"),
            "requires_user_approval": record.get("requires_user_approval", not bool(metrics.get("kaggle_submitted"))),
        },
        db_path,
    )
    return 1


def _manual_submission_record(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    submission_file = metrics.get("submission_file")
    lb_score = metrics.get("kaggle_lb_score") or metrics.get("lb_score")
    if not submission_file and lb_score is None:
        return {}
    return {
        "platform": "kaggle",
        "submission_file": submission_file,
        "status": metrics.get("kaggle_status") or ("submitted" if metrics.get("kaggle_submitted") else "recorded"),
        "submitted_lb_score": lb_score,
        "submitted_rank": metrics.get("kaggle_rank"),
        "requires_user_approval": False,
    }


def _trial_primary_axis(
    trial_path: Path,
    *,
    plan: dict[str, Any],
    decision: dict[str, Any],
    manual_metrics: dict[str, Any],
    executed_facts: dict[str, Any],
) -> str | None:
    pipeline_plan = _read_trial_json(trial_path, "pipeline_improvement_plan.json")
    candidates = [
        executed_facts.get("primary_change_axis"),
        plan.get("primary_change_axis"),
        plan.get("primary_axis"),
        decision.get("active_axis"),
        decision.get("change_axis"),
        manual_metrics.get("change_axis"),
        pipeline_plan.get("primary_change_axis"),
        pipeline_plan.get("primary_axis"),
        _next_experiment_strategy(trial_path),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _trial_plan_summary(plan: dict[str, Any]) -> str | None:
    for key in ["plan_title", "strategy", "objective", "rationale"]:
        value = str(plan.get(key) or "").strip()
        if value:
            return value
    return None


def _next_experiment_strategy(trial_path: Path) -> str | None:
    path = trial_path / "next_experiment.md"
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    match = re.search(r"(?ims)^##\s+Strategy\s*$\s*([A-Za-z0-9_-]+)\s*$", text)
    return match.group(1).strip() if match else None


def _sync_token_usage(competition: str, db_path: Path) -> int:
    path = paths.competition_memory_dir(competition) / "token_usage.jsonl"
    count = 0
    for line_number, row in _read_jsonl(path):
        record_token_usage(
            {
                "source_key": f"{_project_relative(path)}:{line_number}",
                "competition_id": row.get("competition") or competition,
                "trial_id": row.get("trial_id"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "call_type": row.get("call_type"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "total_tokens": row.get("total_tokens"),
                "created_at": row.get("time") or row.get("created_at"),
            },
            db_path,
        )
        count += 1
    return count


def _sync_submission_log(competition: str, db_path: Path, valid_trial_ids: set[str]) -> int:
    path = paths.competition_submissions_dir(competition) / "submission_log.jsonl"
    count = 0
    for _, row in _read_jsonl(path):
        trial_id = row.get("trial_id")
        if not trial_id:
            continue
        if valid_trial_ids and trial_id not in valid_trial_ids:
            continue
        upsert_submission(
            {
                "competition_id": row.get("competition") or competition,
                "trial_id": trial_id,
                "platform": row.get("platform") or "unknown",
                "submission_file": row.get("submission_file") or "",
                "status": row.get("status") or "recorded",
                "lb_score": row.get("submitted_lb_score") or row.get("lb_score"),
                "rank": row.get("submitted_rank") or row.get("rank"),
                "submitted_at": row.get("submitted_at") or row.get("time"),
                "requires_user_approval": row.get("requires_user_approval", False),
            },
            db_path,
        )
        count += 1
    return count


def _refresh_best_score_flags(competition: str, db_path: Path, competition_record: dict[str, Any]) -> None:
    objective = str(competition_record.get("objective") or "maximize").lower()
    descending = objective != "minimize"
    with state_db_connection(db_path) as connection:
        connection.execute(
            "UPDATE trial_scores SET is_best_local = 0, is_best_lb = 0 WHERE competition_id = ?",
            [competition],
        )
        for score_column, flag_column in [("local_score", "is_best_local"), ("lb_score", "is_best_lb")]:
            rows = connection.execute(
                f"""
                SELECT trial_id, {score_column} AS score
                FROM trial_scores
                WHERE competition_id = ? AND {score_column} IS NOT NULL
                """,
                [competition],
            ).fetchall()
            if not rows:
                continue
            best = sorted(
                [dict(row) for row in rows],
                key=lambda row: (float(row["score"]), str(row["trial_id"])),
                reverse=descending,
            )[0]
            connection.execute(
                f"UPDATE trial_scores SET {flag_column} = 1 WHERE competition_id = ? AND trial_id = ?",
                [competition, best["trial_id"]],
            )


def _trial_status(
    *,
    plan: dict[str, Any],
    result_cycle: dict[str, Any],
    demo_cycle: dict[str, Any],
    graph_cycle: dict[str, Any],
    workspace_run: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    for source in [graph_cycle, demo_cycle, result_cycle, workspace_run]:
        status = str(source.get("status") or "").strip() if isinstance(source, dict) else ""
        if status:
            return status
    if metrics:
        return "completed"
    plan_status = str(plan.get("status") or "").strip() if isinstance(plan, dict) else ""
    if plan_status in {"ready", "planned"}:
        return "planned"
    if plan_status:
        return plan_status
    return "discovered"


def _competition_record(competition: str) -> dict[str, Any]:
    profile = simple_yaml.load(paths.competition_dir(competition) / "execution_profile.yaml", default={})
    state = simple_yaml.load(paths.competition_dir(competition) / "state.yaml", default={})
    workspace_config = _workspace_config(profile)
    competition_state = state.get("competition") if isinstance(state, dict) else {}
    return {
        "competition_id": competition,
        "platform": profile.get("platform") or workspace_config.get("platform") or competition_state.get("platform"),
        "topic": (
            workspace_config.get("topic")
            or competition_state.get("topic")
            or competition_state.get("name")
            or competition
        ),
        "metric": workspace_config.get("metric") or competition_state.get("metric"),
        "objective": workspace_config.get("objective") or competition_state.get("objective"),
        "status": _competition_status(competition),
        "workspace_path": profile.get("project_root"),
    }


def _workspace_config(profile: dict[str, Any]) -> dict[str, Any]:
    project_root = profile.get("project_root") if isinstance(profile, dict) else None
    if not project_root:
        return {}
    return _read_json(Path(str(project_root)) / "workspace_config.json")


def _competition_status(competition: str) -> str:
    trials = _trial_sources(competition)
    if trials:
        return "has_trials"
    if paths.competition_dir(competition).exists():
        return "initialized"
    return "discovered"


def _discover_competitions() -> list[str]:
    names = set()
    for root in [paths.competition_dir("..").parent, paths.experiments_dir(), paths.memory_dir()]:
        if root.is_dir():
            names.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(name for name in names if name and name != "..")


def _trial_dirs(competition: str) -> list[Path]:
    root = paths.experiment_dir(competition)
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("trial_")],
        key=lambda item: item.name,
    )


def _trial_sources(competition: str) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}
    for trial_path in _trial_dirs(competition):
        sources[trial_path.name] = {"trial_path": trial_path}
    manual_root = _manual_trials_dir(competition)
    if manual_root.is_dir():
        for manual_path in sorted(manual_root.iterdir(), key=lambda item: item.name):
            if not manual_path.is_dir() or not manual_path.name.startswith("trial_"):
                continue
            source = sources.setdefault(manual_path.name, {"trial_path": manual_path})
            source["manual_path"] = manual_path
    return dict(sorted(sources.items()))


def _manual_trials_dir(competition: str) -> Path:
    profile = simple_yaml.load(paths.competition_dir(competition) / "execution_profile.yaml", default={})
    workspace_root = profile.get("project_root") if isinstance(profile, dict) else None
    if workspace_root:
        return Path(str(workspace_root)) / "manual_trials"
    return paths.project_root() / "demo_workspaces" / competition / "manual_trials"


def _read_trial_json(trial_path: Path, name: str) -> dict[str, Any]:
    return _read_json(trial_path / name) or _read_json(trial_path / "internal" / name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append((index, value))
    return rows


def _submission_score(submission: dict[str, Any]) -> float | None:
    candidates = [
        submission.get("submitted_lb_score"),
        submission.get("lb_score"),
    ]
    recorded = submission.get("recorded_submission")
    if isinstance(recorded, dict):
        candidates.append(recorded.get("submitted_lb_score"))
    for value in candidates:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.project_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _count_competition_rows(db_path: Path, table: str, competition: str) -> int:
    if table not in {"submissions", "token_usage", "trial_artifacts", "trials"}:
        raise ValueError(f"Unsupported table for sync count: {table}")
    with closing(sqlite3.connect(db_path)) as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE competition_id = ?",
                [competition],
            ).fetchone()[0]
        )
