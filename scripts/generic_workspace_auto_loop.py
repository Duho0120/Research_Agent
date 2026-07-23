from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_research_agent.agents.submission import submit_trial  # noqa: E402
from kaggle_research_agent.demo_one_cycle import prepare_workspace_trial_plan  # noqa: E402
from kaggle_research_agent.execution_profile import load_execution_profile, validate_execution_profile  # noqa: E402
from kaggle_research_agent.paths import trial_dir  # noqa: E402
from kaggle_research_agent.state_db_sync import sync_state_db  # noqa: E402
from kaggle_research_agent.store import load_state, write_text  # noqa: E402
from kaggle_research_agent.trial_artifacts import organize_trial_artifacts  # noqa: E402
from kaggle_research_agent.workspace_after_coding import run_workspace_after_coding  # noqa: E402
from kaggle_research_agent.workspace_code_writer import run_workspace_code_writer  # noqa: E402
from kaggle_research_agent.workspace_coding_handoff import prepare_workspace_coding_handoff  # noqa: E402
from kaggle_research_agent.workspace_metrics_collector import collect_workspace_metrics  # noqa: E402
from kaggle_research_agent.workspace_next_gate import plan_next_workspace_trial  # noqa: E402
from kaggle_research_agent.workspace_result_cycle import process_workspace_result  # noqa: E402
from kaggle_research_agent.workspace_runner import run_workspace_pipeline  # noqa: E402
from kaggle_research_agent.user_insight_policy import (  # noqa: E402
    build_next_trial_user_insight_override,
    latest_user_insight_record,
)


DEFAULT_RUNTIME_DIR = ROOT / "demo_workspaces" / "_runtime"
RUNTIME_DIR = Path(os.environ.get("RESEARCH_AGENT_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))).expanduser().resolve()
STATE_PATH = RUNTIME_DIR / "auto_loop_state.json"
LOCK_PATH = RUNTIME_DIR / "auto_loop.lock"
PAUSE_REQUEST_PATH = RUNTIME_DIR / "pause.request"


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_loop_state(**updates: Any) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    state = load_json(STATE_PATH)
    state.update(updates)
    state["updated_at"] = now_text()
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)
    return state


def acquire_loop_lock() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = load_lock_owner()
            if owner and process_is_alive(owner):
                raise SystemExit(f"Another auto loop is already running (PID {owner}).")
            try:
                LOCK_PATH.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps({"pid": os.getpid(), "started_at": now_text()}) + "\n")
        atexit.register(release_loop_lock)
        return
    raise SystemExit("Could not acquire the auto-loop lock.")


def load_lock_owner() -> int | None:
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("pid")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def release_loop_lock() -> None:
    if load_lock_owner() != os.getpid():
        return
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def pause_is_requested() -> bool:
    return PAUSE_REQUEST_PATH.exists() or bool(load_json(STATE_PATH).get("pause_requested"))


def clear_pause_request() -> None:
    try:
        PAUSE_REQUEST_PATH.unlink()
    except FileNotFoundError:
        pass


def next_trial_id(trial_id: str) -> str:
    prefix, _, number = trial_id.rpartition("_")
    try:
        return f"{prefix}_{int(number) + 1:03d}"
    except ValueError:
        return "trial_002"


def previous_trial_id(trial_id: str) -> str | None:
    prefix, _, number = trial_id.rpartition("_")
    try:
        value = int(number)
    except ValueError:
        return None
    if value <= 1:
        return None
    return f"{prefix}_{value - 1:03d}"


def trial_sequence(start_trial: str, max_trials: int) -> list[str]:
    trials = [start_trial]
    while len(trials) < max_trials:
        trials.append(next_trial_id(trials[-1]))
    return trials


def objective_for(competition: str, profile: dict[str, Any]) -> str:
    state = load_state(competition)
    value = (
        profile.get("objective")
        or (state.get("competition") or {}).get("objective")
        or (state.get("current_state") or {}).get("objective")
        or "maximize"
    )
    return str(value) if str(value) in {"maximize", "minimize"} else "maximize"


def submission_artifact_path(profile: dict[str, Any]) -> str | None:
    paths = profile.get("artifacts", {}).get("submission") or []
    if not paths:
        return None
    path = Path(profile["project_root"]) / str(paths[0])
    return str(path.resolve())


def local_score_for(trial_id: str, metrics_collection: dict[str, Any]) -> float | None:
    value = metrics_collection.get("cv_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    metrics = load_json(trial_dir(str(metrics_collection.get("competition")), trial_id) / "metrics.json")
    value = metrics.get("cv_score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def run_one_trial(
    competition: str,
    trial_id: str,
    *,
    submit: bool,
    kaggle_slug: str | None,
    poll_attempts: int,
    poll_interval_seconds: float,
    code_writer: bool = False,
    model: str = "gpt-5",
    provider: str = "openai",
    allow_api: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
) -> dict[str, Any]:
    save_loop_state(
        status="running",
        phase="validating_profile",
        competition=competition,
        current_trial=trial_id,
        next_trial=trial_id,
    )
    validation = validate_execution_profile(competition)
    if validation["status"] != "ready":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "blocked_execution_profile",
            "validation": validation,
        }
    profile = load_execution_profile(competition)
    code_stage_result: dict[str, Any] | None = None
    if code_writer and has_coding_plan(competition, trial_id):
        save_loop_state(phase="coding")
        run_result = run_code_writer_trial(
            competition,
            trial_id,
            model=model,
            provider=provider,
            allow_api=allow_api,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
        )
        if run_result.get("status") != "completed":
            return run_result
        code_stage_result = run_result
        workspace_run = run_result.get("workspace_run") or {}
        metrics_collection = run_result.get("metrics_collection") or {}
        result_cycle = run_result.get("workspace_result_cycle") or {}
    else:
        save_loop_state(phase="executing")
        workspace_run = run_workspace_pipeline(competition, trial_id, run_now=True)
        if workspace_run.get("status") != "completed":
            return {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"workspace_run_{workspace_run.get('status')}",
                "workspace_run": workspace_run,
            }
        metrics_collection = collect_workspace_metrics(competition, trial_id)
        if metrics_collection.get("status") != "collected":
            return {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"metrics_{metrics_collection.get('status')}",
                "workspace_run": workspace_run,
                "metrics_collection": metrics_collection,
            }
        result_cycle = process_workspace_result(competition, trial_id)

    if not workspace_run or workspace_run.get("status") != "completed":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": f"workspace_run_{workspace_run.get('status')}",
            "workspace_run": workspace_run,
        }
    if not metrics_collection or metrics_collection.get("status") != "collected":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": f"metrics_{metrics_collection.get('status')}",
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
        }
    submission_run = None
    if submit:
        save_loop_state(phase="submitting")
        submission_file = submission_artifact_path(profile)
        if not submission_file:
            status = "blocked_missing_submission_artifact"
            submission_run = {"status": status, "reason": "execution_profile artifacts.submission is missing"}
        else:
            message = f"{competition} {trial_id} local CV {local_score_for(trial_id, metrics_collection) or 0:.6f}"
            submission_run = submit_trial(
                competition=competition,
                trial_id=trial_id,
                version_name=f"{trial_id}_auto",
                submission_file=submission_file,
                objective=objective_for(competition, profile),
                kaggle_competition_slug=kaggle_slug or competition,
                kaggle_message=message,
                poll_leaderboard=False,
                poll_attempts=poll_attempts,
                poll_interval_seconds=poll_interval_seconds,
                notes="Submitted by generic_workspace_auto_loop.py",
            )
        submission_block = _submission_blocking_issue(submission_run, requires_score=bool(kaggle_slug))
        if submission_block:
            result = {
                "competition": competition,
                "trial_id": trial_id,
                "status": submission_block,
                "workspace_run": workspace_run,
                "metrics_collection": metrics_collection,
                "workspace_result_cycle": result_cycle,
                "submission_run": submission_run,
            }
            if code_stage_result:
                result["handoff"] = code_stage_result.get("handoff")
                result["code_writer"] = code_stage_result.get("code_writer")
                result["after_coding"] = code_stage_result.get("after_coding")
            write_loop_trial_result(result)
            return result
    save_loop_state(phase="organizing_artifacts")
    artifact_summary = organize_trial_artifacts(competition, trial_id, low_cost_user_summary=False)
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "completed",
        "workspace_run": workspace_run,
        "metrics_collection": metrics_collection,
        "workspace_result_cycle": result_cycle,
        "submission_run": submission_run,
        "artifact_summary": artifact_summary,
    }
    if code_stage_result:
        result["handoff"] = code_stage_result.get("handoff")
        result["code_writer"] = code_stage_result.get("code_writer")
        result["after_coding"] = code_stage_result.get("after_coding")
    write_loop_trial_result(result)
    return result


def _submission_blocking_issue(submission_run: dict[str, Any] | None, *, requires_score: bool) -> str | None:
    if not submission_run:
        return "submission_missing_result"
    status = str(submission_run.get("status") or "")
    if status not in {"submitted", "recorded"}:
        return f"submission_{status or 'unknown'}"
    if requires_score and submission_run.get("submitted_lb_score") is None:
        return "submission_missing_leaderboard_score"
    return None


def has_coding_plan(competition: str, trial_id: str) -> bool:
    out_dir = trial_dir(competition, trial_id)
    plan_exists = (out_dir / "next_experiment.md").exists() or (out_dir / "demo_experiment_plan.json").exists()
    return plan_exists and (out_dir / "continuation_context.json").exists()


PLANNED_STATUSES = {
    "planned",
    "planned_with_deferred_review",
    "planned_with_pending_review",
}


def prepare_trial_for_code(args: argparse.Namespace, trial_id: str) -> dict[str, Any]:
    if has_coding_plan(args.competition, trial_id):
        revised = revise_planned_trial_for_pending_insight(args, trial_id)
        return revised or {"status": "planned", "trial_id": trial_id, "existing": True}

    source_trial = previous_trial_id(trial_id)
    if source_trial is None:
        save_loop_state(phase="planning", current_trial=trial_id, next_trial=trial_id)
        return prepare_workspace_trial_plan(
            args.competition,
            trial_id,
            source_trial_id=None,
            model=args.model,
            provider=args.provider,
            allow_api=args.allow_api,
            trial_llm_calls=args.trial_llm_calls,
            strategy_calls_today=args.strategy_calls_today,
        )
    return plan_following_trial(args, source_trial, trial_id)


def plan_following_trial(
    args: argparse.Namespace,
    source_trial: str,
    target_trial: str,
) -> dict[str, Any]:
    save_loop_state(
        phase="planning_next",
        current_trial=None,
        next_trial=target_trial,
    )
    result = plan_next_workspace_trial(
        args.competition,
        source_trial,
        target_trial,
        allow_api=args.allow_api,
    )
    if result.get("status") not in PLANNED_STATUSES:
        return result
    next_experiment = result.get("next_experiment") if isinstance(result.get("next_experiment"), dict) else {}
    override = (
        next_experiment.get("user_insight_override")
        if isinstance(next_experiment.get("user_insight_override"), dict)
        else None
    )
    if override and override.get("status") == "active":
        save_loop_state(phase="revising_plan", current_trial=None, next_trial=target_trial)
        return prepare_workspace_trial_plan(
            args.competition,
            target_trial,
            source_trial_id=source_trial,
            model=args.model,
            provider=args.provider,
            allow_api=args.allow_api,
            trial_llm_calls=args.trial_llm_calls,
            strategy_calls_today=args.strategy_calls_today,
            user_insight_override=override,
            force_replan=True,
        )
    return result


def revise_planned_trial_for_pending_insight(
    args: argparse.Namespace,
    trial_id: str,
) -> dict[str, Any] | None:
    record = latest_user_insight_record(args.competition)
    if not record:
        return None
    source_trial = previous_trial_id(trial_id)
    if source_trial is None:
        return None
    target_trial = str(record.get("target_trial") or next_trial_id(str(record.get("source_trial_id") or "")))
    if target_trial != trial_id:
        return None
    existing_plan = load_json(trial_dir(args.competition, trial_id) / "demo_experiment_plan.json")
    applied_override = (
        existing_plan.get("user_insight_override")
        if isinstance(existing_plan.get("user_insight_override"), dict)
        else {}
    )
    if applied_override.get("insight_id") == record.get("insight_id"):
        return None
    override = build_next_trial_user_insight_override(
        args.competition,
        source_trial,
        trial_id,
        allow_api=args.allow_api,
    )
    if not override or override.get("status") != "active":
        return None
    save_loop_state(phase="revising_plan", current_trial=None, next_trial=trial_id)
    return prepare_workspace_trial_plan(
        args.competition,
        trial_id,
        source_trial_id=source_trial,
        model=args.model,
        provider=args.provider,
        allow_api=args.allow_api,
        trial_llm_calls=args.trial_llm_calls,
        strategy_calls_today=args.strategy_calls_today,
        user_insight_override=override,
        force_replan=True,
    )


def run_code_writer_trial(
    competition: str,
    trial_id: str,
    *,
    model: str,
    provider: str,
    allow_api: bool,
    trial_llm_calls: int | None,
    strategy_calls_today: int | None,
) -> dict[str, Any]:
    last_blocked: dict[str, Any] | None = None
    for attempt in range(1, 3):
        expanded_retry = attempt > 1
        if expanded_retry:
            print("Code writer blocked by missing code snapshot context; regenerating expanded handoff and retrying...", flush=True)
        handoff = prepare_workspace_coding_handoff(
            competition,
            trial_id,
            expanded_snapshot=expanded_retry,
            retry_reason="code_writer_blocked_snapshot_context" if expanded_retry else None,
        )
        if handoff.get("status") != "ready":
            return {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"handoff_{handoff.get('status')}",
                "handoff": handoff,
                "code_writer_attempt": attempt,
            }
        code_writer_result = run_workspace_code_writer(
            competition,
            trial_id,
            model=model,
            provider=provider,
            allow_api=allow_api,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
        )
        if code_writer_result.get("status") != "accepted":
            last_blocked = {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"code_writer_{code_writer_result.get('status')}",
                "handoff": handoff,
                "code_writer": code_writer_result,
                "code_writer_attempt": attempt,
            }
            if attempt == 1 and _should_retry_code_writer_block(last_blocked):
                continue
            return last_blocked
        after_coding = run_workspace_after_coding(competition, trial_id, run_now=True)
        if after_coding.get("status") != "completed":
            return {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"after_coding_{after_coding.get('status')}",
                "handoff": handoff,
                "code_writer": code_writer_result,
                "after_coding": after_coding,
                "code_writer_attempt": attempt,
            }
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "completed",
            "handoff": handoff,
            "code_writer": code_writer_result,
            "after_coding": after_coding,
            "workspace_run": after_coding.get("workspace_run"),
            "metrics_collection": after_coding.get("metrics_collection"),
            "workspace_result_cycle": after_coding.get("workspace_result_cycle"),
            "code_writer_attempt": attempt,
        }
    return last_blocked or {"competition": competition, "trial_id": trial_id, "status": "code_writer_blocked"}


def _should_retry_code_writer_block(result: dict[str, Any]) -> bool:
    if result.get("status") != "code_writer_blocked":
        return False
    code_writer = result.get("code_writer") if isinstance(result.get("code_writer"), dict) else {}
    issues = [
        str(item)
        for item in [
            *list(code_writer.get("blocking_issues") or []),
            *list(code_writer.get("issues") or []),
        ]
    ]
    issue_text = "\n".join(issues).lower()
    retriable_markers = [
        "no_code_snapshot_provided",
        "exact file contents are not provided",
        "patch_only_mode_requires_exact_find_text",
        "exact target code is not visible",
        "missing code snapshot",
        "missing full code context",
        "truncated",
        "safe localized edits",
        "existing anchors",
        "exact find/replace anchors",
    ]
    non_retriable_markers = [
        "token_policy_blocked",
        "api_call_not_enabled",
        "api_error:",
        "quota",
        "insufficient_quota",
    ]
    return any(marker in issue_text for marker in retriable_markers) and not any(
        marker in issue_text for marker in non_retriable_markers
    )


def write_loop_trial_result(result: dict[str, Any]) -> None:
    out_dir = trial_dir(result["competition"], result["trial_id"])
    write_text(out_dir / "auto_loop_trial.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    clear_pause_request()
    acquire_loop_lock()
    selected = trial_sequence(args.start_trial, args.max_trials)
    save_loop_state(
        competition=args.competition,
        status="running",
        pid=os.getpid(),
        start_trial=selected[0],
        end_trial=selected[-1],
        next_trial=selected[0],
        phase="planning",
        pause_requested=False,
    )
    rows = []
    for trial_id in selected:
        successor_trial = next_trial_id(trial_id)
        print(f"\n=== {args.competition} / {trial_id} ===", flush=True)
        if args.code_writer:
            plan_result = prepare_trial_for_code(args, trial_id)
            if plan_result.get("status") not in PLANNED_STATUSES:
                rows.append({"trial_id": trial_id, "status": plan_result.get("status")})
                save_loop_state(
                    status="failed",
                    phase="planning",
                    current_trial=None,
                    next_trial=trial_id,
                    error=plan_result.get("status"),
                    pid=None,
                )
                sync_state_db(args.competition)
                return {"competition": args.competition, "status": "failed", "trials": rows}
        result = run_one_trial(
            args.competition,
            trial_id,
            submit=args.submit,
            kaggle_slug=args.kaggle_slug,
            poll_attempts=args.poll_attempts,
            poll_interval_seconds=args.poll_interval_seconds,
            code_writer=args.code_writer,
            model=args.model,
            provider=args.provider,
            allow_api=args.allow_api,
            trial_llm_calls=args.trial_llm_calls,
            strategy_calls_today=args.strategy_calls_today,
        )
        rows.append({"trial_id": trial_id, "status": result.get("status")})
        if result.get("status") != "completed":
            save_loop_state(
                status="failed",
                phase="blocked",
                current_trial=None,
                next_trial=trial_id,
                error=result.get("status"),
                pid=None,
            )
            sync_state_db(args.competition)
            return {"competition": args.competition, "status": "failed", "trials": rows}
        next_plan = plan_following_trial(args, trial_id, successor_trial)
        if next_plan.get("status") not in PLANNED_STATUSES:
            sync_state_db(args.competition)
            save_loop_state(
                status="failed",
                phase="planning_next",
                current_trial=None,
                last_completed_trial=trial_id,
                next_trial=successor_trial,
                error=next_plan.get("status"),
                pid=None,
            )
            return {
                "competition": args.competition,
                "status": "failed",
                "trials": rows,
                "next_plan_status": next_plan.get("status"),
            }
        sync_state_db(args.competition)
        save_loop_state(
            status="running",
            phase="planned",
            current_trial=None,
            last_completed_trial=trial_id,
            next_trial=successor_trial,
            error=None,
        )
        if pause_is_requested():
            save_loop_state(
                status="paused",
                phase="planned",
                current_trial=None,
                next_trial=successor_trial,
                pid=None,
            )
            return {"competition": args.competition, "status": "paused", "trials": rows}
    final_next_trial = next_trial_id(selected[-1])
    save_loop_state(
        status="completed",
        phase="planned",
        current_trial=None,
        next_trial=final_next_trial,
        pause_requested=False,
        error=None,
        pid=None,
    )
    return {"competition": args.competition, "status": "completed", "trials": rows}


def request_pause() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PAUSE_REQUEST_PATH.write_text(now_text() + "\n", encoding="utf-8")
    save_loop_state(pause_requested=True)
    print("Pause requested. The loop will stop after the current trial completes and the next trial plan is prepared.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a workspace experiment loop from execution_profile.yaml.")
    parser.add_argument("--competition", required=True)
    parser.add_argument("--start-trial", default="trial_001")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--kaggle-slug", default=None)
    parser.add_argument("--code-writer", action="store_true")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--trial-llm-calls", type=int, default=None)
    parser.add_argument("--strategy-calls-today", type=int, default=None)
    parser.add_argument("--poll-attempts", type=int, default=5)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--request-pause", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(json.dumps(load_json(STATE_PATH), ensure_ascii=False, indent=2))
        return 0
    if args.request_pause:
        request_pause()
        return 0
    result = run_loop(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "paused"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        save_loop_state(status="failed", current_trial=None, error=str(error), pid=None)
        print(f"Workspace auto loop failed: {error}", file=sys.stderr)
        raise
