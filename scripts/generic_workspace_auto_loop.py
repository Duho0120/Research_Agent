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

from kaggle_research_agent.agents.memory import log_decision  # noqa: E402
from kaggle_research_agent.agents.submission import submit_trial  # noqa: E402
from kaggle_research_agent.cli_app import (  # noqa: E402
    check_dacon_submission_limit,
    dacon_auto_submit_allowed,
)
from kaggle_research_agent.demo_one_cycle import prepare_workspace_trial_plan  # noqa: E402
from kaggle_research_agent.execution_profile import load_execution_profile, validate_execution_profile  # noqa: E402
from kaggle_research_agent.graph.workspace_loop_graph import (  # noqa: E402
    WorkspaceLoopCallbacks,
    resume_workspace_loop_graph,
    run_workspace_loop_graph,
    workspace_loop_checkpointer,
    workspace_loop_thread_has_pending_work,
)
from kaggle_research_agent.paths import trial_dir  # noqa: E402
from kaggle_research_agent.state_db import list_pending_actions  # noqa: E402
from kaggle_research_agent.state_db_sync import sync_state_db  # noqa: E402
from kaggle_research_agent.store import load_state, write_text  # noqa: E402
from kaggle_research_agent.trial_artifacts import (  # noqa: E402
    organize_trial_artifacts,
    reconcile_trial_execution_metadata,
    trial_artifact_path,
)
from kaggle_research_agent.workspace_after_coding import run_workspace_after_coding  # noqa: E402
from kaggle_research_agent.workspace_code_writer import run_workspace_code_writer  # noqa: E402
from kaggle_research_agent.workspace_coding_handoff import (  # noqa: E402
    DATA_LOADER_INIT_TRIAL_ID,
    HARNESS_INIT_TRIAL_ID,
    generate_data_loader,
    generate_scoring_harness,
    prepare_workspace_coding_handoff,
)
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
    dacon_competition_id: str | None = None,
    dacon_team_name: str | None = None,
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
    recovered_execution = _recover_completed_execution(competition, trial_id) if code_writer else None
    if recovered_execution:
        save_loop_state(phase="resuming_after_execution")
        workspace_run = recovered_execution["workspace_run"]
        metrics_collection = recovered_execution["metrics_collection"]
        result_cycle = {}
        code_stage_result = {
            "status": "completed",
            "code_writer": {
                "status": "resumed",
                "resume_stage": "after_execution",
            },
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
        }
    elif code_writer and has_coding_plan(competition, trial_id):
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
        result_cycle = {}
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
        result_cycle = {}

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
    save_loop_state(phase="validating_execution_facts")
    consistency = reconcile_trial_execution_metadata(competition, trial_id)
    if consistency.get("status") == "blocked":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "blocked_execution_metadata",
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
            "execution_consistency": consistency,
        }
    submission_run = None
    # Resolve the daily limit here (not just from cache) so a competition
    # whose dashboard was never opened still gets checked before the loop
    # ever attempts a submission -- otherwise an unresolved "unknown" limit
    # was silently treated the same as "no limit" and auto-submitted anyway,
    # defeating the safe-by-default off setting.
    dacon_limit_known = (
        bool(submit and dacon_competition_id)
        and check_dacon_submission_limit(competition).get("daily_submission_limit") is not None
    )
    if submit and dacon_competition_id and dacon_limit_known and not dacon_auto_submit_allowed(competition):
        # A known daily submission limit exists and the researcher has not
        # opted in to auto-submit -- skip the actual DACON API call entirely
        # rather than risk burning the scarce daily quota automatically.
        # Local results and next-trial planning proceed as normal; the
        # trial can be submitted on demand later from the dashboard.
        submission_run = {
            "status": "skipped_daily_limit_known",
            "reason": (
                "Daily DACON submission limit is known and auto-submit is not enabled -- "
                "skipped automatic submission. Submit this trial manually when ready."
            ),
        }
    elif submit:
        save_loop_state(phase="submitting")
        submission_file = submission_artifact_path(profile)
        if not submission_file:
            status = "blocked_missing_submission_artifact"
            submission_run = {"status": status, "reason": "execution_profile artifacts.submission is missing"}
        else:
            message = f"{competition} {trial_id} local CV {local_score_for(trial_id, metrics_collection) or 0:.6f}"
            # Route to whichever platform adapter this competition declared.
            # DACON's submission API returns no leaderboard score or rank, so
            # only the Kaggle path can require post-submission score evidence.
            if dacon_competition_id:
                platform_kwargs: dict[str, Any] = {
                    "dacon_competition_id": dacon_competition_id,
                    "dacon_team_name": dacon_team_name,
                    "dacon_message": message,
                }
            else:
                platform_kwargs = {
                    "kaggle_competition_slug": kaggle_slug or competition,
                    "kaggle_message": message,
                }
            submission_run = submit_trial(
                competition=competition,
                trial_id=trial_id,
                version_name=f"{trial_id}_auto",
                submission_file=submission_file,
                objective=objective_for(competition, profile),
                poll_leaderboard=False,
                poll_attempts=poll_attempts,
                poll_interval_seconds=poll_interval_seconds,
                notes="Submitted by generic_workspace_auto_loop.py",
                **platform_kwargs,
            )
        submission_block = _submission_blocking_issue(
            submission_run,
            requires_score=bool(kaggle_slug) and not dacon_competition_id,
        )
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
    save_loop_state(phase="analyzing_result")
    result_cycle = process_workspace_result(competition, trial_id)
    if result_cycle.get("status") == "blocked":
        result = {
            "competition": competition,
            "trial_id": trial_id,
            "status": "result_cycle_blocked",
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
            "execution_consistency": consistency,
            "workspace_result_cycle": result_cycle,
            "submission_run": submission_run,
        }
        write_loop_trial_result(result)
        return result
    save_loop_state(phase="organizing_artifacts")
    artifact_summary = organize_trial_artifacts(
        competition, trial_id, low_cost_user_summary=False, allow_api=allow_api
    )
    result = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "completed",
        "workspace_run": workspace_run,
        "metrics_collection": metrics_collection,
        "workspace_result_cycle": result_cycle,
        "execution_consistency": consistency,
        "submission_run": submission_run,
        "artifact_summary": artifact_summary,
    }
    if code_stage_result:
        result["handoff"] = code_stage_result.get("handoff")
        result["code_writer"] = code_stage_result.get("code_writer")
        result["after_coding"] = code_stage_result.get("after_coding")
    write_loop_trial_result(result)
    return result


def _recover_completed_execution(competition: str, trial_id: str) -> dict[str, Any] | None:
    out_dir = trial_dir(competition, trial_id)
    validation = load_json(trial_artifact_path(out_dir, "workspace_coding_result_validation.json"))
    workspace_run = load_json(trial_artifact_path(out_dir, "workspace_run.json"))
    metrics_collection = load_json(trial_artifact_path(out_dir, "metrics_collection.json"))
    if (
        validation.get("status") == "accepted"
        and workspace_run.get("status") == "completed"
        and metrics_collection.get("status") == "collected"
    ):
        return {
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
        }
    return None


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
    # Always ask the LLM-driven planner for a concrete, evidence-specific plan --
    # not only when a user insight happens to be active. It already uses
    # decision_context (active_axis, axis_attempt_count, rejected_axes) as primary
    # evidence on its own, so it can name a specific next candidate even with no
    # insight text at all, instead of falling back to generic axis-template
    # wording. If the LLM call itself is unavailable or budget-blocked, the
    # rule-based plan already written by plan_next_workspace_trial above remains
    # as the fallback.
    save_loop_state(phase="revising_plan", current_trial=None, next_trial=target_trial)
    detailed = prepare_workspace_trial_plan(
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
    if detailed.get("status") == "planned":
        return detailed
    if detailed.get("status") == "duplicate_candidate_blocked":
        # The LLM planner repeated an already-rejected candidate even after one
        # forced retry. Surface this as a real block instead of silently falling
        # back to the rule-based plan, which is not checked against
        # rejected_candidates_by_axis and could reintroduce the same duplicate.
        return detailed
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


def _previous_cycle_ignored_feedback_source_trial(competition: str, trial_id: str) -> str | None:
    """If the last persisted attempt for this trial was itself a corrective
    retry (coding_feedback was non-empty) and it still hit the exact same
    guardrail issue it was warned about, a fresh cycle retrying the same way
    again is pointless -- return the source trial to re-plan from instead.

    Returns None when there is nothing to act on (no previous attempt, the
    previous attempt succeeded, or it failed on a different/new issue).
    """
    out_dir = trial_dir(competition, trial_id)
    handoff_path = out_dir / "workspace_coding_handoff.json"
    result_path = out_dir / "workspace_coding_result.json"
    if not handoff_path.exists() or not result_path.exists():
        return None
    try:
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if result.get("status") == "completed":
        return None
    feedback = handoff.get("coding_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return None
    rejected = {str(item) for item in feedback.get("rejected_issues", []) or []}
    if not rejected:
        return None
    issues = {str(item) for item in result.get("blocking_issues", []) or []}
    if not (issues & rejected):
        return None
    return str(handoff.get("source_trial_id") or "") or None


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
    ignored_feedback_source = _previous_cycle_ignored_feedback_source_trial(competition, trial_id)
    if ignored_feedback_source is not None:
        print(
            "Previous cycle's corrective retry repeated the exact same violation it was warned about; "
            "forcing a re-plan instead of retrying the same way again...",
            flush=True,
        )
        _force_replan_same_trial(
            competition,
            trial_id,
            source_trial_id=ignored_feedback_source,
            model=model,
            provider=provider,
            allow_api=allow_api,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
        )
    last_blocked: dict[str, Any] | None = None
    runtime_failure_context: dict[str, Any] | None = None
    coding_feedback: dict[str, Any] | None = None
    for attempt in range(1, 3):
        expanded_retry = attempt > 1
        if runtime_failure_context:
            print("Workspace execution failed; asking the code writer to repair the same trial once...", flush=True)
        elif coding_feedback:
            print("Code writer's previous attempt was blocked by automated review; retrying with that feedback...", flush=True)
        elif expanded_retry:
            print("Code writer blocked by missing code snapshot context; regenerating expanded handoff and retrying...", flush=True)
        handoff = prepare_workspace_coding_handoff(
            competition,
            trial_id,
            expanded_snapshot=expanded_retry,
            retry_reason=(
                "workspace_runtime_failure"
                if runtime_failure_context
                else "code_writer_blocked_review_feedback"
                if coding_feedback
                else ("code_writer_blocked_snapshot_context" if expanded_retry else None)
            ),
            runtime_failure_context=runtime_failure_context,
            coding_feedback=coding_feedback,
        )
        if handoff.get("status") != "ready":
            if attempt == 1 and _should_replan_for_plan_consistency(handoff.get("blocking_issues")):
                print(
                    "Code writer blocked because the plan does not match the base trial's actual code; "
                    "forcing a re-plan before retrying...",
                    flush=True,
                )
                _force_replan_same_trial(
                    competition,
                    trial_id,
                    source_trial_id=handoff.get("source_trial_id"),
                    model=model,
                    provider=provider,
                    allow_api=allow_api,
                    trial_llm_calls=trial_llm_calls,
                    strategy_calls_today=strategy_calls_today,
                )
                continue
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
            if attempt == 1 and _should_replan_after_code_writer_mismatch(last_blocked):
                print(
                    "Code writer blocked by a patch that does not match the base trial's actual code; "
                    "forcing a re-plan before retrying...",
                    flush=True,
                )
                _force_replan_same_trial(
                    competition,
                    trial_id,
                    source_trial_id=handoff.get("source_trial_id"),
                    model=model,
                    provider=provider,
                    allow_api=allow_api,
                    trial_llm_calls=trial_llm_calls,
                    strategy_calls_today=strategy_calls_today,
                )
                continue
            if attempt == 1 and _should_retry_code_writer_block(last_blocked):
                coding_feedback = _coding_feedback_from_blocked_result(last_blocked)
                continue
            if coding_feedback:
                rejected = {str(item) for item in coding_feedback.get("rejected_issues", []) or []}
                new_issues = {
                    str(item)
                    for item in [
                        *list(code_writer_result.get("blocking_issues") or []),
                        *list(code_writer_result.get("issues") or []),
                    ]
                }
                repeated = sorted(rejected & new_issues)
                if repeated:
                    last_blocked["feedback_ignored"] = True
                    log_decision(
                        competition,
                        trial_id,
                        decision_type="code_writer_feedback_ignored",
                        decision="blocked",
                        reason="Corrective retry repeated the exact same guardrail violation it was warned about.",
                        evidence={"rejected_issues": repeated},
                        next_action="force_replan_next_cycle",
                    )
            return last_blocked
        if allow_api and trial_id not in {HARNESS_INIT_TRIAL_ID, DATA_LOADER_INIT_TRIAL_ID}:
            # The loader comes first: the harness calls into it, so a harness
            # generated against a broken loader can only be wrong.
            loader_status = generate_data_loader(
                competition, model=model, provider=provider, allow_api=allow_api
            )
            if loader_status.get("status") == "blocked":
                print(
                    "One-time data loader generation was blocked; continuing this trial without "
                    "the score stage for now (see loader_status for details).",
                    flush=True,
                )
        if allow_api and trial_id != HARNESS_INIT_TRIAL_ID:
            # Only attempted once the code writer's own result was accepted
            # -- that means predict_step.py just passed the scoring interface
            # check (load_samples()/predict() defined), so the harness has a
            # real, verified interface to call into. Attempting this before
            # any trial has ever written a compliant predict_step.py (e.g.
            # before trial_001) would always fail: the scaffold's default
            # predict_step.py doesn't define these functions either.
            harness_status = generate_scoring_harness(competition, model=model, provider=provider, allow_api=allow_api)
            if harness_status.get("status") == "blocked":
                print(
                    "One-time scoring harness generation was blocked; continuing this trial without "
                    "the score stage for now (see harness_status for details).",
                    flush=True,
                )
        after_coding = run_workspace_after_coding(
            competition,
            trial_id,
            run_now=True,
            finalize_result=False,
        )
        if after_coding.get("status") != "execution_completed":
            last_blocked = {
                "competition": competition,
                "trial_id": trial_id,
                "status": f"after_coding_{after_coding.get('status')}",
                "handoff": handoff,
                "code_writer": code_writer_result,
                "after_coding": after_coding,
                "code_writer_attempt": attempt,
            }
            if attempt == 1 and _should_retry_after_coding_failure(after_coding):
                runtime_failure_context = _runtime_failure_context(after_coding)
                continue
            return last_blocked
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


# A blocking reason belonging to this list means the same underlying thing:
# the plan asked for a change against code that does not actually contain
# what the plan assumes (e.g. a model-family axis continued against a base
# trial that never got that family's change accepted into it). Showing more
# code context cannot fix this -- the plan itself needs to be regenerated --
# so these are handled separately from _should_retry_code_writer_block's
# "show more context and retry the same plan" path. This list only needs to
# cover the handoff's own deterministic issue codes (workspace_coding_handoff.py
# generates these itself, so the exact strings are reliable); the code
# writer's own free-text/structured blocking reasons are far less
# predictable and are handled by _should_replan_after_code_writer_mismatch's
# catch-all instead of trying to enumerate every possible phrasing here.
_PLAN_CONSISTENCY_MARKERS = [
    "plan_find_target_missing_in_base_code",
    "patch_find_not_found",
    "patch_target_missing",
]

_NON_RECOVERABLE_MARKERS = [
    "token_policy_blocked",
    "api_call_not_enabled",
    "api_error:",
    "quota",
    "insufficient_quota",
]


def _should_replan_for_plan_consistency(issues: list[Any] | None) -> bool:
    lowered = [str(item).lower() for item in issues or []]
    return any(marker in issue for issue in lowered for marker in _PLAN_CONSISTENCY_MARKERS)


def _should_replan_after_code_writer_mismatch(result: dict[str, Any]) -> bool:
    """Catch-all for code-writer blocks that are neither a known "needs more
    context" issue nor a non-recoverable one (budget/API/quota).

    The code writer's own explanation for a plan/code mismatch is free text
    (or even a structured dict) written by an LLM, so it is impractical to
    enumerate every phrasing it might use ("there is no Ridge init to patch",
    "model mismatch between delta plan and authoritative code", etc.).
    Instead: if the block wrote no files, isn't a context problem the normal
    retry already covers, and isn't a budget/API dead end, the most likely
    explanation left is that the plan itself doesn't match the base code --
    so default to regenerating the plan rather than trying to guess the
    LLM's exact wording.
    """
    if result.get("status") != "code_writer_blocked":
        return False
    code_writer = result.get("code_writer") if isinstance(result.get("code_writer"), dict) else {}
    if code_writer.get("changed_files"):
        return False
    issues = [
        str(item)
        for item in [*list(code_writer.get("blocking_issues") or []), *list(code_writer.get("issues") or [])]
    ]
    issue_text = "\n".join(issues).lower()
    if any(marker in issue_text for marker in _NON_RECOVERABLE_MARKERS):
        return False
    return not _should_retry_code_writer_block(result)


def _force_replan_same_trial(
    competition: str,
    trial_id: str,
    *,
    source_trial_id: str | None,
    model: str,
    provider: str,
    allow_api: bool,
    trial_llm_calls: int | None,
    strategy_calls_today: int | None,
) -> dict[str, Any]:
    return prepare_workspace_trial_plan(
        competition,
        trial_id,
        source_trial_id=source_trial_id,
        model=model,
        provider=provider,
        allow_api=allow_api,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
        force_replan=True,
    )


_GUARDRAIL_BLOCK_MARKERS = (
    "local_validation_not_computed",
    "predict_script_changed_without_test_script_sync",
    "possible_fabricated_data_fallback",
    "scoring_interface_missing_function",
    "predict_script_does_not_iterate_sample_files",
    "changed_file_not_allowed",
    "forbidden_path_touched",
)


def _guardrail_block_issues(result: dict[str, Any]) -> list[str]:
    """The subset of a blocked result's issues that came from
    workspace_code_writer.py's own mechanical guardrails, as opposed to a
    missing-context reason (handled separately via expanded_snapshot)."""
    code_writer = result.get("code_writer") if isinstance(result.get("code_writer"), dict) else {}
    issues = [
        str(item)
        for item in [*list(code_writer.get("blocking_issues") or []), *list(code_writer.get("issues") or [])]
    ]
    return [issue for issue in issues if any(marker in issue.lower() for marker in _GUARDRAIL_BLOCK_MARKERS)]


def _coding_feedback_from_blocked_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Build the corrective-feedback payload rendered into the retry prompt
    (see workspace_coding_handoff.py's "## Previous Attempt Was Rejected"),
    when the block came from one of our own mechanical guardrails.

    Without this, a retry only got more base-code context (expanded_snapshot)
    -- it never actually told the model which of its own checks it had
    tripped, so a retry just reproduced the exact same mistake.
    """
    rejected_issues = _guardrail_block_issues(result)
    if not rejected_issues:
        return None
    code_writer = result.get("code_writer") if isinstance(result.get("code_writer"), dict) else {}
    return {
        "changed_files": list(code_writer.get("changed_files") or []),
        "rejected_issues": rejected_issues,
    }


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
        "authoritative base snapshot does not include",
        "not present in the provided authoritative base code snapshot",
        "truncated",
        "safe localized edits",
        "existing anchors",
        "exact find/replace anchors",
        "missing authoritative source for",
        "is required for patch-only mode",
        # Mechanical guardrail blocks (workspace_code_writer.py's own
        # validators) -- these are real problems in what the code writer
        # produced, not a missing-context issue, but a retry that explains
        # exactly what was flagged (see coding_feedback below) has a real
        # chance of producing compliant code, unlike blindly retrying with
        # more base-code context.
        "local_validation_not_computed",
        "predict_script_changed_without_test_script_sync",
        "possible_fabricated_data_fallback",
        "scoring_interface_missing_function",
        "predict_script_does_not_iterate_sample_files",
        "changed_file_not_allowed",
        "forbidden_path_touched",
    ]
    return any(marker in issue_text for marker in retriable_markers) and not any(
        marker in issue_text for marker in _NON_RECOVERABLE_MARKERS
    )


def _should_retry_after_coding_failure(after_coding: dict[str, Any]) -> bool:
    status = str(after_coding.get("status") or "")
    if status not in {
        "workspace_run_failed",
        "workspace_run_incomplete_artifacts",
        "workspace_run_invalid_artifacts",
        "metrics_blocked",
        "metrics_invalid",
    }:
        return False
    workspace_run = after_coding.get("workspace_run")
    failure = workspace_run.get("failure") if isinstance(workspace_run, dict) else {}
    failure_type = str((failure or {}).get("failure_type") or "")
    return failure_type not in {"resource_cpu_memory", "resource_gpu_missing", "permission_error"}


def _runtime_failure_context(after_coding: dict[str, Any]) -> dict[str, Any]:
    workspace_run = after_coding.get("workspace_run")
    workspace_run = workspace_run if isinstance(workspace_run, dict) else {}
    failure = workspace_run.get("failure")
    failure = failure if isinstance(failure, dict) else {}
    failed_command = next(
        (
            item
            for item in reversed(workspace_run.get("command_results") or [])
            if isinstance(item, dict) and int(item.get("returncode", 0) or 0) != 0
        ),
        {},
    )
    log_path = Path(str(failed_command.get("log_path") or ""))
    log_tail = ""
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-45:])[-5000:]
    return {
        "after_coding_status": after_coding.get("status"),
        "failed_stage": failed_command.get("stage"),
        "failed_command": failed_command.get("command"),
        "returncode": failed_command.get("returncode"),
        "failure_type": failure.get("failure_type"),
        "matched_pattern": failure.get("matched_pattern"),
        "issues": list(after_coding.get("issues") or []),
        "log_tail": log_tail,
    }


def write_loop_trial_result(result: dict[str, Any]) -> None:
    out_dir = trial_dir(result["competition"], result["trial_id"])
    write_text(out_dir / "auto_loop_trial.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "legacy_runtime", False)):
        return run_loop_legacy(args)
    return run_loop_langgraph(args)


def run_loop_langgraph(args: argparse.Namespace) -> dict[str, Any]:
    clear_pause_request()
    acquire_loop_lock()
    selected = trial_sequence(args.start_trial, args.max_trials)
    previous_state = load_json(STATE_PATH)
    resumable = _can_resume_graph_process(previous_state, args.competition, selected[0])
    graph_thread_id = (
        str(previous_state.get("graph_thread_id"))
        if resumable
        else f"{args.competition}:{selected[0]}:{time.time_ns()}"
    )
    save_loop_state(
        competition=args.competition,
        status="running",
        pid=os.getpid(),
        start_trial=selected[0],
        end_trial=selected[-1],
        last_completed_trial=previous_trial_id(selected[0]),
        next_trial=selected[0],
        phase="planning",
        pause_requested=False,
        error=None,
        resume_from_status=None,
        graph_runtime="langgraph",
        graph_thread_id=graph_thread_id,
    )
    settings = {
        "competition": args.competition,
        "submit": bool(args.submit),
        "kaggle_slug": args.kaggle_slug,
        "dacon_competition_id": getattr(args, "dacon_competition_id", None),
        "dacon_team_name": getattr(args, "dacon_team_name", None),
        "poll_attempts": args.poll_attempts,
        "poll_interval_seconds": args.poll_interval_seconds,
        "code_writer": bool(args.code_writer),
        "model": args.model,
        "provider": args.provider,
        "allow_api": bool(args.allow_api),
        "trial_llm_calls": args.trial_llm_calls,
        "strategy_calls_today": args.strategy_calls_today,
    }
    callbacks = WorkspaceLoopCallbacks(
        prepare_trial=prepare_trial_for_code,
        run_trial=_run_trial_from_graph,
        plan_successor=plan_following_trial,
        sync_state=sync_state_db,
        save_state=save_loop_state,
        pause_requested=pause_is_requested,
        human_review_resolved=_human_review_resolved,
    )
    checkpoint_path = Path(
        os.environ.get(
            "RESEARCH_AGENT_GRAPH_CHECKPOINT_DB",
            str(RUNTIME_DIR / "langgraph_checkpoints.sqlite3"),
        )
    ).expanduser()
    initial = {
        "competition": args.competition,
        "selected_trials": selected,
        "trial_index": 0,
        "settings": settings,
        "status": "running",
        "phase": "planning",
        "current_trial": None,
        "next_trial": selected[0],
        "rows": [],
        "steps": [],
        "graph_runtime": "langgraph",
        "graph_thread_id": graph_thread_id,
    }
    with workspace_loop_checkpointer(checkpoint_path) as (checkpointer, checkpoint_backend):
        if (
            resumable
            and checkpoint_backend == "sqlite"
            and not workspace_loop_thread_has_pending_work(
                callbacks, checkpointer=checkpointer, thread_id=graph_thread_id
            )
        ):
            # The saved thread already ran to completion (or to a terminal
            # failure). Resuming it would replay that terminal result without
            # executing anything, so every restart would report the same old
            # failure, write no state, and exit -- the launcher's "running"
            # state then reconciles into a confusing process_not_running.
            # Start a fresh thread so the loop can actually make progress.
            print(
                "Saved graph thread already reached a terminal state; starting a fresh run instead of resuming...",
                flush=True,
            )
            resumable = False
            graph_thread_id = f"{args.competition}:{selected[0]}:{time.time_ns()}"
            initial["graph_thread_id"] = graph_thread_id
            save_loop_state(graph_thread_id=graph_thread_id, resume_from_status=None)
        save_loop_state(graph_checkpoint_backend=checkpoint_backend, graph_checkpoint_path=str(checkpoint_path))
        if resumable and checkpoint_backend == "sqlite":
            result = resume_workspace_loop_graph(
                callbacks,
                checkpointer=checkpointer,
                thread_id=graph_thread_id,
                resume_value=(
                    {"action": "feedback_applied"}
                    if previous_state.get("status") == "awaiting_human_review"
                    else None
                ),
            )
        else:
            result = run_workspace_loop_graph(
                initial,
                callbacks,
                checkpointer=checkpointer,
                thread_id=graph_thread_id,
            )
    response = {
        "competition": args.competition,
        "status": result.get("status") or "failed",
        "trials": list(result.get("rows") or []),
        "graph_runtime": "langgraph",
        "graph_thread_id": graph_thread_id,
        "graph_steps": list(result.get("steps") or []),
    }
    if result.get("failure_stage") == "planning_next":
        response["next_plan_status"] = (result.get("next_plan_result") or {}).get("status")
    return response


def _can_resume_graph_process(state: dict[str, Any], competition: str, start_trial: str) -> bool:
    resume_status = str(state.get("resume_from_status") or state.get("status") or "").casefold()
    return bool(
        state.get("graph_runtime") == "langgraph"
        and resume_status in {"running", "awaiting_human_review"}
        and state.get("competition") == competition
        and state.get("next_trial") == start_trial
        and state.get("graph_thread_id")
    )


def _human_review_resolved(competition: str, trial_id: str) -> bool:
    if list_pending_actions(competition, status="pending"):
        return False
    cycle = load_json(trial_dir(competition, trial_id) / "workspace_result_cycle.json")
    return cycle.get("status") in {
        "completed",
        "completed_feedback_applied",
        "completed_review_deferred",
        "already_processed",
    }


def _run_trial_from_graph(args: argparse.Namespace, trial_id: str) -> dict[str, Any]:
    print(f"\n=== {args.competition} / {trial_id} ===", flush=True)
    return run_one_trial(
        args.competition,
        trial_id,
        submit=args.submit,
        kaggle_slug=args.kaggle_slug,
        dacon_competition_id=getattr(args, "dacon_competition_id", None),
        dacon_team_name=getattr(args, "dacon_team_name", None),
        poll_attempts=args.poll_attempts,
        poll_interval_seconds=args.poll_interval_seconds,
        code_writer=args.code_writer,
        model=args.model,
        provider=args.provider,
        allow_api=args.allow_api,
        trial_llm_calls=args.trial_llm_calls,
        strategy_calls_today=args.strategy_calls_today,
    )


def run_loop_legacy(args: argparse.Namespace) -> dict[str, Any]:
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
        graph_runtime="legacy",
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
            dacon_competition_id=getattr(args, "dacon_competition_id", None),
            dacon_team_name=getattr(args, "dacon_team_name", None),
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
    parser.add_argument("--dacon-competition-id", dest="dacon_competition_id", default=None)
    parser.add_argument("--dacon-team-name", dest="dacon_team_name", default=None)
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
    parser.add_argument(
        "--legacy-runtime",
        action="store_true",
        help="Use the pre-LangGraph Python loop as an emergency compatibility path.",
    )
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
