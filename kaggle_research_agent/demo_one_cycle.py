from __future__ import annotations

import ast
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from .agents.code_writer_adapter import FileResponseClient, create_llm_client, provider_log_name
from .agents.memory import log_decision, log_token_usage
from .agents.policy_gate import log_llm_decision
from .agents.submission import prepare_submission
from .execution_profile import load_execution_profile, validate_execution_profile
from .incremental_trial import enrich_delta_plan, write_base_summary
from .paths import competition_dir, competition_memory_dir, trial_dir
from .policies import load_policy, select_model_for_call
from .rag_policy import evaluate_rag_policy
from .retrieval.context_pack import build_context_pack, context_pack_prompt_summary
from .state_db_auto import sync_trial_state_after_finish
from .store import load_recent_trials, load_state, now_iso, read_text, write_text
from .trial_artifacts import organize_trial_artifacts, trial_artifact_path
from .trial_decision import load_latest_decision_context, write_trial_decision_card
from .trial_memory_card import write_trial_memory_card
from .workspace_code_writer import run_workspace_code_writer, validate_workspace_coding_result
from .workspace_coding_handoff import prepare_workspace_coding_handoff
from .workspace_metrics_collector import collect_workspace_metrics
from .workspace_runner import run_workspace_pipeline


def run_demo_one_cycle(
    competition: str,
    trial_id: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    allow_api: bool = False,
    mock_plan_file: str | None = None,
    mock_response_file: str | None = None,
    run_now: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
    show_progress: bool = False,
    low_cost_user_summary: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = DemoStatusReporter(competition, trial_id, show_progress=show_progress)
    model_policy = resolve_demo_model_policy(model_override=model, provider_override=provider)

    reporter.start("F-01", "Loading problem context", 1, "Reading Execution Profile, state, inventory, and recent file memory.")
    context = load_demo_context(competition, trial_id)
    context["model_policy"] = model_policy
    context["planning_context_pack"] = _build_planning_context_pack_if_needed(competition, trial_id, context)
    write_text(out_dir / "demo_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_context.md", render_demo_context(context))
    if context["status"] != "ready":
        reporter.fail("F-01", "Problem context is blocked.", next_action="fix-execution-profile")
        return _finish(
            competition,
            trial_id,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "run_now": run_now,
                "steps": ["F-01"],
                "context": context,
                "model_policy": model_policy,
                "plan": None,
                "handoff": None,
                "code_writer": None,
                "workspace_run": None,
                "metrics_collection": None,
                "record": None,
                "issues": context["issues"],
                "next_action": "fix-execution-profile",
            },
            reporter=reporter,
            low_cost_user_summary=low_cost_user_summary,
            allow_api=allow_api,
        )
    reporter.complete(
        "F-01",
        f"Context ready: metric={context.get('metric')} objective={context.get('objective')}.",
        next_action="create-demo-experiment-plan",
        details={
            "metric": context.get("metric"),
            "objective": context.get("objective"),
            "platform": context.get("platform"),
        },
    )

    reporter.start("F-02", "Planning one experiment", 2, "Calling mock/API LLM for one practical first-cycle plan.")
    plan = None if mock_plan_file else _load_ready_demo_plan(out_dir)
    if plan is None:
        plan = create_demo_experiment_plan(
            competition,
            trial_id,
            context,
            model_config=model_policy["experiment_planning"],
            allow_api=allow_api,
            mock_plan_file=mock_plan_file,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
        )
    else:
        plan["resumed_from_existing_artifact"] = True
    if plan["status"] != "ready":
        reporter.fail("F-02", "Experiment planning is blocked.", next_action=plan["next_action"])
        return _finish(
            competition,
            trial_id,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "run_now": run_now,
                "steps": ["F-01", "F-02"],
                "context": context,
                "model_policy": model_policy,
                "plan": plan,
                "handoff": None,
                "code_writer": None,
                "workspace_run": None,
                "metrics_collection": None,
                "record": None,
                "issues": plan["issues"],
                "next_action": plan["next_action"],
            },
            reporter=reporter,
        )
    reporter.complete(
        "F-02",
        f"Plan ready: {plan.get('plan_title')}.",
        next_action="prepare-workspace-handoff",
        details={"token_usage": plan.get("token_usage")},
    )

    reporter.start("F-03", "Writing pipeline code", 3, "Preparing scoped handoff and applying mock/API code updates.")
    _write_continuation_context(competition, trial_id)
    handoff = prepare_workspace_coding_handoff(competition, trial_id)
    if handoff["status"] != "ready":
        reporter.fail("F-03", "Coding handoff is blocked.", next_action=handoff["next_action"])
        return _finish(
            competition,
            trial_id,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "run_now": run_now,
                "steps": ["F-01", "F-02", "F-03:handoff"],
                "context": context,
                "model_policy": model_policy,
                "plan": plan,
                "handoff": handoff,
                "code_writer": None,
                "workspace_run": None,
                "metrics_collection": None,
                "record": None,
                "issues": handoff.get("blocking_issues", []),
                "next_action": handoff["next_action"],
            },
            reporter=reporter,
        )

    code_writer = None if mock_response_file else _load_accepted_workspace_code_writer(competition, trial_id)
    if code_writer is None:
        client = FileResponseClient(mock_response_file) if mock_response_file else None
        code_writer = run_workspace_code_writer(
            competition,
            trial_id,
            client=client,
            model=model_policy["workspace_code_writing"]["model"],
            provider=model_policy["workspace_code_writing"]["provider"],
            allow_api=allow_api,
            trial_llm_calls=trial_llm_calls,
            strategy_calls_today=strategy_calls_today,
        )
    if code_writer["status"] != "accepted":
        reporter.fail("F-03", "Code writer result is blocked.", next_action=code_writer["next_action"])
        return _finish(
            competition,
            trial_id,
            {
                "competition": competition,
                "trial_id": trial_id,
                "status": "blocked",
                "run_now": run_now,
                "steps": ["F-01", "F-02", "F-03"],
                "context": context,
                "model_policy": model_policy,
                "plan": plan,
                "handoff": handoff,
                "code_writer": code_writer,
                "workspace_run": None,
                "metrics_collection": None,
                "record": None,
                "issues": code_writer.get("issues", []) or code_writer.get("blocking_issues", []),
                "next_action": code_writer["next_action"],
            },
            reporter=reporter,
        )
    reporter.complete(
        "F-03",
        f"Code accepted: changed_files={', '.join(code_writer.get('changed_files', []) or ['None'])}.",
        next_action="run-local-pipeline",
        details={"changed_files": code_writer.get("changed_files", [])},
    )

    reporter.start("F-04", "Running local pipeline", 4, "Processing test/train/predict commands from the Execution Profile.")
    workspace_run = run_workspace_pipeline(competition, trial_id, run_now=run_now)
    metrics_collection = None
    record = None
    status = "planned"
    issues: list[str] = []
    next_action = "rerun-demo-one-cycle-with-run-now"
    if run_now:
        if workspace_run["status"] == "completed":
            reporter.complete(
                "F-04",
                "Local pipeline completed.",
                next_action="record-demo-result",
                details={
                    "commands": [
                        item.get("stage")
                        for item in workspace_run.get("command_results", [])
                        if item.get("returncode") == 0
                    ]
                },
            )
            reporter.start("F-06", "Recording result", 5, "Collecting metrics and writing demo result files.")
            metrics_collection = collect_workspace_metrics(competition, trial_id)
            if metrics_collection["status"] == "collected":
                record = record_demo_cycle_result(
                    competition,
                    trial_id,
                    context=context,
                    plan=plan,
                    code_writer=code_writer,
                    workspace_run=workspace_run,
                    metrics_collection=metrics_collection,
                )
                submission_manifest = prepare_demo_submission(
                    competition,
                    trial_id,
                    record=record,
                    notes="Demo cycle completed local execution. Submit only after user approval.",
                )
                status = "completed"
                next_action = "review-submit-manifest"
                reporter.complete(
                    "F-06",
                    f"Result recorded: local_score={record.get('local_score')}.",
                    next_action=next_action,
                    details={
                        "metric": record.get("metric"),
                        "local_score": record.get("local_score"),
                        "user_view": f"runs/{competition}/{trial_id}",
                    },
                )
            else:
                status = "blocked"
                issues = metrics_collection.get("issues", [])
                next_action = metrics_collection["next_action"]
                reporter.fail("F-06", "Metrics collection is blocked.", next_action=next_action)
        else:
            status = "failed" if workspace_run["status"] == "failed" else "blocked"
            issues = _workspace_run_issues(workspace_run)
            next_action = workspace_run["next_action"]
            reporter.fail("F-04", f"Local pipeline status={workspace_run['status']}.", next_action=next_action)
    else:
        reporter.complete("F-04", "Local pipeline planned only. Re-run with --run-now to execute.", next_action=next_action)

    return _finish(
        competition,
        trial_id,
        {
            "competition": competition,
            "trial_id": trial_id,
            "status": status,
            "run_now": run_now,
            "steps": ["F-01", "F-02", "F-03", "F-04", "F-06" if record else "F-06:pending"],
            "context": context,
            "model_policy": model_policy,
            "plan": plan,
            "handoff": handoff,
            "code_writer": code_writer,
            "workspace_run": workspace_run,
            "metrics_collection": metrics_collection,
            "record": record,
            "submission_manifest": locals().get("submission_manifest"),
            "issues": issues,
            "next_action": next_action,
        },
        reporter=reporter,
        low_cost_user_summary=low_cost_user_summary,
        allow_api=allow_api,
    )


def prepare_demo_submission(
    competition: str,
    trial_id: str,
    *,
    record: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Prepare a guarded submission manifest after local metrics are available."""

    profile = load_execution_profile(competition)
    project_root = Path(profile.get("project_root") or ".")
    submission_relatives = profile.get("artifacts", {}).get("submission") or ["outputs/submission.csv"]
    submission_relative = str(submission_relatives[0])
    submission_file = project_root / submission_relative
    objective = (
        (record or {}).get("objective")
        or profile.get("objective")
        or load_state(competition).get("competition", {}).get("objective")
        or "maximize"
    )
    version_name = f"{trial_id}_local_submission"
    return prepare_submission(
        competition=competition,
        trial_id=trial_id,
        version_name=version_name,
        submission_file=str(submission_file),
        objective=str(objective),
        notes=notes,
    )


class DemoStatusReporter:
    def __init__(self, competition: str, trial_id: str, *, show_progress: bool = False):
        self.competition = competition
        self.trial_id = trial_id
        self.show_progress = show_progress
        self.out_dir = trial_dir(competition, trial_id)
        self.status_path = self.out_dir / "agent_status.json"
        self.events_path = self.out_dir / "agent_events.jsonl"
        self.started_at = now_iso()
        self.total_steps = 5

    def start(self, stage: str, label: str, progress: int, message: str, *, next_action: str | None = None) -> None:
        self._record(
            event="stage_started",
            status="running",
            stage=stage,
            label=label,
            progress=progress,
            message=message,
            next_action=next_action,
        )

    def complete(
        self,
        stage: str,
        message: str,
        *,
        next_action: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            event="stage_completed",
            status="running",
            stage=stage,
            label=stage,
            progress=_stage_progress(stage),
            message=message,
            next_action=next_action,
            details=details,
        )

    def fail(self, stage: str, message: str, *, next_action: str | None = None) -> None:
        self._record(
            event="stage_failed",
            status="blocked" if stage != "F-04" else "failed",
            stage=stage,
            label=stage,
            progress=_stage_progress(stage),
            message=message,
            next_action=next_action,
        )

    def finish(self, status: str, message: str, *, next_action: str | None = None) -> None:
        event = "cycle_completed" if status in {"completed", "planned"} else "cycle_blocked"
        self._record(
            event=event,
            status=status,
            stage="done",
            label="Demo cycle finished",
            progress=self.total_steps,
            message=message,
            next_action=next_action,
        )

    def _record(
        self,
        *,
        event: str,
        status: str,
        stage: str,
        label: str,
        progress: int,
        message: str,
        next_action: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = now_iso()
        row = {
            "time": now,
            "event": event,
            "competition": self.competition,
            "trial_id": self.trial_id,
            "status": status,
            "current_stage": stage,
            "stage_label": label,
            "message": message,
            "progress": progress,
            "total_steps": self.total_steps,
            "started_at": self.started_at,
            "updated_at": now,
            "pid": os.getpid(),
            "next_action": next_action,
            "details": details or {},
            "status_path": str(self.status_path.as_posix()),
            "events_path": str(self.events_path.as_posix()),
        }
        write_text(self.status_path, json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.show_progress:
            print(format_demo_event(row), flush=True)


def read_demo_agent_status(competition: str, trial_id: str) -> dict[str, Any]:
    path = trial_dir(competition, trial_id) / "agent_status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "status": "missing",
            "competition": competition,
            "trial_id": trial_id,
            "message": "No demo status file exists yet.",
            "status_path": str(path.as_posix()),
            "next_action": "run-demo-one-cycle",
        }
    except json.JSONDecodeError:
        return {
            "status": "invalid",
            "competition": competition,
            "trial_id": trial_id,
            "message": "Demo status file is not valid JSON.",
            "status_path": str(path.as_posix()),
            "next_action": "inspect-status-file",
        }
    return status if isinstance(status, dict) else {"status": "invalid", "message": "Demo status file is not an object."}


def read_demo_agent_events(competition: str, trial_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
    path = trial_dir(competition, trial_id) / "agent_events.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows[-limit:]


def _load_ready_demo_plan(out_dir: Path) -> dict[str, Any] | None:
    plan = _load_json(trial_artifact_path(out_dir, "demo_experiment_plan.json"), default={})
    if isinstance(plan, dict) and plan.get("status") == "ready":
        context = _load_json(out_dir / "demo_context.json", default={})
        if isinstance(context, dict) and context:
            plan = _normalize_plan(plan, context)
            write_text(out_dir / "demo_experiment_plan.md", render_demo_plan(plan))
            if not (out_dir / "next_experiment.md").exists():
                write_text(out_dir / "next_experiment.md", render_demo_plan(plan))
        return plan
    return None


def _load_accepted_workspace_code_writer(competition: str, trial_id: str) -> dict[str, Any] | None:
    out_dir = trial_dir(competition, trial_id)
    existing_validation = _load_json(
        trial_artifact_path(out_dir, "workspace_coding_result_validation.json"),
        default={},
    )
    if isinstance(existing_validation, dict) and existing_validation.get("status") == "accepted":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "accepted",
            "issues": existing_validation.get("issues", []),
            "coding_result_status": existing_validation.get("coding_result_status"),
            "changed_files": existing_validation.get("changed_files", []),
            "allowed_write_paths": existing_validation.get("allowed_write_paths", []),
            "forbidden_paths": existing_validation.get("forbidden_paths", []),
            "blocking_issues": [],
            "next_action": "run-workspace-validation-commands",
            "resumed_from_existing_artifact": True,
            "resume_source": "workspace_coding_result_validation",
        }
    result_path = trial_artifact_path(out_dir, "workspace_coding_result.json")
    if not result_path.exists():
        return None
    try:
        validation = validate_workspace_coding_result(competition, trial_id)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None
    if validation.get("status") != "accepted":
        return None
    coding_result = _load_json(result_path, default={})
    return {
        "competition": competition,
        "trial_id": trial_id,
        "status": "accepted",
        "issues": validation.get("issues", []),
        "coding_result_status": validation.get("coding_result_status"),
        "changed_files": validation.get("changed_files", []),
        "allowed_write_paths": validation.get("allowed_write_paths", []),
        "forbidden_paths": validation.get("forbidden_paths", []),
        "blocking_issues": coding_result.get("blocking_issues", []) if isinstance(coding_result, dict) else [],
        "next_action": "run-workspace-validation-commands",
        "resumed_from_existing_artifact": True,
    }


def resolve_demo_model_policy(
    *,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> dict[str, Any]:
    policy = load_policy("model_policy")
    experiment_planning = select_model_for_call("experiment_planning", policy=policy)
    workspace_code_writing = select_model_for_call("workspace_code_writing", policy=policy)
    low_cost = dict(policy.get("low_cost", {}))
    if model_override:
        experiment_planning["model"] = model_override
        workspace_code_writing["model"] = model_override
        experiment_planning["override"] = True
        workspace_code_writing["override"] = True
    if provider_override:
        experiment_planning["provider"] = provider_override
        workspace_code_writing["provider"] = provider_override
        experiment_planning["override"] = True
        workspace_code_writing["override"] = True
    return {
        "experiment_planning": experiment_planning,
        "workspace_code_writing": workspace_code_writing,
        "low_cost": {
            "tier": "low_cost",
            "provider": low_cost.get("provider"),
            "api": low_cost.get("api"),
            "model": low_cost.get("model"),
            "call_types": low_cost.get("call_types", []),
        },
    }


def render_demo_agent_watch(competition: str, trial_id: str, *, event_limit: int = 8) -> str:
    status = read_demo_agent_status(competition, trial_id)
    events = read_demo_agent_events(competition, trial_id, limit=event_limit)
    lines = [
        f"데모 에이전트 상태: {competition} / {trial_id}",
        f"- 전체 상태: {_status_label_ko(str(status.get('status', '-')))}",
        f"- 현재 단계: {_stage_name_ko(str(status.get('current_stage', '-')))}",
        f"- 진행률: {status.get('progress', 0)}/{status.get('total_steps', 5)}",
        f"- 다음 행동: {status.get('next_action')}",
        f"- 갱신 시각: {status.get('updated_at', '-')}",
        f"- 프로세스 ID: {status.get('pid', '-')}",
        f"- 상태 파일: {status.get('status_path')}",
        "",
        "최근 진행 기록:",
    ]
    if events:
        lines.extend(f"- {format_demo_event(event)}" for event in events)
    else:
        lines.append("- 아직 기록된 이벤트가 없습니다.")
    return "\n".join(lines)


def watch_demo_agent(
    competition: str,
    trial_id: str,
    *,
    follow: bool = False,
    interval_seconds: float = 1.0,
    max_refreshes: int | None = None,
) -> dict[str, Any]:
    refreshes = 0
    last_rendered = ""
    while True:
        rendered = render_demo_agent_watch(competition, trial_id)
        if rendered != last_rendered:
            print(rendered, flush=True)
            last_rendered = rendered
        refreshes += 1
        status = read_demo_agent_status(competition, trial_id)
        if not follow or status.get("status") in {"completed", "planned", "blocked", "failed"}:
            return status
        if max_refreshes is not None and refreshes >= max_refreshes:
            return status
        time.sleep(interval_seconds)


def format_demo_event(row: dict[str, Any]) -> str:
    event = str(row.get("event", "event"))
    stage = str(row.get("current_stage", "-"))
    status = str(row.get("status", "-"))
    progress = f"{row.get('progress', 0)}/{row.get('total_steps', 5)}"
    prefix = "[완료]" if event in {"stage_completed", "cycle_completed"} else "[진행]"
    if event in {"stage_failed", "cycle_blocked"} or status in {"blocked", "failed"}:
        prefix = "[중단]"
    return f"{prefix} {progress} {stage} {_stage_name_ko(stage)} - {_event_summary_ko(row)}"


def _stage_name_ko(stage: str) -> str:
    return {
        "F-01": "대회/데이터 맥락 확인",
        "F-02": "실험 계획 생성",
        "F-03": "파이프라인 코드 작성",
        "F-04": "로컬 실행",
        "F-06": "결과 기록",
        "done": "1회 실험 종료",
    }.get(stage, stage or "-")


def _status_label_ko(status: str) -> str:
    return {
        "running": "진행 중",
        "completed": "완료",
        "planned": "실행 대기",
        "blocked": "중단됨",
        "failed": "실패",
        "missing": "상태 파일 없음",
        "invalid": "상태 파일 오류",
    }.get(status, status)


def _event_summary_ko(row: dict[str, Any]) -> str:
    event = str(row.get("event", "event"))
    stage = str(row.get("current_stage", "-"))
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    message = str(row.get("message", "")).strip()
    if event == "stage_started":
        return {
            "F-01": "설정, 데이터 목록, 이전 실험 기록을 읽는 중",
            "F-02": "LLM으로 첫 실험 계획 1개를 만드는 중",
            "F-03": "계획을 코드 변경으로 변환하는 중",
            "F-04": "로컬에서 테스트/학습/예측 명령을 실행하는 중",
            "F-06": "점수와 산출물 경로를 저장하는 중",
        }.get(stage, message or "진행 중")
    if event == "stage_completed":
        return _stage_completion_summary_ko(stage, details, message)
    if event == "cycle_completed":
        return "1회 실험 사이클이 완료되었습니다."
    if event in {"stage_failed", "cycle_blocked"}:
        next_action = row.get("next_action")
        suffix = f" 다음 조치: {next_action}" if next_action else ""
        return f"{message or '진행이 중단되었습니다.'}{suffix}"
    return message or event


def _stage_completion_summary_ko(stage: str, details: dict[str, Any], message: str) -> str:
    if stage == "F-01":
        metric = details.get("metric", "-")
        objective = details.get("objective", "-")
        platform = details.get("platform", "-")
        return f"준비 완료: platform={platform}, metric={metric}, objective={objective}"
    if stage == "F-02":
        usage = details.get("token_usage") or {}
        token_text = ""
        if isinstance(usage, dict) and usage:
            total = usage.get("total_tokens")
            token_text = f", tokens={total}" if total is not None else ""
        return f"계획 생성 완료{token_text}"
    if stage == "F-03":
        changed = details.get("changed_files") or []
        if not changed:
            return "코드 변경 완료: 변경 파일 없음"
        preview = ", ".join(str(item) for item in changed[:3])
        more = f" 외 {len(changed) - 3}개" if len(changed) > 3 else ""
        return f"코드 변경 완료: {preview}{more}"
    if stage == "F-04":
        commands = [str(item) for item in details.get("commands", []) if item]
        return "로컬 실행 완료" + (f": {', '.join(commands)} 성공" if commands else "")
    if stage == "F-06":
        score = details.get("local_score")
        metric = details.get("metric")
        user_view = details.get("user_view")
        parts = ["결과 저장 완료"]
        if metric or score is not None:
            parts.append(f"{metric or 'score'}={score}")
        if user_view:
            parts.append(f"확인 폴더={user_view}")
        return ", ".join(parts)
    return message or "완료"


def render_demo_agent_watch(competition: str, trial_id: str, *, event_limit: int = 8) -> str:
    status = read_demo_agent_status(competition, trial_id)
    events = read_demo_agent_events(competition, trial_id, limit=event_limit)
    lines = [
        f"데모 에이전트 상태: {competition} / {trial_id}",
        f"- 전체 상태: {_status_label_ko(str(status.get('status', '-')))}",
        f"- 현재 단계: {_stage_name_ko(str(status.get('current_stage', '-')))}",
        f"- 진행률: {status.get('progress', 0)}/{status.get('total_steps', 5)}",
        f"- 다음 행동: {status.get('next_action')}",
        f"- 갱신 시각: {status.get('updated_at', '-')}",
        f"- 상태 파일: {status.get('status_path')}",
        "",
        "최근 진행 기록:",
    ]
    if events:
        lines.extend(f"- {format_demo_event(event)}" for event in events)
    else:
        lines.append("- 아직 기록된 이벤트가 없습니다.")
    return "\n".join(lines)


def format_demo_event(row: dict[str, Any]) -> str:
    event = str(row.get("event", "event"))
    stage = str(row.get("current_stage", "-"))
    status = str(row.get("status", "-"))
    progress = f"{row.get('progress', 0)}/{row.get('total_steps', 5)}"
    if event == "stage_started":
        return f"[진행] {progress} {_stage_name_ko(stage)}\n      진행 중: {_event_summary_ko(row)}"
    if event == "stage_completed":
        return f"      [완료] {_event_summary_ko(row)}"
    if event == "cycle_completed":
        return f"\n{_event_summary_ko(row)}"
    if event in {"stage_failed", "cycle_blocked"} or status in {"blocked", "failed"}:
        return f"      [중단] {_event_summary_ko(row)}"
    return f"[상태] {progress} {_stage_name_ko(stage)} - {_event_summary_ko(row)}"


def render_demo_cycle_cli_summary(result: dict[str, Any]) -> str:
    competition = result.get("competition", "-")
    trial_id = result.get("trial_id", "-")
    status = str(result.get("status", "-"))
    record = result.get("record") if isinstance(result.get("record"), dict) else {}
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    code_writer = result.get("code_writer") if isinstance(result.get("code_writer"), dict) else {}
    submission_manifest = (
        result.get("submission_manifest") if isinstance(result.get("submission_manifest"), dict) else {}
    )
    metric = record.get("metric") or (result.get("context") or {}).get("metric")
    score = record.get("local_score")
    changed_files = code_writer.get("changed_files") if isinstance(code_writer, dict) else []
    user_view = f"runs/{competition}/{trial_id}"
    lines = [
        "",
        "실험 요약",
        f"- 실험: {competition} / {trial_id}",
        f"- 상태: {_status_label_ko(status)}",
    ]
    if plan.get("plan_title"):
        lines.append(f"- 계획: {plan['plan_title']}")
    if changed_files:
        preview = ", ".join(str(item) for item in changed_files[:3])
        more = f" 외 {len(changed_files) - 3}개" if len(changed_files) > 3 else ""
        lines.append(f"- 코드 변경: {len(changed_files)}개 파일 ({preview}{more})")
    if score is not None:
        lines.append(f"- 점수: {metric or 'score'} = {score}")
    if submission_manifest:
        lines.append(
            "- 제출 준비: "
            f"{submission_manifest.get('status')} "
            f"({submission_manifest.get('submission_file')})"
        )
    if status in {"completed", "planned"}:
        lines.append(f"- 확인 폴더: {user_view}")
    next_action = result.get("next_action")
    if next_action:
        lines.append(f"- 다음 조치: {next_action}")
    return "\n".join(lines)


def _stage_name_ko(stage: str) -> str:
    return {
        "F-01": "대회와 데이터 정보 확인",
        "F-02": "첫 실험 계획 생성",
        "F-03": "파이프라인 코드 작성",
        "F-04": "로컬 실행",
        "F-06": "결과 저장",
        "P-01": "제출 준비",
        "done": "1회 실험 종료",
    }.get(stage, stage or "-")


def _status_label_ko(status: str) -> str:
    return {
        "running": "진행 중",
        "completed": "완료",
        "planned": "실행 대기",
        "blocked": "중단됨",
        "failed": "실패",
        "missing": "상태 파일 없음",
        "invalid": "상태 파일 오류",
    }.get(status, status)


def _event_summary_ko(row: dict[str, Any]) -> str:
    event = str(row.get("event", "event"))
    stage = str(row.get("current_stage", "-"))
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    message = str(row.get("message", "")).strip()
    if event == "stage_started":
        return {
            "F-01": "설정, 데이터 목록, 이전 실험 기록을 읽고 있습니다.",
            "F-02": "LLM으로 첫 실험 계획 1개를 만들고 있습니다.",
            "F-03": "계획을 실제 파이프라인 코드 변경으로 바꾸고 있습니다.",
            "F-04": "로컬에서 test -> train -> predict 명령을 실행하고 있습니다.",
            "F-06": "점수와 산출물 경로를 정리하고 있습니다.",
        }.get(stage, message or "진행 중입니다.")
    if event == "stage_completed":
        return _stage_completion_summary_ko(stage, details, message)
    if event == "cycle_completed":
        return "1회 실험 사이클 완료"
    if event in {"stage_failed", "cycle_blocked"}:
        next_action = row.get("next_action")
        suffix = f" 다음 조치: {next_action}" if next_action else ""
        return f"{message or '진행이 중단되었습니다.'}{suffix}"
    return message or event


def _stage_completion_summary_ko(stage: str, details: dict[str, Any], message: str) -> str:
    if stage == "F-01":
        metric = details.get("metric", "-")
        objective = details.get("objective", "-")
        platform = details.get("platform", "-")
        return f"{platform}, {metric} {objective} 문제로 준비되었습니다."
    if stage == "F-02":
        usage = details.get("token_usage") or {}
        token_text = ""
        if isinstance(usage, dict) and usage:
            total = usage.get("total_tokens")
            token_text = f" LLM 사용: {total} tokens." if total is not None else ""
        return f"실험 계획을 생성했습니다.{token_text}"
    if stage == "F-03":
        changed = details.get("changed_files") or []
        if not changed:
            return "코드 변경을 확인했습니다. 변경 파일은 없습니다."
        preview = ", ".join(str(item) for item in changed[:3])
        more = f" 외 {len(changed) - 3}개" if len(changed) > 3 else ""
        return f"{len(changed)}개 파일을 수정했습니다. 주요 파일: {preview}{more}"
    if stage == "F-04":
        commands = [str(item) for item in details.get("commands", []) if item]
        return "로컬 실행이 완료되었습니다." + (f" 실행: {', '.join(commands)}" if commands else "")
    if stage == "F-06":
        score = details.get("local_score")
        metric = details.get("metric")
        user_view = details.get("user_view")
        parts = ["결과를 저장했습니다."]
        if metric or score is not None:
            parts.append(f"{metric or 'score'}={score}")
        if user_view:
            parts.append(f"확인 폴더={user_view}")
        return ", ".join(parts)
    return message or "완료"


def load_demo_context(competition: str, trial_id: str) -> dict[str, Any]:
    validation = validate_execution_profile(competition)
    profile: dict[str, Any] = {}
    if validation["status"] == "ready":
        profile = load_execution_profile(competition)
    state = load_state(competition)
    inventory = _load_json(competition_dir(competition) / "workspace_inventory.json", default={})
    recent_trials = _load_demo_recent_trials(competition, current_trial_id=trial_id, limit=3)
    decision_context = load_latest_decision_context(competition)
    source_trial_id = _select_demo_source_trial_id(recent_trials, trial_id, decision_context)
    base_summary = write_base_summary(competition, trial_id, source_trial_id)
    competition_docs = _load_demo_competition_docs(competition)
    data_profile = _build_demo_data_profile(competition, profile, state)
    context = {
        "competition": competition,
        "trial_id": trial_id,
        "plan_type": "continuation_delta_plan" if source_trial_id else "initial_pipeline_plan",
        "source_trial_id": source_trial_id,
        "base_summary": base_summary,
        "status": "ready" if validation["status"] == "ready" else "blocked",
        "issues": validation["issues"],
        "metric": state.get("competition", {}).get("metric", "unknown"),
        "objective": state.get("competition", {}).get("objective", "maximize"),
        "platform": profile.get("platform") or state.get("competition", {}).get("platform"),
        "project_root": profile.get("project_root"),
        "commands": profile.get("commands", {}),
        "artifacts": profile.get("artifacts", {}),
        "write_scope": profile.get("write_scope", {}),
        "metrics_contract": profile.get("metrics_contract", {}),
        "artifact_policy": load_policy("artifact_policy"),
        "inventory_summary": _inventory_summary(inventory),
        "data_profile": data_profile,
        "baseline_guardrails": _baseline_guardrails(data_profile),
        "competition_docs": competition_docs,
        "recent_trials": recent_trials,
        "decision_context": decision_context,
        "llm_call": False,
    }
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "demo_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_context.md", render_demo_context(context))
    log_decision(
        competition,
        trial_id,
        decision_type="demo_context_load",
        decision=context["status"],
        reason="Demo F-01 loaded problem context with rule-based file/profile inspection.",
        evidence={
            "metric": context["metric"],
            "objective": context["objective"],
            "profile_status": validation["status"],
            "recent_trial_count": len(recent_trials),
            "competition_doc_count": len([value for value in competition_docs.values() if value]),
            "data_profile_status": data_profile.get("status"),
            "data_files": [item.get("name") for item in data_profile.get("files", [])],
            "recommended_base_trial": decision_context.get("recommended_base_trial"),
            "rejected_axes": decision_context.get("rejected_axes", []),
        },
        next_action="create-demo-experiment-plan" if context["status"] == "ready" else "fix-execution-profile",
    )
    return context


def prepare_workspace_trial_plan(
    competition: str,
    trial_id: str,
    *,
    source_trial_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    allow_api: bool = False,
    trial_llm_calls: int | None = None,
    strategy_calls_today: int | None = None,
    user_insight_override: dict[str, Any] | None = None,
    force_replan: bool = False,
) -> dict[str, Any]:
    """Prepare a runnable trial plan without starting code generation."""

    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_ready_demo_plan(out_dir)
    if existing is not None and not force_replan:
        _write_continuation_context(competition, trial_id, source_trial_id=source_trial_id)
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "planned",
            "plan": existing,
            "resumed_from_existing_artifact": True,
        }

    if force_replan:
        _archive_plan_revision(out_dir)

    context = load_demo_context(competition, trial_id)
    if source_trial_id is not None:
        context["source_trial_id"] = source_trial_id
        context["plan_type"] = "continuation_delta_plan"
        context["base_summary"] = write_base_summary(competition, trial_id, source_trial_id)
    if user_insight_override:
        decision_context = (
            dict(context.get("decision_context") or {})
            if isinstance(context.get("decision_context"), dict)
            else {}
        )
        decision_context.update(
            {
                "user_insight_override": user_insight_override,
                "active_axis": user_insight_override.get("active_axis"),
                "axis_attempt_count": user_insight_override.get("axis_attempt_count"),
                "axis_attempt_limit": user_insight_override.get("axis_attempt_limit"),
                "recommended_base_trial": user_insight_override.get("base_trial_id"),
                "planner_constraints": list(decision_context.get("planner_constraints") or [])
                + list(user_insight_override.get("planner_constraints") or []),
            }
        )
        context["decision_context"] = decision_context
        context["user_insight_override"] = user_insight_override

    model_policy = resolve_demo_model_policy(model_override=model, provider_override=provider)
    context["model_policy"] = model_policy
    context["planning_context_pack"] = _build_planning_context_pack_if_needed(competition, trial_id, context)
    write_text(out_dir / "demo_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_context.md", render_demo_context(context))
    if context.get("status") != "ready":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "blocked_context",
            "issues": list(context.get("issues") or []),
            "context": context,
        }

    plan = create_demo_experiment_plan(
        competition,
        trial_id,
        context,
        model_config=model_policy["experiment_planning"],
        allow_api=allow_api,
        mock_plan_file=None,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
    )
    if plan.get("status") != "ready":
        return {
            "competition": competition,
            "trial_id": trial_id,
            "status": "blocked_planning",
            "issues": list(plan.get("issues") or []),
            "plan": plan,
        }
    if user_insight_override:
        plan["user_insight_override"] = user_insight_override
        plan["plan_revision"] = {
            "reason": "user_insight_before_code",
            "insight_id": user_insight_override.get("insight_id"),
        }
        _write_plan_result(competition, trial_id, plan)
    _write_continuation_context(competition, trial_id, source_trial_id=source_trial_id)
    return {
        "competition": competition,
        "trial_id": trial_id,
        "status": "planned",
        "plan": plan,
        "resumed_from_existing_artifact": False,
    }


def _archive_plan_revision(out_dir: Path) -> None:
    names = [
        "demo_experiment_plan.json",
        "demo_experiment_plan.md",
        "next_experiment.md",
        "delta_plan.json",
        "delta_plan.md",
        "demo_context.json",
        "demo_context.md",
    ]
    existing = [out_dir / name for name in names if (out_dir / name).is_file()]
    if not existing:
        return
    revision_dir = out_dir / "internal" / "plan_revisions" / time.strftime("%Y%m%d_%H%M%S")
    revision_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        write_text(revision_dir / path.name, path.read_text(encoding="utf-8"))


def _build_planning_context_pack_if_needed(competition: str, trial_id: str, context: dict[str, Any]) -> dict[str, Any]:
    policy = _planning_rag_policy(context)
    if not policy["use_rag"]:
        return {
            "task": "experiment_planning",
            "document_count": 0,
            "documents": [],
            "skipped": True,
            "skip_reason": policy["reason"],
            "policy": policy,
        }
    context_pack = _compact_context_pack(
        build_context_pack(
            competition,
            trial_id,
            task="experiment_planning",
            query=(
                "competition overview metric data notes source materials execution profile "
                "competition data card data profile target id columns feature recommendations "
                "previous trial plan metrics pipeline structure result"
            ),
        )
    )
    context_pack["policy"] = policy
    return context_pack


def _should_use_planning_rag(context: dict[str, Any]) -> bool:
    return bool(_planning_rag_policy(context)["use_rag"])


def _planning_rag_policy(context: dict[str, Any]) -> dict[str, Any]:
    return evaluate_rag_policy(
        context,
        task="experiment_planning",
        is_first_trial=not bool(context.get("source_trial_id")),
    )


def create_demo_experiment_plan(
    competition: str,
    trial_id: str,
    context: dict[str, Any],
    *,
    model_config: dict[str, Any],
    allow_api: bool,
    mock_plan_file: str | None,
    trial_llm_calls: int | None,
    strategy_calls_today: int | None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    token_decision = log_llm_decision(
        competition,
        trial_id,
        "experiment_planning",
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
        prompt_summary_path=f"experiments/{competition}/{trial_id}/demo_context.md",
    )
    if token_decision["decision"] != "call_llm":
        return _write_plan_result(
            competition,
            trial_id,
            {
                "status": "blocked",
                "issues": ["token_policy_blocked"],
                "token_decision": token_decision,
                "next_action": "revise-token-policy-or-use-rule-based-plan",
            },
        )

    if mock_plan_file:
        raw_response = json.loads(Path(mock_plan_file).read_text(encoding="utf-8"))
    else:
        if not allow_api:
            return _write_plan_result(
                competition,
                trial_id,
                {
                    "status": "blocked",
                    "issues": ["api_call_not_enabled"],
                    "token_decision": token_decision,
                    "next_action": "provide-mock-plan-file-or-run-with-allow-api",
                },
            )
        client = create_llm_client(str(model_config["provider"]))
        model = str(model_config["model"])
        payload = build_demo_plan_payload(context, model=model)
        write_text(out_dir / "demo_plan_api_request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        try:
            raw_response = client.create_response(payload)
        except RuntimeError as error:
            return _write_plan_result(
                competition,
                trial_id,
                {
                    "status": "blocked",
                    "issues": [f"api_error:{error}"],
                    "token_decision": token_decision,
                    "next_action": "fix-api-configuration-or-use-mock-plan-file",
                },
            )

    write_text(out_dir / "demo_plan_api_response.json", json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n")
    usage = raw_response.get("usage")
    token_usage = (
        log_token_usage(
            competition,
            trial_id,
            provider=provider_log_name(str(model_config["provider"])),
            model=str(model_config["model"]),
            call_type="experiment_planning",
            usage=usage,
            request_id=raw_response.get("id"),
        )
        if isinstance(usage, dict)
        else None
    )
    plan = _normalize_plan(_extract_plan(raw_response), context)
    plan.update(
        {
            "competition": competition,
            "trial_id": trial_id,
            "status": "ready" if not plan.get("issues") else "blocked",
            "token_decision": token_decision,
            "token_usage": token_usage,
            "model_config": model_config,
            "next_action": "prepare-workspace-handoff" if not plan.get("issues") else "revise-demo-plan",
        }
    )
    return _write_plan_result(competition, trial_id, plan)


def build_demo_plan_payload(context: dict[str, Any], *, model: str) -> dict[str, Any]:
    if _is_delta_patch_context(context):
        return build_delta_plan_payload(context, model=model)

    context_pack = context.get("planning_context_pack", {})
    prompt_data_profile = _compact_data_profile_for_prompt(context.get("data_profile", {}))
    prompt_context = _compact_demo_context_for_prompt(context)
    recent_trials = context.get("recent_trials", [])
    source_trial_id = context.get("source_trial_id")
    rag_skipped = bool(isinstance(context_pack, dict) and context_pack.get("skipped"))
    evidence_instruction = (
        [
            "RAG is intentionally skipped for this continuation step. Use decision_context, the source/base trial summary, rejected candidates, and the compact structured context instead.",
            "Do not request broad historical evidence unless user feedback, human review, external sources, axis exhaustion, or unexplained failures require it.",
        ]
        if rag_skipped
        else [
            "Use the RAG context pack as evidence. Prefer concrete details from retrieved documents over generic ML advice.",
        ]
    )
    cycle_instruction = (
        "Create exactly one continuation delta plan for this ML workspace."
        if source_trial_id
        else "Create exactly one first-cycle experiment plan for this ML workspace."
    )
    continuation_instruction = (
        [
            "This is not the first trial. Use previous trial evidence from structured context and the RAG context only when it is provided.",
            f"The source trial is `{source_trial_id}`. Treat its pipeline as the base pipeline.",
            "The source trial may be the best/base trial while decision_context.active_axis comes from a different failed trial.",
            "When active_axis is present and axis_attempt_count is below axis_attempt_limit, apply a new candidate from that same axis on top of the source/base trial.",
            "Use decision_context before long notes: latest_decision_card, active_axis, recommended_base_trial, rejected_axes, rejected_candidates, and planner_constraints are the highest-priority evidence.",
            "If decision_context.active_axis is present, keep that same primary axis and choose a different candidate or parameter variant within the axis.",
            "Do not switch away from active_axis until the decision context explicitly marks that axis as rejected.",
            "If recommended_base_trial is different from the latest completed trial, start from recommended_base_trial and avoid stacking on the rejected latest change.",
            "Do not keep a rejected axis unless the new primary axis is explicitly rollback/validation-reliability.",
            "Do not repeat any rejected_candidate; use it as evidence for what was already tried inside the active axis.",
            "Do not redesign the whole pipeline unless the evidence explicitly says the current pipeline is invalid.",
            "Write the plan as a delta from the source trial: keep most of the pipeline fixed and change exactly one primary improvement axis.",
            "Explicitly state the previous best/local score you used, what changed in the previous pipeline, and exactly one improvement axis for this trial.",
            "Avoid generic wording such as 'improve performance'; name the concrete split, preprocessing, feature, model, or parameter change.",
            "Do not call the plan first-cycle or baseline unless the evidence truly shows no completed prior trial.",
            "Return continuation fields when applicable: primary_change_axis, keep_unchanged, change_details, affected_stages, required_code_symbols, expected_metadata_changes, code_change_targets, success_criteria, failure_decision.",
            "The plan should be concise. Do not repeat the full pipeline blueprint except for the parts affected by the chosen change axis.",
        ]
        if source_trial_id
        else [
            "Because this is the first trial, prioritize a simple reproducible baseline and submission-format verification.",
            "For the first trial, the compact data_profile and baseline_guardrails are mandatory evidence. Do not fall back to a generic CSV schema-discovery baseline when concrete columns are available.",
            "For tabular supervised classification, prefer a stable baseline such as imputation + low-cardinality categorical encoding + logistic-regression/linear classifier, or a similarly conservative tree baseline.",
            "Avoid Gaussian Naive Bayes as the first mixed-tabular baseline unless the data profile strongly supports independent numeric Gaussian features.",
            "Do not include high-cardinality free-text/id-like columns in the first baseline merely because they exist. Exclude or defer them unless you explicitly engineer a safe derived feature.",
            "For initial_pipeline_plan, these fields are required: plan_type, plan_title, objective, rationale, pipeline_blueprint, code_change_targets, success_criteria, implementation_notes, expected_outputs.",
            "Do not omit pipeline_blueprint, code_change_targets, or success_criteria; they are the coder handoff and next-trial memory.",
        ]
    )
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You plan one practical ML experiment. Return only JSON.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        cycle_instruction,
                        "Do not include leaderboard submission, human review, ensembling, or multi-trial axis switching.",
                        "Model/checkpoint artifacts are optional. Follow artifact_policy: do not persist a trained model by default.",
                        "If a model artifact is needed, justify it using an allowed_when reason such as required_for_separate_predict_command.",
                        "Always preserve metrics, submission output, code snapshot, and pipeline summary as the primary trial memory.",
                        "Return JSON with plan_type, plan_title, objective, rationale, implementation_notes, expected_outputs.",
                        "Keep the JSON concise but concrete: do not restate full evidence, avoid generic performance wording, and prefer short bullets with actual columns, split values, preprocessing, model, and output files.",
                        "For pipeline_blueprint/change_details, include only the planned pipeline facts needed by the coder and the next trial memory; avoid essay-style explanations.",
                        *evidence_instruction,
                        "Treat the compact data_profile and baseline_guardrails below as higher priority than broad RAG summaries for first-trial baseline design.",
                        "Mention the applied data, split, preprocessing, model, and output assumptions when they are available in evidence.",
                        *continuation_instruction,
                        "",
                        "## Compact Data Profile",
                        "",
                        json.dumps(prompt_data_profile, ensure_ascii=False, indent=2),
                        "",
                        "## Baseline Guardrails",
                        "",
                        json.dumps(context.get("baseline_guardrails", {}), ensure_ascii=False, indent=2),
                        "",
                        "## RAG Context Pack",
                        "",
                        _planning_context_pack_prompt_summary(context_pack) if context_pack else "No RAG context pack available.",
                        "",
                        "## Structured Demo Context",
                        "",
                        json.dumps(prompt_context, ensure_ascii=False, indent=2),
                    ]
                ),
            },
        ],
    }


def build_delta_plan_payload(context: dict[str, Any], *, model: str) -> dict[str, Any]:
    decision_context = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
    compact_context = _compact_delta_context_for_prompt(context)
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You choose one small ML pipeline patch. Return only compact JSON.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Create exactly one delta_patch plan for the next trial.",
                        "RAG is intentionally skipped for this active-axis refinement.",
                        "This is not a new experiment design. Do not rewrite the baseline, full pipeline, or long rationale.",
                        "Use the source/base trial as the code base.",
                        "Keep the active_axis because its attempt_count is below attempt_limit.",
                        "Do not switch away from active_axis before the attempt limit is reached.",
                        "keep that same primary axis; do not switch away from active_axis before the attempt limit is reached.",
                        "Choose one new candidate or parameter variant inside active_axis; choose a different candidate or parameter variant within the axis.",
                        "Do not repeat rejected candidates.",
                        "Do not repeat any rejected_candidate listed in the context.",
                        "Change exactly one primary axis and keep split/model/preprocessing unchanged unless the active axis explicitly targets them.",
                        "Return JSON with exactly these top-level fields:",
                        "plan_type, plan_title, source_trial_id, primary_change_axis, candidate, do_not_repeat, keep_unchanged, change_details, affected_stages, required_code_symbols, expected_metadata_changes, code_change_targets, success_criteria, failure_decision, expected_outputs.",
                        "Keep every list short. candidate must include name, description, and implementation_hint.",
                        "",
                        "## Delta Context",
                        "",
                        json.dumps(compact_context, ensure_ascii=False, indent=2),
                        "",
                        "## Active Axis State",
                        "",
                        json.dumps(
                            {
                                "active_axis": decision_context.get("active_axis"),
                                "axis_attempt_count": decision_context.get("axis_attempt_count"),
                                "axis_attempt_limit": decision_context.get("axis_attempt_limit"),
                                "recommended_base_trial": decision_context.get("recommended_base_trial"),
                                "active_axis_rejected_candidates": decision_context.get("active_axis_rejected_candidates", []),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ]
                ),
            },
        ],
    }


def _compact_delta_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    decision_context = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
    latest = decision_context.get("latest_decision_card") if isinstance(decision_context.get("latest_decision_card"), dict) else {}
    best = decision_context.get("best_decision_card") if isinstance(decision_context.get("best_decision_card"), dict) else {}
    data_profile = context.get("data_profile") if isinstance(context.get("data_profile"), dict) else {}
    return {
        "competition": context.get("competition"),
        "trial_id": context.get("trial_id"),
        "plan_type": "delta_patch",
        "source_trial_id": context.get("source_trial_id"),
        "metric": context.get("metric"),
        "objective": context.get("objective"),
        "target_column": data_profile.get("target_column"),
        "id_column": data_profile.get("id_column"),
        "base_summary": context.get("base_summary", {}),
        "base_trial": {
            "trial_id": context.get("source_trial_id"),
            "local_score": best.get("local_score"),
            "change_axis": best.get("change_axis"),
        },
        "latest_trial": {
            "trial_id": latest.get("trial_id"),
            "decision": latest.get("decision"),
            "local_score": latest.get("local_score"),
            "change_axis": latest.get("change_axis"),
            "candidate_label": latest.get("candidate_label"),
        },
        "active_axis": decision_context.get("active_axis"),
        "axis_attempt_count": decision_context.get("axis_attempt_count"),
        "axis_attempt_limit": decision_context.get("axis_attempt_limit"),
        "rejected_axes": decision_context.get("rejected_axes", []),
        "rejected_candidates_by_axis": decision_context.get("rejected_candidates_by_axis", {}),
        "active_axis_rejected_candidates": decision_context.get("active_axis_rejected_candidates", []),
        "planner_constraints": decision_context.get("planner_constraints", []),
    }


def _is_delta_patch_context(context: dict[str, Any]) -> bool:
    if not context.get("source_trial_id"):
        return False
    decision_context = context.get("decision_context") if isinstance(context.get("decision_context"), dict) else {}
    active_axis = str(decision_context.get("active_axis") or "").strip()
    try:
        attempt_count = int(decision_context.get("axis_attempt_count") or 0)
    except (TypeError, ValueError):
        attempt_count = 0
    try:
        attempt_limit = int(decision_context.get("axis_attempt_limit") or 3)
    except (TypeError, ValueError):
        attempt_limit = 3
    return bool(active_axis and attempt_count < attempt_limit)


def record_demo_cycle_result(
    competition: str,
    trial_id: str,
    *,
    context: dict[str, Any],
    plan: dict[str, Any],
    code_writer: dict[str, Any],
    workspace_run: dict[str, Any],
    metrics_collection: dict[str, Any],
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    record = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "recorded",
        "metric": metrics_collection.get("metric"),
        "objective": metrics_collection.get("objective"),
        "local_score": metrics_collection.get("cv_score"),
        "score_source": metrics_collection.get("score_source"),
        "plan_path": f"experiments/{competition}/{trial_id}/next_experiment.md",
        "code_result_path": f"experiments/{competition}/{trial_id}/workspace_coding_result.json",
        "changed_files": code_writer.get("changed_files", []),
        "workspace_run_path": f"experiments/{competition}/{trial_id}/workspace_run.json",
        "metrics_path": f"experiments/{competition}/{trial_id}/metrics.json",
        "log_paths": [
            item.get("log_path")
            for item in workspace_run.get("command_results", [])
            if item.get("log_path")
        ],
        "project_root": context.get("project_root"),
        "plan_title": plan.get("plan_title"),
        "rationale": plan.get("rationale"),
    }
    write_text(out_dir / "demo_cycle_record.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_cycle_record.md", render_demo_cycle_record(record))
    decision_card = write_trial_decision_card(
        competition,
        trial_id,
        plan=plan,
        metrics=metrics_collection,
    )
    record["decision"] = decision_card.get("decision")
    record["decision_card_path"] = f"experiments/{competition}/{trial_id}/decision_card.json"
    memory_card = write_trial_memory_card(
        competition,
        trial_id,
        plan=plan,
        metrics=metrics_collection,
        decision_card=decision_card,
    )
    record["trial_memory_card_path"] = f"experiments/{competition}/{trial_id}/trial_memory_card.json"
    record["memory_card_summary"] = {
        "change_axis": memory_card.get("change_axis"),
        "decision": memory_card.get("decision"),
        "recommended_base_trial": memory_card.get("recommended_base_trial"),
    }
    write_text(out_dir / "demo_cycle_record.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_cycle_record.md", render_demo_cycle_record(record))
    index_path = competition_memory_dir(competition) / "demo_trial_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_decision(
        competition,
        trial_id,
        decision_type="demo_result_record",
        decision="recorded",
        reason="Demo F-06 recorded one-cycle local result without SQLite.",
        evidence={
            "local_score": record["local_score"],
            "metric": record["metric"],
            "changed_files": record["changed_files"],
        },
        next_action="inspect-demo-cycle-artifacts",
    )
    return record


def render_demo_context(context: dict[str, Any]) -> str:
    lines = [
        f"# {context['competition']} / {context['trial_id']} Demo Context",
        "",
        f"- status: {context['status']}",
        f"- platform: {context.get('platform')}",
        f"- metric: {context.get('metric')}",
        f"- objective: {context.get('objective')}",
        f"- project_root: {context.get('project_root')}",
        "",
        "## Model Policy",
        "",
    ]
    model_policy = context.get("model_policy", {})
    if model_policy:
        for call_type in ("experiment_planning", "workspace_code_writing"):
            item = model_policy.get(call_type, {})
            lines.append(
                f"- {call_type}: {item.get('provider')} / {item.get('model')} ({item.get('tier')})"
            )
        low_cost = model_policy.get("low_cost", {})
        lines.append(f"- low_cost: {low_cost.get('provider')} / {low_cost.get('model')}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
        "## Commands",
        "",
        ]
    )
    for stage, commands in context.get("commands", {}).items():
        lines.extend(f"- {stage}: `{command}`" for command in commands)
    lines.extend(["", "## Artifacts", ""])
    for kind, values in context.get("artifacts", {}).items():
        lines.extend(f"- {kind}: {value}" for value in values)
    artifact_policy = context.get("artifact_policy", {})
    if artifact_policy:
        lines.extend(
            [
                "",
                "## Artifact Policy",
                "",
                "- trained model default save: "
                + str(artifact_policy.get("save_model", {}).get("default"))
                if isinstance(artifact_policy.get("save_model"), dict)
                else "- trained model default save: unknown",
                "- primary memory: metrics, submission, code snapshot, pipeline summary",
            ]
        )
    context_pack = context.get("planning_context_pack", {})
    if context_pack:
        lines.extend(
            [
                "",
                "## RAG Context Pack",
                "",
                f"- task: {context_pack.get('task')}",
                f"- documents: {context_pack.get('document_count')}",
                f"- context_pack: `{context_pack.get('context_pack_md_file')}`",
                f"- manifest: `{context_pack.get('retrieval_manifest_file')}`",
            ]
        )
    lines.extend(["", "## Competition Documents", ""])
    competition_docs = context.get("competition_docs", {})
    if competition_docs:
        for name, content in competition_docs.items():
            lines.extend([f"### {name}", "", content or "Not provided.", ""])
    else:
        lines.append("- None")
    lines.extend(["", "## Recent Trials", ""])
    recent_trials = context.get("recent_trials", [])
    if recent_trials:
        lines.extend(f"- {row.get('trial_id')}: {row.get('cv_score')}" for row in recent_trials)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _compact_context_pack(context_pack: dict[str, Any]) -> dict[str, Any]:
    documents = []
    budget = context_pack.get("budget") if isinstance(context_pack.get("budget"), dict) else {}
    max_chars = 900
    if budget.get("max_chars_per_document") is not None:
        try:
            max_chars = min(max_chars, int(budget["max_chars_per_document"]))
        except (TypeError, ValueError):
            pass
    for doc in context_pack.get("documents", []):
        documents.append(
            {
                "source_path": doc.get("source_path"),
                "source_kind": doc.get("source_kind"),
                "trial_id": doc.get("trial_id"),
                "score": doc.get("score"),
                "text": str(doc.get("text", ""))[:max_chars],
            }
        )
    compact = dict(context_pack)
    compact["documents"] = documents
    return compact


def _planning_context_pack_prompt_summary(context_pack: dict[str, Any]) -> str:
    compact = dict(context_pack)
    compact["documents"] = [
        doc
        for doc in context_pack.get("documents", [])
        if doc.get("source_kind") not in {"data_profile", "competition_data_card"}
    ]
    return context_pack_prompt_summary(compact)


def render_demo_plan(plan: dict[str, Any]) -> str:
    plan_type = plan.get("plan_type")
    lines = [
        f"# {plan.get('trial_id')} Demo Experiment Plan",
        "",
        f"- status: {plan.get('status')}",
        f"- plan_type: {plan_type}",
        f"- source_trial_id: {plan.get('source_trial_id')}",
        f"- title: {plan.get('plan_title')}",
        f"- next_action: {plan.get('next_action')}",
        "",
        "## Objective",
        "",
        plan.get("objective", ""),
        "",
        "## Rationale",
        "",
        plan.get("rationale", ""),
        "",
    ]
    if plan_type in {"continuation_delta_plan", "delta_patch"}:
        lines.extend(
            [
                "## Primary Change Axis",
                "",
                str(plan.get("primary_change_axis") or "Not specified"),
                "",
                "## Keep Unchanged",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("keep_unchanged")) or ["None"])
        lines.extend(["", "## Change Details", ""])
        lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("change_details")) or ["None"])
    else:
        lines.extend(["## Pipeline Blueprint", ""])
        lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("pipeline_blueprint")) or ["None"])
    lines.extend(
        [
            "",
            "## Code Change Targets",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("code_change_targets")) or ["None"])
    lines.extend(
        [
            "",
            "## Implementation Notes",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("implementation_notes")) or ["None"])
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("success_criteria")) or ["None"])
    if plan_type in {"continuation_delta_plan", "delta_patch"}:
        lines.extend(["", "## Failure Decision", ""])
        lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("failure_decision")) or ["None"])
    lines.extend(["", "## Expected Outputs", ""])
    lines.extend(f"- {item}" for item in _normalize_plan_items(plan.get("expected_outputs")) or ["None"])
    lines.extend(["", "## Issues", ""])
    lines.extend(f"- {item}" for item in plan.get("issues", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def render_demo_cycle(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['competition']} / {result['trial_id']} Demo One Cycle",
        "",
        f"- status: {result['status']}",
        f"- run_now: {result['run_now']}",
        f"- next_action: {result['next_action']}",
        "",
        "## Steps",
        "",
    ]
    lines.extend(f"- {step}" for step in result.get("steps", []))
    lines.extend(["", "## Key Artifacts", ""])
    for path in [
        "demo_context.md",
        "next_experiment.md",
        "workspace_coding_result.md",
        "workspace_run.md",
        "metrics_collection.md",
        "demo_cycle_record.md",
    ]:
        lines.append(f"- experiments/{result['competition']}/{result['trial_id']}/{path}")
    lines.extend(["", "## Issues", ""])
    lines.extend(f"- {item}" for item in result.get("issues", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def render_demo_cycle_record(record: dict[str, Any]) -> str:
    lines = [
        f"# {record['competition']} / {record['trial_id']} Demo Result Record",
        "",
        f"- status: {record['status']}",
        f"- metric: {record.get('metric')}",
        f"- objective: {record.get('objective')}",
        f"- local_score: {record.get('local_score')}",
        f"- score_source: {record.get('score_source')}",
        f"- project_root: {record.get('project_root')}",
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- {item}" for item in record.get("changed_files", []) or ["None"])
    lines.extend(["", "## Logs", ""])
    lines.extend(f"- {item}" for item in record.get("log_paths", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _write_plan_result(competition: str, trial_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "demo_experiment_plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_experiment_plan.md", render_demo_plan(plan))
    if plan.get("status") == "ready":
        write_text(out_dir / "next_experiment.md", render_demo_plan(plan))
        if plan.get("source_trial_id"):
            write_text(out_dir / "delta_plan.json", json.dumps(_delta_plan_contract(plan), ensure_ascii=False, indent=2) + "\n")
            write_text(out_dir / "delta_plan.md", render_delta_plan_contract(plan))
    log_decision(
        competition,
        trial_id,
        decision_type="demo_experiment_planning",
        decision=plan["status"],
        reason="Demo F-02 produced one experiment plan for the first cycle.",
        evidence={
            "plan_title": plan.get("plan_title"),
            "issues": plan.get("issues", []),
            "token_decision": plan.get("token_decision", {}),
            "token_usage": plan.get("token_usage"),
        },
        next_action=plan["next_action"],
    )
    return plan


def _delta_plan_contract(plan: dict[str, Any]) -> dict[str, Any]:
    plan = enrich_delta_plan(plan)
    candidate = plan.get("candidate") if isinstance(plan.get("candidate"), dict) else {}
    return {
        "schema_version": "1.0",
        "plan_type": "delta_patch",
        "trial_id": plan.get("trial_id"),
        "source_trial_id": plan.get("source_trial_id"),
        "primary_change_axis": plan.get("primary_change_axis"),
        "candidate": {
            "name": candidate.get("name") or plan.get("plan_title"),
            "description": candidate.get("description") or "",
            "implementation_hint": candidate.get("implementation_hint") or "",
        },
        "do_not_repeat": _normalize_plan_items(plan.get("do_not_repeat")),
        "keep_unchanged": _normalize_plan_items(plan.get("keep_unchanged")),
        "change_details": _normalize_plan_items(plan.get("change_details")),
        "code_change_targets": _normalize_plan_items(plan.get("code_change_targets")),
        "affected_stages": _normalize_plan_items(plan.get("affected_stages")),
        "required_code_symbols": _normalize_plan_items(plan.get("required_code_symbols")),
        "expected_metadata_changes": _normalize_plan_items(plan.get("expected_metadata_changes")),
        "success_criteria": _normalize_plan_items(plan.get("success_criteria")),
        "failure_decision": _normalize_plan_items(plan.get("failure_decision")),
        "expected_outputs": _normalize_plan_items(plan.get("expected_outputs")),
    }


def render_delta_plan_contract(plan: dict[str, Any]) -> str:
    contract = _delta_plan_contract(plan)
    candidate = contract["candidate"]
    lines = [
        f"# {contract.get('trial_id')} Delta Patch Plan",
        "",
        f"- base/source trial: {contract.get('source_trial_id')}",
        f"- active axis: {contract.get('primary_change_axis')}",
        f"- candidate: {candidate.get('name')}",
        f"- description: {candidate.get('description')}",
        f"- implementation_hint: {candidate.get('implementation_hint')}",
        "",
        "## Do Not Repeat",
        "",
    ]
    lines.extend(f"- {item}" for item in contract.get("do_not_repeat", []) or ["None"])
    lines.extend(["", "## Change Details", ""])
    lines.extend(f"- {item}" for item in contract.get("change_details", []) or ["None"])
    lines.extend(["", "## Keep Unchanged", ""])
    lines.extend(f"- {item}" for item in contract.get("keep_unchanged", []) or ["None"])
    lines.extend(["", "## Code Change Targets", ""])
    lines.extend(f"- {item}" for item in contract.get("code_change_targets", []) or ["None"])
    lines.extend(["", "## Success Criteria", ""])
    lines.extend(f"- {item}" for item in contract.get("success_criteria", []) or ["None"])
    lines.append("")
    return "\n".join(lines)


def _finish(
    competition: str,
    trial_id: str,
    result: dict[str, Any],
    *,
    reporter: DemoStatusReporter | None = None,
    low_cost_user_summary: bool = False,
    allow_api: bool = False,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "demo_one_cycle.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_one_cycle.md", render_demo_cycle(result))
    artifact_summary = None
    if result.get("status") == "completed":
        artifact_summary = organize_trial_artifacts(
            competition,
            trial_id,
            low_cost_user_summary=low_cost_user_summary,
            allow_api=allow_api,
        )
    if reporter is not None:
        reporter.finish(result["status"], f"Demo one-cycle finished with status={result['status']}.", next_action=result["next_action"])
    log_decision(
        competition,
        trial_id,
        decision_type="demo_one_cycle",
        decision=result["status"],
        reason="Two-week demo one-cycle orchestrator finished.",
        evidence={"steps": result.get("steps", []), "issues": result.get("issues", [])},
        next_action=result["next_action"],
    )
    if artifact_summary is not None:
        result["artifact_summary"] = artifact_summary
    result["state_db_sync"] = sync_trial_state_after_finish(competition, trial_id)
    write_text(out_dir / "demo_one_cycle.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_one_cycle.md", render_demo_cycle(result))
    return result


def _write_continuation_context(competition: str, trial_id: str, source_trial_id: str | None = None) -> None:
    out_dir = trial_dir(competition, trial_id)
    decision_context = load_latest_decision_context(competition)
    source_trial_id = source_trial_id or _select_demo_source_trial_id(
        _load_demo_recent_trials(competition, current_trial_id=trial_id, limit=3),
        trial_id,
        decision_context,
    )
    context = {
        "competition": competition,
        "trial_id": trial_id,
        "source_trial_id": source_trial_id,
        "next_trial_id": trial_id,
        "continuation_mode": "can_continue",
        "pending_human_review": False,
        "demo_scope": "one_cycle_without_F05_F07_submission_or_UI",
        "decision_context": decision_context,
    }
    write_text(out_dir / "continuation_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(
        out_dir / "continuation_context.md",
        "\n".join(
            [
                f"# {trial_id} Continuation Context",
                "",
                f"- source_trial_id: {source_trial_id}",
                "- continuation_mode: can_continue",
                "- pending_human_review: false",
                "- demo_scope: one cycle only",
                "",
            ]
        ),
    )


def _build_demo_data_profile(competition: str, profile: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(profile.get("project_root") or ""))
    data_dir = project_root / "data"
    workspace_config = _load_json(project_root / "workspace_config.json", default={}) if project_root else {}
    competition_state = state.get("competition", {}) if isinstance(state, dict) else {}
    target_column = profile.get("target_column") or workspace_config.get("target_column") or competition_state.get("target_column")
    id_column = profile.get("id_column") or workspace_config.get("id_column") or competition_state.get("id_column")
    required_files = list(
        profile.get("required_data_files", [])
        or workspace_config.get("required_data_files", [])
        or competition_state.get("required_data_files", [])
        or []
    )
    files: list[dict[str, Any]] = []
    if data_dir.is_dir():
        for path in sorted(data_dir.glob("*.csv")):
            files.append(_profile_demo_csv(path, data_dir, target_column=target_column, id_column=id_column))
    train = next((item for item in files if item.get("role") == "train"), None)
    test = next((item for item in files if item.get("role") == "test"), None)
    sample = next((item for item in files if item.get("role") == "sample_submission"), None)
    data_profile = {
        "schema_version": "1.0",
        "competition": competition,
        "status": "ready" if files else "missing_local_csv",
        "project_root": str(project_root) if project_root else None,
        "data_dir": str(data_dir) if data_dir else None,
        "required_data_files": required_files,
        "target_column": target_column,
        "id_column": id_column,
        "submission_prediction_column": workspace_config.get("submission_prediction_column") or target_column,
        "validation_size": workspace_config.get("validation_size"),
        "random_seed": workspace_config.get("random_seed"),
        "task_type": _infer_demo_task_type(train, target_column),
        "files": files,
        "train_file": train.get("name") if train else None,
        "test_file": test.get("name") if test else None,
        "sample_submission_file": sample.get("name") if sample else None,
        "baseline_recommendation": _demo_baseline_recommendation(train, target_column=target_column, id_column=id_column),
    }
    root = competition_dir(competition)
    write_text(root / "data_profile.json", json.dumps(data_profile, ensure_ascii=False, indent=2) + "\n")
    write_text(root / "data_profile.md", _render_demo_data_profile(data_profile))
    write_text(root / "competition_data_card.json", json.dumps(data_profile, ensure_ascii=False, indent=2) + "\n")
    write_text(root / "competition_data_card.md", _render_demo_data_profile(data_profile))
    return data_profile


def _profile_demo_csv(path: Path, data_dir: Path, *, target_column: str | None, id_column: str | None) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        for index, row in enumerate(reader):
            if index >= 80:
                break
            rows.append(dict(row))
    role = _infer_demo_file_role(path.name, columns, target_column)
    return {
        "name": path.relative_to(data_dir).as_posix(),
        "role": role,
        "format": path.suffix.lower().lstrip("."),
        "columns": columns,
        "rows_sampled": len(rows),
        "column_profiles": [
            _profile_demo_column(column, [row.get(column, "") for row in rows], target_column=target_column, id_column=id_column)
            for column in columns
        ],
    }


def _profile_demo_column(column: str, values: list[str], *, target_column: str | None, id_column: str | None) -> dict[str, Any]:
    non_missing = [value for value in values if value not in {"", None, "NA", "N/A", "nan", "NaN", "null", "None"}]
    unique_values = sorted({str(value) for value in non_missing})
    inferred_type = _infer_demo_column_type(non_missing)
    unique_count = len(unique_values)
    sample_values = unique_values[:8]
    role = "feature"
    if target_column and column == target_column:
        role = "target"
    elif id_column and column == id_column:
        role = "id"
    lower = column.lower()
    looks_text_like = inferred_type == "categorical_or_text" and (
        unique_count > 20 or any(token in lower for token in ["name", "ticket", "text", "description", "comment"])
    )
    looks_sparse = len(non_missing) < max(3, int(len(values) * 0.5)) if values else False
    return {
        "name": column,
        "role": role,
        "type": inferred_type,
        "missing_in_sample": len(values) - len(non_missing),
        "unique_in_sample": unique_count,
        "sample_values": sample_values,
        "baseline_use": _baseline_column_use(role, inferred_type, unique_count, looks_text_like, looks_sparse),
    }


def _infer_demo_file_role(name: str, columns: list[str], target_column: str | None) -> str:
    lowered = name.lower()
    stem = Path(lowered).stem
    if "sample_submission" in lowered or "gender_submission" in lowered:
        return "sample_submission"
    if stem == "train" or (target_column and target_column in columns):
        return "train"
    if stem == "test" or (target_column and target_column not in columns):
        return "test"
    return "unknown"


def _infer_demo_column_type(values: list[str]) -> str:
    if not values:
        return "unknown"
    numeric = 0
    for value in values:
        try:
            float(value)
            numeric += 1
        except (TypeError, ValueError):
            pass
    return "numeric" if numeric / len(values) >= 0.9 else "categorical_or_text"


def _baseline_column_use(role: str, inferred_type: str, unique_count: int, looks_text_like: bool, looks_sparse: bool) -> str:
    if role in {"target", "id"}:
        return "exclude"
    if looks_text_like:
        return "defer_or_engineer"
    if looks_sparse and inferred_type == "categorical_or_text":
        return "defer_sparse_categorical"
    if inferred_type == "numeric":
        return "include_numeric"
    if unique_count <= 20:
        return "include_low_cardinality_categorical"
    return "defer_high_cardinality_categorical"


def _infer_demo_task_type(train: dict[str, Any] | None, target_column: str | None) -> str:
    if not train or not target_column:
        return "unknown"
    target = next((item for item in train.get("column_profiles", []) if item.get("name") == target_column), None)
    if not target:
        return "tabular_unknown_target"
    if target.get("unique_in_sample", 0) <= 20:
        return "tabular_classification"
    return "tabular_regression_or_ranking"


def _demo_baseline_recommendation(
    train: dict[str, Any] | None,
    *,
    target_column: str | None,
    id_column: str | None,
) -> dict[str, Any]:
    columns = train.get("column_profiles", []) if train else []
    include = [item["name"] for item in columns if str(item.get("baseline_use", "")).startswith("include_")]
    defer = [item["name"] for item in columns if str(item.get("baseline_use", "")).startswith("defer")]
    exclude = [item["name"] for item in columns if item.get("baseline_use") == "exclude"]
    return {
        "first_trial_policy": "stable_supervised_tabular_baseline",
        "target_column": target_column,
        "id_column": id_column,
        "include_features_first": include,
        "defer_features_first": defer,
        "exclude_columns": exclude,
        "preferred_model_families": [
            "logistic_regression_or_linear_classifier_for_binary_classification",
            "small_random_forest_or_gradient_boosted_tree_if_available",
        ],
        "avoid_first_trial": [
            "gaussian_naive_bayes_for_mixed_numeric_and_categorical_data",
            "raw_high_cardinality_text_or_identifier_one_hot_features",
            "broad_schema_discovery_when_target_id_and_columns_are_known",
        ],
        "notes": [
            "Use concrete data_profile columns before generic RAG summaries.",
            "Keep the first baseline simple, supervised, deterministic, and submission-format checked.",
        ],
    }


def _baseline_guardrails(data_profile: dict[str, Any]) -> dict[str, Any]:
    recommendation = data_profile.get("baseline_recommendation", {}) if isinstance(data_profile, dict) else {}
    return {
        "priority": "first_trial_quality_over_generic_flexibility",
        "data_profile_is_mandatory": data_profile.get("status") == "ready",
        "recommended_features": recommendation.get("include_features_first", []),
        "deferred_features": recommendation.get("defer_features_first", []),
        "excluded_columns": recommendation.get("exclude_columns", []),
        "preferred_model_families": recommendation.get("preferred_model_families", []),
        "avoid_first_trial": recommendation.get("avoid_first_trial", []),
    }


def _compact_data_profile_for_prompt(data_profile: Any) -> dict[str, Any]:
    if not isinstance(data_profile, dict):
        return {}
    recommendation = data_profile.get("baseline_recommendation", {})
    files = data_profile.get("files", []) if isinstance(data_profile.get("files"), list) else []
    compact_files: list[dict[str, Any]] = []
    train_columns: set[str] = set()
    for file_profile in files:
        if not isinstance(file_profile, dict):
            continue
        role = file_profile.get("role")
        if role not in {"train", "test", "sample_submission"}:
            continue
        column_profiles = [item for item in file_profile.get("column_profiles", []) if isinstance(item, dict)]
        if role == "train":
            columns = []
            for column in column_profiles:
                name = str(column.get("name") or "")
                if name:
                    train_columns.add(name)
                columns.append(
                    {
                        "name": column.get("name"),
                        "role": column.get("role"),
                        "type": column.get("type"),
                        "unique": column.get("unique_in_sample"),
                        "missing": column.get("missing_in_sample"),
                        "baseline_use": column.get("baseline_use"),
                    }
                )
            compact_files.append(
                {
                    "name": file_profile.get("name"),
                    "role": role,
                    "columns": columns,
                }
            )
            continue
        column_names = [column.get("name") for column in column_profiles]
        file_summary: dict[str, Any] = {
            "name": file_profile.get("name"),
            "role": role,
            "columns": column_names,
        }
        if role == "test":
            expected_test_columns = {name for name in train_columns if name != data_profile.get("target_column")}
            file_summary["same_feature_schema_as_train"] = set(str(name) for name in column_names) == expected_test_columns
            file_summary["target_present"] = data_profile.get("target_column") in column_names
        compact_files.append(file_summary)
    return {
        "status": data_profile.get("status"),
        "task_type": data_profile.get("task_type"),
        "target_column": data_profile.get("target_column"),
        "id_column": data_profile.get("id_column"),
        "submission_prediction_column": data_profile.get("submission_prediction_column"),
        "train_file": data_profile.get("train_file"),
        "test_file": data_profile.get("test_file"),
        "validation_size": data_profile.get("validation_size"),
        "random_seed": data_profile.get("random_seed"),
        "include_features_first": recommendation.get("include_features_first", []),
        "defer_features_first": recommendation.get("defer_features_first", []),
        "exclude_columns": recommendation.get("exclude_columns", []),
        "preferred_model_families": recommendation.get("preferred_model_families", []),
        "avoid_first_trial": recommendation.get("avoid_first_trial", []),
        "files": compact_files,
    }


def _compact_demo_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    decision_context = context.get("decision_context", {})
    return {
        "competition": context.get("competition"),
        "trial_id": context.get("trial_id"),
        "plan_type": context.get("plan_type"),
        "source_trial_id": context.get("source_trial_id"),
        "base_summary": context.get("base_summary", {}),
        "metric": context.get("metric"),
        "objective": context.get("objective"),
        "platform": context.get("platform"),
        "project_root": context.get("project_root"),
        "commands": context.get("commands", {}),
        "write_scope": context.get("write_scope", {}),
        "artifact_policy_summary": {
            "save_metrics": context.get("artifact_policy", {}).get("save_metrics"),
            "save_submission": context.get("artifact_policy", {}).get("save_submission"),
            "save_pipeline_summary": context.get("artifact_policy", {}).get("save_pipeline_summary"),
            "save_model_default": context.get("artifact_policy", {}).get("save_model_default"),
        },
        "recent_trials": context.get("recent_trials", []),
        "user_insight_override": context.get("user_insight_override"),
            "decision_context": {
                "decision": decision_context.get("decision"),
                "active_axis": decision_context.get("active_axis"),
                "axis_attempt_count": decision_context.get("axis_attempt_count"),
                "axis_attempt_limit": decision_context.get("axis_attempt_limit"),
                "recommended_base_trial": decision_context.get("recommended_base_trial"),
                "rejected_axes": decision_context.get("rejected_axes", []),
                "rejected_candidates": decision_context.get("rejected_candidates", [])[:12],
                "active_axis_rejected_candidates": decision_context.get("active_axis_rejected_candidates", [])[:5],
                "planner_constraints": decision_context.get("planner_constraints", []),
            },
    }


def _render_demo_data_profile(profile: dict[str, Any]) -> str:
    rec = profile.get("baseline_recommendation", {})
    lines = [
        f"# Competition Data Card: {profile.get('competition')}",
        "",
        f"- status: {profile.get('status')}",
        f"- task_type: {profile.get('task_type')}",
        f"- target_column: {profile.get('target_column')}",
        f"- id_column: {profile.get('id_column')}",
        f"- train_file: {profile.get('train_file')}",
        f"- test_file: {profile.get('test_file')}",
        "",
        "## First Baseline Recommendation",
        "",
        f"- include_features_first: {', '.join(rec.get('include_features_first', [])) or 'None'}",
        f"- defer_features_first: {', '.join(rec.get('defer_features_first', [])) or 'None'}",
        f"- exclude_columns: {', '.join(rec.get('exclude_columns', [])) or 'None'}",
        f"- preferred_model_families: {', '.join(rec.get('preferred_model_families', [])) or 'None'}",
        f"- avoid_first_trial: {', '.join(rec.get('avoid_first_trial', [])) or 'None'}",
        "",
        "## Files",
        "",
    ]
    for file_profile in profile.get("files", []):
        lines.append(f"### {file_profile.get('name')} [{file_profile.get('role')}]")
        for column in file_profile.get("column_profiles", []):
            lines.append(
                "- "
                f"{column.get('name')}: role={column.get('role')}, type={column.get('type')}, "
                f"unique_sample={column.get('unique_in_sample')}, missing_sample={column.get('missing_in_sample')}, "
                f"baseline_use={column.get('baseline_use')}"
            )
        lines.append("")
    return "\n".join(lines)


def _load_demo_competition_docs(competition: str, *, max_chars_per_file: int = 3000) -> dict[str, str]:
    root = competition_dir(competition)
    docs = {}
    for name in ["overview.md", "data_notes.md", "metric.md", "source_materials.md"]:
        content = read_text(root / name, default="").strip()
        if content:
            docs[name] = content[:max_chars_per_file]
    return docs


def _extract_plan(raw_response: dict[str, Any]) -> dict[str, Any]:
    text = raw_response.get("output_text") or _extract_output_text(raw_response)
    if not text:
        return {"issues": ["missing_output_text"]}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "plan_title": "One-cycle experiment",
            "objective": text.strip(),
            "rationale": "Planner returned plain text rather than JSON.",
            "implementation_notes": [text.strip()],
            "expected_outputs": ["metrics.json", "submission.csv"],
            "issues": [],
        }
    return parsed if isinstance(parsed, dict) else {"issues": ["invalid_json_output"]}


def _extract_output_text(raw_response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in raw_response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(part for part in parts if part)


def _normalize_plan(plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(plan)
    plan_type = context.get("plan_type") or ("continuation_delta_plan" if context.get("source_trial_id") else "initial_pipeline_plan")
    normalized.setdefault("schema_version", "1.0")
    normalized.setdefault("plan_type", plan_type)
    normalized.setdefault("source_trial_id", context.get("source_trial_id"))
    normalized.setdefault(
        "plan_title",
        "Initial local baseline experiment" if plan_type == "initial_pipeline_plan" else "Continuation delta experiment",
    )
    normalized.setdefault("objective", f"Create and run one local experiment for {context['competition']}.")
    normalized.setdefault("rationale", "Use the available Execution Profile and local metric artifact to verify the loop.")
    normalized.setdefault("pipeline_blueprint", [])
    normalized.setdefault("primary_change_axis", "")
    normalized.setdefault("keep_unchanged", [])
    normalized.setdefault("change_details", [])
    normalized.setdefault("code_change_targets", [])
    normalized.setdefault("success_criteria", [])
    normalized.setdefault("failure_decision", [])
    normalized.setdefault("implementation_notes", [])
    normalized.setdefault("expected_outputs", ["metrics.json", "submission.csv"])
    normalized.setdefault("candidate", {})
    normalized.setdefault("do_not_repeat", [])
    normalized.setdefault("affected_stages", [])
    normalized.setdefault("required_code_symbols", [])
    normalized.setdefault("expected_metadata_changes", [])
    normalized.setdefault("issues", [])
    normalized["plan_title"] = _normalize_plan_text(normalized.get("plan_title"))
    normalized["objective"] = _normalize_plan_text(normalized.get("objective"))
    normalized["rationale"] = _normalize_plan_text(normalized.get("rationale"))
    normalized["primary_change_axis"] = _normalize_plan_text(normalized.get("primary_change_axis"))
    normalized["pipeline_blueprint"] = _normalize_plan_items(normalized.get("pipeline_blueprint"))
    normalized["keep_unchanged"] = _normalize_plan_items(normalized.get("keep_unchanged"))
    normalized["change_details"] = _normalize_plan_items(normalized.get("change_details"))
    normalized["code_change_targets"] = _normalize_plan_items(normalized.get("code_change_targets"))
    normalized["success_criteria"] = _normalize_plan_items(normalized.get("success_criteria"))
    normalized["failure_decision"] = _normalize_plan_items(normalized.get("failure_decision"))
    normalized["implementation_notes"] = _normalize_plan_items(normalized.get("implementation_notes"))
    normalized["expected_outputs"] = _normalize_plan_items(normalized.get("expected_outputs"))
    normalized["do_not_repeat"] = _normalize_plan_items(normalized.get("do_not_repeat"))
    normalized["affected_stages"] = _normalize_plan_items(normalized.get("affected_stages"))
    normalized["required_code_symbols"] = _normalize_plan_items(normalized.get("required_code_symbols"))
    normalized["expected_metadata_changes"] = _normalize_plan_items(normalized.get("expected_metadata_changes"))
    if not isinstance(normalized.get("candidate"), dict):
        normalized["candidate"] = {"name": _normalize_plan_text(normalized.get("candidate"))}
    else:
        normalized["candidate"] = {
            "name": _normalize_plan_text(normalized["candidate"].get("name")),
            "description": _normalize_plan_text(normalized["candidate"].get("description")),
            "implementation_hint": _normalize_plan_text(normalized["candidate"].get("implementation_hint")),
        }
    if plan_type == "continuation_delta_plan":
        _promote_implementation_notes(normalized)
    if normalized.get("source_trial_id"):
        normalized = enrich_delta_plan(normalized)
    if not isinstance(normalized["issues"], list):
        normalized["issues"] = [str(normalized["issues"])]
    return normalized


def _promote_implementation_notes(plan: dict[str, Any]) -> None:
    promoted = _implementation_note_buckets(plan.get("implementation_notes", []))
    if not plan.get("keep_unchanged") and promoted["keep_unchanged"]:
        plan["keep_unchanged"] = promoted["keep_unchanged"]
    if not plan.get("change_details") and promoted["change_details"]:
        plan["change_details"] = promoted["change_details"]
    if not plan.get("code_change_targets") and promoted["code_change_targets"]:
        plan["code_change_targets"] = promoted["code_change_targets"]


def _implementation_note_buckets(notes: list[str]) -> dict[str, list[str]]:
    buckets = {"keep_unchanged": [], "change_details": [], "code_change_targets": []}
    prefixes = {
        "keep unchanged": "keep_unchanged",
        "change details": "change_details",
        "code change targets": "code_change_targets",
    }
    for note in notes:
        if not isinstance(note, str):
            continue
        label, separator, body = note.partition(":")
        if not separator:
            continue
        bucket = prefixes.get(label.strip().casefold())
        if bucket and body.strip():
            buckets[bucket].append(body.strip())
    return buckets


def _load_demo_recent_trials(competition: str, *, current_trial_id: str, limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(load_recent_trials(competition, limit=limit * 2))
    demo_index = competition_memory_dir(competition) / "demo_trial_index.jsonl"
    if demo_index.exists():
        for line in demo_index.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("competition") == competition:
                rows.append(row)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        trial = str(row.get("trial_id") or "")
        if not trial or trial == current_trial_id or trial in seen:
            continue
        seen.add(trial)
        deduped.append(row)
    return deduped[-limit:]


def _latest_source_trial_id(recent_trials: list[dict[str, Any]], current_trial_id: str) -> str | None:
    for row in reversed(recent_trials):
        trial_id = row.get("trial_id")
        if trial_id and trial_id != current_trial_id:
            return str(trial_id)
    return None


def _latest_demo_source_trial_id(competition: str, trial_id: str) -> str | None:
    recent_trials = _load_demo_recent_trials(competition, current_trial_id=trial_id, limit=3)
    decision_context = load_latest_decision_context(competition)
    return _select_demo_source_trial_id(recent_trials, trial_id, decision_context)


def _select_demo_source_trial_id(
    recent_trials: list[dict[str, Any]],
    current_trial_id: str,
    decision_context: dict[str, Any],
) -> str | None:
    recommended = decision_context.get("recommended_base_trial")
    if recommended and recommended != current_trial_id:
        return str(recommended)
    return _latest_source_trial_id(recent_trials, current_trial_id)


def _normalize_plan_items(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in items:
        normalized.extend(_flatten_plan_item(_coerce_plan_item(item)))
    return [item for item in normalized if item]


def _normalize_plan_text(value: Any) -> str:
    items = _normalize_plan_items(value)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "\n".join(f"- {item}" for item in items)


def _coerce_plan_item(item: Any) -> Any:
    if not isinstance(item, str):
        return item
    stripped = item.strip()
    if not stripped or stripped[0] not in "[{":
        return item
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return item


def _flatten_plan_item(item: Any, *, label: str | None = None) -> list[str]:
    if isinstance(item, dict):
        lines: list[str] = []
        for key, value in item.items():
            key_label = _human_plan_label(str(key))
            lines.extend(_flatten_plan_item(value, label=key_label))
        return lines
    if isinstance(item, list):
        lines = []
        for value in item:
            lines.extend(_flatten_plan_item(value, label=label))
        return lines
    text = str(item).strip()
    if not text:
        return []
    return [f"{label}: {text}" if label else text]


def _human_plan_label(key: str) -> str:
    normalized = "_".join(key.strip().lower().replace("-", " ").split())
    labels = {
        "applied_data_assumptions": "데이터 가정",
        "applied_data": "데이터 적용",
        "split": "검증 분리",
        "validation_split": "검증 분리",
        "preprocessing": "전처리",
        "model": "모델",
        "commands": "실행 명령",
        "prediction_output": "예측/제출 출력",
        "testing": "테스트",
        "artifact_policy": "산출물 정책",
        "workspace_constraints": "작업 범위",
        "write_scope": "수정 범위",
        "metrics": "지표 파일",
        "submission": "제출 파일",
        "code_snapshot": "코드 스냅샷",
        "pipeline_summary": "파이프라인 요약",
        "model_artifact": "모델 파일",
    }
    return labels.get(normalized, key.replace("_", " ").strip().title())


def _workspace_run_issues(workspace_run: dict[str, Any]) -> list[str]:
    if workspace_run.get("failure"):
        return [f"local_failure:{workspace_run['failure'].get('failure_type')}"]
    if workspace_run.get("profile_issues"):
        return list(workspace_run["profile_issues"])
    return [f"workspace_run_status:{workspace_run.get('status')}"]


def _stage_progress(stage: str) -> int:
    return {"F-01": 1, "F-02": 2, "F-03": 3, "F-04": 4, "F-06": 5, "done": 5}.get(stage, 0)


def _inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    files = inventory.get("files", []) if isinstance(inventory, dict) else []
    return {
        "file_count": inventory.get("file_count", len(files)) if isinstance(inventory, dict) else len(files),
        "truncated": bool(inventory.get("truncated")) if isinstance(inventory, dict) else False,
        "data_files": [item.get("path") for item in files if item.get("category") == "data"][:20],
        "code_files": [item.get("path") for item in files if item.get("category") == "code"][:20],
    }


def _load_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default
