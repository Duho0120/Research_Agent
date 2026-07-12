from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .agents.code_writer_adapter import FileResponseClient, create_llm_client, provider_log_name
from .agents.memory import log_decision, log_token_usage
from .agents.policy_gate import log_llm_decision
from .execution_profile import load_execution_profile, validate_execution_profile
from .paths import competition_dir, competition_memory_dir, trial_dir
from .policies import load_policy, select_model_for_call
from .store import load_recent_trials, load_state, now_iso, write_text
from .workspace_code_writer import run_workspace_code_writer
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
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = DemoStatusReporter(competition, trial_id, show_progress=show_progress)
    model_policy = resolve_demo_model_policy(model_override=model, provider_override=provider)

    reporter.start("F-01", "Loading problem context", 1, "Reading Execution Profile, state, inventory, and recent file memory.")
    context = load_demo_context(competition, trial_id)
    context["model_policy"] = model_policy
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
        )
    reporter.complete(
        "F-01",
        f"Context ready: metric={context.get('metric')} objective={context.get('objective')}.",
        next_action="create-demo-experiment-plan",
    )

    reporter.start("F-02", "Planning one experiment", 2, "Calling mock/API LLM for one practical first-cycle plan.")
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
            reporter.complete("F-04", "Local pipeline completed.", next_action="record-demo-result")
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
                status = "completed"
                next_action = "inspect-demo-cycle-artifacts"
                reporter.complete(
                    "F-06",
                    f"Result recorded: local_score={record.get('local_score')}.",
                    next_action=next_action,
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
            "issues": issues,
            "next_action": next_action,
        },
        reporter=reporter,
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
        f"Demo agent status: {competition} / {trial_id}",
        f"status        : {status.get('status')}",
        f"stage         : {status.get('current_stage', '-')}",
        f"progress      : {status.get('progress', 0)}/{status.get('total_steps', 5)}",
        f"message       : {status.get('message', '')}",
        f"next_action   : {status.get('next_action')}",
        f"updated_at    : {status.get('updated_at', '-')}",
        f"pid           : {status.get('pid', '-')}",
        f"status_file   : {status.get('status_path')}",
        "",
        "Recent events:",
    ]
    if events:
        lines.extend(f"- {format_demo_event(event)}" for event in events)
    else:
        lines.append("- No events recorded yet.")
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
    message = str(row.get("message", ""))
    status = str(row.get("status", "-"))
    progress = f"{row.get('progress', 0)}/{row.get('total_steps', 5)}"
    prefix = "[OK]" if event in {"stage_completed", "cycle_completed"} else "[RUN]"
    if event in {"stage_failed", "cycle_blocked"} or status in {"blocked", "failed"}:
        prefix = "[STOP]"
    return f"{prefix} {stage} {progress} {event}: {message}"


def load_demo_context(competition: str, trial_id: str) -> dict[str, Any]:
    validation = validate_execution_profile(competition)
    profile: dict[str, Any] = {}
    if validation["status"] == "ready":
        profile = load_execution_profile(competition)
    state = load_state(competition)
    inventory = _load_json(competition_dir(competition) / "workspace_inventory.json", default={})
    recent_trials = load_recent_trials(competition, limit=3)
    context = {
        "competition": competition,
        "trial_id": trial_id,
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
        "inventory_summary": _inventory_summary(inventory),
        "recent_trials": recent_trials,
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
        },
        next_action="create-demo-experiment-plan" if context["status"] == "ready" else "fix-execution-profile",
    )
    return context


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
                        "Create exactly one first-cycle experiment plan for this ML workspace.",
                        "Do not include leaderboard submission, human review, ensembling, or multi-trial axis switching.",
                        "Return JSON with plan_title, objective, rationale, implementation_notes, expected_outputs.",
                        "",
                        json.dumps(context, ensure_ascii=False, indent=2),
                    ]
                ),
            },
        ],
    }


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
    lines.extend(["", "## Recent Trials", ""])
    recent_trials = context.get("recent_trials", [])
    if recent_trials:
        lines.extend(f"- {row.get('trial_id')}: {row.get('cv_score')}" for row in recent_trials)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def render_demo_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan.get('trial_id')} Demo Experiment Plan",
        "",
        f"- status: {plan.get('status')}",
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
        "## Implementation Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in plan.get("implementation_notes", []) or ["None"])
    lines.extend(["", "## Expected Outputs", ""])
    lines.extend(f"- {item}" for item in plan.get("expected_outputs", []) or ["None"])
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


def _finish(
    competition: str,
    trial_id: str,
    result: dict[str, Any],
    *,
    reporter: DemoStatusReporter | None = None,
) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "demo_one_cycle.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "demo_one_cycle.md", render_demo_cycle(result))
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
    return result


def _write_continuation_context(competition: str, trial_id: str) -> None:
    out_dir = trial_dir(competition, trial_id)
    context = {
        "competition": competition,
        "trial_id": trial_id,
        "continuation_mode": "can_continue",
        "pending_human_review": False,
        "demo_scope": "one_cycle_without_F05_F07_submission_or_UI",
    }
    write_text(out_dir / "continuation_context.json", json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    write_text(
        out_dir / "continuation_context.md",
        "\n".join(
            [
                f"# {trial_id} Continuation Context",
                "",
                "- continuation_mode: can_continue",
                "- pending_human_review: false",
                "- demo_scope: one cycle only",
                "",
            ]
        ),
    )


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
    normalized.setdefault("schema_version", "1.0")
    normalized.setdefault("plan_title", "First local baseline experiment")
    normalized.setdefault("objective", f"Create and run one local experiment for {context['competition']}.")
    normalized.setdefault("rationale", "Use the available Execution Profile and local metric artifact to verify the loop.")
    normalized.setdefault("implementation_notes", [])
    normalized.setdefault("expected_outputs", ["metrics.json", "submission.csv"])
    normalized.setdefault("issues", [])
    if not isinstance(normalized["implementation_notes"], list):
        normalized["implementation_notes"] = [str(normalized["implementation_notes"])]
    if not isinstance(normalized["expected_outputs"], list):
        normalized["expected_outputs"] = [str(normalized["expected_outputs"])]
    if not isinstance(normalized["issues"], list):
        normalized["issues"] = [str(normalized["issues"])]
    return normalized


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
