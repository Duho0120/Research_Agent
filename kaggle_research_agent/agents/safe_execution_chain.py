from __future__ import annotations

import json
from typing import Any

from ..paths import trial_dir
from ..store import write_text
from .code_writer_adapter import CodeWriterClient, run_code_writer
from .memory import log_decision
from .post_validation_executor import run_after_validation


def run_safe_execution_chain(
    competition: str,
    trial_id: str,
    *,
    client: CodeWriterClient | None = None,
    model: str = "gpt-5",
    allow_api: bool = False,
    trial_llm_calls: int = 0,
    strategy_calls_today: int = 0,
    command: str | None = None,
    backend: str = "local",
    run_now: bool = False,
) -> dict[str, Any]:
    code_writer = run_code_writer(
        competition,
        trial_id,
        client=client,
        model=model,
        allow_api=allow_api,
        trial_llm_calls=trial_llm_calls,
        strategy_calls_today=strategy_calls_today,
        run_validation_after=True,
    )
    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "status": _chain_status(code_writer),
        "code_writer_status": code_writer.get("status"),
        "validation_status": code_writer.get("validation_command_status"),
        "execution_status": None,
        "execution_decision": None,
        "run_command": command,
        "backend": backend,
        "run_now": run_now,
        "issues": [],
        "next_action": "revise-code-result",
    }

    if code_writer.get("status") != "accepted":
        result["issues"].append("code_writer_not_accepted")
        return _write_result(competition, trial_id, result)
    if code_writer.get("validation_command_status") != "passed":
        result["issues"].append("validation_commands_not_passed")
        result["next_action"] = "fix-validation"
        return _write_result(competition, trial_id, result)

    execution = run_after_validation(
        competition,
        trial_id,
        command=command,
        backend=backend,
        run_now=run_now,
    )
    result["status"] = execution["status"]
    result["execution_status"] = execution["status"]
    result["execution_decision"] = execution.get("execution_decision")
    result["issues"] = execution.get("issues", [])
    result["next_action"] = execution.get("next_action")
    return _write_result(competition, trial_id, result)


def render_safe_execution_chain(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['trial_id']} Safe Execution Chain",
        "",
        f"- status: {result['status']}",
        f"- code_writer_status: {result.get('code_writer_status')}",
        f"- validation_status: {result.get('validation_status')}",
        f"- execution_status: {result.get('execution_status')}",
        f"- execution_decision: {result.get('execution_decision')}",
        f"- next_action: {result.get('next_action')}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {item}" for item in result["issues"] or ["None"])
    if result.get("run_command"):
        lines.extend(["", "## Run Command", "", "```powershell", result["run_command"], "```"])
    lines.append("")
    return "\n".join(lines)


def _chain_status(code_writer: dict[str, Any]) -> str:
    if code_writer.get("status") != "accepted":
        return "code_writer_blocked"
    if code_writer.get("validation_command_status") != "passed":
        return "validation_failed"
    return "ready_for_execution"


def _write_result(competition: str, trial_id: str, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = trial_dir(competition, trial_id)
    write_text(out_dir / "safe_execution_chain.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "safe_execution_chain.md", render_safe_execution_chain(result))
    log_decision(
        competition,
        trial_id,
        decision_type="safe_execution_chain",
        decision=result["status"],
        reason="Safe execution chain ran code writer, validation commands, and post-validation execution gates.",
        evidence={
            "code_writer_status": result.get("code_writer_status"),
            "validation_status": result.get("validation_status"),
            "execution_status": result.get("execution_status"),
            "execution_decision": result.get("execution_decision"),
            "issues": result.get("issues", []),
        },
        next_action=result.get("next_action") or "",
    )
    return result
