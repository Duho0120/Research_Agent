"""Asking the code writer for model.py, under the contract model.

The legacy path spends 3,476 lines describing five writable files and then
reading the returned text through twelve checks. Almost all of that exists to
compensate for a missing contract: without one, the only way to ask "did
training reach prediction" is to inspect the source, and source inspection
loses -- every check here has at some point been satisfied in letter and
skipped in substance.

Here the contract does that work. The agent writes one file with two
functions, and the verdict comes from running it: execution_core reports which
stage failed and why, and that report is the retry feedback. Two static checks
remain, both about things that would waste a run rather than fake a result --
whether the file parses and defines the two functions, and whether it wrote
outside its scope.

The prompt is assembled once. The legacy path built the audit markdown and the
API payload separately, and a section that appeared in the report but never in
the request cost several trials before anyone noticed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from execution_core.contract import (
    LOADER_MODULE,
    MODEL_MODULE,
    loader_contract_source,
    model_contract_source,
)

from .agents.memory import log_decision
from .code_snapshot import load_trial_code_snapshot
from .execution_profile import contract_data_dir, load_execution_profile
from .paths import trial_dir
from .store import write_text
from .trial_decision import DATA_LOADING_AXIS, SCORING_LOGIC_AXIS


MODEL_FILENAME = f"{MODEL_MODULE}.py"
LOADER_FILENAME = f"{LOADER_MODULE}.py"
REQUIRED_MODEL_FUNCTIONS = ("fit", "predict")


def build_contract_handoff(
    competition: str,
    trial_id: str,
    *,
    plan: str,
    primary_axis: str = "",
    source_trial_id: str | None = None,
    feedback: dict[str, Any] | None = None,
    loader_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the code writer is given, in one place.

    When source_trial_id names a prior trial, its code is restored into the
    workspace and shown in the prompt as the thing to modify. Real incident:
    without this, a continuation plan built on an empty base_summary asked the
    writer to tune a LogisticRegression that never existed -- the model was a
    polynomial extrapolator -- because nothing told it what was actually there.
    """
    profile = load_execution_profile(competition)
    writable = _writable_files(primary_axis)
    project_root = Path(str(profile.get("project_root", "")))
    base_code, base_issues = _restore_base_code(competition, source_trial_id, project_root, writable)
    return {
        "competition": competition,
        "trial_id": trial_id,
        "execution_model": "contract",
        "project_root": str(project_root),
        "primary_change_axis": primary_axis,
        "source_trial_id": source_trial_id,
        "allowed_paths": writable,
        "forbidden_paths": _forbidden_files(writable),
        "base_code_restored": sorted(base_code),
        "base_code_issues": base_issues,
        "prompt": _build_prompt(
            competition,
            plan=plan,
            primary_axis=primary_axis,
            writable=writable,
            loader_sample=loader_sample,
            feedback=feedback,
            base_code=base_code,
        ),
    }


def _restore_base_code(
    competition: str, source_trial_id: str | None, project_root: Path, writable: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Write the base trial's writable files into the workspace and return them.

    Falls back to scratch -- issues recorded, nothing invented -- when there is
    no source trial or its snapshot does not exist (a legacy trial, or one that
    predates this mechanism). A missing base is reported, not silently guessed
    around.
    """
    if not source_trial_id:
        return {}, []
    snapshot = dict(load_trial_code_snapshot(trial_dir(competition, source_trial_id)))
    restored: dict[str, str] = {}
    issues: list[str] = []
    for path in writable:
        content = snapshot.get(path)
        if content is None:
            issues.append(f"no_base_snapshot:{path}:{source_trial_id}")
            continue
        write_text(project_root / path, content)
        restored[path] = content
    return restored, issues


def _writable_files(primary_axis: str) -> list[str]:
    """model.py always; the loader only when the trial is explicitly about it.

    The loader is a competition-level asset -- rewriting it changes what every
    past score meant -- so unlocking it is a decision the axis has to make out
    loud, exactly as in the legacy path.
    """
    if str(primary_axis or "").strip() == DATA_LOADING_AXIS:
        return [MODEL_FILENAME, LOADER_FILENAME]
    return [MODEL_FILENAME]


def _forbidden_files(writable: list[str]) -> list[str]:
    everything = [LOADER_FILENAME, "outputs/metrics.json", "outputs/submission.csv", "data/"]
    return [path for path in everything if path not in writable]


def _build_prompt(
    competition: str,
    *,
    plan: str,
    primary_axis: str,
    writable: list[str],
    loader_sample: dict[str, Any] | None,
    feedback: dict[str, Any] | None,
    base_code: dict[str, str] | None = None,
) -> str:
    base_code = base_code or {}
    opening = (
        "You are modifying one file for an automated ML experiment."
        if base_code
        else "You are writing one file for an automated ML experiment, from scratch."
    )
    sections = [
        opening,
        "",
        f"Writable files, and nothing else: {', '.join(writable)}",
        "",
        "## The contract",
        "",
        model_contract_source(),
    ]
    if LOADER_FILENAME in writable:
        sections += [
            "",
            "This trial's change axis is the data loading itself, so the loader is",
            "unlocked as well. It is otherwise locked -- rewriting it changes what",
            "every previous score meant.",
            "",
            loader_contract_source(),
        ]

    sections += [
        "",
        "## How this will be run and judged",
        "",
        "The framework loads the data, splits off a labelled holdout, calls fit()",
        "on the training part, then calls predict() once per holdout sample with",
        "whatever fit() returned, and scores the result with the competition's",
        "metric. You do not write any of that, and you cannot influence the score",
        "except through the quality of your predictions.",
        "",
        "Rejections you should expect if you cut corners:",
        "- returning the same prediction for every sample stops the trial outright;",
        "- an exception anywhere is reported with the stage it happened in;",
        "- writing to a file outside the list above is rejected before anything runs.",
    ]

    if loader_sample:
        sections += [
            "",
            "## What a sample looks like",
            "",
            "One real training sample, as load_samples() returns it:",
            "",
            "```json",
            json.dumps(loader_sample, ensure_ascii=False, indent=2)[:4000],
            "```",
        ]

    if base_code:
        sections += [
            "",
            "## Current code (this is what you are modifying)",
            "",
            "This is the actual, currently-scoring implementation. Change exactly what",
            "the plan below calls for and leave the rest alone -- do not rewrite it from",
            "scratch, and do not invent behaviour that is not in this source.",
        ]
        for path, content in base_code.items():
            sections += ["", f"### {path}", "", "```python", content.strip(), "```"]

    sections += ["", "## This trial's plan", "", plan.strip()]
    if primary_axis:
        sections += ["", f"Primary change axis: {primary_axis}"]
        if primary_axis == SCORING_LOGIC_AXIS:
            sections += [
                "",
                "This axis is about the validation structure, which the framework owns.",
                "Change it through the profile's holdout settings, not by rewriting how",
                "the score is computed -- the metric is fixed by the competition.",
            ]

    if feedback:
        sections += ["", "## The previous attempt was rejected", ""]
        if feedback.get("failed_stage"):
            sections.append(f"Failed at stage: {feedback['failed_stage']}")
        for issue in feedback.get("issues") or []:
            sections.append(f"- {issue}")
        if feedback.get("error"):
            sections += ["", "```", str(feedback["error"])[:3000], "```"]
        sections += ["", "Do not repeat that attempt."]

    sections += [
        "",
        "## Response format",
        "",
        "Reply with JSON only:",
        '{"files": [{"path": "model.py", "content": "<the complete file>"}],',
        ' "summary": "<one sentence>"}',
    ]
    return "\n".join(sections)


def validate_contract_code(
    project_root: Path | str, handoff: dict[str, Any], changed_files: list[str]
) -> list[str]:
    """The two checks worth making before spending a run.

    Deliberately not a substitute for running the code. These catch what would
    waste the run -- a file that cannot be imported, or a write outside scope --
    and nothing else. Judging whether the model is any good is the trial's job.
    """
    issues: list[str] = []
    allowed = set(handoff.get("allowed_paths") or [])
    for path in changed_files:
        if path not in allowed:
            issues.append(f"wrote_outside_scope:{path}")

    model_path = Path(project_root) / MODEL_FILENAME
    if not model_path.is_file():
        return issues + [f"missing_file:{MODEL_FILENAME}"]
    try:
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return issues + [f"model_does_not_parse:line_{error.lineno}"]

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in REQUIRED_MODEL_FUNCTIONS:
        if name not in defined:
            issues.append(f"model_missing_function:{name}")
    return issues


def load_sample_preview(competition: str) -> dict[str, Any] | None:
    """One real training sample, so the plan is written against actual keys.

    Long sequences are truncated: the model needs the shape, and a full
    time series would crowd out the rest of the prompt.
    """
    profile = load_execution_profile(competition)
    root = Path(str(profile.get("project_root", "")))
    if not (root / LOADER_FILENAME).is_file():
        return None
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_contract_loader_preview", root / LOADER_FILENAME)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_contract_loader_preview"] = module
    try:
        spec.loader.exec_module(module)
        samples = module.load_samples(str(contract_data_dir(profile)), "train")
    except Exception:
        return None
    finally:
        sys.modules.pop("_contract_loader_preview", None)
    return _truncate(samples[0]) if samples else None


def _truncate(sample: dict[str, Any], *, keep: int = 3) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, list) and len(value) > keep:
            preview[key] = [*value[:keep], f"... ({len(value)} values total)"]
        else:
            preview[key] = value
    return preview


def run_contract_code_writer(
    competition: str,
    trial_id: str,
    handoff: dict[str, Any],
    *,
    client: Any | None = None,
    model: str = "gpt-5",
    provider: str = "openai",
    allow_api: bool = False,
) -> dict[str, Any]:
    """Ask for model.py and write what comes back.

    One prompt string is built, saved, and sent -- the same object for all
    three. The legacy path derived the audit copy and the request separately,
    and they drifted.
    """
    from .agents.code_writer_adapter import (
        _extract_coding_result,
        _record_response_token_usage,
        create_llm_client,
        normalize_provider,
    )

    out_dir = trial_dir(competition, trial_id)
    provider = normalize_provider(provider)
    result: dict[str, Any] = {
        "competition": competition,
        "trial_id": trial_id,
        "status": "blocked",
        "changed_files": [],
        "issues": [],
    }

    if client is None:
        if not allow_api:
            result["issues"] = ["api_call_not_enabled"]
            return _write_coding_result(out_dir, result)
        client = create_llm_client(provider)

    payload = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": "You are a careful ML engineer. Return only JSON that follows the requested schema.",
            },
            {"role": "user", "content": handoff["prompt"]},
        ],
    }
    write_text(out_dir / "contract_coding_request.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    try:
        raw_response = client.create_response(payload)
    except Exception as error:
        result["issues"] = [f"api_call_failed:{type(error).__name__}"]
        result["error"] = str(error)[:600]
        return _write_coding_result(out_dir, result)

    write_text(out_dir / "contract_coding_response.json", json.dumps(raw_response, ensure_ascii=False, indent=2) + "\n")
    result["token_usage"] = _record_response_token_usage(
        competition, trial_id, raw_response, provider=provider, model=model, call_type="code_writing"
    )

    parsed = _extract_coding_result(raw_response)
    if parsed.get("status") == "blocked":
        result["issues"] = parsed.get("blocking_issues") or ["unparseable_response"]
        return _write_coding_result(out_dir, result)

    written = _apply_files(handoff, parsed)
    result["summary"] = parsed.get("summary")
    result["changed_files"] = written["changed_files"]
    result["issues"] = written["issues"]
    result["status"] = "accepted" if not written["issues"] else "rejected"
    return _write_coding_result(out_dir, result)


def _apply_files(handoff: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    """Write only inside the declared scope, and refuse the whole response otherwise.

    Refused as a unit rather than partially applied: a half-written change
    leaves a workspace that matches neither the previous trial nor this one.
    """
    files = parsed.get("files")
    if not isinstance(files, list) or not files:
        return {"changed_files": [], "issues": ["response_declared_no_files"]}

    allowed = set(handoff.get("allowed_paths") or [])
    root = Path(handoff["project_root"])
    staged: list[tuple[Path, str]] = []
    issues: list[str] = []
    for entry in files:
        path = str((entry or {}).get("path") or "").strip().replace("\\", "/")
        content = (entry or {}).get("content")
        # An entry with neither a path nor content asks for nothing. Models
        # pad their array with one of these; rejecting the whole response over
        # it threw away a correct model.py and a 5,000-token call. A named
        # path with empty content is a different matter and still fails below.
        if not path and not str(content or "").strip():
            continue
        if path not in allowed:
            issues.append(f"wrote_outside_scope:{path or '<empty>'}")
            continue
        if not isinstance(content, str) or not content.strip():
            issues.append(f"empty_content:{path}")
            continue
        staged.append((root / path, content))
    if issues:
        return {"changed_files": [], "issues": issues}

    for target, content in staged:
        write_text(target, content if content.endswith("\n") else content + "\n")
    return {"changed_files": [str(path.name) for path, _ in staged], "issues": []}


def _write_coding_result(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    write_text(out_dir / "contract_coding_result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    log_decision(
        result["competition"],
        result["trial_id"],
        decision_type="contract_code_writer",
        decision=result["status"],
        reason="Contract model: the code writer returned model.py.",
        evidence={"changed_files": result["changed_files"], "issues": result["issues"]},
        next_action="run-contract-pipeline" if result["status"] == "accepted" else "retry-code-writer",
    )
    return result


def write_contract_coding_request(competition: str, trial_id: str, handoff: dict[str, Any]) -> Path:
    """Save the exact text that was sent.

    The same string the API receives, not a re-rendering of it. A report that
    describes a request rather than reproducing it can disagree with it, and
    once did."""
    out_dir = trial_dir(competition, trial_id)
    path = out_dir / "contract_coding_request.md"
    write_text(path, handoff["prompt"] + "\n")
    log_decision(
        competition,
        trial_id,
        decision_type="contract_coding_request",
        decision="prepared",
        reason="Contract model: the code writer was asked for model.py.",
        evidence={
            "allowed_paths": handoff.get("allowed_paths"),
            "primary_change_axis": handoff.get("primary_change_axis"),
            "source_trial_id": handoff.get("source_trial_id"),
            "base_code_restored": handoff.get("base_code_restored"),
            "base_code_issues": handoff.get("base_code_issues"),
            "prompt_chars": len(handoff["prompt"]),
        },
        next_action="run-code-writer",
    )
    return path
