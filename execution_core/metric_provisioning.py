"""Deciding how a competition gets scored -- once, before any trial runs.

The file this produces lives under competitions/<id>/, not in the workspace.
That single placement is the whole ownership move: the agent's write scope is
the workspace, so a scorer kept outside it is beyond reach as a matter of
filesystem layout rather than of rules. Previously scoring_harness.py sat in
the workspace next to the model, and "do not touch your own scorer" was a
request.

Generation is a last resort, not a default. A competition whose metric is
already implemented costs nothing to score, and most are.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from . import metrics as metrics_module
from .metric_spec import MetricDeclaration, normalize_metric_name, parse_metric_spec
from .metric_verification import verify_metric


METRIC_MODULE_FILENAME = "competition_metric.py"
GenerateFn = Callable[[str], str]


def provision_metric(
    competition_dir: Path | str,
    *,
    target_keys: list[str],
    generate: GenerateFn | None = None,
    truths: list[dict[str, Any]] | None = None,
    regenerate: bool = False,
) -> dict[str, Any]:
    """Resolve this competition's metric and make it available for scoring.

    The ladder, cheapest and most trustworthy first:

      1. an already-provisioned module for this competition -- reuse it;
      2. a name the built-in registry knows -- use that, no generation at all;
      3. a name it does not know -- generate an implementation, verify it, and
         record how much the definition it was given can be trusted;
      4. nothing usable -- refuse, rather than score with something unproven.
    """
    directory = Path(competition_dir)
    spec = parse_metric_spec(directory / "metric.md")
    if spec is None:
        return _refused("metric_spec_missing_or_unnamed", directory)

    key = normalize_metric_name(spec.name)
    result: dict[str, Any] = {
        "competition_dir": str(directory),
        "declared_name": spec.name,
        "metric": key,
        "objective": spec.objective,
        "spec_confidence": spec.confidence,
    }

    module_path = directory / METRIC_MODULE_FILENAME
    if module_path.is_file() and not regenerate:
        # Re-verified against the spec on every load, not trusted because it
        # was accepted once: metric.md may have gained a definition or an
        # example since, and an implementation that no longer matches the
        # competition's own words should stop being used.
        loaded = _register_from_file(
            module_path,
            key,
            spec.objective,
            target_keys=target_keys,
            truths=truths,
            worked_example=spec.worked_example,
        )
        return {**result, **loaded, "source": "provisioned"}

    builtin = metrics_module.METRICS.get(key)
    if builtin is not None:
        if builtin.objective != spec.objective:
            return {
                **result,
                "status": "blocked",
                "source": "builtin",
                "issues": [
                    f"objective_conflicts_with_builtin:competition_says_{spec.objective}"
                    f"_builtin_says_{builtin.objective}"
                ],
            }
        return {
            **result,
            "status": "ready",
            "source": "builtin",
            "confidence": "high",
            "issues": [],
            "spec": builtin,
        }

    if generate is None:
        return {
            **result,
            "status": "blocked",
            "source": "none",
            "issues": [f"metric_not_implemented_and_generation_unavailable:{key}"],
        }

    return {**result, **_generate_and_verify(module_path, spec, key, target_keys, truths, generate)}


def build_generation_prompt(spec: MetricDeclaration, target_keys: list[str]) -> str:
    """What the implementer is told. Carries the competition's own words verbatim."""
    lines = [
        f"Implement the competition metric {spec.name!r} as a Python module.",
        "",
        "Required module contents, exactly these names:",
        "",
        f"    NAME = {normalize_metric_name(spec.name)!r}",
        f"    OBJECTIVE = {spec.objective!r}",
        "",
        "    def compute(predictions, truths, target_keys):",
        "        \"\"\"predictions and truths are equal-length lists of dicts.",
        "        Each dict holds every key in target_keys. Return one float.\"\"\"",
        "",
        f"For this competition target_keys is {target_keys!r}.",
        "",
        "Rules:",
        "- Standard library only, plus numpy if it genuinely helps.",
        "- Deterministic: no sampling, no shuffling, no reliance on dict order.",
        "- Read every value out of the dicts; never return a constant.",
        "- Raise on malformed input rather than substituting a default.",
    ]
    if spec.has_definition:
        lines += ["", "The competition defines the metric as follows:", "", spec.prose]
    else:
        lines += [
            "",
            "The competition supplied only the metric's name and direction. Implement the",
            "standard definition of that name and state any assumption you had to make",
            "in a module-level docstring -- especially units and thresholds.",
        ]
    if spec.worked_example is not None:
        example = spec.worked_example
        lines += [
            "",
            "This example is hand-checked and your implementation must reproduce it:",
            f"  predictions = {example.predictions}",
            f"  truths      = {example.truths}",
            f"  target_keys = {example.target_keys}",
            f"  compute(...) == {example.expected}",
        ]
    return "\n".join(lines)


def _generate_and_verify(
    module_path: Path,
    spec: MetricDeclaration,
    key: str,
    target_keys: list[str],
    truths: list[dict[str, Any]] | None,
    generate: GenerateFn,
) -> dict[str, Any]:
    """Generate, check, and delete on rejection.

    A rejected implementation is removed rather than left on disk: a file that
    exists is treated as provisioned by the step above, so leaving a failed one
    behind would quietly promote it on the next run.
    """
    prompt = build_generation_prompt(spec, target_keys)
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        try:
            source = generate(prompt if attempt == 1 else _retry_prompt(prompt, attempts[-1]["issues"]))
        except Exception as error:
            return {
                "status": "blocked",
                "source": "generated",
                "issues": [f"metric_generation_failed:{type(error).__name__}"],
                "error": str(error)[:400],
                "attempts": attempts,
            }
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(source, encoding="utf-8")

        loaded = _register_from_file(
            module_path,
            key,
            spec.objective,
            target_keys=target_keys,
            truths=truths,
            worked_example=spec.worked_example,
        )
        attempts.append({"attempt": attempt, "issues": loaded.get("issues", [])})
        if loaded.get("status") == "ready":
            return {
                **loaded,
                "source": "generated",
                # The spec's confidence caps the metric's: an implementation
                # verified against a definition nobody wrote is still only as
                # good as the name it was guessed from.
                "confidence": spec.confidence,
                "attempts": attempts,
            }
        module_path.unlink(missing_ok=True)
    return {
        "status": "blocked",
        "source": "generated",
        "issues": attempts[-1]["issues"],
        "attempts": attempts,
    }


def _retry_prompt(prompt: str, issues: list[str]) -> str:
    return "\n".join(
        [
            prompt,
            "",
            "A previous attempt was rejected by the following checks. Do not repeat it:",
            *(f"- {issue}" for issue in issues),
        ]
    )


def _register_from_file(
    module_path: Path,
    key: str,
    objective: str,
    *,
    target_keys: list[str] | None = None,
    truths: list[dict[str, Any]] | None = None,
    worked_example: Any | None = None,
) -> dict[str, Any]:
    module = _load_module(module_path)
    if isinstance(module, str):
        return {"status": "blocked", "issues": [module]}
    compute = getattr(module, "compute", None)
    if not callable(compute):
        return {"status": "blocked", "issues": ["metric_module_has_no_compute"]}
    declared_objective = str(getattr(module, "OBJECTIVE", objective))
    if declared_objective != objective:
        return {
            "status": "blocked",
            "issues": [f"metric_module_objective_mismatch:{declared_objective}_vs_{objective}"],
        }

    verification = verify_metric(
        compute,
        objective=objective,
        target_keys=target_keys or list(getattr(module, "TARGET_KEYS", []) or []),
        truths=truths,
        # The competition's example, never the module's own. An implementation
        # supplying the case it is checked against proves nothing.
        worked_example=worked_example,
    )
    if verification["issues"]:
        return {"status": "blocked", "issues": verification["issues"], "verification": verification}

    # Returned, not registered. The caller passes this spec to run_trial for
    # this competition only; METRICS keeps holding built-ins alone.
    return {
        "status": "ready",
        "issues": [],
        "verification": verification,
        "spec": metrics_module.MetricSpec(key, objective, compute),
    }


def _load_module(module_path: Path) -> Any | str:
    name = f"_competition_metric_{module_path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        return "metric_module_unloadable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        return f"metric_module_raised_on_import:{type(error).__name__}"
    return module


def _refused(issue: str, directory: Path) -> dict[str, Any]:
    return {
        "competition_dir": str(directory),
        "status": "blocked",
        "source": "none",
        "issues": [issue],
    }
