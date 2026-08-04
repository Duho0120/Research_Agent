"""Execution-based verification of a generated data loader.

Static text checks on generated code were bypassed seven times in a row
during the DACON 236716 debugging session: a required key got hardcoded, a
required function was defined with an empty body, a rejected file was simply
left untouched, format checks were substituted for scoring. Every one of
those satisfied the letter of a text pattern while skipping the substance.

Running the loader and asserting on what it actually returns cannot be
bypassed the same way -- a loader that returns ids with no features, or
reads the wrong split, fails the assertion no matter how the code is
written.

Deliberately structure-agnostic. DACON competitions differ in file layout
from one competition to the next, so nothing here assumes folders, a flat
table, or any particular format. Every assertion is derived from the two
anchors every competition has: the submission template (which ids, in which
order) and the label source (where the answers are).

Split in two on purpose:
- ``run_sample_loading_probe`` executes the loader in a subprocess (the
  generated module may import heavy libraries or have import side effects)
  and reports plain facts.
- ``evaluate_loader_contract`` is pure, so the rules are unit-testable
  without generating or running any code.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROBE_SOURCE = '''
import json, os, sys, importlib

# The probe script lives in a temp dir, so Python puts that dir on sys.path
# instead of the workspace. Add the workspace (cwd) so the generated module
# and its own relative imports resolve.
sys.path.insert(0, os.getcwd())

def _facts(module_name, data_dir):
    mod = importlib.import_module(module_name)
    loader = getattr(mod, "load_samples", None)
    if loader is None:
        return {"error": "load_samples_not_defined"}
    from pathlib import Path as _P
    root = _P(data_dir)
    out = {}
    for split in ("train", "test"):
        try:
            samples = list(loader(root, split))
        except Exception as exc:
            out[split] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        ids, feature_keys = [], set()
        for s in samples[:200]:
            if isinstance(s, dict):
                ids.append(str(s.get("id")))
                feature_keys.update(k for k in s.keys() if k != "id")
            else:
                ids.append(str(getattr(s, "id", "")))
                feature_keys.update(
                    k for k in getattr(s, "__dict__", {}) if k != "id"
                )
        out[split] = {
            "count": len(samples),
            "head_ids": ids[:50],
            "feature_keys": sorted(feature_keys)[:40],
        }
    try:
        again = list(loader(root, "train"))
        out["deterministic"] = len(again) == out.get("train", {}).get("count")
    except Exception:
        out["deterministic"] = None
    return out

print("<<PROBE>>" + json.dumps(_facts(sys.argv[1], sys.argv[2])))
'''


def run_sample_loading_probe(
    project_root: Path,
    python: str,
    module_name: str,
    *,
    data_dir: Path | str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run the loader in a subprocess and return what it actually produced.

    ``data_dir`` is what gets handed to ``load_samples`` as its first
    argument, and it must be the competition's real data directory. Passing
    the project root instead made a correct loader look broken: it raised
    (as instructed -- no silent fallback) because the split directories sit
    under data/, not beside the module.
    """
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "_loader_probe.py"
        probe.write_text(PROBE_SOURCE, encoding="utf-8")
        try:
            completed = subprocess.run(
                [python, str(probe), module_name, str(data_dir or project_root)],
                cwd=str(project_root),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": "loader_probe_timed_out"}
    marker = "<<PROBE>>"
    for line in (completed.stdout or "").splitlines():
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                break
    detail = (completed.stderr or completed.stdout or "").strip()[-300:]
    return {"error": f"loader_probe_failed: {detail}"}


def evaluate_loader_contract(
    facts: dict[str, Any],
    *,
    label_ids: set[str] | None = None,
    submission_ids: list[str] | None = None,
) -> list[str]:
    """Assertions a real loader satisfies, whatever the file layout is.

    ``label_ids`` and ``submission_ids`` come from the agent's own declaration
    of where the labels and the submission template live -- so a wrong
    declaration also fails here, which is the point: we verify the claim
    instead of trying to infer it ourselves.
    """
    if not isinstance(facts, dict) or facts.get("error"):
        return [f"loader_probe_error:{(facts or {}).get('error', 'unknown')}"]

    issues: list[str] = []
    for split in ("train", "test"):
        info = facts.get(split)
        if not isinstance(info, dict):
            issues.append(f"loader_split_missing:{split}")
            continue
        if info.get("error"):
            issues.append(f"loader_split_raised:{split}:{info['error']}")
            continue
        if not info.get("count"):
            issues.append(f"loader_returned_no_samples:{split}")
            continue
        # An id list is not a dataset. This is the shape that let a loader
        # look correct while reading only sample_submission.csv.
        if not info.get("feature_keys"):
            issues.append(f"loader_samples_have_no_features:{split}")

    train = facts.get("train") if isinstance(facts.get("train"), dict) else {}
    test = facts.get("test") if isinstance(facts.get("test"), dict) else {}

    if label_ids and train.get("head_ids"):
        if not (set(train["head_ids"]) & label_ids):
            # Catches loading the unlabelled split under the name "train".
            issues.append("loader_train_ids_do_not_match_labels")
        elif train.get("count") and _magnitude(train["count"]) != _magnitude(len(label_ids)):
            issues.append(
                f"loader_train_count_far_from_label_count:{train['count']}_vs_{len(label_ids)}"
            )

    if submission_ids and test.get("head_ids"):
        expected = submission_ids[: len(test["head_ids"])]
        if test["head_ids"] != expected:
            issues.append("loader_test_ids_do_not_match_submission_template")
        elif test.get("count") and test["count"] != len(submission_ids):
            issues.append(
                f"loader_test_count_differs_from_submission:{test['count']}_vs_{len(submission_ids)}"
            )

    if facts.get("deterministic") is False:
        issues.append("loader_is_not_deterministic")
    return issues


def _magnitude(value: int) -> int:
    return len(str(max(int(value), 1)))


PERTURBATION_PROBE_SOURCE = '''
import json, os, sys, runpy, importlib
sys.path.insert(0, os.getcwd())

harness_module, metrics_path, perturb = sys.argv[1], sys.argv[2], sys.argv[3] == "1"

if perturb:
    # Replace the trial's predictions with deliberately wrong ones before the
    # harness runs. A harness that really scores predictions must move; one
    # that hardcodes a number, or only format-checks, cannot.
    predict_module = importlib.import_module(sys.argv[4])
    original = predict_module.predict

    def _wrecked(sample):
        value = original(sample)
        if isinstance(value, dict):
            return {k: (v * 0 - 999.0 if isinstance(v, (int, float)) else v) for k, v in value.items()}
        if isinstance(value, (int, float)):
            return -999.0
        return value

    predict_module.predict = _wrecked

try:
    runpy.run_module(harness_module, run_name="__main__")
except SystemExit:
    pass
except Exception as exc:
    print("<<PROBE>>" + json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))
    raise SystemExit(0)

score = None
try:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        score = json.load(handle).get(sys.argv[5])
except Exception as exc:
    print("<<PROBE>>" + json.dumps({"error": f"metrics_unreadable: {exc}"[:200]}))
    raise SystemExit(0)
print("<<PROBE>>" + json.dumps({"score": score}))
'''


def run_scoring_perturbation_probe(
    project_root: Path,
    python: str,
    *,
    harness_module: str,
    predict_module: str,
    metrics_path: Path | str,
    score_key: str,
    timeout: int = 900,
) -> dict[str, Any]:
    """Score once normally, once with predictions deliberately wrecked.

    This is the check a fabricated harness cannot survive. Text checks were
    bypassed by hardcoding the score key and by substituting format checks
    for scoring; neither of those reacts to the predictions changing.
    """
    # The probe runs the real harness, which writes the real metrics
    # artifact -- including on the perturbed pass, whose score is deliberate
    # nonsense. Leaving that behind would let a detection device become a
    # source of exactly the fabricated score it exists to catch, so the
    # artifact is restored to its pre-probe state no matter how this exits.
    metrics_file = Path(metrics_path)
    before = metrics_file.read_bytes() if metrics_file.is_file() else None
    try:
        return _run_perturbation_passes(
            project_root,
            python,
            harness_module=harness_module,
            predict_module=predict_module,
            metrics_path=metrics_path,
            score_key=score_key,
            timeout=timeout,
        )
    finally:
        if before is None:
            if metrics_file.is_file():
                metrics_file.unlink()
        else:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_bytes(before)


def _run_perturbation_passes(
    project_root: Path,
    python: str,
    *,
    harness_module: str,
    predict_module: str,
    metrics_path: Path | str,
    score_key: str,
    timeout: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label, perturb in (("baseline", "0"), ("perturbed", "1")):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "_score_probe.py"
            probe.write_text(PERTURBATION_PROBE_SOURCE, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [
                        python,
                        str(probe),
                        harness_module,
                        str(metrics_path),
                        perturb,
                        predict_module,
                        score_key,
                    ],
                    cwd=str(project_root),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return {"error": f"scoring_probe_timed_out:{label}"}
        payload = _probe_payload(completed.stdout, completed.stderr)
        if payload.get("error"):
            return {"error": f"{label}:{payload['error']}"}
        results[label] = payload.get("score")
    return results


def evaluate_scoring_sensitivity(results: dict[str, Any]) -> list[str]:
    """A real scorer reacts to its inputs."""
    if not isinstance(results, dict) or results.get("error"):
        return [f"scoring_probe_error:{(results or {}).get('error', 'unknown')}"]
    baseline, perturbed = results.get("baseline"), results.get("perturbed")
    if not isinstance(baseline, (int, float)):
        return [f"scoring_produced_no_numeric_score:{baseline!r}"[:120]]
    if not isinstance(perturbed, (int, float)):
        return [f"scoring_produced_no_numeric_score_when_perturbed:{perturbed!r}"[:120]]
    if baseline == perturbed:
        return ["scoring_ignores_predictions:score_unchanged_when_predictions_wrecked"]
    return []


def _probe_payload(stdout: str, stderr: str) -> dict[str, Any]:
    marker = "<<PROBE>>"
    for line in (stdout or "").splitlines():
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                break
    return {"error": f"probe_failed: {(stderr or stdout or '').strip()[-200:]}"}


PREDICT_SENSITIVITY_PROBE_SOURCE = '''
import json, os, sys, importlib
sys.path.insert(0, os.getcwd())

loader_module, predict_module, data_dir = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    loader = importlib.import_module(loader_module)
    predictor = importlib.import_module(predict_module)
    from pathlib import Path as _P
    samples = list(loader.load_samples(_P(data_dir), "train"))[:8]
    if len(samples) < 2:
        print("<<PROBE>>" + json.dumps({"error": "not_enough_samples_to_compare"}))
        raise SystemExit(0)
    outputs = [repr(predictor.predict(s)) for s in samples]
except SystemExit:
    raise
except Exception as exc:
    print("<<PROBE>>" + json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))
    raise SystemExit(0)
print("<<PROBE>>" + json.dumps({"distinct": len(set(outputs)), "compared": len(outputs)}))
'''


def run_predict_sensitivity_probe(
    project_root: Path,
    python: str,
    *,
    loader_module: str,
    predict_module: str,
    data_dir: Path | str,
    timeout: int = 900,
) -> dict[str, Any]:
    """Ask predict() for several different samples and see if it varies."""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "_predict_probe.py"
        probe.write_text(PREDICT_SENSITIVITY_PROBE_SOURCE, encoding="utf-8")
        try:
            completed = subprocess.run(
                [python, str(probe), loader_module, predict_module, str(data_dir)],
                cwd=str(project_root),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": "predict_probe_timed_out"}
    return _probe_payload(completed.stdout, completed.stderr)


def evaluate_predict_sensitivity(results: dict[str, Any]) -> list[str]:
    """A model that returns the same answer for every input is not using its
    input. Real incident: predict() was `return {"x": 0.0, "y": 0.0, "z":
    0.0}` for many trials, and the plan it was supposed to implement (Ridge
    on per-sample features) never got built -- yet every text-level check
    passed, because the function existed and returned a valid shape.
    """
    if not isinstance(results, dict) or results.get("error"):
        return [f"predict_probe_error:{(results or {}).get('error', 'unknown')}"]
    compared = results.get("compared") or 0
    if compared < 2:
        return ["predict_probe_error:not_enough_samples_to_compare"]
    if results.get("distinct", 0) < 2:
        return ["predict_ignores_input:same_output_for_every_sample"]
    return []
