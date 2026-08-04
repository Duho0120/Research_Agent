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

def _facts(module_name):
    mod = importlib.import_module(module_name)
    loader = getattr(mod, "load_samples", None)
    if loader is None:
        return {"error": "load_samples_not_defined"}
    from pathlib import Path as _P
    root = _P(mod.__file__).resolve().parent
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

print("<<PROBE>>" + json.dumps(_facts(sys.argv[1])))
'''


def run_sample_loading_probe(
    project_root: Path, python: str, module_name: str, *, timeout: int = 600
) -> dict[str, Any]:
    """Run the loader in a subprocess and return what it actually produced."""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "_loader_probe.py"
        probe.write_text(PROBE_SOURCE, encoding="utf-8")
        try:
            completed = subprocess.run(
                [python, str(probe), module_name],
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
