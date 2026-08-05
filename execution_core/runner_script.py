"""The script that runs inside the child process -- framework-authored.

Agent code is executed in a separate process so a hang or a crash cannot take
the framework down with it, but the framework decides what crosses the
boundary: the child returns predictions, never a model and never a score.
Results go to a file rather than stdout because agent code is free to print,
and a stray print used to be enough to corrupt a parsed result.
"""

from __future__ import annotations


def build_runner_source(
    *,
    core_path: str,
    out_path: str,
    data_dir: str,
    loader_module: str,
    model_module: str,
    holdout_ratio: float,
    seed: int,
) -> str:
    header = "\n".join(
        [
            "CORE_PATH = " + repr(str(core_path)),
            "OUT_PATH = " + repr(str(out_path)),
            "DATA_DIR = " + repr(str(data_dir)),
            "LOADER_MODULE = " + repr(str(loader_module)),
            "MODEL_MODULE = " + repr(str(model_module)),
            "HOLDOUT_RATIO = " + repr(float(holdout_ratio)),
            "SEED = " + repr(int(seed)),
        ]
    )
    return header + "\n" + _BODY


_BODY = '''
import hashlib
import importlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, CORE_PATH)
sys.path.insert(0, os.getcwd())

from execution_core.splitting import split_samples


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:
            pass
    return str(value)


def finish(payload):
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    sys.exit(0)


def fail(stage, error):
    finish({"ok": False, "stage": stage, "error": error})


stage = "import_loader"
try:
    loader = importlib.import_module(LOADER_MODULE)

    stage = "import_model"
    model = importlib.import_module(MODEL_MODULE)

    stage = "loader_declarations"
    label_keys = [str(key) for key in loader.label_keys()]
    submission_columns = [str(column) for column in loader.submission_columns()]
    id_column = str(loader.id_column()) if hasattr(loader, "id_column") else submission_columns[0]

    # Checked here, before any data is loaded or any model is fitted. Left to
    # the parent it would only surface after fit() had already crashed on the
    # key it could not find, which names the symptom instead of the cause.
    submission_targets = [column for column in submission_columns if column != id_column]
    if sorted(label_keys) != sorted(submission_targets):
        fail(
            "loader_declarations",
            "label_keys() "
            + repr(sorted(label_keys))
            + " does not match the submission's non-id columns "
            + repr(sorted(submission_targets))
            + "; they must name the same values.",
        )

    stage = "load_train"
    train_all = loader.load_samples(DATA_DIR, "train")

    stage = "split"
    train, holdout = split_samples(
        train_all, label_keys, holdout_ratio=HOLDOUT_RATIO, seed=SEED
    )
    truths = [{key: jsonable(sample[key]) for key in label_keys} for sample in holdout]
    # Recorded, not acted on. Two scores are only comparable if they came from
    # the same holdout, and when a trial deliberately changes the validation
    # structure this is the only trace of it left in the artifacts.
    holdout_fingerprint = hashlib.sha256(
        "|".join(str(sample["id"]) for sample in holdout).encode("utf-8")
    ).hexdigest()[:16]

    stage = "fit"
    started = time.time()
    fitted = model.fit(train)
    fit_seconds = time.time() - started

    stage = "predict_holdout"
    holdout_predictions = [jsonable(model.predict(fitted, sample)) for sample in holdout]

    stage = "load_test"
    test = loader.load_samples(DATA_DIR, "test")

    stage = "test_ids"
    test_ids = [jsonable(sample["id"]) for sample in test]

    stage = "predict_test"
    test_predictions = [jsonable(model.predict(fitted, sample)) for sample in test]

    finish(
        {
            "ok": True,
            "label_keys": label_keys,
            "submission_columns": submission_columns,
            "id_column": id_column,
            "n_train_total": len(train_all),
            "n_train": len(train),
            "n_holdout": len(holdout),
            "n_test": len(test),
            "split": {
                "ratio": HOLDOUT_RATIO,
                "seed": SEED,
                "fingerprint": holdout_fingerprint,
            },
            "truths": truths,
            "holdout_predictions": holdout_predictions,
            "test_ids": test_ids,
            "test_predictions": test_predictions,
            "fit_returned": type(fitted).__name__,
            "fit_seconds": round(fit_seconds, 3),
        }
    )
except SystemExit:
    raise
except BaseException:
    fail(stage, traceback.format_exc(limit=12))
'''
