"""Stage 1 of the execution-core rewrite.

Each test here names the real incident it makes impossible. The point of the
rewrite is that most of these are now ruled out by the framework owning the
flow, rather than by a checker inspecting agent code after the fact.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from execution_core import run_trial, split_samples, write_submission
from execution_core.contract import ContractViolation
from execution_core.metrics import compute, objective_of


LOADER = '''
def load_samples(data_dir, split):
    if split == "train":
        return [{"id": i, "x": float(i), "y": float(i) * 2.0} for i in range(20)]
    return [{"id": 100 + i, "x": float(i)} for i in range(5)]

def label_keys():
    return ["y"]

def submission_columns():
    return ["id", "y"]
'''

GOOD_MODEL = '''
def fit(train_samples):
    n = len(train_samples)
    return sum(s["y"] / s["x"] for s in train_samples if s["x"]) / max(1, n - 1)

def predict(fitted, sample):
    return {"y": fitted * sample["x"]}
'''

CONSTANT_MODEL = '''
def fit(train_samples):
    return None

def predict(fitted, sample):
    return {"y": 0.0}
'''


def _workspace(root: Path, loader: str = LOADER, model: str | None = GOOD_MODEL) -> None:
    (root / "data_loader.py").write_text(loader, encoding="utf-8")
    if model is not None:
        (root / "model.py").write_text(model, encoding="utf-8")


def _run(root: Path, **kwargs):
    return run_trial(root, sys.executable, data_dir=root, metric="rmse", timeout=120, **kwargs)


class SplittingTest(unittest.TestCase):
    """The split is framework code, so an unlabeled holdout cannot be built.

    Real incident: the agent's own harness split off a slice with no labels
    and scored against it; the check that was supposed to catch that was a
    text search for "train" in the harness source."""

    SAMPLES = [{"id": i, "y": float(i)} for i in range(10)]

    def test_split_is_deterministic(self):
        first = split_samples(self.SAMPLES, ["y"], seed=7)
        second = split_samples(self.SAMPLES, ["y"], seed=7)
        self.assertEqual([s["id"] for s in first[1]], [s["id"] for s in second[1]])

    def test_split_covers_every_sample_exactly_once(self):
        train, holdout = split_samples(self.SAMPLES, ["y"])
        self.assertEqual(
            sorted(s["id"] for s in train + holdout), [s["id"] for s in self.SAMPLES]
        )
        self.assertEqual(set(), {s["id"] for s in train} & {s["id"] for s in holdout})

    def test_unlabeled_sample_is_rejected_loudly(self):
        samples = [{"id": 0, "y": 1.0}, {"id": 1}, {"id": 2, "y": 3.0}]
        with self.assertRaises(ContractViolation) as caught:
            split_samples(samples, ["y"])
        self.assertIn("index 1", str(caught.exception))

    def test_both_sides_are_non_empty_even_for_a_tiny_dataset(self):
        train, holdout = split_samples(self.SAMPLES[:2], ["y"], holdout_ratio=0.01)
        self.assertEqual(1, len(train))
        self.assertEqual(1, len(holdout))


class SubmissionWriterTest(unittest.TestCase):
    """The framework holds the pen, so column order comes from the template."""

    def test_columns_follow_the_template_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            write_submission(
                path,
                columns=["id", "b", "a"],
                id_column="id",
                ids=[1, 2],
                predictions=[{"a": 10, "b": 20}, {"a": 30, "b": 40}],
            )
            rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual([["id", "b", "a"], ["1", "20", "10"], ["2", "40", "30"]], rows)

    def test_missing_target_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractViolation):
                write_submission(
                    Path(tmp) / "s.csv",
                    columns=["id", "a", "b"],
                    id_column="id",
                    ids=[1],
                    predictions=[{"a": 1.0}],
                )

    def test_non_finite_value_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractViolation):
                write_submission(
                    Path(tmp) / "s.csv",
                    columns=["id", "a"],
                    id_column="id",
                    ids=[1],
                    predictions=[{"a": float("nan")}],
                )


class MetricsOwnershipTest(unittest.TestCase):
    """Scoring lives outside anything the agent writes."""

    def test_rmse_and_objective(self):
        score = compute("rmse", [{"y": 2.0}, {"y": 4.0}], [{"y": 1.0}, {"y": 2.0}], ["y"])
        self.assertAlmostEqual((1 + 4) ** 0.5 / (2 ** 0.5), score)
        self.assertEqual("minimize", objective_of("rmse"))

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            compute("not_a_metric", [{"y": 1}], [{"y": 1}], ["y"])

    def test_empty_holdout_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            compute("rmse", [], [], ["y"])


class RunTrialTest(unittest.TestCase):
    """The single execution path: fit, predict, score, write."""

    def test_happy_path_scores_and_writes_both_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            result = _run(root)

            self.assertEqual("completed", result["status"], result.get("error"))
            self.assertAlmostEqual(0.0, result["cv_score"], places=6)
            self.assertEqual(5, result["n_test"])
            metrics = json.loads((root / "outputs" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("execution_core", metrics["generated_by"])
            self.assertEqual(result["cv_score"], metrics["cv_score"])
            rows = (root / "outputs" / "submission.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual("id,y", rows[0])
            self.assertEqual(6, len(rows))

    def test_training_reaches_prediction_without_a_check(self):
        """fit()'s return value arrives at predict() as an argument.

        Real incident: train_step.py never saved a model and predict_step.py
        never loaded one, and both exited 0. Here the framework passes the
        object, so there is nothing to skip -- a model that scores 0.0 error
        can only have done so by using what fit() returned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            result = _run(root)
            self.assertAlmostEqual(0.0, result["cv_score"], places=6)
            self.assertEqual("float", result["fit_returned"])

    def test_constant_predictor_is_blocked_and_writes_nothing(self):
        """Blocking before the artifacts exist is what makes this unbypassable.

        Real incident: the old check sat inside the coding function, so
        resuming a completed trial skipped it and a constant {x:0,y:0,z:0}
        predictor was recorded as a result twice."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model=CONSTANT_MODEL)
            result = _run(root)

            self.assertEqual("blocked_constant_predictor", result["status"])
            self.assertEqual(1, result["distinct_holdout_predictions"])
            self.assertIsNone(result["cv_score"])
            self.assertFalse((root / "outputs" / "metrics.json").exists())
            self.assertFalse((root / "outputs" / "submission.csv").exists())

    def test_first_trial_may_be_a_constant_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model=CONSTANT_MODEL)
            result = _run(root, allow_constant_predictions=True)
            self.assertEqual("completed", result["status"], result.get("error"))
            self.assertTrue((root / "outputs" / "metrics.json").exists())

    def test_model_exception_reports_the_stage_it_died_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model="def fit(t):\n    raise ValueError('boom')\n\ndef predict(f, s):\n    return {}\n")
            result = _run(root)
            self.assertEqual("failed", result["status"])
            self.assertEqual("fit", result["failed_stage"])
            self.assertIn("boom", result["error"])

    def test_agent_print_output_cannot_corrupt_the_result(self):
        """Results come back through a file, not stdout.

        Real incident: parsed-from-stdout results broke the moment a model
        printed a progress line."""
        noisy = 'import sys\n' + GOOD_MODEL.replace(
            "def fit(train_samples):",
            "def fit(train_samples):\n    print('{\"ok\": false}'); sys.stderr.write('noise\\n')",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model=noisy)
            result = _run(root)
            self.assertEqual("completed", result["status"], result.get("error"))

    def test_missing_model_module_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model=None)
            result = _run(root)
            self.assertEqual("blocked", result["status"])
            self.assertEqual(["missing_module:model"], result["issues"])

    def test_label_keys_must_match_the_submission_columns(self):
        """Scored on one thing, submitted for another, is not allowed."""
        mismatched = LOADER.replace('return ["y"]', 'return ["z"]').replace(
            '{"id": i, "x": float(i), "y": float(i) * 2.0}',
            '{"id": i, "x": float(i), "z": float(i) * 2.0}',
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, loader=mismatched)
            result = _run(root)
            self.assertEqual("failed", result["status"])
            # Caught before anything is loaded or fitted, so the message names
            # the mismatch rather than a downstream KeyError inside fit().
            self.assertEqual("loader_declarations", result["failed_stage"])
            self.assertIn("does not match", result["error"])

    def test_loader_failure_surfaces_as_a_loader_stage(self):
        broken = LOADER.replace(
            "def load_samples(data_dir, split):",
            "def load_samples(data_dir, split):\n    raise FileNotFoundError('no such layout')",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, loader=broken)
            result = _run(root)
            self.assertEqual("failed", result["status"])
            self.assertEqual("load_train", result["failed_stage"])

    def test_the_score_is_recomputed_by_the_framework_not_read_from_the_agent(self):
        """A model that writes its own metrics.json cannot influence the score.

        Real incident: a harness wrote a hardcoded score, and another wrote a
        real-looking one that never touched the predictions."""
        liar = GOOD_MODEL.replace(
            "def fit(train_samples):",
            "def fit(train_samples):\n"
            "    import json, os\n"
            "    os.makedirs('outputs', exist_ok=True)\n"
            "    json.dump({'cv_score': 0.0001}, open('outputs/metrics.json', 'w'))",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root, model=liar)
            result = _run(root)

            metrics = json.loads((root / "outputs" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("execution_core", metrics["generated_by"])
            self.assertEqual(result["cv_score"], metrics["cv_score"])
            self.assertNotEqual(0.0001, metrics["cv_score"])


class LoaderAnchorTest(unittest.TestCase):
    """The framework never learns this competition's layout.

    It only checks the loader's output against an anchor the competition
    itself supplies -- the submission template's id column. That is what lets
    every DACON competition keep its own directory structure without the
    framework guessing at any of them."""

    def _template(self, root: Path, ids) -> Path:
        path = root / "sample_submission.csv"
        path.write_text(
            "id,y\n" + "".join(f"{identifier},0\n" for identifier in ids), encoding="utf-8"
        )
        return path

    def test_matching_ids_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            template = self._template(root, range(100, 105))
            result = _run(root, submission_template=template)

            self.assertEqual("completed", result["status"], result.get("error"))
            self.assertEqual("verified", result["template_verification"]["status"])

    def test_loader_reading_the_wrong_ids_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            template = self._template(root, range(900, 905))
            result = _run(root, submission_template=template)

            self.assertEqual("failed", result["status"])
            self.assertEqual("loader_anchor", result["failed_stage"])
            self.assertTrue(result["issues"][0].startswith("test_ids_do_not_match_template"))
            self.assertFalse((root / "outputs" / "submission.csv").exists())

    def test_wrong_row_count_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            template = self._template(root, range(100, 103))
            result = _run(root, submission_template=template)
            self.assertIn("test_count_differs_from_template:5_vs_3", result["issues"])

    def test_right_ids_in_the_wrong_order_is_caught(self):
        """A permuted submission is valid on disk and scores as noise."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            template = self._template(root, [104, 103, 102, 101, 100])
            result = _run(root, submission_template=template)
            self.assertEqual(
                ["test_ids_match_template_but_in_a_different_order"], result["issues"]
            )

    def test_missing_template_is_reported_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            result = _run(root, submission_template=root / "nope.csv")
            self.assertEqual("failed", result["status"])
            self.assertEqual("loader_anchor", result["failed_stage"])

    def test_skipping_verification_is_recorded_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace(root)
            result = _run(root)
            self.assertEqual("completed", result["status"], result.get("error"))
            self.assertEqual("skipped", result["template_verification"]["status"])
            self.assertEqual("no_template_declared", result["template_verification"]["reason"])


if __name__ == "__main__":
    unittest.main()
