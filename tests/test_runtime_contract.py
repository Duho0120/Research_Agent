import sys
import tempfile
import unittest
from pathlib import Path

from research_agent.runtime_contract import (
    evaluate_loader_contract,
    run_sample_loading_probe,
)


class LoaderContractRulesTest(unittest.TestCase):
    """The pure rule layer. Every assertion must hold without knowing whether
    the competition stores one file per sample, a flat table, or anything
    else -- DACON layouts differ per competition."""

    def _facts(self, **overrides):
        facts = {
            "train": {"count": 10000, "head_ids": ["TRAIN_1", "TRAIN_2"], "feature_keys": ["x", "y"]},
            "test": {"count": 2, "head_ids": ["TEST_1", "TEST_2"], "feature_keys": ["x", "y"]},
            "deterministic": True,
        }
        facts.update(overrides)
        return facts

    def test_a_real_loader_passes(self):
        issues = evaluate_loader_contract(
            self._facts(),
            label_ids={f"TRAIN_{i}" for i in range(1, 10001)},
            submission_ids=["TEST_1", "TEST_2"],
        )
        self.assertEqual([], issues)

    def test_id_only_samples_are_rejected(self):
        # The exact shape that let a loader look correct while reading only
        # sample_submission.csv: ids present, no features at all.
        facts = self._facts(train={"count": 10000, "head_ids": ["TRAIN_1"], "feature_keys": []})
        self.assertIn("loader_samples_have_no_features:train", evaluate_loader_contract(facts))

    def test_loading_the_unlabelled_split_as_train_is_rejected(self):
        facts = self._facts(train={"count": 10, "head_ids": ["TEST_1"], "feature_keys": ["x"]})
        issues = evaluate_loader_contract(facts, label_ids={"TRAIN_1", "TRAIN_2"})
        self.assertIn("loader_train_ids_do_not_match_labels", issues)

    def test_loading_only_a_handful_of_train_samples_is_rejected(self):
        facts = self._facts(train={"count": 3, "head_ids": ["TRAIN_1"], "feature_keys": ["x"]})
        issues = evaluate_loader_contract(facts, label_ids={f"TRAIN_{i}" for i in range(1, 10001)})
        self.assertTrue(any(i.startswith("loader_train_count_far_from_label_count") for i in issues))

    def test_test_ids_must_match_the_submission_template_in_order(self):
        facts = self._facts(test={"count": 2, "head_ids": ["TEST_2", "TEST_1"], "feature_keys": ["x"]})
        issues = evaluate_loader_contract(facts, submission_ids=["TEST_1", "TEST_2"])
        self.assertIn("loader_test_ids_do_not_match_submission_template", issues)

    def test_empty_split_and_raising_split_are_reported(self):
        self.assertIn(
            "loader_returned_no_samples:train",
            evaluate_loader_contract(self._facts(train={"count": 0, "head_ids": [], "feature_keys": []})),
        )
        self.assertIn(
            "loader_split_raised:train:FileNotFoundError: train.csv",
            evaluate_loader_contract(self._facts(train={"error": "FileNotFoundError: train.csv"})),
        )

    def test_non_deterministic_loader_is_reported(self):
        self.assertIn("loader_is_not_deterministic", evaluate_loader_contract(self._facts(deterministic=False)))

    def test_probe_failure_surfaces_as_a_single_issue(self):
        self.assertEqual(
            ["loader_probe_error:loader_probe_timed_out"],
            evaluate_loader_contract({"error": "loader_probe_timed_out"}),
        )


class LoaderProbeExecutionTest(unittest.TestCase):
    def _run(self, source: str):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "predict_step.py").write_text(source, encoding="utf-8")
            return run_sample_loading_probe(root, sys.executable, "predict_step", timeout=120)

    def test_probe_reports_real_features_from_a_working_loader(self):
        facts = self._run(
            "def load_samples(data_dir, split):\n"
            "    n = 4 if split == 'train' else 2\n"
            "    tag = 'TRAIN' if split == 'train' else 'TEST'\n"
            "    return [{'id': f'{tag}_{i}', 'x': i * 1.5} for i in range(1, n + 1)]\n"
        )
        self.assertEqual(4, facts["train"]["count"])
        self.assertEqual(["x"], facts["train"]["feature_keys"])
        self.assertEqual(["TEST_1", "TEST_2"], facts["test"]["head_ids"])
        self.assertTrue(facts["deterministic"])

    def test_probe_exposes_an_id_only_loader(self):
        # Running it is what makes this unfakeable: the code defines
        # load_samples and returns rows, so any text-level check passes.
        facts = self._run(
            "def load_samples(data_dir, split):\n"
            "    return [{'id': f'TEST_{i}'} for i in range(1, 3)]\n"
        )
        self.assertEqual([], facts["train"]["feature_keys"])
        self.assertIn("loader_samples_have_no_features:train", evaluate_loader_contract(facts))

    def test_probe_reports_a_missing_loader(self):
        facts = self._run("def something_else():\n    return []\n")
        self.assertEqual("load_samples_not_defined", facts.get("error"))
        self.assertEqual(
            ["loader_probe_error:load_samples_not_defined"], evaluate_loader_contract(facts)
        )


class ScoringSensitivityTest(unittest.TestCase):
    """A scorer that does not react to its predictions is not scoring. This
    is the check the fabricated harnesses could not have survived: one
    hardcoded `"cv_score": 0.0` with a comment saying the key was required,
    another substituted format checks for scoring. Neither moves when the
    predictions change."""

    def _workspace(self, root: Path, harness_body: str):
        (root / "predict_step.py").write_text(
            "def predict(sample):\n    return {'x': 1.0}\n", encoding="utf-8"
        )
        (root / "outputs").mkdir(exist_ok=True)
        (root / "scoring_harness.py").write_text(harness_body, encoding="utf-8")

    def _run(self, harness_body):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._workspace(root, harness_body)
            from research_agent.runtime_contract import run_scoring_perturbation_probe

            return run_scoring_perturbation_probe(
                root,
                sys.executable,
                harness_module="scoring_harness",
                predict_module="predict_step",
                metrics_path=root / "outputs" / "metrics.json",
                score_key="cv_score",
                timeout=120,
            )

    def test_a_real_scorer_moves_when_predictions_are_wrecked(self):
        from research_agent.runtime_contract import evaluate_scoring_sensitivity

        results = self._run(
            "import json\n"
            "from predict_step import predict\n"
            "if __name__ == '__main__':\n"
            "    truth = [1.0, 1.0]\n"
            "    preds = [predict({'id': i})['x'] for i in range(2)]\n"
            "    err = sum(abs(p - t) for p, t in zip(preds, truth)) / len(truth)\n"
            "    json.dump({'cv_score': err}, open('outputs/metrics.json', 'w'))\n"
        )
        self.assertNotEqual(results["baseline"], results["perturbed"])
        self.assertEqual([], evaluate_scoring_sensitivity(results))

    def test_hardcoded_score_is_caught(self):
        from research_agent.runtime_contract import evaluate_scoring_sensitivity

        results = self._run(
            "import json\n"
            "if __name__ == '__main__':\n"
            "    json.dump({'cv_score': 0.0}, open('outputs/metrics.json', 'w'))\n"
        )
        self.assertEqual(
            ["scoring_ignores_predictions:score_unchanged_when_predictions_wrecked"],
            evaluate_scoring_sensitivity(results),
        )

    def test_format_only_harness_is_caught(self):
        from research_agent.runtime_contract import evaluate_scoring_sensitivity

        # Mirrors the real one: it calls predict, counts rows, and reports a
        # constant -- busy-looking, but blind to prediction quality.
        results = self._run(
            "import json\n"
            "from predict_step import predict\n"
            "if __name__ == '__main__':\n"
            "    rows = [predict({'id': i}) for i in range(3)]\n"
            "    json.dump({'cv_score': 0.0, 'n': len(rows)}, open('outputs/metrics.json', 'w'))\n"
        )
        self.assertIn(
            "scoring_ignores_predictions:score_unchanged_when_predictions_wrecked",
            evaluate_scoring_sensitivity(results),
        )

    def test_missing_score_key_is_reported(self):
        from research_agent.runtime_contract import evaluate_scoring_sensitivity

        results = self._run(
            "import json\n"
            "if __name__ == '__main__':\n"
            "    json.dump({'other': 1}, open('outputs/metrics.json', 'w'))\n"
        )
        self.assertTrue(
            evaluate_scoring_sensitivity(results)[0].startswith("scoring_produced_no_numeric_score")
        )


if __name__ == "__main__":
    unittest.main()
