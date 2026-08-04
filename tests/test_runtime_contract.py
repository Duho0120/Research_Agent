import sys
import tempfile
import unittest
from pathlib import Path

from kaggle_research_agent.runtime_contract import (
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


if __name__ == "__main__":
    unittest.main()
