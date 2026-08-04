import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from kaggle_research_agent.runtime_contract import (
    evaluate_predict_sensitivity,
    run_predict_sensitivity_probe,
)


LOADER = (
    "def load_samples(data_dir, split):\n"
    "    return [{'id': f'S{i}', 'x': float(i)} for i in range(4)]\n"
)


class PredictSensitivityTest(unittest.TestCase):
    """A model returning the same answer for every input is not using its
    input. Real incident: predict() stayed a constant {x:0, y:0, z:0} across
    trials while the plan it was meant to implement (Ridge on per-sample
    features) never got built -- and every text-level check passed, because
    the function existed and returned a valid shape."""

    def _run(self, predict_body: str):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data_loader.py").write_text(LOADER, encoding="utf-8")
            (root / "predict_step.py").write_text(predict_body, encoding="utf-8")
            return run_predict_sensitivity_probe(
                root,
                sys.executable,
                loader_module="data_loader",
                predict_module="predict_step",
                data_dir=root,
                timeout=120,
            )

    def test_constant_predictor_is_caught(self):
        results = self._run("def predict(sample):\n    return {'y': 0.0}\n")
        self.assertEqual(1, results["distinct"])
        self.assertEqual(
            ["predict_ignores_input:same_output_for_every_sample"],
            evaluate_predict_sensitivity(results),
        )

    def test_input_dependent_predictor_passes(self):
        results = self._run("def predict(sample):\n    return {'y': sample['x'] * 2}\n")
        self.assertGreater(results["distinct"], 1)
        self.assertEqual([], evaluate_predict_sensitivity(results))

    def test_probe_failure_is_reported(self):
        results = self._run("def predict(sample):\n    raise ValueError('boom')\n")
        self.assertTrue(evaluate_predict_sensitivity(results)[0].startswith("predict_probe_error"))

    def test_too_few_samples_is_not_treated_as_a_constant_predictor(self):
        self.assertEqual(
            ["predict_probe_error:not_enough_samples_to_compare"],
            evaluate_predict_sensitivity({"distinct": 1, "compared": 1}),
        )




class ConstantPredictorGateTest(unittest.TestCase):
    """Wiring: only continuation trials are held to this. The first trial is
    legitimately allowed to be a constant submission-format baseline."""

    def test_first_trial_is_exempt(self):
        from scripts import generic_workspace_auto_loop as loop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            (trial / "continuation_context.json").write_text("{}", encoding="utf-8")
            with unittest.mock.patch("kaggle_research_agent.paths.project_root", return_value=root):
                with unittest.mock.patch.object(loop, "run_predict_sensitivity_probe") as probe:
                    self.assertEqual([], loop._constant_predictor_issues("demo", "trial_001"))
            probe.assert_not_called()

    def test_probe_failures_do_not_block(self):
        from kaggle_research_agent.runtime_contract import evaluate_predict_sensitivity

        issues = evaluate_predict_sensitivity({"error": "ImportError: no module"})
        self.assertTrue(issues[0].startswith("predict_probe_error"))
        self.assertEqual([], [i for i in issues if i.startswith("predict_ignores_input")])



class PerturbationDoesNotPolluteMetricsTest(unittest.TestCase):
    """The probe runs the real harness, which writes the real metrics
    artifact -- including on the perturbed pass, whose score is deliberate
    nonsense. Real incident: 1732.01 (a wrecked-prediction score) was left
    sitting in outputs/metrics.json, so the device built to catch fabricated
    scores had become a source of one."""

    HARNESS = (
        "import json, os\n"
        "from predict_step import predict\n"
        "if __name__ == '__main__':\n"
        "    p = predict({'x': 1.0})['y']\n"
        "    os.makedirs('outputs', exist_ok=True)\n"
        "    json.dump({'cv_score': abs(p - 1.0)}, open('outputs/metrics.json', 'w'))\n"
    )

    def _workspace(self, root: Path):
        (root / "predict_step.py").write_text(
            "def predict(sample):\n    return {'y': sample['x'] * 2}\n", encoding="utf-8"
        )
        (root / "outputs").mkdir(exist_ok=True)
        (root / "scoring_harness.py").write_text(self.HARNESS, encoding="utf-8")

    def _probe(self, root: Path):
        from kaggle_research_agent.runtime_contract import run_scoring_perturbation_probe

        return run_scoring_perturbation_probe(
            root,
            sys.executable,
            harness_module="scoring_harness",
            predict_module="predict_step",
            metrics_path=root / "outputs" / "metrics.json",
            score_key="cv_score",
            timeout=120,
        )

    def test_pre_existing_metrics_are_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._workspace(root)
            original = '{"cv_score": 0.5, "metric": "real"}'
            (root / "outputs" / "metrics.json").write_text(original, encoding="utf-8")

            results = self._probe(root)

            self.assertNotEqual(results["baseline"], results["perturbed"])
            self.assertEqual(original, (root / "outputs" / "metrics.json").read_text(encoding="utf-8"))

    def test_no_metrics_file_is_left_behind_when_none_existed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._workspace(root)

            self._probe(root)

            self.assertFalse((root / "outputs" / "metrics.json").exists())



class ConstantPredictorBlocksAfterRetryTest(unittest.TestCase):
    """Detecting a constant predictor and then running anyway would make the
    check pointless: the trial's score cannot say anything about the change
    it was meant to test."""

    def _run(self, constant_per_attempt):
        from scripts import generic_workspace_auto_loop as loop

        after = {
            "status": "execution_completed",
            "workspace_run": {"status": "completed"},
            "metrics_collection": {"status": "collected"},
        }
        with unittest.mock.patch.object(loop, "generate_scoring_harness", return_value={"status": "already_exists"}):
            with unittest.mock.patch.object(loop, "generate_data_loader", return_value={"status": "already_exists"}):
                with unittest.mock.patch.object(
                    loop, "prepare_workspace_coding_handoff", return_value={"status": "ready"}
                ):
                    with unittest.mock.patch.object(
                        loop,
                        "run_workspace_code_writer",
                        return_value={"status": "accepted", "changed_files": ["predict_step.py"]},
                    ):
                        with unittest.mock.patch.object(loop, "run_workspace_after_coding", return_value=after):
                            with unittest.mock.patch.object(
                                loop, "_constant_predictor_issues", side_effect=constant_per_attempt
                            ):
                                with unittest.mock.patch.object(loop, "log_decision"):
                                    return loop.run_code_writer_trial(
                                        "demo",
                                        "trial_002",
                                        model="gpt-5",
                                        provider="openai",
                                        allow_api=True,
                                        trial_llm_calls=None,
                                        strategy_calls_today=None,
                                    )

    def test_still_constant_after_retry_is_blocked_not_run(self):
        issues = ["predict_ignores_input:same_output_for_every_sample"]
        result = self._run([issues, issues])
        self.assertEqual("code_writer_blocked", result["status"])
        self.assertTrue(result["feedback_ignored"])
        self.assertEqual(issues, result["constant_predictor_issues"])

    def test_fixed_on_retry_proceeds_normally(self):
        result = self._run([["predict_ignores_input:same_output_for_every_sample"], []])
        self.assertEqual("completed", result["status"])

if __name__ == "__main__":
    unittest.main()
