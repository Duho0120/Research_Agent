"""Stage 2: the competition's metric stops being something the agent writes.

The scorer moves out of the workspace into competitions/<id>/, which is
outside the agent's write scope -- so "do not rewrite your own scorer" stops
being a rule and becomes a fact about where files live.
"""

import tempfile
import unittest
from pathlib import Path

from execution_core import METRICS, compute, normalize_metric_name, parse_metric_spec, verify_metric
from execution_core.metric_provisioning import build_generation_prompt, provision_metric


NAME_ONLY = """# Metric

- name: R-Hit@1cm
- objective: maximize
"""

WITH_PROSE = NAME_ONLY + """
## 정의
정답 좌표와의 유클리드 거리가 1cm 이내인 예측의 비율.

## 단위
좌표 x, y, z 는 미터(m). 따라서 임계값은 0.01.
"""

WITH_EXAMPLE = WITH_PROSE + """
## 검산 예시
```json
{"predictions": [{"x": 0.0}, {"x": 1.0}],
 "truths": [{"x": 0.005}, {"x": 1.5}],
 "target_keys": ["x"],
 "expected": 0.5}
```
"""

# A correct implementation: metres, threshold 0.01, higher is better.
GOOD_METRIC = '''
import math
NAME = "r_hit_at_1cm"
OBJECTIVE = "maximize"

def compute(predictions, truths, target_keys):
    hits = 0
    for predicted, actual in zip(predictions, truths):
        squared = sum((float(predicted[k]) - float(actual[k])) ** 2 for k in target_keys)
        if math.sqrt(squared) <= 0.01:
            hits += 1
    return hits / len(truths)
'''

# Reacts to predictions, peaks on perfect ones -- and is off by a factor of
# 100 because it read the threshold as millimetres.
WRONG_UNITS_METRIC = GOOD_METRIC.replace("<= 0.01", "<= 1.0")

# Mean distance instead of hit rate: reacts, is monotonic, and points the
# opposite way from the declared objective.
INVERTED_METRIC = '''
import math
NAME = "r_hit_at_1cm"
OBJECTIVE = "maximize"

def compute(predictions, truths, target_keys):
    total = 0.0
    for predicted, actual in zip(predictions, truths):
        total += math.sqrt(sum((float(predicted[k]) - float(actual[k])) ** 2 for k in target_keys))
    return total / len(truths)
'''

CONSTANT_METRIC = '''
NAME = "r_hit_at_1cm"
OBJECTIVE = "maximize"

def compute(predictions, truths, target_keys):
    return 0.87
'''

RANDOM_METRIC = '''
import random
NAME = "r_hit_at_1cm"
OBJECTIVE = "maximize"

def compute(predictions, truths, target_keys):
    hits = sum(1 for p, a in zip(predictions, truths) if abs(p[target_keys[0]] - a[target_keys[0]]) <= 0.01)
    return hits / len(truths) + random.random() * 1e-9
'''


def _competition(root: Path, metric_md: str) -> Path:
    directory = root / "236716"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metric.md").write_text(metric_md, encoding="utf-8")
    return directory


def _generator(source: str):
    def generate(_prompt: str) -> str:
        return source
    return generate


class MetricNameTest(unittest.TestCase):
    def test_competition_spellings_fold_to_one_key(self):
        self.assertEqual("r_hit_at_1cm", normalize_metric_name("R-Hit@1cm"))
        self.assertEqual("rmsle", normalize_metric_name("RMSLE"))
        self.assertEqual("mean_iou", normalize_metric_name("Mean IoU"))


class MetricSpecTest(unittest.TestCase):
    """Confidence reflects what the competition actually told us."""

    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.md"
            path.write_text(text, encoding="utf-8")
            return parse_metric_spec(path)

    def test_name_only_is_low_confidence(self):
        spec = self._parse(NAME_ONLY)
        self.assertEqual("R-Hit@1cm", spec.name)
        self.assertEqual("maximize", spec.objective)
        self.assertEqual("low", spec.confidence)

    def test_prose_raises_confidence_and_reaches_the_prompt(self):
        spec = self._parse(WITH_PROSE)
        self.assertEqual("medium", spec.confidence)
        self.assertIn("미터(m)", build_generation_prompt(spec, ["x", "y", "z"]))

    def test_a_worked_example_is_the_highest_confidence(self):
        spec = self._parse(WITH_EXAMPLE)
        self.assertEqual("high", spec.confidence)
        self.assertEqual(0.5, spec.worked_example.expected)
        self.assertNotIn("```json", spec.prose)

    def test_missing_file_is_not_an_invented_metric(self):
        self.assertIsNone(parse_metric_spec(Path("nope") / "metric.md"))


class MetricVerificationTest(unittest.TestCase):
    """What each check is for, stated as the failure it catches."""

    def _verify(self, source: str, **kwargs):
        namespace: dict = {}
        exec(source, namespace)
        return verify_metric(
            namespace["compute"], objective="maximize", target_keys=["x"], **kwargs
        )

    def test_a_correct_metric_passes(self):
        self.assertEqual([], self._verify(GOOD_METRIC)["issues"])

    def test_a_constant_scorer_is_caught(self):
        issues = self._verify(CONSTANT_METRIC)["issues"]
        self.assertIn("metric_ignores_predictions:score_unchanged_when_predictions_wrecked", issues)

    def test_a_metric_pointing_the_wrong_way_is_caught(self):
        """Mean distance where a hit rate was declared.

        This is the failure a sensitivity check cannot see: the score does
        react to predictions. It just orders them backwards, so the agent
        optimises away from the answer while every rule reads as normal."""
        issues = self._verify(INVERTED_METRIC)["issues"]
        self.assertIn("metric_direction_contradicts_objective:maximize", issues)

    def test_a_non_deterministic_metric_is_caught(self):
        self.assertIn("metric_is_not_deterministic", self._verify(RANDOM_METRIC)["issues"])

    def test_wrong_units_survive_every_check_but_the_worked_example(self):
        """The whole argument for asking a human for one example.

        Read "1cm" as millimetres and the metric still reacts, still peaks on
        perfect predictions, still points the right way -- it is simply
        measuring the wrong thing, by a factor of a hundred."""
        self.assertEqual([], self._verify(WRONG_UNITS_METRIC)["issues"])

        with tempfile.TemporaryDirectory() as tmp:
            spec = parse_metric_spec(_competition(Path(tmp), WITH_EXAMPLE) / "metric.md")
            caught = self._verify(WRONG_UNITS_METRIC, worked_example=spec.worked_example)
        self.assertTrue(
            any(i.startswith("metric_disagrees_with_worked_example") for i in caught["issues"]),
            caught["issues"],
        )


class ProvisioningLadderTest(unittest.TestCase):
    """Generation is the last rung, not the first."""

    def test_a_known_metric_costs_no_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(
                Path(tmp), "# Metric\n\n- name: RMSLE\n- objective: minimize\n"
            )
            calls = []

            def generate(prompt):
                calls.append(prompt)
                return ""

            result = provision_metric(directory, target_keys=["count"], generate=generate)

        self.assertEqual("ready", result["status"])
        self.assertEqual("builtin", result["source"])
        self.assertEqual([], calls)

    def test_an_unknown_metric_is_generated_verified_and_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_EXAMPLE)
            result = provision_metric(
                directory, target_keys=["x"], generate=_generator(GOOD_METRIC)
            )

            self.assertEqual("ready", result["status"], result.get("issues"))
            self.assertEqual("generated", result["source"])
            self.assertEqual("high", result["confidence"])
            self.assertTrue((directory / "competition_metric.py").is_file())

        # Returned for this competition, not registered globally: a metric
        # provisioned for one competition must not change how another is
        # scored inside the same process.
        self.assertAlmostEqual(
            0.5, compute(result["spec"], [{"x": 0.0}, {"x": 1.0}], [{"x": 0.005}, {"x": 1.5}], ["x"])
        )
        self.assertNotIn("r_hit_at_1cm", METRICS)

    def test_a_rejected_implementation_is_deleted_not_left_behind(self):
        """A file left on disk is treated as provisioned by the next run.

        Real incident, in the old harness path: a rejected artifact stayed
        put and an 'already exists' branch promoted it on the following
        cycle."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_PROSE)
            result = provision_metric(
                directory, target_keys=["x"], generate=_generator(CONSTANT_METRIC)
            )

            self.assertEqual("blocked", result["status"])
            self.assertFalse((directory / "competition_metric.py").exists())
            self.assertEqual(2, len(result["attempts"]))

    def test_the_retry_is_told_what_was_wrong(self):
        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return CONSTANT_METRIC

        with tempfile.TemporaryDirectory() as tmp:
            provision_metric(_competition(Path(tmp), WITH_PROSE), target_keys=["x"], generate=generate)

        self.assertEqual(2, len(prompts))
        self.assertIn("metric_ignores_predictions", prompts[1])

    def test_a_name_only_spec_yields_a_low_confidence_metric_not_a_refusal(self):
        """Optional by design.

        Requiring a definition would just get the field filled in to satisfy
        the requirement -- the same pattern every static check here has already
        lost to. Running with the uncertainty visible beats a formality."""
        with tempfile.TemporaryDirectory() as tmp:
            result = provision_metric(
                _competition(Path(tmp), NAME_ONLY), target_keys=["x"], generate=_generator(GOOD_METRIC)
            )
        self.assertEqual("ready", result["status"])
        self.assertEqual("low", result["confidence"])

    def test_generation_unavailable_blocks_rather_than_guesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = provision_metric(_competition(Path(tmp), NAME_ONLY), target_keys=["x"])
        self.assertEqual("blocked", result["status"])
        self.assertTrue(result["issues"][0].startswith("metric_not_implemented"))

    def test_a_disagreeing_objective_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(
                Path(tmp), "# Metric\n\n- name: RMSLE\n- objective: maximize\n"
            )
            result = provision_metric(directory, target_keys=["count"])
        self.assertEqual("blocked", result["status"])
        self.assertTrue(result["issues"][0].startswith("objective_conflicts_with_builtin"))

    def test_an_existing_module_is_reused_without_regenerating(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_EXAMPLE)
            provision_metric(directory, target_keys=["x"], generate=_generator(GOOD_METRIC))

            calls = []
            again = provision_metric(
                directory,
                target_keys=["x"],
                generate=lambda prompt: calls.append(prompt) or GOOD_METRIC,
            )
        self.assertEqual("provisioned", again["source"])
        self.assertEqual([], calls)
        # The label has to survive reuse -- it is what tells a reader how much
        # to trust the score, and it is read downstream, not here.
        self.assertEqual("high", again["confidence"])


class WorkedExampleReachesProvisioningTest(unittest.TestCase):
    """The example must travel from metric.md into the check that uses it.

    Real incident: it did not. verify_metric was reading the example off the
    generated module instead of off the competition's spec, so an
    implementation was effectively checked against a case it supplied itself.
    Every unit test passed -- they called verify_metric directly and handed it
    the example -- and the gap only appeared when a real metric.md went
    through the real provisioning path: an implementation reading the
    threshold as 1.0 instead of 0.01 was accepted.

    The same shape of mistake as an audit report that showed a section the API
    payload never carried: the check existed, and nothing routed to it."""

    def test_wrong_threshold_is_rejected_through_provision_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_EXAMPLE)
            result = provision_metric(
                directory,
                target_keys=["x"],
                generate=_generator(GOOD_METRIC.replace("<= 0.01", "<= 1.0")),
            )
        self.assertEqual("blocked", result["status"])
        self.assertTrue(
            any(i.startswith("metric_disagrees_with_worked_example") for i in result["issues"]),
            result["issues"],
        )

    def test_a_module_cannot_supply_its_own_worked_example(self):
        """Grading your own homework proves nothing."""
        self_serving = GOOD_METRIC.replace("<= 0.01", "<= 1.0") + (
            "\nfrom execution_core.metric_spec import WorkedExample\n"
            "WORKED_EXAMPLE = WorkedExample([{'x': 0.0}], [{'x': 0.5}], ['x'], 1.0)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_EXAMPLE)
            result = provision_metric(directory, target_keys=["x"], generate=_generator(self_serving))
        self.assertEqual("blocked", result["status"])

    def test_a_previously_accepted_module_is_rechecked_against_the_current_spec(self):
        """metric.md may gain an example after the module was provisioned."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = _competition(Path(tmp), WITH_PROSE)
            wrong = GOOD_METRIC.replace("<= 0.01", "<= 1.0")
            accepted = provision_metric(directory, target_keys=["x"], generate=_generator(wrong))
            self.assertEqual("ready", accepted["status"])  # prose alone cannot see it

            (directory / "metric.md").write_text(WITH_EXAMPLE, encoding="utf-8")
            rechecked = provision_metric(directory, target_keys=["x"])

        self.assertEqual("blocked", rechecked["status"])
        self.assertEqual("provisioned", rechecked["source"])


class BuiltinMetricTest(unittest.TestCase):
    def test_rmsle_matches_its_definition(self):
        import math

        score = compute("rmsle", [{"c": 2.0}], [{"c": 3.0}], ["c"])
        self.assertAlmostEqual(abs(math.log1p(2.0) - math.log1p(3.0)), score)

    def test_rmsle_clips_negative_predictions_but_rejects_negative_truths(self):
        self.assertGreater(compute("rmsle", [{"c": -5.0}], [{"c": 3.0}], ["c"]), 0)
        with self.assertRaises(ValueError):
            compute("rmsle", [{"c": 1.0}], [{"c": -3.0}], ["c"])


if __name__ == "__main__":
    unittest.main()
