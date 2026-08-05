"""Stage 3b-2: asking for model.py, and what happens to the answer.

The legacy coding path reads returned source through twelve checks before
running anything, and every one of them has at some point been satisfied in
letter while skipped in substance. Here two checks remain, both about things
that would waste a run rather than fake a result, and the verdict comes from
executing the code.
"""

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from research_agent import contract_coding
from research_agent.contract_coding import _apply_files, validate_contract_code


GOOD_MODEL = "def fit(train_samples):\n    return None\n\n\ndef predict(fitted, sample):\n    return {'y': 1}\n"


def _handoff(root: Path, allowed=("model.py",)) -> dict:
    return {"project_root": str(root), "allowed_paths": list(allowed)}


class WritableScopeTest(unittest.TestCase):
    """The loader is unlocked only by the axis that is about the loader.

    Rewriting it changes what every past score meant, so it is a decision the
    plan has to make out loud -- the same gate the legacy path applied, kept."""

    def test_model_only_by_default(self):
        self.assertEqual(["model.py"], contract_coding._writable_files(""))
        self.assertEqual(["model.py"], contract_coding._writable_files("feature_engineering"))

    def test_the_data_loading_axis_unlocks_the_loader(self):
        self.assertEqual(
            ["model.py", "data_loader.py"], contract_coding._writable_files("data_loading")
        )

    def test_whatever_is_not_writable_is_named_forbidden(self):
        forbidden = contract_coding._forbidden_files(["model.py"])
        self.assertIn("data_loader.py", forbidden)
        self.assertIn("outputs/metrics.json", forbidden)
        self.assertNotIn("model.py", forbidden)


class ApplyFilesTest(unittest.TestCase):
    def test_a_file_in_scope_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _apply_files(
                _handoff(root), {"files": [{"path": "model.py", "content": GOOD_MODEL}]}
            )
            self.assertEqual([], result["issues"])
            self.assertEqual(["model.py"], result["changed_files"])
            self.assertIn("def fit", (root / "model.py").read_text(encoding="utf-8"))

    def test_an_empty_padding_entry_is_ignored(self):
        """Real incident: a correct response was thrown away over one of these.

        The model returned model.py plus a trailing {"path": "", "content": ""}.
        Refusing the whole response cost a 5,000-token call and produced the
        useless verdict wrote_outside_scope:<empty>. An entry asking for
        nothing is a no-op, not a scope violation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _apply_files(
                _handoff(root),
                {"files": [{"path": "model.py", "content": GOOD_MODEL}, {"path": "", "content": ""}]},
            )
            self.assertEqual([], result["issues"])
            self.assertEqual(["model.py"], result["changed_files"])

    def test_a_named_file_with_no_content_still_fails(self):
        """Declaring a file and supplying nothing is not padding."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _apply_files(
                _handoff(Path(tmp)), {"files": [{"path": "model.py", "content": "   "}]}
            )
            self.assertEqual(["empty_content:model.py"], result["issues"])

    def test_out_of_scope_writes_nothing_at_all(self):
        """Refused as a unit: a half-applied change leaves a workspace that
        matches neither the previous trial nor this one."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _apply_files(
                _handoff(root),
                {
                    "files": [
                        {"path": "model.py", "content": GOOD_MODEL},
                        {"path": "data_loader.py", "content": "x = 1"},
                    ]
                },
            )
            self.assertEqual(["wrote_outside_scope:data_loader.py"], result["issues"])
            self.assertFalse((root / "model.py").exists())


class StaticCheckTest(unittest.TestCase):
    """Only what would waste a run. Whether the model is any good is the
    trial's job, and asking source code that question is what kept failing."""

    def test_a_valid_model_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text(GOOD_MODEL, encoding="utf-8")
            self.assertEqual([], validate_contract_code(root, _handoff(root), ["model.py"]))

    def test_a_missing_function_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text("def fit(t):\n    return None\n", encoding="utf-8")
            self.assertEqual(
                ["model_missing_function:predict"],
                validate_contract_code(root, _handoff(root), ["model.py"]),
            )

    def test_a_file_that_cannot_be_imported_is_caught_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text("def fit(:\n", encoding="utf-8")
            issues = validate_contract_code(root, _handoff(root), ["model.py"])
            self.assertTrue(issues[0].startswith("model_does_not_parse"), issues)

    def test_a_constant_predictor_is_not_rejected_here(self):
        """Deliberately allowed through. It is caught by running it, where a
        source check can be talked around and an observation cannot."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text(
                "def fit(t):\n    return None\n\n\ndef predict(f, s):\n    return {'x': 0.0}\n",
                encoding="utf-8",
            )
            self.assertEqual([], validate_contract_code(root, _handoff(root), ["model.py"]))


class PromptTest(unittest.TestCase):
    """One prompt string, built once.

    Real incident: the legacy path rendered the audit copy and the API payload
    from separate code, and a section that appeared in the report was never in
    the request -- for several trials, unnoticed."""

    def _prompt(self, **kwargs) -> str:
        defaults = {"plan": "PLAN-MARKER", "primary_axis": "feature_engineering",
                    "writable": ["model.py"], "loader_sample": None, "feedback": None}
        return contract_coding._build_prompt("demo", **{**defaults, **kwargs})

    def test_the_contract_and_the_plan_are_both_in_it(self):
        prompt = self._prompt()
        self.assertIn("def fit(train_samples", prompt)
        self.assertIn("def predict(fitted", prompt)
        self.assertIn("PLAN-MARKER", prompt)

    def test_the_loader_contract_appears_only_when_the_loader_is_writable(self):
        self.assertNotIn("def load_samples", self._prompt())
        self.assertIn("def load_samples", self._prompt(writable=["model.py", "data_loader.py"]))

    def test_rejection_feedback_carries_the_stage_and_the_traceback(self):
        prompt = self._prompt(
            feedback={"failed_stage": "fit", "issues": ["execution_failed:fit"], "error": "ZeroDivisionError-MARKER"}
        )
        self.assertIn("Failed at stage: fit", prompt)
        self.assertIn("ZeroDivisionError-MARKER", prompt)

    def test_the_scoring_axis_is_told_the_metric_is_not_negotiable(self):
        prompt = self._prompt(primary_axis="scoring_logic")
        self.assertIn("the metric is fixed by the competition", prompt)

    def test_the_saved_request_is_the_string_that_was_sent(self):
        handoff = {"prompt": "EXACT-TEXT", "allowed_paths": ["model.py"], "primary_change_axis": ""}
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(contract_coding, "trial_dir", return_value=Path(tmp)):
                with unittest.mock.patch.object(contract_coding, "log_decision"):
                    path = contract_coding.write_contract_coding_request("demo", "trial_001", handoff)
            self.assertEqual("EXACT-TEXT\n", path.read_text(encoding="utf-8"))


class CodeWriterTest(unittest.TestCase):
    def test_no_api_permission_blocks_without_calling_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(contract_coding, "trial_dir", return_value=Path(tmp)):
                with unittest.mock.patch.object(contract_coding, "log_decision"):
                    result = contract_coding.run_contract_code_writer(
                        "demo", "trial_001", {"prompt": "x", "project_root": tmp, "allowed_paths": []},
                        allow_api=False,
                    )
        self.assertEqual("blocked", result["status"])
        self.assertEqual(["api_call_not_enabled"], result["issues"])

    def test_a_response_is_applied_and_recorded(self):
        class Client:
            def create_response(self, _payload):
                return {
                    "output_text": json.dumps(
                        {"files": [{"path": "model.py", "content": GOOD_MODEL}], "summary": "ok"}
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with unittest.mock.patch.object(contract_coding, "trial_dir", return_value=root):
                with unittest.mock.patch.object(contract_coding, "log_decision"):
                    result = contract_coding.run_contract_code_writer(
                        "demo", "trial_001", _handoff(root) | {"prompt": "x"}, client=Client()
                    )
            self.assertEqual("accepted", result["status"])
            self.assertEqual(["model.py"], result["changed_files"])
            self.assertTrue((root / "contract_coding_result.json").is_file())


if __name__ == "__main__":
    unittest.main()
