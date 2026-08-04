"""Stage 3b: routing a competition to the contract model, one profile at a time.

The migration rule the plan commits to is that nothing changes for a
competition until its own profile says so. These tests hold that line: the
dispatch is driven by the profile and by nothing else, and a competition still
on the old model must take the old path unchanged.
"""

import unittest
import unittest.mock
from pathlib import Path

from research_agent import workspace_runner
from research_agent.execution_profile import (
    CONTRACT_MODEL,
    LEGACY_MODEL,
    contract_data_dir,
    contract_submission_template,
    execution_model,
)


class ExecutionModelTest(unittest.TestCase):
    """Legacy is the default, so a profile written before the contract model
    existed keeps behaving exactly as it did."""

    def test_absent_field_means_legacy(self):
        self.assertEqual(LEGACY_MODEL, execution_model({}))

    def test_contract_is_opt_in_by_exact_value(self):
        self.assertEqual(CONTRACT_MODEL, execution_model({"execution_model": "contract"}))
        self.assertEqual(CONTRACT_MODEL, execution_model({"execution_model": " Contract "}))
        self.assertEqual(LEGACY_MODEL, execution_model({"execution_model": "contractual"}))
        self.assertEqual(LEGACY_MODEL, execution_model({"execution_model": ""}))


class ContractPathResolutionTest(unittest.TestCase):
    def test_absolute_data_dir_is_used_as_given(self):
        profile = {"project_root": r"C:\ws", "data_dir": r"D:\elsewhere\data"}
        self.assertEqual(Path(r"D:\elsewhere\data"), contract_data_dir(profile))

    def test_relative_data_dir_resolves_against_project_root(self):
        profile = {"project_root": r"C:\ws", "data_dir": "data"}
        self.assertEqual(Path(r"C:\ws") / "data", contract_data_dir(profile))

    def test_template_resolves_against_the_data_dir_not_the_workspace(self):
        """The template ships with the data, wherever that was unpacked."""
        profile = {"project_root": r"C:\ws", "data_dir": r"D:\data", "submission_template": "sample_submission.csv"}
        self.assertEqual(Path(r"D:\data") / "sample_submission.csv", contract_submission_template(profile))

    def test_no_template_declared_is_none_not_a_guess(self):
        self.assertIsNone(contract_submission_template({"project_root": r"C:\ws", "data_dir": "data"}))


class DispatchTest(unittest.TestCase):
    """One dispatch point, driven by the profile.

    Callers do not choose the execution model -- if they could, a competition
    would migrate on some entry points and not others, which is how the old
    constant-predictor check ended up reachable from one path out of three."""

    def _run(self, profile: dict):
        with unittest.mock.patch.object(
            workspace_runner, "validate_execution_profile", return_value={"status": "ready", "issues": []}
        ):
            with unittest.mock.patch.object(workspace_runner, "load_execution_profile", return_value=profile):
                with unittest.mock.patch(
                    "research_agent.contract_execution.run_contract_pipeline",
                    return_value={"status": "completed", "execution_model": "contract"},
                ) as contract:
                    with unittest.mock.patch.object(workspace_runner, "_execute_commands") as legacy:
                        with unittest.mock.patch.object(workspace_runner, "log_decision"):
                            with unittest.mock.patch.object(workspace_runner, "_write_result"):
                                result = workspace_runner.run_workspace_pipeline(
                                    "demo", "trial_001", run_now=True
                                )
        return result, contract, legacy

    LEGACY_PROFILE = {
        "project_root": r"C:\ws",
        "python": "python",
        "commands": {"train": ["{python} train_step.py"]},
        "artifacts": {},
    }

    def test_a_contract_profile_takes_the_new_path(self):
        result, contract, legacy = self._run({**self.LEGACY_PROFILE, "execution_model": "contract"})
        self.assertEqual("contract", result["execution_model"])
        contract.assert_called_once()
        legacy.assert_not_called()

    def test_a_legacy_profile_is_untouched(self):
        """The Kaggle competitions run on this path and must not move."""
        _result, contract, legacy = self._run(self.LEGACY_PROFILE)
        contract.assert_not_called()
        legacy.assert_called_once()


class RealProfilesTest(unittest.TestCase):
    """The migration is per competition. This is the regression guard: a change
    made for one competition must not silently move another.

    Real incident: an interface requirement added for 236716 blocked
    bike-sharing-demand, which had been working, and the symptom was noticed by
    the user rather than by any check."""

    def test_each_competition_is_on_the_model_its_profile_declares(self):
        from research_agent.execution_profile import load_execution_profile

        expected = {
            "236716": CONTRACT_MODEL,
            "titanic": LEGACY_MODEL,
            "bike-sharing-demand": LEGACY_MODEL,
        }
        for competition, model in expected.items():
            try:
                profile = load_execution_profile(competition)
            except (FileNotFoundError, ValueError):
                self.skipTest(f"{competition} is not configured in this checkout")
            self.assertEqual(model, execution_model(profile), competition)

    def test_every_configured_competition_still_validates(self):
        from research_agent.execution_profile import validate_execution_profile

        for competition in ("236716", "titanic", "bike-sharing-demand"):
            try:
                result = validate_execution_profile(competition)
            except (FileNotFoundError, ValueError):
                self.skipTest(f"{competition} is not configured in this checkout")
            self.assertEqual("ready", result["status"], f"{competition}: {result['issues']}")


if __name__ == "__main__":
    unittest.main()
