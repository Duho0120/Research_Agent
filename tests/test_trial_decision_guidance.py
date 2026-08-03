from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.trial_decision import (
    _base_trial_change_axis,
    _planner_constraints,
    write_trial_decision_card,
)


class PlannerConstraintsTest(unittest.TestCase):
    def test_continue_axis_refinement_states_base_lacks_the_axis_as_a_fact(self):
        # Regression for trial_027: after trial_026's model-family switch to
        # Ridge was not accepted as the new base, trial_027 continued the
        # same axis with recommended_base_trial=trial_014 (still
        # HistGradientBoostingRegressor). The old guidance only said "use
        # trial_014 as the base even if the axis was attempted elsewhere",
        # which the planner read as "Ridge already exists there, just tweak
        # its params" -- producing a patch that looked for Ridge( in code
        # that never had it. This is now a computed FACT (trial_014's own
        # change_axis differs from the active axis), not a hint to infer.
        constraints = _planner_constraints(
            "continue_axis_refinement",
            current_axis="model_hyperparameters:Ridge",
            recommended_base_trial="trial_014",
            base_trial_axis="model_params:increase_l2_regularization",
        )
        combined = " ".join(constraints)
        self.assertIn("FACT:", combined)
        self.assertIn("does NOT contain the `model_hyperparameters:Ridge` change yet", combined)
        self.assertIn("trial_014", combined)

    def test_continue_axis_refinement_states_base_already_reflects_the_axis(self):
        constraints = _planner_constraints(
            "continue_axis_refinement",
            current_axis="model_hyperparameters:Ridge",
            recommended_base_trial="trial_020",
            base_trial_axis="model_hyperparameters:Ridge",
        )
        combined = " ".join(constraints)
        self.assertIn("FACT:", combined)
        self.assertIn("already reflects this axis", combined)

    def test_continue_axis_refinement_without_known_base_axis_omits_the_fact(self):
        constraints = _planner_constraints(
            "continue_axis_refinement",
            current_axis="model_hyperparameters:Ridge",
            recommended_base_trial="trial_014",
        )
        combined = " ".join(constraints)
        self.assertNotIn("FACT:", combined)


class BaseTrialChangeAxisTest(unittest.TestCase):
    def test_looks_up_the_recommended_base_trials_own_axis(self):
        cards = [
            {"trial_id": "trial_013", "change_axis": "feature_selection:x"},
            {"trial_id": "trial_014", "change_axis": "model_params:increase_l2_regularization"},
        ]
        self.assertEqual(
            "model_params:increase_l2_regularization",
            _base_trial_change_axis(cards, "trial_014"),
        )

    def test_returns_none_when_base_trial_not_found(self):
        self.assertEqual(None, _base_trial_change_axis([{"trial_id": "trial_001"}], "trial_099"))

    def test_returns_none_when_no_base_trial_given(self):
        self.assertEqual(None, _base_trial_change_axis([{"trial_id": "trial_001"}], None))


class WriteTrialDecisionCardAxisFactTest(unittest.TestCase):
    def test_continue_axis_refinement_card_carries_the_base_axis_mismatch_fact(self):
        # End-to-end version of the trial_014 -> trial_026 -> trial_027
        # scenario: trial_014 was planned under a different axis, trial_026
        # tried model_hyperparameters:Ridge but did not improve enough to
        # become the new base, so trial_027 continues that axis with
        # trial_014 (still the old axis's code) as the base.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_014",
                    plan={"plan_type": "continuation_delta_plan", "primary_change_axis": "model_params:increase_l2"},
                    metrics={"cv_score": 0.39, "objective": "minimize"},
                )
                # Mild (non-catastrophic) regressions: the axis keeps its
                # refinement attempts, so the base-axis-mismatch FACT applies.
                # Catastrophic regressions now reject immediately and are
                # covered by CatastrophicRegressionTest instead.
                write_trial_decision_card(
                    "demo",
                    "trial_026",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_014",
                        "primary_change_axis": "model_hyperparameters:Ridge",
                    },
                    metrics={"cv_score": 0.40, "objective": "minimize"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_027",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_026",
                        "primary_change_axis": "model_hyperparameters:Ridge",
                    },
                    metrics={"cv_score": 0.41, "objective": "minimize"},
                )

        self.assertEqual("trial_014", card["recommended_base_trial"])
        combined = " ".join(card["planner_constraints"])
        self.assertIn("FACT:", combined)
        self.assertIn("does NOT contain the `model_hyperparameters:Ridge` change yet", combined)


class CatastrophicRegressionTest(unittest.TestCase):
    def test_catastrophic_regression_rejects_axis_immediately_without_refinement_attempts(self):
        # Regression for trial_023-025: feature_selection dropped the
        # dominant hour signal and scored 1.248 vs base 0.391 (3.2x worse).
        # The status-only decision treated that like a 0.1% dip and granted
        # the axis two more full trials of "refinement" on an obviously
        # broken idea. A first-attempt regression this large must reject the
        # axis immediately.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_014",
                    plan={"plan_type": "continuation_delta_plan", "primary_change_axis": "model_params:increase_l2"},
                    metrics={"cv_score": 0.3912, "objective": "minimize"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_023",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_014",
                        "primary_change_axis": "feature_selection:use_only_recommended_numeric_features",
                    },
                    metrics={"cv_score": 1.2483, "objective": "minimize"},
                )

        self.assertTrue(card["catastrophic_regression"])
        self.assertEqual("reject_or_hold", card["decision"])
        self.assertIn("feature_selection:use_only_recommended_numeric_features", card["rejected_axes"])
        self.assertTrue(any("far beyond tuning noise" in item for item in card["planner_constraints"]))

    def test_mild_regression_still_gets_refinement_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.3912, "objective": "minimize"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering:new_feature",
                    },
                    metrics={"cv_score": 0.3950, "objective": "minimize"},
                )

        self.assertFalse(card["catastrophic_regression"])
        self.assertEqual("continue_axis_refinement", card["decision"])


class EstimatorFamilySwapTest(unittest.TestCase):
    def test_rejected_estimator_swap_also_rejects_model_family_axis(self):
        # Regression for trial_026-028: "model_hyperparameters:Ridge" on an
        # HGBR base is a model-family swap wearing a tuning axis's name, and
        # the free-text axis comparison let it bypass the already-rejected
        # model_family axis. When such a trial is rejected, model_family must
        # land in rejected_axes and the planner must be told explicitly.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_014",
                    plan={"plan_type": "continuation_delta_plan", "primary_change_axis": "model_params:increase_l2"},
                    metrics={"cv_score": 0.3912, "objective": "minimize", "model_type": "HistGradientBoostingRegressor"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_026",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_014",
                        "primary_change_axis": "model_hyperparameters:Ridge",
                    },
                    metrics={"cv_score": 0.8628, "objective": "minimize", "model_type": "Ridge"},
                )

        self.assertTrue(card["estimator_family_changed"])
        self.assertIn("model_family", card["rejected_axes"])
        self.assertTrue(any("estimator family" in item for item in card["planner_constraints"]))


class NoChangeSuspectedTest(unittest.TestCase):
    def test_bit_identical_score_to_code_base_is_flagged(self):
        # Regression for trial_008-010: three consecutive trials scored
        # bit-identically to trial_003 because the vague plans changed
        # nothing, yet each burned an axis attempt as if a real experiment
        # had run and failed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={"plan_type": "continuation_delta_plan", "primary_change_axis": "model_family"},
                    metrics={"cv_score": 0.3540896976524032, "objective": "minimize"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_008",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_003",
                        "primary_change_axis": "sota_architecture_attempt",
                    },
                    metrics={"cv_score": 0.3540896976524032, "objective": "minimize"},
                )

        self.assertTrue(card["no_change_suspected"])
        self.assertTrue(any("bit-identical" in item for item in card["planner_constraints"]))


class AxisNormalizationTest(unittest.TestCase):
    def test_prefix_synonyms_and_spacing_count_as_the_same_axis(self):
        from kaggle_research_agent.trial_decision import _normalize_axis

        self.assertEqual(
            _normalize_axis("model_params:increase_l2"),
            _normalize_axis("model_hyperparameters: increase_l2"),
        )
        self.assertEqual(
            _normalize_axis("model_hyperparameters: HistGradientBoostingRegressor"),
            _normalize_axis("model_hyperparameters:HistGradientBoostingRegressor"),
        )
        self.assertNotEqual(
            _normalize_axis("model_hyperparameters:Ridge"),
            _normalize_axis("feature_engineering:Ridge"),
        )


class CandidateLabelTest(unittest.TestCase):
    def test_class_names_are_not_truncated_mid_word(self):
        from kaggle_research_agent.trial_decision import _candidate_label

        # Regression: details were pre-truncated to 90 chars before signal
        # extraction, so long class names near the boundary were stored as
        # fragments ("TransformedTarg") the planner could not match.
        detail = (
            "Wrap the estimator so predictions come back on the original scale using "
            "TransformedTargetRegressor around the existing pipeline model step"
        )
        label = _candidate_label(
            {
                "primary_change_axis": "model_family",
                "plan_title": "Wrap estimator in a target transformer",
                "change_details": [detail],
            }
        )
        self.assertIn("TransformedTargetRegressor", label)
        self.assertNotIn("TransformedTarg |", label)


if __name__ == "__main__":
    unittest.main()
