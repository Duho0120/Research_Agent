from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.trial_user_view import write_proposed_plan_preview


class WriteProposedPlanPreviewTest(unittest.TestCase):
    def test_writes_korean_labeled_preview_from_proposed_plan_without_execution_facts(self):
        # Before a trial has been coded/executed, resolve_trial_plan() (used by
        # the post-execution render path) has nothing to resolve. This preview
        # must render directly from the plan the planner just produced, so the
        # dashboard has a Korean-labeled document even before code writing runs.
        plan = {
            "plan_type": "continuation_delta_plan",
            "source_trial_id": "trial_012",
            "primary_change_axis": "feature_engineering:new_simple_numeric_features",
            "plan_title": "Add windspeed_zero indicator on top of trial_012",
            "rationale": "Keep Poisson HGBR fixed; add one low-risk feature.",
            "change_details": ["Name: windspeed_is_zero", "Formula: (windspeed <= 1e-3).astype(int)"],
            "keep_unchanged": ["model_definition: HistGradientBoostingRegressor with current params"],
            "candidate": {"name": "windspeed_is_zero", "description": "flag near-zero windspeed readings"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                preview_path = write_proposed_plan_preview(
                    "demo", "trial_013", plan, metric="rmsle", objective="minimize"
                )
                content = preview_path.read_text(encoding="utf-8")

            self.assertTrue(preview_path.is_file())
            self.assertIn("trial_013 실험 계획", content)
            self.assertIn("기준 trial", content)
            self.assertIn("trial_012", content)
            self.assertIn("windspeed_is_zero", content)
            self.assertIn("rmsle", content)

    def test_preview_is_overwritten_by_effective_render_after_execution(self):
        # The preview lives at the same path (01_plan.ko.md) as the
        # post-execution "effective" document. organize_trial_artifacts writes
        # whatever render_user_view_files returns to that same path, so once
        # the trial actually runs, the preview is naturally replaced with the
        # ground-truth version -- no staleness.
        from research_agent.store import write_text
        from research_agent.trial_user_view import render_user_view_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                preview_path = write_proposed_plan_preview(
                    "demo",
                    "trial_013",
                    {"plan_title": "proposed", "primary_change_axis": "feature_engineering"},
                )
                with patch(
                    "research_agent.plan_translation.render_effective_user_plan",
                    return_value="# effective content",
                ):
                    files = render_user_view_files(
                        preview_path.parent.parent, {"competition": "demo", "trial_id": "trial_013"}, []
                    )
                write_text(preview_path, files["01_plan.ko.md"])

            content = preview_path.read_text(encoding="utf-8")
            self.assertEqual("# effective content", content)


if __name__ == "__main__":
    unittest.main()
