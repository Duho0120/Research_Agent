from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.execution_plan_snapshot import (
    capture_pending_execution_plan,
    ensure_pending_execution_plan_snapshot,
    finalize_execution_plan_snapshot,
    load_execution_plan_snapshot,
)
from kaggle_research_agent.paths import trial_dir
from kaggle_research_agent.store import write_text


class ExecutionPlanSnapshotTest(unittest.TestCase):
    def test_pending_plan_can_change_before_finalization_but_final_plan_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_request = self._request("trial_001", "model_family", "Use RandomForest")
            revised_request = self._request("trial_003", "validation_review", "Use TimeSeriesSplit")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                capture_pending_execution_plan(
                    "demo",
                    "trial_006",
                    request_text=first_request,
                    request_id="demo:trial_006:workspace-coding",
                )
                pending = capture_pending_execution_plan(
                    "demo",
                    "trial_006",
                    request_text=revised_request,
                    request_id="demo:trial_006:workspace-coding",
                )
                finalized = finalize_execution_plan_snapshot("demo", "trial_006")
                capture_pending_execution_plan(
                    "demo",
                    "trial_006",
                    request_text=first_request,
                    request_id="demo:trial_006:workspace-coding:retry",
                )
                unchanged = finalize_execution_plan_snapshot("demo", "trial_006")
                loaded = load_execution_plan_snapshot("demo", "trial_006")

            self.assertEqual("validation_review", pending["plan"]["primary_change_axis"])
            self.assertEqual("trial_003", finalized["plan"]["source_trial_id"])
            self.assertEqual("Use TimeSeriesSplit", finalized["plan"]["plan_title"])
            self.assertEqual(finalized, unchanged)
            self.assertEqual(finalized, loaded)

    def test_ensure_pending_snapshot_recaptures_after_a_force_replan_between_coding_attempts(self):
        # A blocked code-writing attempt writes a pending snapshot from the plan
        # in effect at that time. If the plan is then force-replanned (e.g. a
        # stale continuation_context.json fix, or a duplicate-candidate replan)
        # before the next coding attempt, ensure_pending_execution_plan_snapshot
        # must not keep serving the old cached snapshot -- otherwise
        # finalize_execution_plan_snapshot freezes the wrong plan as the trial's
        # recorded fact, corrupting axis/candidate history for every later trial.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_request = self._request("trial_001", "model_family", "Use RandomForest")
            replanned_request = self._request("trial_003", "validation_review", "Use TimeSeriesSplit")

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                out_dir = trial_dir("demo", "trial_006")
                write_text(out_dir / "workspace_coding_agent_request.md", first_request)
                first = ensure_pending_execution_plan_snapshot(
                    "demo", "trial_006", request_id="demo:trial_006:workspace-coding"
                )
                self.assertEqual("model_family", first["plan"]["primary_change_axis"])

                # Retrying with the identical request (e.g. expanded_snapshot retry
                # with the same underlying plan) must reuse the cached snapshot.
                same_again = ensure_pending_execution_plan_snapshot(
                    "demo", "trial_006", request_id="demo:trial_006:workspace-coding:retry"
                )
                self.assertEqual(first["captured_at"], same_again["captured_at"])

                # A genuinely replanned request must be detected and recaptured.
                write_text(out_dir / "workspace_coding_agent_request.md", replanned_request)
                after_replan = ensure_pending_execution_plan_snapshot(
                    "demo", "trial_006", request_id="demo:trial_006:workspace-coding:retry-2"
                )
                finalized = finalize_execution_plan_snapshot("demo", "trial_006")

            self.assertEqual("validation_review", after_replan["plan"]["primary_change_axis"])
            self.assertEqual("validation_review", finalized["plan"]["primary_change_axis"])
            self.assertEqual("trial_003", finalized["plan"]["source_trial_id"])

    @staticmethod
    def _request(source_trial: str, axis: str, title: str) -> str:
        return "\n".join(
            [
                "# Coding request",
                "",
                "## Next Experiment",
                "",
                "# trial_006 Demo Experiment Plan",
                "",
                "- plan_type: delta_patch",
                f"- source_trial_id: {source_trial}",
                f"- title: {title}",
                "",
                "## Primary Change Axis",
                "",
                axis,
                "",
                "## Change Details",
                "",
                f"- {title}",
            ]
        )


if __name__ == "__main__":
    unittest.main()
