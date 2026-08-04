from __future__ import annotations

import unittest

from research_agent.local_app import _render_trial_overview


class RenderTrialOverviewTest(unittest.TestCase):
    def test_overview_keeps_reading_order_and_submission_score_together(self):
        # _render_trial_overview used to exist as two duplicate definitions in this
        # module; the second silently shadowed the first and dropped its "추천 읽기
        # 순서" section and closing usage-hint paragraph while adding a submission
        # score line. The merged version must keep both.
        trial = {
            "competition": "bike-sharing-demand",
            "trial_id": "trial_003",
            "status": "completed",
            "is_best_lb": True,
            "local_score": 0.354,
            "lb_score": 0.41035,
            "metric": "rmsle",
            "change_axis": "model_family",
            "decision": "accept",
            "recommended_base_trial": "trial_002",
            "token_total": 1200,
        }

        overview = _render_trial_overview(trial, [])

        self.assertIn("제출 점수", overview)
        self.assertIn("Best: LB", overview)
        self.assertIn("추천 읽기 순서", overview)
        self.assertIn("판단: accept", overview)
        self.assertIn("파일 열기", overview)

    def test_overview_handles_missing_trial(self):
        self.assertEqual("Trial을 선택하면 핵심 요약이 표시됩니다.", _render_trial_overview({}, []))


if __name__ == "__main__":
    unittest.main()
