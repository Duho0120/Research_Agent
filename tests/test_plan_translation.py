from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.plan_translation import render_plan_ko


class RenderPlanKoTest(unittest.TestCase):
    PLAN = {
        "plan_type": "continuation_delta_plan",
        "source_trial_id": "trial_012",
        "primary_change_axis": "feature_engineering:new_simple_numeric_features",
        "plan_title": "Add windspeed_zero indicator on top of trial_012",
        "rationale": "Keep Poisson HGBR fixed; add one low-risk feature.",
        "change_details": ["Name: windspeed_is_zero"],
    }

    class _FakeTranslationClient:
        def __init__(self, text: str):
            self._text = text

        def create_response(self, payload):
            # The trial_id must be explicit in the prompt so the model cannot
            # confuse the trial being planned with the source/base trial it
            # references in rationale (this was a real bug found in manual
            # testing: the model wrote the wrong trial number in the title).
            assert '"trial_id": "trial_013"' in payload["input"][1]["content"]
            return {
                "id": "translate-test",
                "model": payload["model"],
                "usage": {"input_tokens": 500, "output_tokens": 300, "total_tokens": 800},
                "output_text": self._text,
            }

    class _FailingTranslationClient:
        def create_response(self, payload):
            raise RuntimeError("simulated API failure")

    class _EmptyTranslationClient:
        def create_response(self, payload):
            return {"id": "empty", "model": payload["model"], "usage": {}, "output_text": ""}

    def test_uses_translated_text_when_api_allowed_and_call_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                content = render_plan_ko(
                    root / "experiments" / "demo" / "trial_013",
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=self._FakeTranslationClient("# trial_013 실험 계획\n\n번역된 본문입니다."),
                )
                token_log = (root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8")

        self.assertEqual("# trial_013 실험 계획\n\n번역된 본문입니다.", content)
        self.assertIn('"call_type": "plan_translation"', token_log)

    def test_falls_back_to_template_when_api_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                content = render_plan_ko(
                    root / "experiments" / "demo" / "trial_013",
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=False,
                )

        self.assertIn("trial_013 실험 계획", content)
        self.assertIn("windspeed_is_zero", content)

    def test_falls_back_to_template_when_translation_call_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                content = render_plan_ko(
                    root / "experiments" / "demo" / "trial_013",
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=self._FailingTranslationClient(),
                )

        self.assertIn("trial_013 실험 계획", content)
        self.assertIn("windspeed_is_zero", content)

    def test_falls_back_to_template_when_translation_response_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("research_agent.paths.project_root", return_value=root):
                content = render_plan_ko(
                    root / "experiments" / "demo" / "trial_013",
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=self._EmptyTranslationClient(),
                )

        self.assertIn("trial_013 실험 계획", content)

    def test_second_call_with_unchanged_plan_reuses_cache_without_a_new_api_call(self):
        # render_plan_ko is called twice per trial in the common case: a
        # pre-execution preview right after planning, then again after
        # execution finishes. If the plan did not change in between (no user
        # insight forced a replan), the second call must not spend another
        # low-cost LLM call translating identical content.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "experiments" / "demo" / "trial_013"
            client = self._CountingTranslationClient("# trial_013 실험 계획\n\n번역된 본문입니다.")
            with patch("research_agent.paths.project_root", return_value=root):
                first = render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=client,
                )
                second = render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=dict(self.PLAN),
                    allow_api=True,
                    client=client,
                )

        self.assertEqual(first, second)
        self.assertEqual(1, client.call_count)

    def test_second_call_with_changed_plan_translates_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "experiments" / "demo" / "trial_013"
            client = self._CountingTranslationClient("# trial_013 실험 계획\n\n번역된 본문입니다.")
            revised_plan = dict(self.PLAN)
            revised_plan["primary_change_axis"] = "user_insight_axis"
            with patch("research_agent.paths.project_root", return_value=root):
                render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=client,
                )
                render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=revised_plan,
                    allow_api=True,
                    client=client,
                )

        self.assertEqual(2, client.call_count)

    def test_cache_survives_user_view_directory_being_wiped(self):
        # organize_trial_artifacts deletes the whole user_view/ directory
        # (shutil.rmtree) right before its final render call -- the cache
        # must live somewhere that survives that, or every "final" call would
        # be a guaranteed miss and the cache would do nothing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "experiments" / "demo" / "trial_013"
            client = self._CountingTranslationClient("# trial_013 실험 계획\n\n번역된 본문입니다.")
            with patch("research_agent.paths.project_root", return_value=root):
                render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=self.PLAN,
                    allow_api=True,
                    client=client,
                )
                import shutil

                shutil.rmtree(out_dir / "user_view", ignore_errors=True)
                render_plan_ko(
                    out_dir,
                    {"competition": "demo", "trial_id": "trial_013"},
                    plan=dict(self.PLAN),
                    allow_api=True,
                    client=client,
                )

        self.assertEqual(1, client.call_count)

    class _CountingTranslationClient:
        def __init__(self, text: str):
            self._text = text
            self.call_count = 0

        def create_response(self, payload):
            self.call_count += 1
            return {
                "id": f"translate-{self.call_count}",
                "model": payload["model"],
                "usage": {"input_tokens": 500, "output_tokens": 300, "total_tokens": 800},
                "output_text": self._text,
            }


if __name__ == "__main__":
    unittest.main()
