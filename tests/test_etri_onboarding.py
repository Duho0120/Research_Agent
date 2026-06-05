import json
import unittest
from pathlib import Path

from kaggle_research_agent import simple_yaml


class EtriOnboardingTest(unittest.TestCase):
    def test_etri_competition_onboarding_files_exist_and_preserve_key_context(self):
        root = Path(__file__).resolve().parents[1]
        competition = "etri_human_understanding"

        state_path = root / "competitions" / competition / "state.yaml"
        notes_path = root / "memory" / competition / "research_notes.md"
        rules_path = root / "memory" / competition / "rules.md"
        trial_index_path = root / "memory" / competition / "trial_index.jsonl"
        v11_metrics_path = root / "experiments" / competition / "trial_v11_public_baseline" / "metrics.json"
        v16_metrics_path = root / "experiments" / competition / "trial_v16_causal_rolling_baseline" / "metrics.json"

        for path in [state_path, notes_path, rules_path, trial_index_path, v11_metrics_path, v16_metrics_path]:
            self.assertTrue(path.exists(), f"Missing ETRI onboarding file: {path}")

        state = simple_yaml.load(state_path)
        self.assertEqual("etri_human_understanding", state["competition"]["name"])
        self.assertEqual("mean_binary_logloss", state["competition"]["metric"])
        self.assertEqual("minimize", state["competition"]["objective"])
        self.assertEqual("dacon", state["competition"]["platform"])
        self.assertEqual("trial_v16_causal_rolling_baseline", state["current_state"]["active_trial"])

        notes = notes_path.read_text(encoding="utf-8")
        self.assertIn("multi-output binary classification", notes)
        self.assertIn("V11", notes)
        self.assertIn("V16", notes)
        self.assertIn("DACON", notes)

        rules = rules_path.read_text(encoding="utf-8")
        self.assertIn("Do not use random KFold", rules)
        self.assertIn("Subject-hole CV", rules)
        self.assertIn("Tail CV", rules)
        self.assertIn("Do not use Kaggle CLI", rules)

        rows = [json.loads(line) for line in trial_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(["trial_v11_public_baseline", "trial_v15_subject_temporal_deviation", "trial_v16_causal_rolling_baseline"], [row["trial_id"] for row in rows])
        self.assertEqual("minimize", rows[0]["objective"])
        self.assertTrue(rows[2]["is_best"])

        v11 = json.loads(v11_metrics_path.read_text(encoding="utf-8"))
        v16 = json.loads(v16_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(0.5984, v11["lb_score"])
        self.assertLess(v16["cv_score"], v11["cv_score"])
        self.assertIsNone(v16["lb_score"])


if __name__ == "__main__":
    unittest.main()
