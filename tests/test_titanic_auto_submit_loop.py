import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import titanic_auto_submit_loop as loop


class TitanicAutoSubmitLoopRuntimeTest(unittest.TestCase):
    def test_lock_rejects_second_live_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with self._runtime_paths(runtime):
                loop.acquire_loop_lock()
                self.assertTrue(loop.LOCK_PATH.exists())
                with self.assertRaises(SystemExit):
                    loop.acquire_loop_lock()
                loop.release_loop_lock()
                self.assertFalse(loop.LOCK_PATH.exists())

    def test_stale_lock_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with self._runtime_paths(runtime):
                runtime.mkdir(parents=True, exist_ok=True)
                loop.LOCK_PATH.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
                with patch("scripts.titanic_auto_submit_loop.process_is_alive", return_value=False):
                    loop.acquire_loop_lock()
                self.assertEqual(os.getpid(), loop.load_lock_owner())
                loop.release_loop_lock()

    def test_pause_signal_is_separate_from_state_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            with self._runtime_paths(runtime):
                loop.request_pause()
                self.assertTrue(loop.pause_is_requested())
                self.assertTrue(loop.PAUSE_REQUEST_PATH.exists())
                loop.clear_pause_request()
                loop.save_loop_state(pause_requested=False)
                self.assertFalse(loop.pause_is_requested())

    def test_main_submits_even_when_local_score_is_lower_than_previous_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            run_dir = root / "manual_trials"
            specs = [
                SimpleNamespace(trial_id="trial_001", change_axis="baseline", feature_mode="baseline"),
                SimpleNamespace(trial_id="trial_002", change_axis="lower_local", feature_mode="baseline"),
            ]
            local_scores = {"trial_001": 0.90, "trial_002": 0.70}
            lb_scores = {"trial_001": 0.80, "trial_002": 0.79}
            submitted: list[str] = []

            def fake_run_trial(spec, *, submit, poll_seconds):
                trial_dir = run_dir / spec.trial_id
                trial_dir.mkdir(parents=True, exist_ok=True)
                submission = trial_dir / "submission.csv"
                submission.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
                return {
                    "trial_id": spec.trial_id,
                    "local_score": local_scores[spec.trial_id],
                    "submission_file": str(submission),
                    "change_axis": spec.change_axis,
                    "feature_mode": spec.feature_mode,
                    "model": "FakeModel",
                }

            def fake_submit(submission_file, message):
                submitted.append(message)

            def fake_wait(message, *, attempts, interval_seconds):
                trial_id = "trial_002" if "trial_002" in message else "trial_001"
                return {
                    "ref": f"ref_{trial_id}",
                    "status": loop.SUBMISSION_STATUS_COMPLETE,
                    "public_score": lb_scores[trial_id],
                    "private_score": None,
                }

            with self._runtime_paths(runtime):
                with patch.multiple(loop, TRIALS=specs, RUN_DIR=run_dir):
                    with patch("scripts.titanic_auto_submit_loop.validate_credential_mode"):
                        with patch("scripts.titanic_auto_submit_loop.run_trial", side_effect=fake_run_trial):
                            with patch("scripts.titanic_auto_submit_loop.submit_to_kaggle", side_effect=fake_submit):
                                with patch("scripts.titanic_auto_submit_loop.wait_for_submission_result", side_effect=fake_wait):
                                    with patch("scripts.titanic_auto_submit_loop.feature_columns", return_value=([], [])):
                                        with patch("scripts.titanic_auto_submit_loop.write_user_artifacts"):
                                            with patch("scripts.titanic_auto_submit_loop.record_project_submission"):
                                                with patch("sys.argv", ["prog", "--start", "trial_001", "--end", "trial_002"]):
                                                    self.assertEqual(0, loop.main())

            self.assertEqual(2, len(submitted))
            self.assertTrue(any("trial_001" in message for message in submitted))
            self.assertTrue(any("trial_002" in message for message in submitted))

    @staticmethod
    def _runtime_paths(runtime: Path):
        return patch.multiple(
            loop,
            RUNTIME_DIR=runtime,
            STATE_PATH=runtime / "auto_loop_state.json",
            LOCK_PATH=runtime / "auto_loop.lock",
            PAUSE_REQUEST_PATH=runtime / "pause.request",
        )


if __name__ == "__main__":
    unittest.main()
