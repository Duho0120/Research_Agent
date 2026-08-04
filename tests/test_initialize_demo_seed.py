import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent import paths, simple_yaml
from scripts.initialize_demo_seed import initialize_demo_seed


class InitializeDemoSeedTest(unittest.TestCase):
    def test_initializes_empty_storage_and_keeps_existing_data_on_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed"
            storage = root / "storage"
            self._write_seed(seed)

            with patch.object(paths, "ROOT", storage):
                first = initialize_demo_seed(seed, storage)
                marker = storage / "experiments" / "titanic" / "trial_001" / "local-note.txt"
                marker.write_text("keep me\n", encoding="utf-8")
                second = initialize_demo_seed(seed, storage)

            profile = simple_yaml.load(storage / "competitions" / "titanic" / "execution_profile.yaml")
            source = json.loads(
                (storage / "competitions" / "titanic" / "workspace_source.json").read_text(encoding="utf-8")
            )
            loop = json.loads((storage / "runtime" / "auto_loop_state.json").read_text(encoding="utf-8"))
            marker_text = marker.read_text(encoding="utf-8")

        self.assertEqual("completed", first["sync_status"])
        self.assertIn("experiments", first["copied"])
        self.assertIn("experiments", second["skipped"])
        self.assertEqual("keep me\n", marker_text)
        self.assertEqual(str((storage / "demo_workspaces" / "titanic").resolve()), profile["project_root"])
        self.assertEqual(str((storage / "demo_workspaces" / "titanic").resolve()), source["source_path"])
        self.assertEqual("trial_001", loop["last_completed_trial"])
        self.assertEqual("trial_002", loop["next_trial"])
        self.assertEqual("planning", loop["phase"])

    @staticmethod
    def _write_seed(seed: Path) -> None:
        competition = seed / "competitions" / "titanic"
        experiment = seed / "experiments" / "titanic" / "trial_001"
        workspace = seed / "demo_workspaces" / "titanic"
        for path in [
            competition,
            experiment,
            workspace,
            seed / "runs" / "titanic",
            seed / "memory" / "titanic",
            seed / "submissions" / "titanic",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "competition": "titanic",
                "platform": "kaggle",
                "project_root": "C:\\old\\titanic",
                "python": "C:\\old\\python.exe",
            },
            competition / "execution_profile.yaml",
        )
        (competition / "workspace_source.json").write_text(
            json.dumps({"source_path": "C:\\old\\titanic", "python": "C:\\old\\python.exe"}),
            encoding="utf-8",
        )
        simple_yaml.dump(
            {
                "competition": {
                    "name": "Titanic",
                    "platform": "kaggle",
                    "metric": "accuracy",
                    "objective": "maximize",
                }
            },
            competition / "state.yaml",
        )
        (experiment / "workspace_result_cycle.json").write_text(
            json.dumps({"status": "completed"}),
            encoding="utf-8",
        )
        (workspace / "workspace_config.json").write_text(
            json.dumps({"platform": "kaggle", "topic": "Titanic", "metric": "accuracy"}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
