import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.demo_train import main


class DemoTrainTest(unittest.TestCase):
    def test_demo_train_reflects_sota_config_in_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            output = root / "out"
            config.write_text(
                json.dumps(
                    {
                        "model": {"type": "skeleton_transformer"},
                        "features": {
                            "use_view_aware_features": True,
                            "use_bed_wandering_aux_head": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "demo_train.py",
                "--config",
                str(config),
                "--output",
                str(output),
                "--score",
                "0.81",
            ]

            with patch.object(sys, "argv", argv):
                code = main()

            self.assertEqual(code, 0)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["model_type"], "skeleton_transformer")
            self.assertGreater(metrics["cv_score"], 0.81)
            self.assertIn("skeleton_transformer", metrics["notes"])

    def test_demo_train_runs_as_direct_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            output = root / "out"
            config.write_text(json.dumps({"model": {"type": "lightgbm"}}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/demo_train.py",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()


