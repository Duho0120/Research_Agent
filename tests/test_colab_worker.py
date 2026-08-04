from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from colab.worker import load_job, run_pending_jobs
from research_agent import simple_yaml


class ColabWorkerTest(unittest.TestCase):
    def test_worker_updates_quoted_yaml_status_and_preserves_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "jobs" / "demo" / "job.yaml"
            job_path.parent.mkdir(parents=True)
            job_path.write_text(
                '\n'.join(
                    [
                        'job_id: "demo_job"',
                        'status: "pending"',
                        'command: "python train.py --label a:b"',
                        "metadata:",
                        "  owner: researcher",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            observed: list[str] = []

            def runner(command, *, shell, cwd):
                observed.append(load_job(job_path)["status"])
                self.assertTrue(shell)
                self.assertEqual(root, cwd)
                return SimpleNamespace(returncode=0)

            run_pending_jobs(root=root, runner=runner)

            updated = simple_yaml.load(job_path)
            self.assertEqual(["running"], observed)
            self.assertEqual("done", updated["status"])
            self.assertEqual("python train.py --label a:b", updated["command"])
            self.assertEqual({"owner": "researcher"}, updated["metadata"])
            self.assertFalse(job_path.with_suffix(".yaml.tmp").exists())

    def test_worker_records_failed_process_without_corrupting_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "jobs" / "job.yaml"
            job_path.parent.mkdir(parents=True)
            simple_yaml.dump(
                {
                    "job_id": "failed_job",
                    "status": "pending",
                    "command": "python fail.py",
                    "options": {"retry": False},
                },
                job_path,
            )

            run_pending_jobs(root=root, runner=lambda *args, **kwargs: SimpleNamespace(returncode=7))

            updated = simple_yaml.load(job_path)
            self.assertEqual("failed", updated["status"])
            self.assertEqual("return_code_7", updated["worker_error"])
            self.assertEqual({"retry": False}, updated["options"])

    def test_worker_marks_missing_command_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_path = root / "jobs" / "job.yaml"
            job_path.parent.mkdir(parents=True)
            simple_yaml.dump({"job_id": "missing", "status": "pending"}, job_path)

            run_pending_jobs(root=root)

            updated = simple_yaml.load(job_path)
            self.assertEqual("failed", updated["status"])
            self.assertEqual("missing_command", updated["worker_error"])


if __name__ == "__main__":
    unittest.main()
