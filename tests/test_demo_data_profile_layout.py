import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.demo_one_cycle import _build_demo_data_profile
from kaggle_research_agent.workspace_coding_handoff import _load_data_card_summary


def _write_per_sample_workspace(root: Path) -> Path:
    """A workspace shaped like competitions that ship one CSV per sample.

    Mirrors the real DACON 236716 layout: data/train/TRAIN_xxxxx.csv +
    data/train_labels.csv (ids match the filename stems), and
    data/test/TEST_xxxxx.csv + data/sample_submission.csv.
    """
    project = root / "demo_workspaces" / "demo"
    data = project / "data"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir(parents=True)
    for index in (1, 2, 3):
        (data / "train" / f"TRAIN_{index:05d}.csv").write_text(
            "timestep_ms,x,y,z\n-400,0.1,0.2,0.3\n0,0.4,0.5,0.6\n", encoding="utf-8"
        )
        (data / "test" / f"TEST_{index:05d}.csv").write_text(
            "timestep_ms,x,y,z\n-400,1.1,1.2,1.3\n0,1.4,1.5,1.6\n", encoding="utf-8"
        )
    (data / "train_labels.csv").write_text(
        "id,x,y,z\nTRAIN_00001,0.7,0.8,0.9\nTRAIN_00002,1.0,1.1,1.2\nTRAIN_00003,1.3,1.4,1.5\n",
        encoding="utf-8",
    )
    (data / "sample_submission.csv").write_text(
        "id,x,y,z\nTEST_00001,0,0,0\nTEST_00002,0,0,0\nTEST_00003,0,0,0\n", encoding="utf-8"
    )
    return project


class PerSampleFileDatasetProfileTest(unittest.TestCase):
    def test_profile_describes_per_sample_directories_instead_of_reporting_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_per_sample_workspace(root)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = _build_demo_data_profile(
                    "demo",
                    {"project_root": str(project)},
                    {"competition": {"metric": "r_hit_at_1cm"}},
                )

            # Before this was handled, only flat data/*.csv were scanned, so
            # a per-sample-file competition looked like it had no train data
            # at all and the code writer fell back to a nonexistent train.csv.
            self.assertEqual("ready", profile["status"])
            self.assertEqual("per_sample_files", profile["dataset_layout"])
            self.assertEqual("train/", profile["train_dir"])
            self.assertEqual("test/", profile["test_dir"])

            by_role = {group["role"]: group for group in profile["directory_datasets"]}
            self.assertEqual({"train", "test"}, set(by_role))
            self.assertEqual(3, by_role["train"]["file_count"])
            self.assertEqual("TRAIN_#####.csv", by_role["train"]["filename_pattern"])
            self.assertEqual(["timestep_ms", "x", "y", "z"], by_role["train"]["per_file_columns"])
            self.assertEqual("filename_stem", by_role["train"]["sample_id_source"])

    def test_profile_links_label_and_submission_files_to_their_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_per_sample_workspace(root)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = _build_demo_data_profile(
                    "demo",
                    {"project_root": str(project)},
                    {"competition": {"metric": "r_hit_at_1cm"}},
                )

            by_role = {group["role"]: group for group in profile["directory_datasets"]}
            self.assertIn("train_labels.csv:id", by_role["train"]["id_matched_files"])
            self.assertIn("sample_submission.csv:id", by_role["test"]["id_matched_files"])

    def test_flat_table_competitions_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "demo_workspaces" / "demo"
            data = project / "data"
            data.mkdir(parents=True)
            (data / "train.csv").write_text("id,feature,target\n1,0.5,1\n", encoding="utf-8")
            (data / "test.csv").write_text("id,feature\n2,0.7\n", encoding="utf-8")
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                profile = _build_demo_data_profile(
                    "demo",
                    {"project_root": str(project), "target_column": "target"},
                    {"competition": {"metric": "accuracy"}},
                )

            self.assertEqual("flat_tables", profile["dataset_layout"])
            self.assertEqual([], profile["directory_datasets"])
            self.assertEqual("train.csv", profile["train_file"])
            self.assertIsNone(profile["train_dir"])

    def test_data_card_summary_forwards_directory_layout_to_the_code_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_per_sample_workspace(root)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                _build_demo_data_profile(
                    "demo",
                    {"project_root": str(project)},
                    {"competition": {"metric": "r_hit_at_1cm"}},
                )
                summary = _load_data_card_summary("demo")

            # The handoff summary picks explicit keys, so a layout the profile
            # knows about is still invisible to the code writer unless it is
            # forwarded here too.
            self.assertEqual("per_sample_files", summary["dataset_layout"])
            self.assertEqual("train/", summary["train_dir"])
            roles = {group["role"] for group in summary["directory_datasets"]}
            self.assertEqual({"train", "test"}, roles)

    def test_rendered_data_card_markdown_documents_the_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_per_sample_workspace(root)
            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                _build_demo_data_profile(
                    "demo",
                    {"project_root": str(project)},
                    {"competition": {"metric": "r_hit_at_1cm"}},
                )
                rendered = (root / "competitions" / "demo" / "competition_data_card.md").read_text(encoding="utf-8")

            self.assertIn("Per-Sample File Directories", rendered)
            self.assertIn("TRAIN_#####.csv", rendered)
            self.assertIn("train_labels.csv:id", rendered)


if __name__ == "__main__":
    unittest.main()
