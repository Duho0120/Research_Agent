import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.cli import main
from research_agent.retrieval.context_pack import build_context_pack
from research_agent.retrieval.document_registry import collect_retrieval_documents
from research_agent.retrieval.index_builder import build_document_index
from research_agent.retrieval.retriever import retrieve_documents


class RetrievalIndexTest(unittest.TestCase):
    def test_collects_competition_memory_and_trial_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)

            with patch("research_agent.paths.project_root", return_value=root):
                docs = collect_retrieval_documents("demo")

            paths = {doc.source_path for doc in docs}
            self.assertIn("competitions/demo/overview.md", paths)
            self.assertIn("memory/demo/trial_index.jsonl", paths)
            self.assertIn("experiments/demo/trial_001/internal/pipeline_structure.json", paths)
            self.assertIn("experiments/demo/trial_001/user_view/02_pipeline_structure.ko.md", paths)
            kinds = {doc.source_kind for doc in docs}
            self.assertIn("competition_overview", kinds)
            self.assertIn("pipeline_structure", kinds)
            self.assertIn("user_pipeline_structure", kinds)

    def test_build_index_writes_jsonl_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)

            with patch("research_agent.paths.project_root", return_value=root):
                manifest = build_document_index("demo")

            memory = root / "memory" / "demo"
            self.assertGreaterEqual(manifest["document_count"], 5)
            self.assertTrue((memory / "document_index.jsonl").exists())
            self.assertTrue((memory / "document_index_manifest.json").exists())
            self.assertTrue((memory / "document_index.md").exists())
            rows = [
                json.loads(line)
                for line in (memory / "document_index.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["document_count"], len(rows))
            self.assertTrue(all("document_id" in row for row in rows))

    def test_retrieves_relevant_documents_from_file_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)

            with patch("research_agent.paths.project_root", return_value=root):
                build_document_index("demo")
                result = retrieve_documents("demo", "pipeline logistic regression FamilySize", limit=3)

            self.assertGreaterEqual(result["result_count"], 1)
            top_paths = [item["source_path"] for item in result["results"]]
            self.assertIn("experiments/demo/trial_001/user_view/02_pipeline_structure.ko.md", top_paths)

    def test_build_context_pack_writes_pack_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)

            with patch("research_agent.paths.project_root", return_value=root):
                pack = build_context_pack(
                    "demo",
                    "trial_001",
                    task="experiment_planning",
                    query="metric pipeline LogisticRegression FamilySize",
                    limit=4,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("experiment_planning", pack["task"])
            self.assertGreaterEqual(pack["document_count"], 1)
            self.assertTrue((trial / "context_pack_experiment_planning.json").exists())
            self.assertTrue((trial / "context_pack_experiment_planning.md").exists())
            self.assertTrue((trial / "retrieval_manifest_experiment_planning.json").exists())
            manifest = json.loads((trial / "retrieval_manifest_experiment_planning.json").read_text(encoding="utf-8"))
            self.assertEqual("memory/demo/document_index.jsonl", manifest["index_file"])
            self.assertLessEqual(pack["document_count"], 5)
            self.assertLessEqual(pack["budget"]["max_chars_per_document"], 900)

    def test_context_pack_prefers_decision_and_memory_cards_under_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)
            memory = root / "memory" / "demo"
            (memory / "latest_decision_card.md").write_text(
                "# Decision\n\n- rejected_axes: feature_engineering_family_size\n- recommended_base_trial: trial_001\n",
                encoding="utf-8",
            )
            (memory / "latest_trial_memory_card.md").write_text(
                "# Memory\n\n- change_axis: feature_engineering_family_size\n- local_score: 0.83\n",
                encoding="utf-8",
            )

            with patch("research_agent.paths.project_root", return_value=root):
                pack = build_context_pack(
                    "demo",
                    "trial_002",
                    task="experiment_planning",
                    query="decision rejected axes memory score pipeline",
                )

            kinds = [doc["source_kind"] for doc in pack["documents"]]
            self.assertIn("decision_card", kinds[:2])
            self.assertIn("trial_memory_card", kinds[:3])
            self.assertLessEqual(pack["document_count"], 5)

    def test_cli_build_retrieval_index_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sample_project(root)

            with patch("research_agent.paths.project_root", return_value=root):
                code = main(["build-retrieval-index", "--competition", "demo"])

            self.assertEqual(0, code)
            self.assertTrue((root / "memory" / "demo" / "document_index.jsonl").exists())

    def _write_sample_project(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True)
        (comp / "overview.md").write_text("# Demo\n\nPredict survival from tabular passenger data.\n", encoding="utf-8")
        (comp / "metric.md").write_text("# Metric\n\naccuracy, maximize\n", encoding="utf-8")
        (comp / "data_notes.md").write_text("# Data\n\nColumns include Age, Fare, Sex, Embarked.\n", encoding="utf-8")

        memory = root / "memory" / "demo"
        memory.mkdir(parents=True)
        (memory / "research_notes.md").write_text("# Notes\n\nFirst baseline uses logistic regression.\n", encoding="utf-8")
        (memory / "trial_index.jsonl").write_text(
            json.dumps({"competition": "demo", "trial_id": "trial_001", "cv_score": 0.83}) + "\n",
            encoding="utf-8",
        )

        trial = root / "experiments" / "demo" / "trial_001"
        (trial / "internal").mkdir(parents=True)
        (trial / "user_view").mkdir(parents=True)
        (trial / "metrics.json").write_text(
            json.dumps({"cv_score": 0.83, "metric": "accuracy", "objective": "maximize"}),
            encoding="utf-8",
        )
        (trial / "internal" / "pipeline_structure.json").write_text(
            json.dumps(
                {
                    "model_definition": {"structured_details": {"estimator": "LogisticRegression"}},
                    "feature_representation": {"structured_details": {"derived_features": ["FamilySize", "IsAlone"]}},
                }
            ),
            encoding="utf-8",
        )
        (trial / "user_view" / "02_pipeline_structure.ko.md").write_text(
            "# Pipeline\n\n- model: LogisticRegression\n- derived features: FamilySize, IsAlone\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
