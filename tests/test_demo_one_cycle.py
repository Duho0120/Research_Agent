import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent import simple_yaml
from kaggle_research_agent.cli import main
from kaggle_research_agent.demo_one_cycle import (
    _demo_baseline_recommendation,
    _infer_demo_file_role,
    _infer_demo_task_type,
    build_demo_plan_payload,
    run_demo_one_cycle,
)
from kaggle_research_agent.graph.demo_auto_loop import run_demo_graph_auto_loop
from kaggle_research_agent.graph.demo_cycle_graph import run_demo_graph_cycle
from kaggle_research_agent.state_db import get_trial_summary
from kaggle_research_agent.trial_artifacts import trial_artifact_exists


class DemoOneCycleTest(unittest.TestCase):
    def test_kaggle_camel_case_sample_submission_is_not_treated_as_train(self):
        self.assertEqual(
            "sample_submission",
            _infer_demo_file_role("sampleSubmission.csv", ["datetime", "count"], "count"),
        )

    def test_rmsle_task_uses_regression_baseline_recommendations(self):
        train = {
            "column_profiles": [
                {
                    "name": "count",
                    "role": "target",
                    "type": "numeric",
                    "unique_in_sample": 1,
                    "baseline_use": "exclude",
                }
            ]
        }
        task_type = _infer_demo_task_type(train, "count", metric="rmsle")
        recommendation = _demo_baseline_recommendation(
            train,
            target_column="count",
            id_column="datetime",
            task_type=task_type,
        )

        self.assertEqual("tabular_regression_or_ranking", task_type)
        self.assertIn("regularized_linear_regression", recommendation["preferred_model_families"][0])

    def test_baseline_recommendation_excludes_train_only_columns(self):
        train = {
            "column_profiles": [
                {"name": "datetime", "baseline_use": "exclude"},
                {"name": "count", "baseline_use": "exclude"},
                {"name": "temp", "baseline_use": "include_numeric"},
                {"name": "casual", "baseline_use": "include_numeric"},
                {"name": "registered", "baseline_use": "include_numeric"},
            ]
        }
        test = {
            "column_profiles": [
                {"name": "datetime", "baseline_use": "exclude"},
                {"name": "temp", "baseline_use": "include_numeric"},
            ]
        }

        recommendation = _demo_baseline_recommendation(
            train,
            test=test,
            target_column="count",
            id_column="datetime",
            task_type="tabular_regression_or_ranking",
        )

        self.assertEqual(["temp"], recommendation["include_features_first"])
        self.assertIn("casual", recommendation["exclude_columns"])
        self.assertIn("registered", recommendation["exclude_columns"])

    def test_demo_one_cycle_runs_mock_plan_code_execution_and_records_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("completed", result["status"])
            self.assertEqual("ready", result["plan"]["status"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertEqual(0.88, result["record"]["local_score"])
            self.assertEqual("ready", result["submission_manifest"]["status"])
            self.assertTrue((trial / "submit_manifest.json").exists())
            self.assertTrue((trial / "submit_manifest.md").exists())
            self.assertTrue(trial_artifact_exists(trial, "decision_card.json"))
            self.assertTrue(trial_artifact_exists(trial, "trial_memory_card.json"))
            self.assertTrue((trial / "demo_context.md").exists())
            self.assertTrue((trial / "next_experiment.md").exists())
            self.assertTrue(trial_artifact_exists(trial, "workspace_coding_result.json"))
            self.assertTrue(trial_artifact_exists(trial, "workspace_run.json"))
            self.assertTrue(trial_artifact_exists(trial, "demo_cycle_record.json"))
            self.assertTrue((trial / "agent_status.json").exists())
            self.assertTrue((trial / "agent_events.jsonl").exists())
            status = json.loads((trial / "agent_status.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", status["status"])
            self.assertEqual("done", status["current_stage"])
            events = (trial / "agent_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "stage_started"' in line for line in events))
            self.assertTrue(any('"event": "cycle_completed"' in line for line in events))
            self.assertTrue((root / "memory" / "demo" / "demo_trial_index.jsonl").exists())
            self.assertTrue((root / "memory" / "demo" / "latest_decision_card.json").exists())
            self.assertTrue((root / "memory" / "demo" / "latest_trial_memory_card.json").exists())
            usage_lines = (root / "memory" / "demo" / "token_usage.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(2, len(usage_lines))
            usages = [json.loads(line) for line in usage_lines]
            self.assertEqual(["experiment_planning", "workspace_code_writing"], [row["call_type"] for row in usages])
            self.assertEqual({"openai_responses"}, {row["provider"] for row in usages})
            self.assertEqual({"gpt-5.5"}, {row["model"] for row in usages})
            self.assertEqual("gpt-5.6-luna", result["model_policy"]["low_cost"]["model"])
            self.assertFalse(result["context"]["artifact_policy"]["save_model"]["default"])
            self.assertIn("improved", (project / "train_step.py").read_text(encoding="utf-8"))
            plan_request = build_demo_plan_payload(result["context"], model="gpt-5.5")
            plan_prompt = plan_request["input"][1]["content"]
            self.assertIn("Model/checkpoint artifacts are optional", plan_prompt)
            self.assertIn("required_for_separate_predict_command", plan_prompt)
            self.assertIn("RAG Context Pack", plan_prompt)
            self.assertIn("retrieved documents", plan_prompt.lower())
            self.assertIn("Artifact Policy", (trial / "demo_context.md").read_text(encoding="utf-8"))
            self.assertTrue((trial / "context_pack_experiment_planning.json").exists())
            self.assertTrue((trial / "retrieval_manifest_experiment_planning.json").exists())
            self.assertEqual("synced", result["state_db_sync"]["status"])
            db_path = root / "memory" / "research_agent.sqlite3"
            self.assertTrue(db_path.exists())
            summary = get_trial_summary("demo", "trial_001", db_path)
            self.assertEqual("completed", summary["status"])
            self.assertEqual(0.88, summary["local_score"])

    def test_demo_graph_cycle_runs_same_one_cycle_with_graph_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            trial = root / "experiments" / "demo" / "trial_001"
            self.assertEqual("completed", result["status"])
            self.assertTrue(result["graph_execution"]["enabled"])
            self.assertEqual("accepted", result["code_writer"]["status"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual(0.88, result["record"]["local_score"])
            self.assertEqual("ready", result["submission_manifest"]["status"])
            self.assertTrue((trial / "demo_graph_cycle.json").exists())
            self.assertTrue((trial / "submit_manifest.json").exists())
            self.assertTrue(trial_artifact_exists(trial, "demo_one_cycle.json"))
            self.assertTrue((trial / "graph_state.json").exists())
            self.assertTrue((trial / "node_events.jsonl").exists())
            self.assertTrue((trial / "graph_rag_manifest.json").exists())
            self.assertTrue((trial / "graph_rag_manifest.md").exists())
            self.assertEqual(
                "experiments/demo/trial_001/graph_rag_manifest.json",
                result["graph_execution"]["graph_rag_manifest_file"],
            )
            manifest = json.loads((trial / "graph_rag_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(["experiment_planning", "workspace_code_writing"], [item["task"] for item in manifest["rag_contexts"]])
            self.assertTrue(all(item["files_exist"]["context_pack_md_file"] for item in manifest["rag_contexts"]))
            graph_state = json.loads((trial / "graph_state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", graph_state["status"])
            self.assertEqual("finalize", graph_state["last_completed_node"])
            events = [
                json.loads(line)
                for line in (trial / "node_events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual("load_context", events[0]["node"])
            self.assertTrue(any(item["node"] == "record_result" and item["event"] == "completed" for item in events))
            self.assertTrue(any(item["node"] == "prepare_submission" and item["event"] == "completed" for item in events))

    def test_demo_graph_auto_loop_runs_two_trials_and_records_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_auto_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=2,
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            self.assertEqual("completed", result["status"])
            self.assertEqual(["trial_001", "trial_002"], [row["trial_id"] for row in result["trials"]])
            self.assertEqual("trial_001", result["best_trial"]["trial_id"])
            self.assertTrue(result["trials"][0]["is_best"])
            self.assertFalse(result["trials"][1]["is_best"])
            memory = root / "memory" / "demo"
            self.assertTrue((memory / "demo_best_trial.json").exists())
            self.assertTrue((memory / "demo_graph_auto_loop.json").exists())
            self.assertTrue((memory / "document_index.jsonl").exists())
            self.assertFalse((root / "experiments" / "demo" / "trial_002" / "context_pack_experiment_planning.json").exists())
            trial_002_manifest = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "graph_rag_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            planning_context = next(
                item for item in trial_002_manifest["rag_contexts"] if item["task"] == "experiment_planning"
            )
            self.assertTrue(planning_context["skipped"])
            self.assertEqual(
                "continuation_uses_structured_memory_and_code_snapshot",
                planning_context["skip_reason"],
            )
            index_text = (memory / "document_index.jsonl").read_text(encoding="utf-8")
            self.assertIn("experiments/demo/trial_001/user_view/02_pipeline_structure.ko.md", index_text)
            state = simple_yaml.load(root / "competitions" / "demo" / "state.yaml")
            self.assertEqual("trial_001", state["current_state"]["best_trial"]["trial_id"])

    def test_demo_graph_auto_loop_stops_after_no_improvement_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_graph_auto_loop(
                    "demo",
                    start_trial_id="trial_001",
                    max_trials=3,
                    stop_no_improvement=1,
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )

            self.assertEqual("stopped_no_improvement", result["status"])
            self.assertEqual(2, len(result["trials"]))
            self.assertEqual(1, result["no_improvement_count"])

    def test_demo_graph_auto_loop_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                code = main(
                    [
                        "demo-graph-auto-loop",
                        "--competition",
                        "demo",
                        "--start-trial",
                        "trial_001",
                        "--max-trials",
                        "2",
                        "--mock-plan-file",
                        str(plan_response),
                        "--mock-response-file",
                        str(code_response),
                        "--run-now",
                    ]
                )

            self.assertEqual(0, code)
            self.assertTrue((root / "memory" / "demo" / "demo_graph_auto_loop.json").exists())

    def test_demo_plan_payload_uses_continuation_instructions_when_recent_trials_exist(self):
        context = {
            "competition": "demo",
            "trial_id": "trial_002",
            "source_trial_id": "trial_001",
            "plan_type": "continuation_delta_plan",
            "recent_trials": [{"trial_id": "trial_001", "cv_score": 0.83}],
            "planning_context_pack": {"documents": []},
        }

        payload = build_demo_plan_payload(context, model="gpt-5.5")
        prompt = payload["input"][1]["content"]

        self.assertIn("continuation delta plan", prompt)
        self.assertIn("source trial is `trial_001`", prompt)
        self.assertIn("Treat its pipeline as the base pipeline", prompt)
        self.assertIn("previous best/local score", prompt)
        self.assertIn("exactly one improvement axis", prompt)
        self.assertIn("primary_change_axis", prompt)
        self.assertNotIn("Create exactly one first-cycle experiment plan", prompt)

    def test_initial_demo_plan_payload_requires_coder_handoff_fields(self):
        context = {
            "competition": "demo",
            "trial_id": "trial_001",
            "source_trial_id": None,
            "plan_type": "initial_pipeline_plan",
            "planning_context_pack": {"documents": []},
            "data_profile": {
                "status": "ready",
                "task_type": "tabular_classification",
                "target_column": "target",
                "id_column": "id",
                "baseline_recommendation": {"include_features_first": ["feature_a"]},
            },
        }

        prompt = build_demo_plan_payload(context, model="gpt-5.5")["input"][1]["content"]

        self.assertIn("For initial_pipeline_plan, these fields are required", prompt)
        self.assertIn("pipeline_blueprint", prompt)
        self.assertIn("code_change_targets", prompt)
        self.assertIn("success_criteria", prompt)

    def test_demo_context_uses_demo_trial_index_as_source_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            memory = root / "memory" / "demo"
            memory.mkdir(parents=True, exist_ok=True)
            (memory / "demo_trial_index.jsonl").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "trial_id": "trial_001",
                        "local_score": 0.83,
                        "cv_score": 0.83,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            from kaggle_research_agent.demo_one_cycle import load_demo_context

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                context = load_demo_context("demo", "trial_002")

            self.assertEqual("continuation_delta_plan", context["plan_type"])
            self.assertEqual("trial_001", context["source_trial_id"])

    def test_demo_context_keeps_active_axis_after_single_failed_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import (
                _build_planning_context_pack_if_needed,
                build_demo_plan_payload,
                load_demo_context,
            )
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_family_size",
                        "change_details": ["Candidate: FamilySize"],
                    },
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                    submission={"submitted_lb_score": 0.75},
                )
                context = load_demo_context("demo", "trial_003")
                context["planning_context_pack"] = _build_planning_context_pack_if_needed("demo", "trial_003", context)

            self.assertEqual("continuation_delta_plan", context["plan_type"])
            self.assertEqual("trial_001", context["source_trial_id"])
            self.assertTrue(context["planning_context_pack"]["skipped"])
            self.assertEqual("trial_001", context["decision_context"]["recommended_base_trial"])
            self.assertEqual("feature_engineering_family_size", context["decision_context"]["active_axis"])
            self.assertEqual(1, context["decision_context"]["axis_attempt_count"])
            self.assertNotIn("feature_engineering_family_size", context["decision_context"]["rejected_axes"])
            self.assertTrue(context["decision_context"]["rejected_candidates"])
            prompt = build_demo_plan_payload(context, model="gpt-5.5")["input"][1]["content"]
            self.assertIn("active_axis", prompt)
            self.assertIn("keep that same primary axis", prompt)
            self.assertIn("rejected_axes", prompt)
            self.assertIn("rejected_candidates", prompt)
            self.assertIn("recommended_base_trial", prompt)
            self.assertIn("RAG is intentionally skipped", prompt)

    def test_planning_rag_policy_skips_plain_continuation_without_trigger(self):
        from kaggle_research_agent.demo_one_cycle import _build_planning_context_pack_if_needed

        context = {
            "source_trial_id": "trial_001",
            "decision_context": {
                "active_axis": "",
                "rejected_axes": ["model_family"],
            },
        }

        with patch("kaggle_research_agent.demo_one_cycle.build_context_pack") as build_pack:
            pack = _build_planning_context_pack_if_needed("demo", "trial_002", context)

        self.assertTrue(pack["skipped"])
        self.assertEqual("continuation_uses_structured_memory_and_code_snapshot", pack["skip_reason"])
        build_pack.assert_not_called()

    def test_planning_rag_policy_uses_rag_when_user_feedback_is_present(self):
        from kaggle_research_agent.demo_one_cycle import _build_planning_context_pack_if_needed

        context = {
            "source_trial_id": "trial_001",
            "user_feedback": "Review previous split experiments before changing model family.",
            "decision_context": {},
        }
        fake_pack = {
            "task": "experiment_planning",
            "query": "q",
            "document_count": 1,
            "budget": {"max_chars_per_document": 100},
            "context_pack_file": "pack.json",
            "context_pack_md_file": "pack.md",
            "retrieval_manifest_file": "manifest.json",
            "documents": [{"source_path": "memory/demo/user_feedback.jsonl", "source_kind": "user_feedback", "text": "x"}],
        }

        with patch("kaggle_research_agent.demo_one_cycle.build_context_pack", return_value=fake_pack) as build_pack:
            pack = _build_planning_context_pack_if_needed("demo", "trial_002", context)

        self.assertFalse(pack.get("skipped", False))
        self.assertEqual("user_feedback_or_human_review_present", pack["policy"]["reason"])
        build_pack.assert_called_once()

    def test_demo_context_keeps_active_axis_after_two_failed_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import build_demo_plan_payload, load_demo_context
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                for index, candidate in enumerate(["RandomForest", "GradientBoosting"], start=2):
                    write_trial_decision_card(
                        "demo",
                        f"trial_00{index}",
                        plan={
                            "plan_type": "continuation_delta_plan",
                            "source_trial_id": "trial_001",
                            "primary_change_axis": "model_family",
                            "change_details": [f"Candidate: {candidate}"],
                        },
                        metrics={"cv_score": 0.80, "objective": "maximize"},
                    )
                context = load_demo_context("demo", "trial_004")

            self.assertEqual("continuation_delta_plan", context["plan_type"])
            self.assertEqual("trial_001", context["source_trial_id"])
            self.assertEqual("model_family", context["decision_context"]["active_axis"])
            self.assertEqual(2, context["decision_context"]["axis_attempt_count"])
            self.assertNotIn("model_family", context["decision_context"]["rejected_axes"])
            self.assertEqual(2, len(context["decision_context"]["rejected_candidates"]))
            prompt = build_demo_plan_payload(context, model="gpt-5.5")["input"][1]["content"]
            self.assertIn("keep that same primary axis", prompt)
            self.assertIn("choose a different candidate or parameter variant within the axis", prompt)
            self.assertIn("Do not repeat any rejected_candidate", prompt)
            self.assertIn("Do not switch away from active_axis", prompt)

    def test_active_axis_refinement_uses_best_base_even_when_axis_was_attempted_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import load_demo_context
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: GradientBoosting"],
                    },
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                failed_axis = write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_title",
                        "change_details": ["Candidate: Title from Name"],
                    },
                    metrics={"cv_score": 0.79, "objective": "maximize"},
                )
                context = load_demo_context("demo", "trial_004")

            self.assertEqual("continue_axis_refinement", failed_axis["decision"])
            self.assertEqual("feature_engineering_title", failed_axis["active_axis"])
            self.assertEqual(1, failed_axis["axis_attempt_count"])
            self.assertEqual("trial_002", failed_axis["recommended_base_trial"])
            self.assertEqual("trial_002", context["source_trial_id"])
            self.assertEqual("feature_engineering_title", context["decision_context"]["active_axis"])
            self.assertEqual("trial_002", context["decision_context"]["recommended_base_trial"])

    def test_delta_patch_plan_result_writes_compact_delta_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.demo_one_cycle import _write_plan_result

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                plan = _write_plan_result(
                    "demo",
                    "trial_002",
                    {
                        "competition": "demo",
                        "trial_id": "trial_002",
                        "status": "ready",
                        "plan_type": "delta_patch",
                        "source_trial_id": "trial_001",
                        "plan_title": "Try compact feature candidate",
                        "objective": "Patch one feature.",
                        "rationale": "Short.",
                        "primary_change_axis": "feature_engineering",
                        "candidate": {
                            "name": "CandidateA",
                            "description": "Add one binary feature.",
                            "implementation_hint": "Patch src/baseline.py.",
                        },
                        "do_not_repeat": ["Candidate0"],
                        "keep_unchanged": ["model"],
                        "change_details": ["Add CandidateA"],
                        "code_change_targets": ["src/baseline.py"],
                        "success_criteria": ["score improves"],
                        "failure_decision": ["reject CandidateA"],
                        "expected_outputs": ["metrics.json"],
                        "issues": [],
                        "next_action": "prepare-workspace-handoff",
                    },
                )

            trial = root / "experiments" / "demo" / "trial_002"
            delta = json.loads((trial / "delta_plan.json").read_text(encoding="utf-8"))
            self.assertEqual("delta_patch", plan["plan_type"])
            self.assertEqual("delta_patch", delta["plan_type"])
            self.assertEqual("CandidateA", delta["candidate"]["name"])
            self.assertEqual(["Candidate0"], delta["do_not_repeat"])
            self.assertTrue((trial / "delta_plan.md").exists())

    def test_decision_card_rejects_axis_after_three_failed_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                for index, candidate in enumerate(["RandomForest", "GradientBoosting", "BalancedLogReg"], start=2):
                    card = write_trial_decision_card(
                        "demo",
                        f"trial_00{index}",
                        plan={
                            "plan_type": "continuation_delta_plan",
                            "source_trial_id": "trial_001",
                            "primary_change_axis": "model_family",
                            "change_details": [f"Candidate: {candidate}"],
                        },
                        metrics={"cv_score": 0.80, "objective": "maximize"},
                    )

            self.assertEqual("reject_or_hold", card["decision"])
            self.assertIn("model_family", card["rejected_axes"])
            self.assertIsNone(card["active_axis"])
            self.assertEqual(3, card["axis_attempt_count"])
            self.assertEqual(3, len(card["rejected_candidates"]))

    def test_decision_card_compacts_candidate_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                card = write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_logistic_regularization",
                        "change_details": [
                            "紐⑤뜽: LogisticRegression",
                            "Replace: penalty=l1, solver=liblinear, C=1.0",
                            "Penalty: elasticnet",
                        ],
                    },
                    metrics={"cv_score": 0.79, "objective": "maximize"},
                )

            self.assertNotIn("紐⑤뜽", card["candidate_label"])
            self.assertIn("LogisticRegression", card["candidate_label"])
            self.assertIn("penalty=l1", card["candidate_label"])
            self.assertLessEqual(len(card["candidate_label"]), 180)
            self.assertEqual(card["rejected_candidates"], card["active_axis_rejected_candidates"])
            self.assertIn("model_logistic_regularization", card["rejected_candidates_by_axis"])

    def test_decision_card_groups_legacy_rejected_candidates_by_label_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.paths import competition_memory_dir
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                memory = competition_memory_dir("demo")
                memory.mkdir(parents=True)
                legacy = {
                    "trial_id": "trial_002",
                    "decision": "continue_axis_refinement",
                    "change_axis": "model_family",
                    "local_score": 0.79,
                    "objective": "maximize",
                    "rejected_candidates": [
                        "feature_engineering_family_size: FamilySize",
                        "model_family: RandomForest",
                    ],
                }
                (memory / "decision_cards.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
                card = write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: GradientBoosting"],
                    },
                    metrics={"cv_score": 0.79, "objective": "maximize"},
                )

            self.assertEqual(["model_family: RandomForest", "model_family: candidate=GradientBoosting"], card["active_axis_rejected_candidates"])
            self.assertEqual(["feature_engineering_family_size: FamilySize"], card["rejected_candidates_by_axis"]["feature_engineering_family_size"])

    def test_decision_card_does_not_promote_recovery_that_only_matches_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                baseline = write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                failed = write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: weaker model"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                recovered = write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: recovered model"],
                    },
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )

            self.assertEqual("baseline_established", baseline["decision"])
            self.assertEqual("continue_axis_refinement", failed["decision"])
            self.assertEqual("continue_axis_refinement", recovered["decision"])
            self.assertEqual("flat", recovered["local_status"])
            self.assertEqual("improved", recovered["previous_local_status"])
            self.assertEqual("trial_001", recovered["recommended_base_trial"])
            self.assertNotIn("model_family", recovered["accepted_axes"])

    def test_decision_card_resets_axis_attempts_after_axis_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.80, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: RandomForest"],
                    },
                    metrics={"cv_score": 0.79, "objective": "maximize"},
                )
                accepted = write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: GradientBoosting"],
                    },
                    metrics={"cv_score": 0.82, "objective": "maximize"},
                )
                after_success_failure = write_trial_decision_card(
                    "demo",
                    "trial_004",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_003",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: ExtraTrees"],
                    },
                    metrics={"cv_score": 0.81, "objective": "maximize"},
                )

            self.assertEqual("provisional_accept", accepted["decision"])
            self.assertEqual(2, accepted["axis_attempt_count"])
            self.assertEqual("continue_axis_refinement", after_success_failure["decision"])
            self.assertEqual(1, after_success_failure["axis_attempt_count"])
            self.assertNotIn("model_family", after_success_failure["rejected_axes"])
            self.assertEqual(
                ["model_family: candidate=ExtraTrees"],
                after_success_failure["active_axis_rejected_candidates"],
            )
            self.assertIn("model_family: candidate=RandomForest", after_success_failure["rejected_candidates"])

    def test_decision_card_ignores_future_trials_when_reprocessing_old_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: weaker model"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_004",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_002",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: future model"],
                    },
                    metrics={"cv_score": 0.82, "objective": "maximize"},
                )
                reprocessed = write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "model_family",
                        "change_details": ["Candidate: recovered model"],
                    },
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )

            self.assertEqual(2, reprocessed["axis_attempt_count"])
            self.assertEqual("continue_axis_refinement", reprocessed["decision"])
            self.assertEqual("trial_001", reprocessed["recommended_base_trial"])

    def test_latest_decision_context_exposes_guidance_and_constraints_for_axis_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            root.mkdir()

            from kaggle_research_agent.trial_decision import load_latest_decision_context, write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_cabin_missing_indicator",
                        "change_details": ["Candidate: cabin known"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_cabin_missing_indicator",
                        "change_details": ["Candidate: cabin known pclass"],
                    },
                    metrics={"cv_score": 0.82, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_004",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "feature_engineering_cabin_missing_indicator",
                        "change_details": ["Candidate: cabin missing pclass"],
                    },
                    metrics={"cv_score": 0.81, "objective": "maximize"},
                )
                context = load_latest_decision_context("demo")

            self.assertIsNone(context["active_axis"])
            self.assertIn("feature_engineering_cabin_missing_indicator", context["rejected_axes"])
            self.assertIn("next_guidance", context)
            self.assertIn("planner_constraints", context)
            self.assertTrue(any("Do not keep stacking" in item for item in context["planner_constraints"]))

    def test_continuation_context_records_source_trial_for_demo_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            memory = root / "memory" / "demo"
            memory.mkdir(parents=True, exist_ok=True)
            (memory / "demo_trial_index.jsonl").write_text(
                json.dumps({"competition": "demo", "trial_id": "trial_001", "local_score": 0.83}) + "\n",
                encoding="utf-8",
            )

            from kaggle_research_agent.demo_one_cycle import _write_continuation_context

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                _write_continuation_context("demo", "trial_002")

            context = json.loads(
                (root / "experiments" / "demo" / "trial_002" / "continuation_context.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("trial_001", context["source_trial_id"])

    def test_continuation_context_exposes_recommended_base_trial_at_top_level(self):
        # workspace_coding_handoff.py reads continuation.get("recommended_base_trial")
        # at the top level (not nested under decision_context) to decide which
        # trial's code snapshot is authoritative. When an axis gets exhausted and
        # rejected, the recommended base trial rolls back to an earlier trial
        # while source_trial_id (the immediately-previous trial) stays put -- both
        # values must be present and must be allowed to differ.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import _write_continuation_context
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "change_details": ["Candidate: time series cv"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_003",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "change_details": ["Candidate: group k-fold"],
                    },
                    metrics={"cv_score": 0.82, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_004",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "change_details": ["Candidate: stratified split"],
                    },
                    metrics={"cv_score": 0.81, "objective": "maximize"},
                )
                # source_trial_id is forced to the last-completed trial (trial_004),
                # mirroring plan_following_trial's real call, even though the axis
                # is now exhausted and the recommended base has rolled back.
                _write_continuation_context("demo", "trial_005", source_trial_id="trial_004")

            context = json.loads(
                (root / "experiments" / "demo" / "trial_005" / "continuation_context.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("trial_004", context["source_trial_id"])
            self.assertEqual("trial_001", context["recommended_base_trial"])
            self.assertEqual(
                "trial_001", context["decision_context"]["recommended_base_trial"]
            )

    def test_find_duplicate_candidate_detects_repeated_candidate_for_same_axis(self):
        from kaggle_research_agent.trial_decision import find_duplicate_candidate

        decision_context = {
            "rejected_candidates_by_axis": {
                "validation_review": ["validation_review: candidate=time series cv"],
            }
        }
        repeated_plan = {
            "primary_change_axis": "validation_review",
            "change_details": ["Candidate: time series cv"],
        }
        different_plan = {
            "primary_change_axis": "validation_review",
            "change_details": ["Candidate: group k-fold"],
        }
        different_axis_plan = {
            "primary_change_axis": "feature_representation",
            "change_details": ["Candidate: time series cv"],
        }

        self.assertIsNotNone(find_duplicate_candidate(decision_context, repeated_plan))
        self.assertIsNone(find_duplicate_candidate(decision_context, different_plan))
        self.assertIsNone(find_duplicate_candidate(decision_context, different_axis_plan))

    def test_prepare_workspace_trial_plan_forces_replan_when_candidate_repeats_and_succeeds_on_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import prepare_workspace_trial_plan
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            duplicate_plan = {
                "status": "ready",
                "plan_type": "continuation_delta_plan",
                "source_trial_id": "trial_001",
                "primary_change_axis": "validation_review",
                "plan_title": "Retry time series cv",
                "change_details": ["Candidate: time series cv"],
            }
            fresh_plan = {
                "status": "ready",
                "plan_type": "continuation_delta_plan",
                "source_trial_id": "trial_001",
                "primary_change_axis": "validation_review",
                "plan_title": "Try group k-fold instead",
                "change_details": ["Candidate: group k-fold"],
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "change_details": ["Candidate: time series cv"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                with patch(
                    "kaggle_research_agent.demo_one_cycle.create_demo_experiment_plan",
                    side_effect=[duplicate_plan, fresh_plan],
                ) as mock_create_plan:
                    result = prepare_workspace_trial_plan(
                        "demo",
                        "trial_003",
                        source_trial_id="trial_002",
                        allow_api=True,
                    )

            self.assertEqual("planned", result["status"])
            self.assertEqual("Try group k-fold instead", result["plan"]["plan_title"])
            self.assertEqual(2, mock_create_plan.call_count)

    def test_prepare_workspace_trial_plan_writes_korean_preview_before_execution(self):
        # trial_013 in production showed raw English next_experiment.md because
        # user_view/01_plan.ko.md only existed after a trial actually ran.
        # prepare_workspace_trial_plan must now write a Korean-labeled preview
        # as soon as planning succeeds, before any code writing/execution.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import prepare_workspace_trial_plan

            fresh_plan = {
                "status": "ready",
                "plan_type": "continuation_delta_plan",
                "source_trial_id": "trial_012",
                "primary_change_axis": "feature_engineering:new_simple_numeric_features",
                "plan_title": "Add windspeed_zero indicator on top of trial_012",
                "change_details": ["Name: windspeed_is_zero"],
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                with patch(
                    "kaggle_research_agent.demo_one_cycle.create_demo_experiment_plan",
                    return_value=fresh_plan,
                ):
                    result = prepare_workspace_trial_plan(
                        "demo",
                        "trial_013",
                        source_trial_id="trial_012",
                        allow_api=True,
                    )
                preview_path = (
                    root / "experiments" / "demo" / "trial_013" / "user_view" / "01_plan.ko.md"
                )
                self.assertTrue(preview_path.is_file())
                preview_content = preview_path.read_text(encoding="utf-8")

            self.assertEqual("planned", result["status"])
            self.assertIn("trial_013 실험 계획", preview_content)
            self.assertIn("windspeed_is_zero", preview_content)

    def test_prepare_workspace_trial_plan_blocks_when_replan_still_repeats_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)

            from kaggle_research_agent.demo_one_cycle import prepare_workspace_trial_plan
            from kaggle_research_agent.trial_decision import write_trial_decision_card

            duplicate_plan = {
                "status": "ready",
                "plan_type": "continuation_delta_plan",
                "source_trial_id": "trial_001",
                "primary_change_axis": "validation_review",
                "plan_title": "Retry time series cv",
                "change_details": ["Candidate: time series cv"],
            }

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                write_trial_decision_card(
                    "demo",
                    "trial_001",
                    plan={"plan_type": "initial_pipeline_plan"},
                    metrics={"cv_score": 0.85, "objective": "maximize"},
                )
                write_trial_decision_card(
                    "demo",
                    "trial_002",
                    plan={
                        "plan_type": "continuation_delta_plan",
                        "source_trial_id": "trial_001",
                        "primary_change_axis": "validation_review",
                        "change_details": ["Candidate: time series cv"],
                    },
                    metrics={"cv_score": 0.83, "objective": "maximize"},
                )
                with patch(
                    "kaggle_research_agent.demo_one_cycle.create_demo_experiment_plan",
                    side_effect=[duplicate_plan, duplicate_plan],
                ) as mock_create_plan:
                    result = prepare_workspace_trial_plan(
                        "demo",
                        "trial_003",
                        source_trial_id="trial_002",
                        allow_api=True,
                    )

            self.assertEqual("duplicate_candidate_blocked", result["status"])
            self.assertEqual(2, mock_create_plan.call_count)

    def test_demo_one_cycle_resumes_from_completed_plan_and_code_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            plan = {
                "schema_version": "1.0",
                "competition": "demo",
                "trial_id": "trial_001",
                "status": "ready",
                "plan_title": "Existing first-cycle plan",
                "objective": "Resume from an already prepared plan.",
                "rationale": "The previous run completed planning.",
                "implementation_notes": ["Reuse existing code result."],
                "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                "issues": [],
                "next_action": "prepare-workspace-handoff",
            }
            (trial / "demo_experiment_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (trial / "next_experiment.md").write_text("# Existing first-cycle plan\n", encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "Existing code change already accepted.",
                        "changed_files": ["train_step.py"],
                        "validation_results": {
                            "commands": [{"command": "{python} test_step.py", "status": "not_run"}]
                        },
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001", run_now=True)

            self.assertEqual("completed", result["status"])
            self.assertTrue(result["plan"]["resumed_from_existing_artifact"])
            self.assertTrue(result["code_writer"]["resumed_from_existing_artifact"])
            self.assertEqual("completed", result["workspace_run"]["status"])
            self.assertEqual("collected", result["metrics_collection"]["status"])
            self.assertEqual(0.5, result["record"]["local_score"])
            self.assertFalse((root / "memory" / "demo" / "token_usage.jsonl").exists())

    def test_demo_one_cycle_resumes_from_existing_accepted_code_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            trial = root / "experiments" / "demo" / "trial_001"
            trial.mkdir(parents=True)
            plan = {
                "schema_version": "1.0",
                "competition": "demo",
                "trial_id": "trial_001",
                "status": "ready",
                "plan_title": "Existing first-cycle plan",
                "objective": "Resume from an already prepared plan.",
                "rationale": "The previous run completed planning.",
                "implementation_notes": ["Reuse existing accepted validation."],
                "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                "issues": [],
                "next_action": "prepare-workspace-handoff",
            }
            (trial / "demo_experiment_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (trial / "next_experiment.md").write_text("# Existing first-cycle plan\n", encoding="utf-8")
            (trial / "workspace_coding_result.json").write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "summary": "A later retry was blocked by token policy.",
                        "changed_files": [],
                        "validation_results": [],
                        "blocking_issues": ["token_policy_blocked"],
                    }
                ),
                encoding="utf-8",
            )
            (trial / "workspace_coding_result_validation.json").write_text(
                json.dumps(
                    {
                        "competition": "demo",
                        "trial_id": "trial_001",
                        "status": "accepted",
                        "issues": [],
                        "coding_result_status": "completed",
                        "changed_files": ["train_step.py"],
                        "allowed_write_paths": ["src/", "tests/", "train_step.py"],
                        "forbidden_paths": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                        "next_action": "run-workspace-validation-commands",
                    }
                ),
                encoding="utf-8",
            )

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001", run_now=True)

            self.assertEqual("completed", result["status"])
            self.assertEqual("workspace_coding_result_validation", result["code_writer"]["resume_source"])
            self.assertEqual(["train_step.py"], result["code_writer"]["changed_files"])
            self.assertEqual("completed", result["workspace_run"]["status"])

    def test_demo_one_cycle_blocks_without_plan_api_or_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                result = run_demo_one_cycle("demo", "trial_001")

            self.assertEqual("blocked", result["status"])
            self.assertIn("api_call_not_enabled", result["issues"])

    def test_demo_one_cycle_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(
                        [
                            "demo-one-cycle",
                            "--competition",
                            "demo",
                            "--trial",
                            "trial_001",
                            "--mock-plan-file",
                            str(plan_response),
                            "--mock-response-file",
                            str(code_response),
                            "--run-now",
                            "--show-progress",
                        ]
                    )

            self.assertEqual(0, code)
            trial = root / "experiments" / "demo" / "trial_001"
            self.assertTrue(trial_artifact_exists(trial, "demo_one_cycle.json"))
            self.assertIn("[진행] 1/5 대회와 데이터 정보 확인", output.getvalue())
            self.assertIn("1회 실험 사이클 완료", output.getvalue())
            self.assertIn("실험 요약", output.getvalue())

    def test_watch_demo_cycle_cli_prints_current_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent"
            project = Path(tmp) / "project"
            root.mkdir()
            project.mkdir()
            self._write_state(root)
            self._write_profile(root, project)
            self._write_project_scripts(project)
            plan_response = self._write_plan_response(root)
            code_response = self._write_code_response(root)

            with patch("kaggle_research_agent.paths.project_root", return_value=root):
                run_demo_one_cycle(
                    "demo",
                    "trial_001",
                    mock_plan_file=str(plan_response),
                    mock_response_file=str(code_response),
                    run_now=True,
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["watch-demo-cycle", "--competition", "demo", "--trial", "trial_001"])

            text = output.getvalue()
            self.assertEqual(0, code)
            self.assertIn("데모 에이전트 상태: demo / trial_001", text)
            self.assertIn("- 전체 상태: 완료", text)
            self.assertIn("최근 진행 기록:", text)

    def _write_state(self, root: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "competition": {"name": "demo", "metric": "accuracy", "objective": "maximize", "platform": "external"},
                "current_state": {
                    "active_trial": None,
                    "best_trial": None,
                    "consecutive_failures": 0,
                    "submissions_today": 0,
                    "validation_suspected": False,
                },
                "strategy": {"current_focus": "baseline", "promising_directions": [], "forbidden_directions": []},
            },
            comp / "state.yaml",
        )
        (root / "memory" / "demo").mkdir(parents=True, exist_ok=True)
        (root / "memory" / "demo" / "trial_index.jsonl").write_text("", encoding="utf-8")

    def _write_profile(self, root: Path, project: Path) -> None:
        comp = root / "competitions" / "demo"
        comp.mkdir(parents=True, exist_ok=True)
        simple_yaml.dump(
            {
                "schema_version": "1.0",
                "competition": "demo",
                "platform": "external",
                "project_root": str(project),
                "python": sys.executable,
                "commands": {
                    "test": ["{python} test_step.py"],
                    "train": ["{python} train_step.py"],
                    "predict": ["{python} predict_step.py"],
                },
                "artifacts": {
                    "metrics": ["outputs/metrics.json"],
                    "submission": ["outputs/submission.csv"],
                },
                "write_scope": {
                    "allowed": ["src/", "tests/", "train_step.py"],
                    "forbidden": ["data/", "outputs/metrics.json", "outputs/submission.csv"],
                },
                "submission_mode": "manual_external",
            },
            comp / "execution_profile.yaml",
        )

    def _write_project_scripts(self, project: Path) -> None:
        (project / "outputs").mkdir()
        (project / "src").mkdir()
        (project / "tests").mkdir()
        (project / "test_step.py").write_text("from pathlib import Path\nPath('order.txt').write_text('test\\n')\n", encoding="utf-8")
        (project / "train_step.py").write_text(
            "from pathlib import Path\n"
            "Path('outputs').mkdir(exist_ok=True)\n"
            "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.5, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n",
            encoding="utf-8",
        )
        (project / "predict_step.py").write_text(
            "from pathlib import Path\n"
            "path = Path('order.txt')\n"
            "path.write_text(path.read_text() + 'predict\\n')\n"
            "Path('outputs/submission.csv').write_text('id,target\\n1,0\\n')\n",
            encoding="utf-8",
        )

    def _write_plan_response(self, root: Path) -> Path:
        path = root / "mock_plan_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_plan",
                    "usage": {"input_tokens": 300, "output_tokens": 90, "total_tokens": 390},
                    "output_text": json.dumps(
                        {
                            "plan_title": "First local baseline",
                            "objective": "Write a small train script and verify local metrics.",
                            "rationale": "A minimal runnable baseline proves the loop before deeper optimization.",
                            "implementation_notes": ["Update train_step.py to emit metrics."],
                            "expected_outputs": ["outputs/metrics.json", "outputs/submission.csv"],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_code_response(self, root: Path) -> Path:
        path = root / "mock_code_response.json"
        path.write_text(
            json.dumps(
                {
                    "id": "resp_code",
                    "usage": {"input_tokens": 500, "output_tokens": 120, "total_tokens": 620},
                    "output_text": json.dumps(
                        {
                            "status": "completed",
                            "summary": "Confirmed the demo train script remains valid.",
                            "changed_files": ["train_step.py"],
                            "patch_updates": [
                                {
                                    "path": "train_step.py",
                                    "find": "Path('outputs').mkdir(exist_ok=True)\n",
                                    "replace": "Path('outputs').mkdir(exist_ok=True)\n",
                                    "reason": "No-op patch used by the mock response to satisfy patch-only edit mode.",
                                }
                            ],
                            "file_updates": [
                                {
                                    "path": "train_step.py",
                                    "content": (
                                        "from pathlib import Path\n"
                                        "# improved demo baseline\n"
                                        "path = Path('order.txt')\n"
                                        "path.write_text(path.read_text() + 'train\\n')\n"
                                        "Path('outputs').mkdir(exist_ok=True)\n"
                                        "Path('outputs/metrics.json').write_text('{\"cv_score\": 0.88, \"metric\": \"accuracy\", \"objective\": \"maximize\"}\\n')\n"
                                    ),
                                }
                            ],
                            "validation_results": [],
                            "blocking_issues": [],
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
