import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from research_agent.graph.workspace_loop_graph import (
    WorkspaceLoopCallbacks,
    resume_workspace_loop_graph,
    run_workspace_loop_graph,
    workspace_loop_checkpointer,
    workspace_loop_thread_has_pending_work,
)


class WorkspaceLoopGraphTest(unittest.TestCase):
    def test_two_trials_keep_existing_plan_execute_successor_order(self):
        events: list[str] = []
        saved: list[dict] = []
        checkpointer = MemorySaver()

        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, f"prepare:{trial}", {"status": "planned"}),
            run_trial=lambda args, trial: _record(events, f"run:{trial}", {"status": "completed"}),
            plan_successor=lambda args, source, target: _record(
                events,
                f"plan_next:{source}->{target}",
                {"status": "planned"},
            ),
            sync_state=lambda competition: events.append(f"sync:{competition}"),
            save_state=lambda **updates: saved.append(updates) or updates,
            pause_requested=lambda: False,
        )

        result = run_workspace_loop_graph(
            _initial_state(["trial_001", "trial_002"]),
            callbacks,
            checkpointer=checkpointer,
            thread_id="demo-two-trials",
        )

        self.assertEqual("completed", result["status"])
        self.assertEqual(
            [
                "prepare:trial_001",
                "run:trial_001",
                "plan_next:trial_001->trial_002",
                "sync:demo",
                "prepare:trial_002",
                "run:trial_002",
                "plan_next:trial_002->trial_003",
                "sync:demo",
            ],
            events,
        )
        self.assertEqual(
            [
                {"trial_id": "trial_001", "status": "completed"},
                {"trial_id": "trial_002", "status": "completed"},
            ],
            result["rows"],
        )
        self.assertEqual("trial_003", result["next_trial"])
        self.assertTrue(
            checkpointer.get_tuple({"configurable": {"thread_id": "demo-two-trials"}})
        )
        self.assertEqual("langgraph", saved[-1]["graph_runtime"])

    def test_pause_happens_after_successor_plan(self):
        events: list[str] = []
        saved: list[dict] = []
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, "prepare", {"status": "planned"}),
            run_trial=lambda args, trial: _record(events, "run", {"status": "completed"}),
            plan_successor=lambda args, source, target: _record(events, "plan_next", {"status": "planned"}),
            sync_state=lambda competition: events.append("sync"),
            save_state=lambda **updates: saved.append(updates) or updates,
            pause_requested=lambda: events.append("pause_gate") is None,
        )

        result = run_workspace_loop_graph(
            _initial_state(["trial_006"]),
            callbacks,
            thread_id="demo-pause",
        )

        self.assertEqual(["prepare", "run", "plan_next", "sync", "pause_gate"], events)
        self.assertEqual("paused", result["status"])
        self.assertEqual("trial_007", result["next_trial"])
        self.assertEqual("planned", result["phase"])
        self.assertEqual("paused", saved[-1]["status"])

    def test_failed_trial_does_not_plan_or_advance(self):
        events: list[str] = []
        saved: list[dict] = []
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, "prepare", {"status": "planned"}),
            run_trial=lambda args, trial: _record(
                events,
                "run",
                {"status": "code_writer_blocked"},
            ),
            plan_successor=lambda args, source, target: _record(events, "plan_next", {"status": "planned"}),
            sync_state=lambda competition: events.append("sync"),
            save_state=lambda **updates: saved.append(updates) or updates,
            pause_requested=lambda: False,
        )

        result = run_workspace_loop_graph(
            _initial_state(["trial_008"]),
            callbacks,
            thread_id="demo-failure",
        )

        self.assertEqual(["prepare", "run", "sync"], events)
        self.assertEqual("failed", result["status"])
        self.assertEqual("blocked", result["phase"])
        self.assertEqual("trial_008", result["next_trial"])
        self.assertEqual("code_writer_blocked", saved[-1]["error"])

    def test_human_review_request_waits_without_marking_graph_failed(self):
        events: list[str] = []
        saved: list[dict] = []
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, "prepare", {"status": "planned"}),
            run_trial=lambda args, trial: _record(events, "run", {"status": "completed"}),
            plan_successor=lambda args, source, target: _record(
                events,
                "plan_next",
                {"status": "blocked_human_review"},
            ),
            sync_state=lambda competition: events.append("sync"),
            save_state=lambda **updates: saved.append(updates) or updates,
            pause_requested=lambda: False,
        )

        result = run_workspace_loop_graph(
            _initial_state(["trial_009"]),
            callbacks,
            thread_id="demo-human-review",
        )

        self.assertEqual(["prepare", "run", "plan_next", "sync"], events)
        self.assertEqual("awaiting_human_review", result["status"])
        self.assertEqual("awaiting_human_review", result["phase"])
        self.assertEqual("trial_010", result["next_trial"])
        self.assertEqual("awaiting_human_review", saved[-1]["status"])

    def test_human_review_resume_continues_at_successor_plan_without_rerunning_trial(self):
        events: list[str] = []
        review = {"resolved": False}
        plan_attempts = {"count": 0}
        checkpointer = MemorySaver()

        def plan_successor(args, source, target):
            plan_attempts["count"] += 1
            events.append(f"plan_next:{source}->{target}:{plan_attempts['count']}")
            return {
                "status": "planned" if review["resolved"] else "blocked_human_review"
            }

        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, f"prepare:{trial}", {"status": "planned"}),
            run_trial=lambda args, trial: _record(events, f"run:{trial}", {"status": "completed"}),
            plan_successor=plan_successor,
            sync_state=lambda competition: events.append(f"sync:{competition}"),
            save_state=lambda **updates: updates,
            pause_requested=lambda: False,
            human_review_resolved=lambda competition, trial: review["resolved"],
        )

        waiting = run_workspace_loop_graph(
            _initial_state(["trial_009"]),
            callbacks,
            checkpointer=checkpointer,
            thread_id="demo-human-review-resume",
        )
        self.assertEqual("awaiting_human_review", waiting["status"])

        review["resolved"] = True
        resumed = resume_workspace_loop_graph(
            callbacks,
            checkpointer=checkpointer,
            thread_id="demo-human-review-resume",
            resume_value={"action": "feedback_applied"},
        )

        self.assertEqual("completed", resumed["status"])
        self.assertEqual(1, events.count("run:trial_009"))
        self.assertIn("plan_next:trial_009->trial_010:2", events)

    def test_many_trials_in_one_batch_do_not_hit_the_default_recursion_limit(self):
        # Regression for the real bike-sharing-demand run: each trial cycles
        # through ~7 graph nodes (initialize/prepare/execute/plan_successor/
        # sync/pause_gate/advance), so LangGraph's default recursion_limit
        # (25) is only good for ~3 trials in a single run/resume call. A
        # 6-trial batch (6*7=42 steps) used to raise GraphRecursionError even
        # though nothing was actually stuck looping.
        events: list[str] = []
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: _record(events, f"prepare:{trial}", {"status": "planned"}),
            run_trial=lambda args, trial: _record(events, f"run:{trial}", {"status": "completed"}),
            plan_successor=lambda args, source, target: _record(
                events, f"plan_next:{source}->{target}", {"status": "planned"}
            ),
            sync_state=lambda competition: events.append(f"sync:{competition}"),
            save_state=lambda **updates: updates,
            pause_requested=lambda: False,
        )

        trials = [f"trial_{index:03d}" for index in range(1, 7)]
        result = run_workspace_loop_graph(
            _initial_state(trials),
            callbacks,
            thread_id="demo-many-trials",
        )

        self.assertEqual("completed", result["status"])
        self.assertEqual(6, len(result["rows"]))

    def test_terminal_thread_reports_no_pending_work(self):
        # Regression: a finished/failed thread has no next node, so resuming
        # it replays the terminal state without executing anything. The loop
        # used to resume such threads forever -- every restart reported the
        # same stale failure, wrote no state, and exited, so the run could
        # never make progress.
        checkpointer = MemorySaver()
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: {"status": "planned"},
            run_trial=lambda args, trial: {"status": "code_writer_blocked"},
            plan_successor=lambda args, source, target: {"status": "planned"},
            sync_state=lambda competition: None,
            save_state=lambda **updates: updates,
            pause_requested=lambda: False,
        )

        result = run_workspace_loop_graph(
            _initial_state(["trial_028"]),
            callbacks,
            checkpointer=checkpointer,
            thread_id="terminal-thread",
        )

        self.assertEqual("failed", result["status"])
        self.assertFalse(
            workspace_loop_thread_has_pending_work(
                callbacks, checkpointer=checkpointer, thread_id="terminal-thread"
            )
        )

    def test_interrupted_thread_reports_pending_work(self):
        checkpointer = MemorySaver()
        attempts = {"run": 0}

        def run_trial(args, trial):
            attempts["run"] += 1
            if attempts["run"] == 1:
                raise RuntimeError("simulated process interruption")
            return {"status": "completed"}

        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: {"status": "planned"},
            run_trial=run_trial,
            plan_successor=lambda args, source, target: {"status": "planned"},
            sync_state=lambda competition: None,
            save_state=lambda **updates: updates,
            pause_requested=lambda: False,
        )

        with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
            run_workspace_loop_graph(
                _initial_state(["trial_001"]),
                callbacks,
                checkpointer=checkpointer,
                thread_id="interrupted-thread",
            )

        self.assertTrue(
            workspace_loop_thread_has_pending_work(
                callbacks, checkpointer=checkpointer, thread_id="interrupted-thread"
            )
        )

    def test_unknown_thread_reports_no_pending_work(self):
        callbacks = WorkspaceLoopCallbacks(
            prepare_trial=lambda args, trial: {"status": "planned"},
            run_trial=lambda args, trial: {"status": "completed"},
            plan_successor=lambda args, source, target: {"status": "planned"},
            sync_state=lambda competition: None,
            save_state=lambda **updates: updates,
            pause_requested=lambda: False,
        )

        self.assertFalse(
            workspace_loop_thread_has_pending_work(
                callbacks, checkpointer=MemorySaver(), thread_id="never-ran"
            )
        )

    def test_sqlite_checkpointer_persists_graph_state_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "langgraph.sqlite3"
            callbacks = WorkspaceLoopCallbacks(
                prepare_trial=lambda args, trial: {"status": "planned"},
                run_trial=lambda args, trial: {"status": "completed"},
                plan_successor=lambda args, source, target: {"status": "planned"},
                sync_state=lambda competition: None,
                save_state=lambda **updates: updates,
                pause_requested=lambda: False,
            )
            with workspace_loop_checkpointer(path) as (checkpointer, backend):
                self.assertEqual("sqlite", backend)
                run_workspace_loop_graph(
                    _initial_state(["trial_001"]),
                    callbacks,
                    checkpointer=checkpointer,
                    thread_id="persistent-thread",
                )

            with workspace_loop_checkpointer(path) as (checkpointer, backend):
                checkpoint = checkpointer.get_tuple(
                    {"configurable": {"thread_id": "persistent-thread"}}
                )

            self.assertEqual("sqlite", backend)
            self.assertIsNotNone(checkpoint)
            self.assertEqual("completed", checkpoint.checkpoint["channel_values"]["status"])

    def test_sqlite_checkpoint_resumes_from_failed_node_without_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "langgraph.sqlite3"
            events: list[str] = []
            attempts = {"run": 0}

            def run_trial(args, trial):
                attempts["run"] += 1
                events.append(f"run:{attempts['run']}")
                if attempts["run"] == 1:
                    raise RuntimeError("simulated process interruption")
                return {"status": "completed"}

            callbacks = WorkspaceLoopCallbacks(
                prepare_trial=lambda args, trial: _record(events, "prepare", {"status": "planned"}),
                run_trial=run_trial,
                plan_successor=lambda args, source, target: _record(
                    events,
                    "plan_next",
                    {"status": "planned"},
                ),
                sync_state=lambda competition: events.append("sync"),
                save_state=lambda **updates: updates,
                pause_requested=lambda: False,
            )
            with workspace_loop_checkpointer(path) as (checkpointer, _):
                with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                    run_workspace_loop_graph(
                        _initial_state(["trial_001"]),
                        callbacks,
                        checkpointer=checkpointer,
                        thread_id="resume-thread",
                    )

            with workspace_loop_checkpointer(path) as (checkpointer, _):
                result = resume_workspace_loop_graph(
                    callbacks,
                    checkpointer=checkpointer,
                    thread_id="resume-thread",
                )

            self.assertEqual("completed", result["status"])
            self.assertEqual(["prepare", "run:1", "run:2", "plan_next", "sync"], events)


def _record(events: list[str], event: str, result: dict) -> dict:
    events.append(event)
    return result


def _initial_state(trials: list[str]) -> dict:
    return {
        "competition": "demo",
        "selected_trials": trials,
        "trial_index": 0,
        "settings": {"competition": "demo", "code_writer": True},
        "status": "running",
        "phase": "planning",
        "rows": [],
        "steps": [],
        "graph_runtime": "langgraph",
        "graph_thread_id": "test-thread",
    }


if __name__ == "__main__":
    unittest.main()
