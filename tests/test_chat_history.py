import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_research_agent.chat_history import (
    answer_chat_question,
    chat_history_snapshot,
    start_new_chat,
)


class ChatHistoryTest(unittest.TestCase):
    def test_answer_is_recorded_and_reloaded_with_trial_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            result_payload = {
                "answer": "제출 점수는 0.77입니다.",
                "mode": "local_evidence",
                "mode_label": "로컬 근거 모드",
                "sources": ["scores.md"],
                "warning": None,
                "interaction": {"access": "read_only"},
            }
            with patch(
                "kaggle_research_agent.chat_history.answer_experiment_question",
                return_value=result_payload,
            ) as answer:
                result = answer_chat_question(
                    "demo",
                    "trial_003",
                    "제출 점수는?",
                    db_path=db_path,
                )

            snapshot = chat_history_snapshot("demo", db_path=db_path)

            self.assertIn("[로컬 근거 모드]", result["rendered_answer"])
            self.assertEqual(["user", "assistant"], [row["role"] for row in snapshot["messages"]])
            self.assertEqual("trial_003", snapshot["messages"][0]["trial_id"])
            self.assertEqual(["scores.md"], snapshot["messages"][1]["metadata"]["sources"])
            answer.assert_called_once()
            self.assertEqual([], answer.call_args.kwargs["conversation"])

    def test_follow_up_uses_only_recent_six_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            with patch(
                "kaggle_research_agent.chat_history.answer_experiment_question",
                return_value={
                    "answer": "답변",
                    "mode": "local_evidence",
                    "mode_label": "로컬 근거 모드",
                    "sources": [],
                    "warning": None,
                    "interaction": {"access": "read_only"},
                },
            ) as answer:
                session_id = None
                for index in range(5):
                    result = answer_chat_question(
                        "demo",
                        f"trial_{index + 1:03d}",
                        f"질문 {index}",
                        session_id=session_id,
                        db_path=db_path,
                    )
                    session_id = result["session"]["session_id"]

            conversation = answer.call_args.kwargs["conversation"]
            self.assertEqual(6, len(conversation))
            self.assertEqual("질문 1", conversation[0]["content"])
            self.assertEqual("assistant", conversation[-1]["role"])

    def test_new_session_keeps_previous_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite3"
            with patch(
                "kaggle_research_agent.chat_history.answer_experiment_question",
                return_value={
                    "answer": "답변",
                    "mode": "local_evidence",
                    "mode_label": "로컬 근거 모드",
                    "sources": [],
                    "warning": None,
                    "interaction": {"access": "read_only"},
                },
            ):
                first = answer_chat_question(
                    "demo",
                    "trial_001",
                    "첫 대화",
                    db_path=db_path,
                )
            second = start_new_chat("demo", db_path=db_path)

            self.assertNotEqual(
                first["session"]["session_id"],
                second["active_session"]["session_id"],
            )
            self.assertEqual(2, len(second["sessions"]))
            self.assertEqual([], second["messages"])


if __name__ == "__main__":
    unittest.main()
