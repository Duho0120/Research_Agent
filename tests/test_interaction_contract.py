import unittest

from research_agent.interaction_contract import interaction_contract


class InteractionContractTest(unittest.TestCase):
    def test_contract_returns_an_independent_copy(self):
        first = interaction_contract("experiment_question")
        first["research_state_mutations"].append("plan")

        second = interaction_contract("experiment_question")

        self.assertEqual([], second["research_state_mutations"])

    def test_unknown_channel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown interaction channel"):
            interaction_contract("implicit_chat_write")


if __name__ == "__main__":
    unittest.main()
