import tempfile
import unittest
from pathlib import Path

from research_agent import simple_yaml


class SimpleYamlRoundTripTest(unittest.TestCase):
    def test_numeric_looking_string_round_trips_as_string(self):
        # Real bug: DACON competition ids are pure digits (e.g. "236716").
        # Writing it bare and reading it back must not silently turn it
        # into an int -- that broke a `competition == competition_id`
        # string comparison in execution_profile.py's mismatch check.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            simple_yaml.dump({"competition": "236716"}, path)
            loaded = simple_yaml.load(path)

        self.assertEqual("236716", loaded["competition"])
        self.assertIsInstance(loaded["competition"], str)

    def test_bool_and_null_looking_strings_round_trip_as_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            simple_yaml.dump({"a": "true", "b": "false", "c": "null", "d": "3.5"}, path)
            loaded = simple_yaml.load(path)

        self.assertEqual({"a": "true", "b": "false", "c": "null", "d": "3.5"}, loaded)

    def test_real_bool_and_int_values_still_round_trip_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            simple_yaml.dump({"flag": True, "count": 236716, "ratio": 3.5, "empty": None}, path)
            loaded = simple_yaml.load(path)

        self.assertEqual({"flag": True, "count": 236716, "ratio": 3.5, "empty": None}, loaded)

    def test_ordinary_string_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            simple_yaml.dump({"competition": "titanic"}, path)
            loaded = simple_yaml.load(path)

        self.assertEqual("titanic", loaded["competition"])


if __name__ == "__main__":
    unittest.main()
