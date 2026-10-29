"""Independent laws for the exact authored live workload and frame protocol."""
import unittest

from demos.live_core import FIELDS, check_result, check_manager_result, frame_sizes
from demos.live_workload import live_workload
from demos.live_intern_workload import live_intern_workload
from token_orchard.contract import AGGREGATE_BASE


class LiveCoreLaws(unittest.TestCase):
    def test_manager_view_uses_the_tracked_goldens_and_reports_unexported_fingerprints(self):
        import json
        from pathlib import Path
        from acceptance.manager_workload import repeated_manager
        cases = json.loads((Path(__file__).resolve().parents[1] / "acceptance/cases.json").read_text())["repeated_manager"]
        for arguments, expected in zip(cases["probes"], cases["expected_probe_results"]):
            rounds = arguments[0]
            result = check_manager_result(repeated_manager(rounds), expected, rounds)
            self.assertEqual(result["checksum"], expected)
            self.assertIsNone(result["fingerprint"])
        with self.assertRaisesRegex(AssertionError, "Native manager"):
            check_manager_result(-14, 7000175, 1)
        with self.assertRaisesRegex(AssertionError, "golden law"):
            check_manager_result(99, 99, 1)
    def test_independent_interning_product_matches_the_extended_workload(self):
        for rounds in (0, 1, 3, 12):
            for collision in (0, 1):
                for method in (0, 1):
                    self.assertEqual(live_intern_workload(rounds, method, collision),
                                     live_workload(rounds, method, collision))
    def test_all_methods_and_collision_modes_preserve_identity(self):
        for rounds in (0, 1, 3, 12):
            for collision in (0, 1):
                for method in range(4):
                    with self.subTest(rounds=rounds, method=method, collision=collision):
                        result = live_workload(rounds, method, collision)
                        metrics = check_result(result, result, rounds, collision)
                        self.assertEqual(metrics["last_token"], AGGREGATE_BASE + 2 * (rounds - 1) if rounds else 0)
                        self.assertEqual(metrics["checks"], rounds * (13 if method == 3 else 10 if method == 2 else 4) + 2)
                        self.assertEqual(metrics["last_length"], 2 if rounds and method >= 2 else 0)

    def test_forced_collisions_do_not_change_mint_or_cover_history(self):
        for method in range(4):
            normal = dict(zip(FIELDS, live_workload(9, method, 0)))
            colliding = dict(zip(FIELDS, live_workload(9, method, 1)))
            del normal["fingerprint"]
            del colliding["fingerprint"]
            self.assertEqual(normal, colliding)

    def test_exact_comparison_cannot_round_away_an_integer_error(self):
        expected = list(live_workload(1, 0, 0))
        expected[9] = 2 ** 60
        actual = expected.copy()
        actual[9] += 1
        with self.assertRaisesRegex(AssertionError, "checksum"):
            check_result(actual, expected, 1, 0)

    def test_python_agreement_alone_does_not_override_independent_laws(self):
        wrong = list(live_workload(1, 0, 0))
        wrong[2] = 0
        with self.assertRaisesRegex(AssertionError, "Independent law epoch"):
            check_result(wrong, wrong, 1, 0)

    def test_live_frames_include_zero_and_exact_requested_target(self):
        self.assertEqual(frame_sizes(0), [0])
        self.assertEqual(frame_sizes(1), [0, 1])
        self.assertEqual(frame_sizes(7), [0, 1, 2, 4, 7])
        self.assertEqual(frame_sizes(8), [0, 1, 2, 4, 8])


if __name__ == "__main__":
    unittest.main()
