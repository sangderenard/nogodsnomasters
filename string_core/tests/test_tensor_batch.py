import unittest

from token_orchard.contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    FINGERPRINT_CONSTANT,
    STATUS_INVALID_ATOM,
    STATUS_LIMIT_EXCEEDED,
    STATUS_OK,
)
from token_orchard.core import Vocabulary
from token_orchard.tensor_batch import preflight_batch


class AbstractTensorBatchTests(unittest.TestCase):
    def assert_preflight_matches_scalar(self, domain, items):
        values = []
        offsets = [0]
        for item in items:
            values.extend(item)
            offsets.append(len(values))
        vocabulary = Vocabulary()
        statuses, fingerprints = preflight_batch(
            domain,
            values,
            offsets,
            max_sequence_atoms=vocabulary.max_sequence_atoms,
            backend="numpy",
        )
        self.assertEqual(statuses, [STATUS_OK] * len(items))
        self.assertEqual(
            fingerprints,
            [vocabulary._fingerprint(domain, item) for item in items],
        )

    def test_text_fingerprints_match_scalar_in_one_padded_ragged_batch(self):
        self.assert_preflight_matches_scalar(
            DOMAIN_TEXT,
            [[], [65], [0x03BB], [65, 66], [0x1F9E0, 0, 0x10FFFF]],
        )

    def test_byte_fingerprints_match_scalar_in_one_padded_ragged_batch(self):
        self.assert_preflight_matches_scalar(
            DOMAIN_BYTES,
            [[], [0], [255], [0, 1], [255, 128, 0, 64]],
        )

    def test_parallel_atom_predicates_preserve_item_statuses(self):
        statuses, unused_fingerprints = preflight_batch(
            DOMAIN_TEXT,
            [65, 0xD800, 66, 67, 68, 69],
            [0, 1, 2, 6],
            max_sequence_atoms=3,
            backend="numpy",
        )
        self.assertEqual(
            statuses,
            [STATUS_OK, STATUS_INVALID_ATOM, STATUS_LIMIT_EXCEEDED],
        )

    def test_constant_fingerprint_mode_still_uses_tensor_validation(self):
        statuses, fingerprints = preflight_batch(
            DOMAIN_BYTES,
            [1, 2, 3, 4],
            [0, 2, 4],
            max_sequence_atoms=4,
            fingerprint_mode=FINGERPRINT_CONSTANT,
            backend="numpy",
        )
        self.assertEqual(statuses, [STATUS_OK, STATUS_OK])
        self.assertEqual(fingerprints, [(0, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
