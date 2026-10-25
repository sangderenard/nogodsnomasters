import hashlib
import struct
import unittest

from token_orchard.contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    FINGERPRINT_CONSTANT,
    STATUS_INVALID_SNAPSHOT,
    STATUS_OK,
    STATUS_UNSUPPORTED_SCHEMA,
)
from token_orchard.core import Vocabulary
from token_orchard.snapshot import (
    DIGEST_BYTES,
    HEADER_FORMAT,
    from_snapshot_bytes,
    snapshot_digest,
    to_snapshot_bytes,
)


class SnapshotTests(unittest.TestCase):
    def populated(self, fingerprint_mode=0):
        vocabulary = Vocabulary(fingerprint_mode=fingerprint_mode)
        vocabulary.intern(DOMAIN_TEXT, [97, 98])
        vocabulary.intern(DOMAIN_TEXT, [97, 98, 99])
        vocabulary.intern(DOMAIN_BYTES, [0, 255, 1])
        return vocabulary

    def test_snapshot_is_deterministic_and_round_trips_authoritative_state(self):
        vocabulary = self.populated()
        first = to_snapshot_bytes(vocabulary)
        second = to_snapshot_bytes(vocabulary)
        self.assertEqual(first, second)
        digest_status, digest = snapshot_digest(first)
        self.assertEqual(digest_status, STATUS_OK)
        self.assertEqual(len(digest), DIGEST_BYTES)
        status, loaded = from_snapshot_bytes(first)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(loaded.state_tuple(), vocabulary.state_tuple())
        self.assertEqual(to_snapshot_bytes(loaded), first)

    def test_reload_continues_minting_identically(self):
        original = self.populated()
        status, loaded = from_snapshot_bytes(to_snapshot_bytes(original))
        self.assertEqual(status, STATUS_OK)
        history = [
            (DOMAIN_TEXT, [97, 98, 99, 100]),
            (DOMAIN_BYTES, [0, 255, 1, 2]),
            (DOMAIN_TEXT, [0x1F9E0, 0x1F680]),
        ]
        original_results = [original.intern(domain, atoms) for domain, atoms in history]
        loaded_results = [loaded.intern(domain, atoms) for domain, atoms in history]
        self.assertEqual(loaded_results, original_results)
        self.assertEqual(loaded.state_tuple(), original.state_tuple())

    def test_constant_fingerprint_snapshot_retains_collision_semantics(self):
        vocabulary = self.populated(FINGERPRINT_CONSTANT)
        snapshot = to_snapshot_bytes(vocabulary)
        status, loaded = from_snapshot_bytes(snapshot)
        self.assertEqual(status, STATUS_OK)
        first = loaded.intern(DOMAIN_TEXT, [1, 2, 3])
        second = loaded.intern(DOMAIN_TEXT, [1, 2, 4])
        self.assertEqual(first[0], STATUS_OK)
        self.assertEqual(second[0], STATUS_OK)
        self.assertNotEqual(first[1], second[1])

    def test_corrupt_payload_or_digest_is_rejected(self):
        snapshot = bytearray(to_snapshot_bytes(self.populated()))
        snapshot[len(snapshot) // 2] ^= 0x80
        status, loaded = from_snapshot_bytes(bytes(snapshot))
        self.assertEqual(status, STATUS_INVALID_SNAPSHOT)
        self.assertIsNone(loaded)

    def test_structural_corruption_with_recomputed_digest_is_rejected(self):
        snapshot = to_snapshot_bytes(self.populated())
        body = bytearray(snapshot[:-DIGEST_BYTES])
        header_size = struct.calcsize(HEADER_FORMAT)
        first_section_header_size = struct.calcsize("<HBBQQ")
        first_payload = header_size + first_section_header_size
        body[first_payload] = 99
        forged = bytes(body) + hashlib.sha256(body).digest()
        status, loaded = from_snapshot_bytes(forged)
        self.assertEqual(status, STATUS_INVALID_SNAPSHOT)
        self.assertIsNone(loaded)

    def test_unknown_schema_is_distinct_from_corruption(self):
        snapshot = to_snapshot_bytes(self.populated())
        body = bytearray(snapshot[:-DIGEST_BYTES])
        schema_offset = 8
        body[schema_offset : schema_offset + 2] = struct.pack("<H", 999)
        forged = bytes(body) + hashlib.sha256(body).digest()
        status, loaded = from_snapshot_bytes(forged)
        self.assertEqual(status, STATUS_UNSUPPORTED_SCHEMA)
        self.assertIsNone(loaded)

    def test_invalid_serialized_limits_are_rejected_as_snapshot_status(self):
        snapshot = to_snapshot_bytes(self.populated())
        body = bytearray(snapshot[:-DIGEST_BYTES])
        max_sequence_offset = 8 + 2 + 2 + 2 + 2 + 8
        body[max_sequence_offset : max_sequence_offset + 8] = struct.pack(
            "<Q", (1 << 32) + 1
        )
        forged = bytes(body) + hashlib.sha256(body).digest()
        status, loaded = from_snapshot_bytes(forged)
        self.assertEqual(status, STATUS_INVALID_SNAPSHOT)
        self.assertIsNone(loaded)

    def test_snapshot_cannot_claim_budget_smaller_than_existing_rows(self):
        snapshot = to_snapshot_bytes(self.populated())
        body = bytearray(snapshot[:-DIGEST_BYTES])
        max_aggregates_offset = 8 + 2 + 2 + 2 + 2 + 8 + 8 + 8
        body[max_aggregates_offset : max_aggregates_offset + 8] = struct.pack(
            "<Q", 0
        )
        forged = bytes(body) + hashlib.sha256(body).digest()
        status, loaded = from_snapshot_bytes(forged)
        self.assertEqual(status, STATUS_INVALID_SNAPSHOT)
        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
