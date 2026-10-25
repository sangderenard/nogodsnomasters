import random
import unittest

from token_orchard.contract import (
    AGGREGATE_BASE,
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    FINGERPRINT_CONSTANT,
    STATUS_INVALID_ATOM,
    STATUS_INVALID_OFFSETS,
    STATUS_INVALID_SLICE,
    STATUS_INVALID_TOKEN,
    STATUS_LIMIT_EXCEEDED,
    STATUS_OK,
)
from token_orchard.core import Vocabulary


class VocabularyLawTests(unittest.TestCase):
    def assert_round_trip(self, vocabulary, domain, atoms):
        status, token = vocabulary.intern(domain, atoms)
        self.assertEqual(status, STATUS_OK)
        expand_status, expanded = vocabulary.expand(token)
        self.assertEqual(expand_status, STATUS_OK)
        self.assertEqual(expanded, atoms)
        return token

    def test_text_round_trips_exact_scalars(self):
        vocabulary = Vocabulary()
        cases = [
            [],
            [0],
            [ord("A")],
            [0x03BB, 0x03BF, 0x03B3, 0x03BF, 0x03C2],
            [ord("e"), 0x0301],
            [0x1F9E0, 0x1F680],
            [ord("x")] * 4096,
        ]
        for atoms in cases:
            self.assert_round_trip(vocabulary, DOMAIN_TEXT, atoms)

    def test_arithmetic_atom_boundaries_round_trip(self):
        vocabulary = Vocabulary()
        for scalar in [0, 0xD7FF, 0xE000, 0x10FFFF]:
            token = self.assert_round_trip(vocabulary, DOMAIN_TEXT, [scalar])
            self.assertEqual(vocabulary.token_domain(token), DOMAIN_TEXT)
        for byte in range(256):
            token = self.assert_round_trip(vocabulary, DOMAIN_BYTES, [byte])
            self.assertEqual(vocabulary.token_domain(token), DOMAIN_BYTES)

    def test_binary_round_trips_every_byte_and_invalid_utf8_shape(self):
        vocabulary = Vocabulary()
        data = list(range(256)) + [0xFF, 0xC0, 0x80, 0]
        self.assert_round_trip(vocabulary, DOMAIN_BYTES, data)

    def test_text_and_bytes_are_disjoint_domains(self):
        vocabulary = Vocabulary()
        text_token = self.assert_round_trip(vocabulary, DOMAIN_TEXT, [65])
        byte_token = self.assert_round_trip(vocabulary, DOMAIN_BYTES, [65])
        self.assertNotEqual(text_token, byte_token)
        self.assertEqual(vocabulary.token_domain(text_token), DOMAIN_TEXT)
        self.assertEqual(vocabulary.token_domain(byte_token), DOMAIN_BYTES)

    def test_intern_is_idempotent_and_injective_under_total_collision(self):
        vocabulary = Vocabulary(fingerprint_mode=FINGERPRINT_CONSTANT)
        first = self.assert_round_trip(vocabulary, DOMAIN_TEXT, [1, 2, 3])
        again = self.assert_round_trip(vocabulary, DOMAIN_TEXT, [1, 2, 3])
        second = self.assert_round_trip(vocabulary, DOMAIN_TEXT, [1, 2, 4])
        binary = self.assert_round_trip(vocabulary, DOMAIN_BYTES, [1, 2, 3])
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, binary)

    def test_cover_uses_only_prior_tokens_and_expands_exactly(self):
        vocabulary = Vocabulary()
        status, token_ab = vocabulary.intern(DOMAIN_TEXT, [97, 98])
        self.assertEqual(status, STATUS_OK)
        status, token_abc = vocabulary.intern(DOMAIN_TEXT, [97, 98, 99])
        self.assertEqual(status, STATUS_OK)
        status, cover = vocabulary.cover(token_abc)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(cover[0], token_ab)
        self.assertTrue(all(child < token_abc for child in cover))
        expanded = []
        for child in cover:
            child_status, child_atoms = vocabulary.expand(child)
            self.assertEqual(child_status, STATUS_OK)
            expanded.extend(child_atoms)
        self.assertEqual(expanded, [97, 98, 99])

    def test_later_better_cover_does_not_change_identity_or_original_cover(self):
        vocabulary = Vocabulary()
        status, whole = vocabulary.intern(DOMAIN_TEXT, [97, 98, 99, 100])
        self.assertEqual(status, STATUS_OK)
        status, initial_cover = vocabulary.cover(whole)
        self.assertEqual(status, STATUS_OK)
        vocabulary.intern(DOMAIN_TEXT, [97, 98])
        vocabulary.intern(DOMAIN_TEXT, [99, 100])
        status, again = vocabulary.intern(DOMAIN_TEXT, [97, 98, 99, 100])
        self.assertEqual(status, STATUS_OK)
        status, retained_cover = vocabulary.cover(again)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(again, whole)
        self.assertEqual(retained_cover, initial_cover)

    def test_concat_slice_length_and_atom_at(self):
        vocabulary = Vocabulary()
        _, left = vocabulary.intern(DOMAIN_TEXT, [10, 20])
        _, right = vocabulary.intern(DOMAIN_TEXT, [30, 40])
        status, combined = vocabulary.concat(left, right)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(vocabulary.expand(combined), (STATUS_OK, [10, 20, 30, 40]))
        self.assertEqual(vocabulary.length(combined), (STATUS_OK, 4))
        status, middle = vocabulary.slice(combined, 1, 3)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(vocabulary.expand(middle), (STATUS_OK, [20, 30]))
        status, atom_token = vocabulary.atom_at(combined, 2)
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(vocabulary.expand(atom_token), (STATUS_OK, [30]))

    def test_invalid_input_does_not_mutate(self):
        vocabulary = Vocabulary(max_sequence_atoms=3)
        before = vocabulary.state_tuple()
        cases = [
            vocabulary.intern(DOMAIN_TEXT, [0xD800]),
            vocabulary.intern(DOMAIN_TEXT, [-1]),
            vocabulary.intern(DOMAIN_BYTES, [256]),
            vocabulary.intern(DOMAIN_TEXT, [1, 2, 3, 4]),
        ]
        self.assertEqual(cases[0][0], STATUS_INVALID_ATOM)
        self.assertEqual(cases[1][0], STATUS_INVALID_ATOM)
        self.assertEqual(cases[2][0], STATUS_INVALID_ATOM)
        self.assertEqual(cases[3][0], STATUS_LIMIT_EXCEEDED)
        self.assertEqual(vocabulary.state_tuple(), before)

    def test_invalid_slice_and_token_are_receipted(self):
        vocabulary = Vocabulary()
        _, token = vocabulary.intern(DOMAIN_TEXT, [1, 2])
        before = vocabulary.state_tuple()
        self.assertEqual(vocabulary.slice(token, -1, 1)[0], STATUS_INVALID_SLICE)
        self.assertEqual(vocabulary.slice(token, 0, 3)[0], STATUS_INVALID_SLICE)
        self.assertEqual(vocabulary.expand(0)[0], STATUS_INVALID_TOKEN)
        self.assertEqual(vocabulary.state_tuple(), before)

    def test_aggregate_limit_is_atomic(self):
        vocabulary = Vocabulary(max_aggregates=0)
        before = vocabulary.state_tuple()
        status, token = vocabulary.intern(DOMAIN_TEXT, [1, 2])
        self.assertEqual((status, token), (STATUS_LIMIT_EXCEEDED, 0))
        self.assertEqual(vocabulary.state_tuple(), before)

    def test_batch_is_flat_ordered_and_envelope_atomic(self):
        vocabulary = Vocabulary()
        status, item_statuses, tokens = vocabulary.intern_batch(
            DOMAIN_TEXT,
            [97, 98, 97, 98, 99],
            [0, 2, 5],
        )
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(item_statuses, [STATUS_OK, STATUS_OK])
        cover_status, cover = vocabulary.cover(tokens[1])
        self.assertEqual(cover_status, STATUS_OK)
        self.assertEqual(cover[0], tokens[0])

        before = vocabulary.state_tuple()
        status, item_statuses, tokens = vocabulary.intern_batch(
            DOMAIN_TEXT, [1, 2], [0, 2, 1]
        )
        self.assertEqual(status, STATUS_INVALID_OFFSETS)
        self.assertEqual(item_statuses, [])
        self.assertEqual(tokens, [])
        self.assertEqual(vocabulary.state_tuple(), before)

    def test_invalid_batch_item_does_not_block_later_valid_item(self):
        vocabulary = Vocabulary()
        status, item_statuses, tokens = vocabulary.intern_batch(
            DOMAIN_BYTES, [256, 1, 2], [0, 1, 3]
        )
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(item_statuses, [STATUS_INVALID_ATOM, STATUS_OK])
        self.assertEqual(tokens[0], 0)
        self.assertEqual(vocabulary.expand(tokens[1]), (STATUS_OK, [1, 2]))

    def test_repeated_history_is_deterministic(self):
        histories = [
            (DOMAIN_TEXT, [97, 98]),
            (DOMAIN_BYTES, [0, 255, 0]),
            (DOMAIN_TEXT, [97, 98, 99]),
            (DOMAIN_TEXT, [0x1F9E0]),
        ]
        left = Vocabulary()
        right = Vocabulary()
        left_tokens = [left.intern(domain, atoms) for domain, atoms in histories]
        right_tokens = [right.intern(domain, atoms) for domain, atoms in histories]
        self.assertEqual(left_tokens, right_tokens)
        self.assertEqual(left.state_tuple(), right.state_tuple())

    def test_aggregate_arrays_are_aligned(self):
        vocabulary = Vocabulary()
        vocabulary.intern(DOMAIN_TEXT, [1, 2])
        vocabulary.intern(DOMAIN_BYTES, [1, 2])
        row_count = len(vocabulary.domain)
        self.assertEqual(AGGREGATE_BASE + row_count - 1, AGGREGATE_BASE + 1)
        for values in (
            vocabulary.atom_length,
            vocabulary.fingerprint_hi,
            vocabulary.fingerprint_lo,
            vocabulary.exact_begin,
            vocabulary.exact_count,
            vocabulary.cover_begin,
            vocabulary.cover_count,
            vocabulary.created_epoch,
        ):
            self.assertEqual(len(values), row_count)

    def test_deterministic_randomized_laws_under_total_collision(self):
        random_source = random.Random(90210)
        vocabulary = Vocabulary(fingerprint_mode=FINGERPRINT_CONSTANT)
        observed = {}
        for unused in range(300):
            domain = DOMAIN_TEXT if random_source.randrange(2) == 0 else DOMAIN_BYTES
            length = random_source.randrange(0, 18)
            if domain == DOMAIN_TEXT:
                atoms = [random_source.randrange(0, 0xD800) for unused_atom in range(length)]
            else:
                atoms = [random_source.randrange(0, 256) for unused_atom in range(length)]
            status, token = vocabulary.intern(domain, atoms)
            self.assertEqual(status, STATUS_OK)
            key = (domain, tuple(atoms))
            if key in observed:
                self.assertEqual(token, observed[key])
            observed[key] = token
            self.assertEqual(vocabulary.expand(token), (STATUS_OK, atoms))
            cover_status, cover = vocabulary.cover(token)
            self.assertEqual(cover_status, STATUS_OK)
            rebuilt = []
            for child in cover:
                if token >= AGGREGATE_BASE:
                    self.assertLess(child, token)
                child_status, child_atoms = vocabulary.expand(child)
                self.assertEqual(child_status, STATUS_OK)
                rebuilt.extend(child_atoms)
            self.assertEqual(rebuilt, atoms)


if __name__ == "__main__":
    unittest.main()
