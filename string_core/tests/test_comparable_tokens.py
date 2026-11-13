import random
import unittest

from token_orchard import Vocabulary, TokenOrchardStr
from token_orchard.contract import (
    DOMAIN_BYTES, DOMAIN_TEXT, FINGERPRINT_CONSTANT, STATUS_OK,
    STATUS_INVALID_TOKEN, STATUS_LIMIT_EXCEEDED, TEXT_ATOM_BASE, BYTE_ATOM_BASE,
)


class ComparableTokenTests(unittest.TestCase):
    def token(self, vocabulary, domain, atoms):
        status, token = vocabulary.intern(domain, atoms)
        self.assertEqual(status, STATUS_OK)
        return token

    def test_exact_unpadded_keys_and_empty_domain(self):
        v = Vocabulary()
        for domain in (DOMAIN_TEXT, DOMAIN_BYTES):
            for atoms in ([], [0], [65], [65, 0, 0]):
                token = self.token(v, domain, atoms)
                before = v.state_tuple()
                status, actual_domain, key = v.comparable_tokens(token)
                self.assertEqual((status, actual_domain), (STATUS_OK, domain))
                self.assertEqual(len(key), len(atoms))
                self.assertEqual(key, [v.atom_at(token, i)[1] for i in range(len(atoms))])
                self.assertEqual(v.state_tuple(), before)
        self.assertNotEqual(v.comparable_tokens(1)[1], v.comparable_tokens(2)[1])

    def test_order_matches_exact_atoms_under_total_hash_collision(self):
        rng = random.Random(13)
        for domain, alphabet in ((DOMAIN_TEXT, [0, 65, 97, 0xD7FF, 0xE000, 0x10FFFF]),
                                 (DOMAIN_BYTES, [0, 65, 127, 128, 255])):
            v = Vocabulary(fingerprint_mode=FINGERPRINT_CONSTANT)
            sequences = [[], [0], [0, 0]] + [
                [rng.choice(alphabet) for _ in range(rng.randrange(7))]
                for _ in range(60)
            ]
            keys = [v.comparable_tokens(self.token(v, domain, atoms))[2]
                    for atoms in sequences]
            for i, left in enumerate(sequences):
                for j, right in enumerate(sequences):
                    self.assertEqual(keys[i] < keys[j], left < right)
                    self.assertEqual(keys[i] == keys[j], left == right)

    def test_atoms_reuse_existing_id_ranges(self):
        v = Vocabulary()
        self.assertEqual(v.comparable_tokens(self.token(v, DOMAIN_TEXT, [0]))[2], [TEXT_ATOM_BASE])
        self.assertEqual(v.comparable_tokens(self.token(v, DOMAIN_BYTES, [0]))[2], [BYTE_ATOM_BASE])
        # The scalar hole is removed; adjacent legal scalars have adjacent IDs.
        a = v.comparable_tokens(self.token(v, DOMAIN_TEXT, [0xD7FF]))[2][0]
        b = v.comparable_tokens(self.token(v, DOMAIN_TEXT, [0xE000]))[2][0]
        self.assertEqual(b, a + 1)

    def test_aggregate_mint_order_and_cover_do_not_define_key_order(self):
        v = Vocabulary()
        z = self.token(v, DOMAIN_TEXT, [122, 122])
        a = self.token(v, DOMAIN_TEXT, [97, 97])
        self.assertLess(z, a)  # allocation order is the opposite of text order
        self.assertGreater(v.comparable_tokens(z)[2], v.comparable_tokens(a)[2])
        aaaa = self.token(v, DOMAIN_TEXT, [97] * 4)
        self.assertEqual(v.cover(aaaa)[1], [a, a])
        self.assertEqual(len(v.comparable_tokens(aaaa)[2]), 4)

    def test_batch_invalid_and_empty_are_distinguished(self):
        v = Vocabulary()
        t = self.token(v, DOMAIN_TEXT, [65, 66])
        status, statuses, domains, values, offsets = v.comparable_tokens_batch([1, 0, t, 2, True])
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(statuses, [STATUS_OK, STATUS_INVALID_TOKEN, STATUS_OK, STATUS_OK, STATUS_INVALID_TOKEN])
        self.assertEqual(domains, [DOMAIN_TEXT, 0, DOMAIN_TEXT, DOMAIN_BYTES, 0])
        self.assertEqual(offsets, [0, 0, 0, 2, 2, 2])
        self.assertEqual(values, v.comparable_tokens(t)[2])
        self.assertEqual(v.comparable_tokens_batch([]), (STATUS_OK, [], [], [], [0]))
        self.assertEqual(Vocabulary(max_batch_items=1).comparable_tokens_batch([1, 2]),
                         (STATUS_LIMIT_EXCEEDED, [], [], [], []))

    def test_independent_vocabularies_and_str_operators(self):
        v, w = Vocabulary(), Vocabulary()
        self.token(w, DOMAIN_TEXT, [999, 998])
        for left, right in [('', '\x00'), ('e\u0301', '\u00e9'), ('A', 'a'), ('\ud7ff', '\ue000'), ('abc', 'ab')]:
            a = TokenOrchardStr.from_text(v, left)
            b = TokenOrchardStr.from_text(w, right)
            self.assertEqual(a < b, left < right)
            self.assertEqual(a <= b, left <= right)
            self.assertEqual(a > b, left > right)
            self.assertEqual(a >= b, left >= right)


if __name__ == '__main__':
    unittest.main()
