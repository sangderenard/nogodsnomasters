import unittest

from token_orchard import TokenOrchardStr, Vocabulary
from token_orchard.contract import (
    DOMAIN_BYTES, FINGERPRINT_CONSTANT, STATUS_LIMIT_EXCEEDED,
)
from token_orchard.str_value import TokenOrchardStrError


class TokenOrchardStrTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = Vocabulary(fingerprint_mode=FINGERPRINT_CONSTANT)

    def text(self, value):
        return TokenOrchardStr.from_text(self.vocabulary, value)

    def test_operators_match_python_for_unicode_and_empty_values(self):
        for left, right in (("", ""), ("ab", "Ω🙂"), ("e\u0301", "é")):
            with self.subTest(left=left, right=right):
                a, b = self.text(left), self.text(right)
                self.assertEqual((a + b).to_text(), left + right)
                self.assertEqual(len(a), len(left))
                self.assertEqual(bool(a), bool(left))
                self.assertEqual(a == b, left == right)
                self.assertEqual(a != b, left != right)
                self.assertEqual(a < b, left < right)
                self.assertEqual(a <= b, left <= right)
                self.assertEqual(a > b, left > right)
                self.assertEqual(a >= b, left >= right)
                self.assertEqual((a * 3).to_text(), left * 3)
                self.assertEqual((3 * a).to_text(), 3 * left)
                self.assertEqual((a * -2).to_text(), left * -2)
                self.assertEqual(b in a, right in left)

    def test_index_slice_and_iteration_match_python(self):
        source = "aΩ🙂z"
        value = self.text(source)
        for index in (-4, -1, 0, 3):
            self.assertEqual(value[index].to_text(), source[index])
        for span in (
            slice(None), slice(1, 3), slice(None, None, -1),
            slice(3, 0, -2), slice(-20, 20, 2),
        ):
            self.assertEqual(value[span].to_text(), source[span])
        self.assertEqual([part.to_text() for part in value], list(source))
        with self.assertRaises(IndexError):
            value[4]
        with self.assertRaises(TypeError):
            value[1.5]
        with self.assertRaises(ValueError):
            value[::0]

    def test_tokens_share_vocabulary_and_distinct_text_does_not_alias(self):
        a, b, c = self.text("ab"), self.text("ab"), self.text("ac")
        self.assertEqual(a.token, b.token)
        self.assertNotEqual(a.token, c.token)
        self.assertEqual((a + c).vocabulary, self.vocabulary)
        self.assertEqual(hash(a), hash(b))
        other_vocabulary = Vocabulary(fingerprint_mode=FINGERPRINT_CONSTANT)
        other = TokenOrchardStr.from_text(other_vocabulary, "ab")
        self.assertEqual(a, other)
        self.assertEqual(hash(a), hash(other))
        with self.assertRaises(ValueError):
            a + other

    def test_rejects_wrong_domain_invalid_scalar_and_limit(self):
        _, byte_token = self.vocabulary.intern(DOMAIN_BYTES, [65])
        with self.assertRaises(ValueError):
            TokenOrchardStr(self.vocabulary, byte_token)
        with self.assertRaises(TokenOrchardStrError):
            self.text("\ud800")
        limited = Vocabulary(max_sequence_atoms=2)
        value = TokenOrchardStr.from_text(limited, "ab")
        with self.assertRaises(TokenOrchardStrError) as caught:
            value + value
        self.assertEqual(caught.exception.status, STATUS_LIMIT_EXCEEDED)


if __name__ == "__main__":
    unittest.main()
