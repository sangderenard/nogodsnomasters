import unittest

from token_orchard.contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    ENCODING_LATIN1,
    ENCODING_UTF16_LE,
    ENCODING_UTF8,
    STATUS_INVALID_ENCODING,
    STATUS_OK,
    STATUS_UNSUPPORTED_ENCODING,
)
from token_orchard.core import Vocabulary
from token_orchard.encoding import (
    decode_bytes,
    encode_scalars,
    encode_text_token,
    intern_encoded,
)


class EncodingReceiptTests(unittest.TestCase):
    def test_utf8_receipt_preserves_both_exact_domains(self):
        vocabulary = Vocabulary()
        byte_values = list("Aλ🧠".encode("utf-8"))
        status, encoding, byte_token, text_token = intern_encoded(
            vocabulary, byte_values, ENCODING_UTF8
        )
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(encoding, ENCODING_UTF8)
        self.assertEqual(vocabulary.token_domain(byte_token), DOMAIN_BYTES)
        self.assertEqual(vocabulary.token_domain(text_token), DOMAIN_TEXT)
        self.assertEqual(vocabulary.expand(byte_token), (STATUS_OK, byte_values))
        self.assertEqual(vocabulary.expand(text_token), (STATUS_OK, [65, 0x03BB, 0x1F9E0]))

    def test_invalid_utf8_is_preserved_as_bytes_without_text_alias(self):
        vocabulary = Vocabulary()
        byte_values = [0xF0, 0x28, 0x8C, 0x28]
        status, encoding, byte_token, text_token = intern_encoded(
            vocabulary, byte_values, ENCODING_UTF8
        )
        self.assertEqual(status, STATUS_INVALID_ENCODING)
        self.assertEqual(encoding, ENCODING_UTF8)
        self.assertNotEqual(byte_token, 0)
        self.assertEqual(text_token, 0)
        self.assertEqual(vocabulary.expand(byte_token), (STATUS_OK, byte_values))

    def test_utf16_and_latin1_are_explicit_distinct_transforms(self):
        scalars = [0x41, 0x00E9]
        utf16_status, utf16 = encode_scalars(scalars, ENCODING_UTF16_LE)
        latin_status, latin = encode_scalars(scalars, ENCODING_LATIN1)
        self.assertEqual(utf16_status, STATUS_OK)
        self.assertEqual(latin_status, STATUS_OK)
        self.assertNotEqual(utf16, latin)
        self.assertEqual(decode_bytes(utf16, ENCODING_UTF16_LE), (STATUS_OK, scalars))
        self.assertEqual(decode_bytes(latin, ENCODING_LATIN1), (STATUS_OK, scalars))

    def test_encode_text_token_returns_reversible_receipt(self):
        vocabulary = Vocabulary()
        _, text_token = vocabulary.intern(DOMAIN_TEXT, [0x41, 0x1F680])
        status, encoding, returned_text, byte_token = encode_text_token(
            vocabulary, text_token, ENCODING_UTF8
        )
        self.assertEqual((status, encoding, returned_text), (STATUS_OK, ENCODING_UTF8, text_token))
        byte_status, byte_values = vocabulary.expand(byte_token)
        self.assertEqual(byte_status, STATUS_OK)
        self.assertEqual(decode_bytes(byte_values, ENCODING_UTF8), (STATUS_OK, [0x41, 0x1F680]))

    def test_no_implicit_unicode_normalization(self):
        vocabulary = Vocabulary()
        _, composed = vocabulary.intern(DOMAIN_TEXT, [0x00E9])
        _, decomposed = vocabulary.intern(DOMAIN_TEXT, [0x0065, 0x0301])
        self.assertNotEqual(composed, decomposed)

    def test_unsupported_encoding_is_explicit(self):
        self.assertEqual(decode_bytes([65], 999), (STATUS_UNSUPPORTED_ENCODING, []))
        self.assertEqual(encode_scalars([65], 999), (STATUS_UNSUPPORTED_ENCODING, []))


if __name__ == "__main__":
    unittest.main()
