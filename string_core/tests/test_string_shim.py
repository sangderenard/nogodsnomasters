import unittest

from token_orchard.api import (
    String,
    StringCoreError,
    _reset_global_runtime_for_tests,
)
from token_orchard.contract import (
    ENCODING_UTF8,
    SEMANTIC_DEPTH_NONE,
    STATUS_INVALID_ENCODING,
    STATUS_OK,
)
from token_orchard.string_shim_contract import (
    STRING_NAMESPACE_IDENTITY,
    required_semantic_depth,
    string_shim_receipt,
)


class StringNamespaceTests(unittest.TestCase):
    def tearDown(self):
        _reset_global_runtime_for_tests()

    def test_namespace_cannot_become_a_string_object(self):
        with self.assertRaises(TypeError):
            String()
        with self.assertRaises(TypeError):
            String.bytes("host text is not ingress")

    def test_bytes_return_stable_integer_tokens_through_global_queue(self):
        first = String.bytes(b"abc")
        second = String.bytes(bytearray(b"abc"))
        empty = String.bytes(memoryview(b""))
        self.assertIsInstance(first, int)
        self.assertEqual(first, second)
        self.assertIsInstance(empty, int)

    def test_bytes_batch_is_flat_and_preserves_order(self):
        tokens = String.bytes_batch((b"a", b"bc", b"a"))
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0], tokens[2])
        self.assertNotEqual(tokens[0], tokens[1])

    def test_explicit_decode_preserves_malformed_bytes_receipt(self):
        receipt = String.decode_receipt(b"\xff", ENCODING_UTF8)
        self.assertEqual(receipt[0], STATUS_INVALID_ENCODING)
        self.assertNotEqual(receipt[2], 0)
        self.assertEqual(receipt[3], 0)
        with self.assertRaises(StringCoreError) as failure:
            String.decode(b"\xff", ENCODING_UTF8)
        self.assertEqual(failure.exception.status, STATUS_INVALID_ENCODING)
        self.assertEqual(failure.exception.byte_token, receipt[2])

    def test_valid_decode_returns_text_token_and_integer_receipt(self):
        status, encoding, byte_token, text_token = String.decode_receipt(
            "λ".encode("utf-8"), ENCODING_UTF8
        )
        self.assertEqual((status, encoding), (STATUS_OK, ENCODING_UTF8))
        self.assertNotEqual(byte_token, 0)
        self.assertNotEqual(text_token, 0)
        self.assertEqual(String.decode("λ".encode("utf-8")), text_token)

    def test_compiler_contract_is_exact_identity_not_method_spelling(self):
        self.assertEqual(String.bytes.__module__, "token_orchard.api")
        self.assertEqual(String.bytes.__qualname__, "String.bytes")
        receipt = string_shim_receipt(("bytes", "decode_receipt", "bytes"))
        self.assertEqual(receipt["namespace_identity"], STRING_NAMESPACE_IDENTITY)
        self.assertEqual(receipt["semantic_depth"], SEMANTIC_DEPTH_NONE)
        self.assertEqual(len(receipt["methods"]), 2)
        self.assertTrue(
            all(
                method["identity"].startswith(STRING_NAMESPACE_IDENTITY + ".")
                for method in receipt["methods"]
            )
        )
        self.assertEqual(required_semantic_depth(()), SEMANTIC_DEPTH_NONE)
        with self.assertRaises(ValueError):
            required_semantic_depth(("not_a_string_operation",))


if __name__ == "__main__":
    unittest.main()
