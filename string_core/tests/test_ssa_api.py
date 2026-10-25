import unittest

from token_orchard.contract import (
    AGGREGATE_BASE,
    BYTE_ATOM_BASE,
    BYTES_EMPTY_TOKEN,
    TEXT_ATOM_BASE,
    TEXT_EMPTY_TOKEN,
    TOKEN_KIND_AGGREGATE,
    TOKEN_KIND_ATOM,
    TOKEN_KIND_EMPTY,
    TOKEN_KIND_INVALID,
)
from token_orchard.ssa_api import token_kind_batch
from token_orchard.tensor_batch import _abstract_tensor_type


class SSABatchApiTests(unittest.TestCase):
    def test_token_kind_batch_is_one_integer_tensor_law(self):
        abstract_tensor = _abstract_tensor_type()
        tokens = [
            0,
            TEXT_EMPTY_TOKEN,
            BYTES_EMPTY_TOKEN,
            TEXT_ATOM_BASE,
            BYTE_ATOM_BASE,
            AGGREGATE_BASE,
            AGGREGATE_BASE + 1,
            AGGREGATE_BASE + 2,
        ]
        with abstract_tensor.use_backend("numpy", "cpu"):
            result = token_kind_batch(
                abstract_tensor.tensor(tokens, dtype="int64"), 2
            ).tolist()
        self.assertEqual(
            result,
            [
                TOKEN_KIND_INVALID,
                TOKEN_KIND_EMPTY,
                TOKEN_KIND_EMPTY,
                TOKEN_KIND_ATOM,
                TOKEN_KIND_ATOM,
                TOKEN_KIND_AGGREGATE,
                TOKEN_KIND_AGGREGATE,
                TOKEN_KIND_INVALID,
            ],
        )


if __name__ == "__main__":
    unittest.main()
