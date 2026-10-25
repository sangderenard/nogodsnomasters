import unittest

from token_orchard.contract import (
    REINTERN_COLD,
    REINTERN_CONDITION_OUT,
    REINTERN_RETAIN_LEARNED,
)
from token_orchard.retention import (
    RetentionNoticeBatch,
    retention_program_abi,
    retention_state_abi,
    validate_reintern_policy,
)
from token_orchard.semantic_tensor import retention_stale_mask
from token_orchard.tensor_batch import _abstract_tensor_type


class RetentionContractTests(unittest.TestCase):
    def test_notices_are_batched_integer_logical_time(self):
        batch = RetentionNoticeBatch.from_tokens(17, [100, 101, 100])
        self.assertEqual(batch.notice_tick, 17)
        self.assertEqual(batch.tokens, (100, 101, 100))
        with self.assertRaises(ValueError):
            RetentionNoticeBatch.from_tokens(-1, [100])
        with self.assertRaises(ValueError):
            RetentionNoticeBatch.from_tokens(1, [])
        with self.assertRaises(ValueError):
            RetentionNoticeBatch.from_tokens(1, [True])

    def test_stale_selection_is_one_tensor_batch_and_respects_protection(self):
        abstract_tensor = _abstract_tensor_type()
        with abstract_tensor.use_backend("numpy", "cpu"):
            mask = retention_stale_mask(
                abstract_tensor.tensor([2, 7, 9, 1], dtype="int64"),
                abstract_tensor.tensor([0, 0, 0, 1], dtype="int64"),
                current_tick=10,
                idle_ticks=3,
            ).tolist()
        self.assertEqual(mask, [1, 1, 0, 0])

    def test_reinternment_semantics_are_explicit_not_an_identity_side_effect(self):
        abi = retention_state_abi()
        self.assertEqual(
            abi["reintern_policies"],
            (REINTERN_COLD, REINTERN_RETAIN_LEARNED, REINTERN_CONDITION_OUT),
        )
        self.assertIn(
            "minimum_pinned_epoch", {field[0] for field in abi["fields"]}
        )
        for policy in abi["reintern_policies"]:
            self.assertEqual(validate_reintern_policy(policy), policy)
        with self.assertRaises(ValueError):
            validate_reintern_policy(3)

    def test_stale_mask_has_an_integer_only_turing_program_abi(self):
        abi = retention_program_abi()
        values = abi["values"]
        self.assertEqual(
            {value["parameter"] for value in values},
            {"last_notice_tick", "protected", "current_tick", "idle_ticks"},
        )
        self.assertTrue(all(value["dtype"] == "int64" for value in values))
        self.assertEqual(
            {value["storage"] for value in values}, {"span", "scalar"}
        )


if __name__ == "__main__":
    unittest.main()
