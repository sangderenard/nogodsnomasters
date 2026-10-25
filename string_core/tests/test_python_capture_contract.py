import unittest

from token_orchard.contract import SEMANTIC_DEPTH_NONE
from token_orchard.python_capture_contract import (
    CAPTURE_CORE_READY,
    CAPTURE_NEEDS_BATCH_OPCODE,
    CAPTURE_NEEDS_EXACT_TRANSFORM,
    PYTHON_STRING_CAPTURES,
    TOKEN_ORCHARD_COMPONENT,
    TURING_EXISTING_STRING_SEAMS,
    python_capture_receipt,
)


class PythonCaptureContractTests(unittest.TestCase):
    def test_exact_reachable_forms_require_one_token_orchard_component(self):
        receipt = python_capture_receipt(
            (
                "source.text_literal",
                "operator.add.text",
                "builtins.len.text_or_bytes",
                "source.text_literal",
            )
        )
        self.assertEqual(receipt["required_component"], TOKEN_ORCHARD_COMPONENT)
        self.assertEqual(receipt["semantic_depth"], SEMANTIC_DEPTH_NONE)
        self.assertEqual(len(receipt["captures"]), 3)

    def test_ready_core_ops_are_separate_from_unimplemented_transforms(self):
        self.assertEqual(
            PYTHON_STRING_CAPTURES["operator.getitem.slice"]["state"],
            CAPTURE_CORE_READY,
        )
        self.assertEqual(
            PYTHON_STRING_CAPTURES["builtins.str.strip"]["state"],
            CAPTURE_NEEDS_EXACT_TRANSFORM,
        )
        for source_form in (
            "operator.contains.text_or_bytes",
            "builtins.str_or_bytes.find",
            "builtins.str_or_bytes.startswith",
            "builtins.str_or_bytes.endswith",
            "builtins.str_or_bytes.count",
        ):
            self.assertEqual(
                PYTHON_STRING_CAPTURES[source_form]["state"],
                CAPTURE_NEEDS_BATCH_OPCODE,
            )

    def test_capture_is_never_inferred_from_a_method_name_alone(self):
        with self.assertRaises(ValueError):
            python_capture_receipt(("strip",))
        self.assertIsNone(python_capture_receipt(())["required_component"])
        self.assertIn("str.strip", TURING_EXISTING_STRING_SEAMS)
        self.assertIn("utf8_encode_alias", TURING_EXISTING_STRING_SEAMS)


if __name__ == "__main__":
    unittest.main()
