import unittest

from token_orchard.contract import (
    OBSERVATION_ADJACENT,
    OBSERVATION_WEIGHT_ONE,
    SEMANTIC_DEPTH_CONTEXT,
    SEMANTIC_DEPTH_NONE,
    SEMANTIC_DEPTH_QUERY,
    SEMANTIC_DEPTH_TRAINING,
)
from token_orchard.semantic import (
    SemanticObservationBatch,
    semantic_entrypoints,
    semantic_phase_roots,
    semantic_program_abi,
    semantic_state_abi,
)
from token_orchard.semantic_tensor import (
    embedding_ordered_composition_batch,
)
from token_orchard.tensor_batch import _abstract_tensor_type


class SemanticContractTests(unittest.TestCase):
    def test_ssa_depths_are_strict_entrypoint_prefixes(self):
        previous = ()
        for depth in range(SEMANTIC_DEPTH_NONE, SEMANTIC_DEPTH_QUERY + 1):
            current = semantic_phase_roots(depth)
            self.assertEqual(current[: len(previous)], previous)
            self.assertEqual(len(current), depth)
            previous = current
        self.assertEqual(
            semantic_entrypoints(SEMANTIC_DEPTH_CONTEXT)[-1],
            "embedding_observation_target_batch",
        )
        with self.assertRaises(NotImplementedError):
            semantic_entrypoints(SEMANTIC_DEPTH_TRAINING)

    def test_observation_abi_is_aligned_integer_only(self):
        batch = SemanticObservationBatch.from_columns(
            7,
            [OBSERVATION_ADJACENT, OBSERVATION_ADJACENT],
            [101, 101],
            [102, 103],
            [200, 200],
            [-1, 1],
            [1, 1],
            [0, 0],
            [OBSERVATION_WEIGHT_ONE, OBSERVATION_WEIGHT_ONE],
        )
        self.assertEqual(batch.count, 2)
        self.assertEqual(batch.end_sequence, 9)
        with self.assertRaises(ValueError):
            SemanticObservationBatch.from_columns(
                0, [OBSERVATION_ADJACENT], [1], [], [0], [0], [1], [0], [1]
            )
        with self.assertRaises(ValueError):
            SemanticObservationBatch.from_columns(
                0,
                [OBSERVATION_ADJACENT],
                [True],
                [2],
                [0],
                [0],
                [1],
                [0],
                [1],
            )
        with self.assertRaises(ValueError):
            SemanticObservationBatch.from_columns(
                0,
                [OBSERVATION_ADJACENT],
                [1],
                [2],
                [0],
                [0],
                [1],
                [2],
                [1],
            )

    def test_state_abi_contains_only_linked_phase_state(self):
        context = semantic_state_abi(SEMANTIC_DEPTH_CONTEXT, 32)
        context_names = {field[0] for field in context["fields"]}
        self.assertIn("observation_columns", context_names)
        self.assertNotIn("model_parameters", context_names)
        training = semantic_state_abi(SEMANTIC_DEPTH_TRAINING, 32)
        training_names = {field[0] for field in training["fields"]}
        self.assertIn("model_parameters", training_names)
        self.assertIn("adam_first_moment", training_names)
        self.assertNotIn("published_vectors", training_names)

    def test_program_abi_is_a_depth_pruned_turing_contract(self):
        position = semantic_program_abi(1)
        self.assertEqual(
            {value["function"] for value in position["values"]},
            {"embedding_position_batch"},
        )
        context = semantic_program_abi(SEMANTIC_DEPTH_CONTEXT)
        self.assertIn(
            "embedding_observation_target_batch",
            {value["function"] for value in context["values"]},
        )

    def test_ordered_coo_target_distinguishes_reversed_children(self):
        abstract_tensor = _abstract_tensor_type()
        with abstract_tensor.use_backend("numpy", "cpu"):
            child_vectors = abstract_tensor.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                dtype="float64",
            )
            result = embedding_ordered_composition_batch(
                child_vectors,
                abstract_tensor.tensor([0, 0, 1, 1], dtype="int64"),
                abstract_tensor.tensor([0, 1, 0, 1], dtype="int64"),
                abstract_tensor.tensor([2, 2, 2, 2], dtype="int64"),
                2,
                2,
            ).tolist()
        self.assertNotEqual(result[0], result[1])


if __name__ == "__main__":
    unittest.main()
