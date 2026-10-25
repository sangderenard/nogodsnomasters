import unittest

from token_orchard.contract import (
    AGGREGATE_BASE,
    DOMAIN_TEXT,
    OBSERVATION_ADJACENT,
    OBSERVATION_OPERATION,
    OBSERVATION_WEIGHT_ONE,
    SEMANTIC_DEPTH_POSITION,
    STATUS_OK,
)
from token_orchard.core import Vocabulary
from token_orchard.embedding import ContinuousEmbeddingLearner
from token_orchard.service import TokenService
from token_orchard.semantic import SemanticObservationBatch


class ContinuousEmbeddingTests(unittest.TestCase):
    def test_graph_delta_is_flat_exact_cover_topology(self):
        vocabulary = Vocabulary()
        status, first = vocabulary.intern(DOMAIN_TEXT, [97, 98])
        self.assertEqual(status, STATUS_OK)
        status, second = vocabulary.intern(DOMAIN_TEXT, [97, 98, 99])
        self.assertEqual(status, STATUS_OK)
        delta = vocabulary.graph_delta_since_row(0)
        begin, end, parents, covers, offsets = delta
        self.assertEqual((begin, end), (0, 2))
        self.assertEqual(parents, [first, second])
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(covers))
        for index, token in enumerate(parents):
            status, exact_cover = vocabulary.cover(token)
            self.assertEqual(status, STATUS_OK)
            self.assertEqual(
                covers[offsets[index] : offsets[index + 1]], exact_cover
            )

    def test_abstract_nn_adam_trains_shared_spherical_adapter(self):
        learner = ContinuousEmbeddingLearner(
            dimension=6, learning_rate=0.02, steps_per_delta=4
        )
        atom_a = 3 + 97
        atom_b = 3 + 98
        self.assertTrue(
            learner.offer_delta(
                0,
                1,
                [AGGREGATE_BASE],
                [atom_a, atom_b],
                [0, 2],
            )
        )
        self.assertTrue(learner.wait_until_epoch(1, 5.0))
        snapshot = learner.snapshot()
        learner.close()
        rows = {token: index for index, token in enumerate(snapshot.tokens)}
        for position in snapshot.positions:
            radius = sum(value * value for value in position) ** 0.5
            self.assertAlmostEqual(radius, 1.0, places=12)
        for position, residual, meaning in zip(
            snapshot.positions, snapshot.residuals, snapshot.vectors
        ):
            for base, learned, actual in zip(position, residual, meaning):
                self.assertAlmostEqual(base + learned, actual, places=12)
        parent = snapshot.vectors[rows[AGGREGATE_BASE]]
        left = snapshot.vectors[rows[atom_a]]
        right = snapshot.vectors[rows[atom_b]]
        target = tuple((a + b) * 0.5 for a, b in zip(left, right))
        learned_error = sum(
            (actual - expected) ** 2
            for actual, expected in zip(parent, target)
        )
        parent_position = snapshot.positions[rows[AGGREGATE_BASE]]
        left_position = snapshot.positions[rows[atom_a]]
        right_position = snapshot.positions[rows[atom_b]]
        positional_target = tuple(
            (a + b) * 0.5
            for a, b in zip(left_position, right_position)
        )
        initial_error = sum(
            (actual - expected) ** 2
            for actual, expected in zip(parent_position, positional_target)
        )
        self.assertLess(learned_error, initial_error)
        self.assertEqual(snapshot.optimizer_steps, 4)
        self.assertIsNotNone(snapshot.last_loss)
        self.assertGreater(
            max(abs(value) for row in snapshot.residuals for value in row),
            0.0,
        )
        self.assertGreater(
            max(
                abs(value)
                for token in (atom_a, atom_b)
                for value in snapshot.residuals[rows[token]]
            ),
            0.0,
        )
        self.assertFalse(snapshot.desynchronized)
        self.assertIsNone(snapshot.failure)

    def test_service_emits_advisory_deltas_without_owning_learner(self):
        learner = ContinuousEmbeddingLearner(dimension=4)
        service = TokenService(embedding_sink=learner)
        status, ticket = service.submit_intern(
            DOMAIN_TEXT, [97, 98, 99], [0, 3], block=True
        )
        self.assertEqual(status, STATUS_OK)
        receipt = ticket.wait(5.0)
        self.assertEqual(receipt[0], STATUS_OK)
        self.assertTrue(learner.wait_until_epoch(receipt[3], 5.0))
        service.close()
        snapshot = learner.snapshot()
        learner.close()
        self.assertIn(receipt[2][0], snapshot.tokens)
        self.assertEqual(snapshot.epoch, receipt[3])

    def test_sink_failure_cannot_fail_authoritative_interning(self):
        class BrokenSink:
            def offer_delta(self, *unused):
                raise RuntimeError("advisory failure")

            def mark_desynchronized(self):
                self.marked = True

        sink = BrokenSink()
        sink.marked = False
        service = TokenService(embedding_sink=sink)
        status, ticket = service.submit_intern(
            DOMAIN_TEXT, [120, 121], [0, 2], block=True
        )
        self.assertEqual(status, STATUS_OK)
        self.assertEqual(ticket.wait(5.0)[0], STATUS_OK)
        service.close()
        self.assertTrue(sink.marked)

    def test_epoch_gap_marks_advisory_state_desynchronized(self):
        learner = ContinuousEmbeddingLearner(dimension=4)
        self.assertTrue(
            learner.offer_delta(
                1, 2, [AGGREGATE_BASE], [3, 4], [0, 2]
            )
        )
        self.assertFalse(learner.wait_until_epoch(2, 5.0))
        snapshot = learner.snapshot()
        learner.close()
        self.assertTrue(snapshot.desynchronized)
        self.assertEqual(snapshot.epoch, 0)

    def test_position_depth_does_not_construct_or_train_network(self):
        learner = ContinuousEmbeddingLearner(
            dimension=4, max_depth=SEMANTIC_DEPTH_POSITION
        )
        self.assertTrue(
            learner.offer_delta(
                0, 1, [AGGREGATE_BASE], [3 + 97, 3 + 98], [0, 2]
            )
        )
        self.assertTrue(learner.wait_until_epoch(1, 5.0))
        snapshot = learner.snapshot()
        learner.close()
        self.assertEqual(snapshot.optimizer_steps, 0)
        self.assertTrue(
            all(value == 0.0 for row in snapshot.residuals for value in row)
        )
        self.assertEqual(snapshot.positions, snapshot.vectors)

    def test_live_observation_batch_trains_and_advances_cursor(self):
        learner = ContinuousEmbeddingLearner(
            dimension=5, learning_rate=0.01, steps_per_delta=2
        )
        batch = SemanticObservationBatch.from_columns(
            0,
            [OBSERVATION_ADJACENT, OBSERVATION_OPERATION],
            [AGGREGATE_BASE, AGGREGATE_BASE],
            [3 + 97, 3 + 98],
            [0, 3 + 99],
            [-1, 2],
            [1, 3],
            [0, 0],
            [OBSERVATION_WEIGHT_ONE, OBSERVATION_WEIGHT_ONE * 2],
        )
        self.assertTrue(learner.offer_observations(batch))
        self.assertTrue(learner.wait_until_observation(2, 5.0))
        snapshot = learner.snapshot()
        learner.close()
        self.assertEqual(snapshot.observation_cursor, 2)
        self.assertEqual(snapshot.optimizer_steps, 2)
        self.assertFalse(snapshot.desynchronized)
        self.assertIsNone(snapshot.failure)

    def test_observation_gap_is_explicit_desynchronization(self):
        learner = ContinuousEmbeddingLearner(dimension=4)
        batch = SemanticObservationBatch.from_columns(
            1,
            [OBSERVATION_ADJACENT],
            [AGGREGATE_BASE],
            [3 + 97],
            [0],
            [1],
            [1],
            [0],
            [OBSERVATION_WEIGHT_ONE],
        )
        self.assertTrue(learner.offer_observations(batch))
        self.assertFalse(learner.wait_until_observation(2, 5.0))
        snapshot = learner.snapshot()
        learner.close()
        self.assertTrue(snapshot.desynchronized)
        self.assertEqual(snapshot.observation_cursor, 0)


if __name__ == "__main__":
    unittest.main()
