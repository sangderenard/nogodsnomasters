import threading
import unittest

from token_orchard.contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    STATUS_OK,
    STATUS_QUEUE_FULL,
    STATUS_SERVICE_CLOSED,
    STATUS_SERVICE_FAILURE,
)
from token_orchard.core import Vocabulary
from token_orchard.service import TokenService


class TokenServiceTests(unittest.TestCase):
    def test_many_producers_receive_one_identity_for_same_sequence(self):
        service = TokenService(queue_capacity=64, drain_limit=32)
        tickets = []
        submission_statuses = []
        tickets_lock = threading.Lock()

        def producer():
            status, ticket = service.submit_intern(
                DOMAIN_TEXT, [97, 98, 99], [0, 3], block=True
            )
            with tickets_lock:
                submission_statuses.append(status)
                tickets.append(ticket)

        threads = [threading.Thread(target=producer) for unused in range(32)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        receipts = [ticket.wait(2.0) for ticket in tickets]
        service.close()
        self.assertEqual(submission_statuses, [STATUS_OK] * 32)
        tokens = [receipt[2][0] for receipt in receipts]
        self.assertEqual(len(set(tokens)), 1)
        self.assertTrue(all(receipt[0] == STATUS_OK for receipt in receipts))
        self.assertFalse(service.is_alive())

    def test_grouped_requests_observe_fifo_prior_tokens(self):
        service = TokenService(queue_capacity=8, drain_limit=8)
        first_status, first = service.submit_intern(
            DOMAIN_TEXT, [97, 98], [0, 2], block=True
        )
        second_status, second = service.submit_intern(
            DOMAIN_TEXT, [97, 98, 99], [0, 3], block=True
        )
        self.assertEqual((first_status, second_status), (STATUS_OK, STATUS_OK))
        first_receipt = first.wait(2.0)
        second_receipt = second.wait(2.0)
        self.assertEqual(first_receipt[0], STATUS_OK)
        self.assertEqual(second_receipt[0], STATUS_OK)
        self.assertGreaterEqual(second_receipt[3], first_receipt[3])
        service.close()

    def test_queued_same_domain_requests_are_fused_without_losing_boundaries(self):
        started = threading.Event()
        release = threading.Event()

        class CountingVocabulary(Vocabulary):
            def __init__(self):
                super().__init__()
                self.batch_item_counts = []

            def intern_preflighted_batch(
                self, domain, values, offsets, item_statuses, fingerprints
            ):
                self.batch_item_counts.append(len(offsets) - 1)
                if len(self.batch_item_counts) == 1:
                    started.set()
                    release.wait(2.0)
                return super().intern_preflighted_batch(
                    domain, values, offsets, item_statuses, fingerprints
                )

        vocabulary = CountingVocabulary()
        service = TokenService(
            vocabulary=vocabulary,
            queue_capacity=8,
            drain_limit=8,
            fusion_atom_limit=64,
        )
        status, first = service.submit_intern(
            DOMAIN_TEXT, [10, 11], [0, 2], block=True
        )
        self.assertEqual(status, STATUS_OK)
        self.assertTrue(started.wait(10.0))

        queued = []
        requests = (
            ([20, 21], [0, 2]),
            ([30, 31, 32], [0, 1, 3]),
            ([40, 41], [0, 2]),
        )
        for values, offsets in requests:
            status, ticket = service.submit_intern(
                DOMAIN_TEXT, values, offsets, block=True
            )
            self.assertEqual(status, STATUS_OK)
            queued.append(ticket)
        release.set()

        self.assertEqual(first.wait(2.0)[0], STATUS_OK)
        receipts = [ticket.wait(2.0) for ticket in queued]
        service.close()
        self.assertEqual(vocabulary.batch_item_counts, [1, 4])
        self.assertEqual([len(receipt[2]) for receipt in receipts], [1, 2, 1])
        self.assertTrue(
            all(status == STATUS_OK for receipt in receipts for status in receipt[1])
        )

    def test_clean_shutdown_accounts_for_accepted_requests_and_rejects_new(self):
        service = TokenService(queue_capacity=16, drain_limit=4)
        tickets = []
        for value in range(12):
            status, ticket = service.submit_intern(
                DOMAIN_BYTES, [value, value], [0, 2], block=True
            )
            self.assertEqual(status, STATUS_OK)
            tickets.append(ticket)
        service.close(wait=True)
        for ticket in tickets:
            receipt = ticket.wait(0.0)
            self.assertEqual(receipt[0], STATUS_OK)
            self.assertEqual(receipt[1], [STATUS_OK])
        status, ticket = service.submit_intern(DOMAIN_BYTES, [1], [0, 1])
        self.assertEqual(status, STATUS_SERVICE_CLOSED)
        self.assertIsNone(ticket)
        state_status, state = service.authoritative_state_after_close()
        self.assertEqual(state_status, STATUS_OK)
        self.assertNotEqual(state, ())

    def test_bounded_queue_reports_backpressure(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingVocabulary(Vocabulary):
            def intern_preflighted_batch(
                self, domain, values, offsets, item_statuses, fingerprints
            ):
                started.set()
                release.wait(2.0)
                return super().intern_preflighted_batch(
                    domain, values, offsets, item_statuses, fingerprints
                )

        service = TokenService(
            vocabulary=BlockingVocabulary(), queue_capacity=1, drain_limit=1
        )
        first_status, first = service.submit_intern(
            DOMAIN_BYTES, [1, 1], [0, 2], block=True
        )
        self.assertEqual(first_status, STATUS_OK)
        self.assertTrue(started.wait(10.0))
        second_status, second = service.submit_intern(
            DOMAIN_BYTES, [2, 2], [0, 2]
        )
        third_status, third = service.submit_intern(
            DOMAIN_BYTES, [3, 3], [0, 2]
        )
        self.assertEqual(second_status, STATUS_OK)
        self.assertEqual(third_status, STATUS_QUEUE_FULL)
        self.assertIsNone(third)
        release.set()
        self.assertEqual(first.wait(2.0)[0], STATUS_OK)
        self.assertEqual(second.wait(2.0)[0], STATUS_OK)
        service.close()

    def test_live_service_does_not_expose_mutable_authoritative_state(self):
        service = TokenService()
        status, state = service.authoritative_state_after_close()
        self.assertEqual(status, STATUS_SERVICE_FAILURE)
        self.assertEqual(state, ())
        service.close()


if __name__ == "__main__":
    unittest.main()
