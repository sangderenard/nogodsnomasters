"""Bounded single-writer queue service for integer Token Orchard batches."""

import queue
import threading

from .contract import (
    STATUS_OK,
    STATUS_QUEUE_FULL,
    STATUS_SERVICE_CLOSED,
    STATUS_SERVICE_FAILURE,
    STATUS_SERVICE_TIMEOUT,
)
from .core import Vocabulary
from .tensor_batch import preflight_batch


class RequestTicket:
    """One accepted request and its integer-only completion receipt."""

    def __init__(self, request_id):
        self.request_id = request_id
        self._event = threading.Event()
        self._batch_status = STATUS_SERVICE_FAILURE
        self._item_statuses = []
        self._tokens = []
        self._published_epoch = 0

    def _complete(self, batch_status, item_statuses, tokens, published_epoch):
        self._batch_status = batch_status
        self._item_statuses = item_statuses
        self._tokens = tokens
        self._published_epoch = published_epoch
        self._event.set()

    def wait(self, timeout=None):
        if not self._event.wait(timeout):
            return STATUS_SERVICE_TIMEOUT, [], [], 0
        return (
            self._batch_status,
            list(self._item_statuses),
            list(self._tokens),
            self._published_epoch,
        )


class TokenService:
    """Own a vocabulary on one writer thread and drain bounded request groups."""

    def __init__(
        self,
        vocabulary=None,
        queue_capacity=1024,
        drain_limit=64,
        fusion_atom_limit=262144,
        tensor_backend="numpy",
        embedding_sink=None,
        thread_daemon=False,
    ):
        if vocabulary is None:
            vocabulary = Vocabulary()
        if (
            queue_capacity <= 0
            or drain_limit <= 0
            or fusion_atom_limit <= 0
            or not isinstance(thread_daemon, bool)
        ):
            raise ValueError("invalid queue, fusion, or thread configuration")
        self._vocabulary = vocabulary
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._drain_limit = drain_limit
        self._fusion_atom_limit = fusion_atom_limit
        self._tensor_backend = tensor_backend
        self._embedding_sink = embedding_sink
        self._state_lock = threading.Lock()
        self._accepting = True
        self._next_request_id = 1
        self._sentinel = object()
        self._thread = threading.Thread(
            target=self._run,
            name="token-orchard-writer",
            daemon=bool(thread_daemon),
        )
        self._thread.start()

    def submit_intern(self, domain, values, offsets, block=False, timeout=None):
        """Submit one flat integer batch; return ``(status, ticket_or_none)``."""
        with self._state_lock:
            if not self._accepting:
                return STATUS_SERVICE_CLOSED, None
            request_id = self._next_request_id
            self._next_request_id += 1
            ticket = RequestTicket(request_id)
            request = (ticket, domain, list(values), list(offsets))
            try:
                if block:
                    self._queue.put(request, block=True, timeout=timeout)
                else:
                    self._queue.put_nowait(request)
            except queue.Full:
                return STATUS_QUEUE_FULL, None
            return STATUS_OK, ticket

    def close(self, wait=True):
        """Stop admission and finish every accepted request before exit."""
        should_signal = False
        with self._state_lock:
            if self._accepting:
                self._accepting = False
                should_signal = True
        if should_signal:
            self._queue.put(self._sentinel)
        if wait:
            self._thread.join()

    def is_alive(self):
        return self._thread.is_alive()

    def authoritative_state_after_close(self):
        """Return state only after shutdown, when mutation has ceased."""
        if self._thread.is_alive():
            return STATUS_SERVICE_FAILURE, ()
        return STATUS_OK, self._vocabulary.state_tuple()

    def _run(self):
        stop_after_group = False
        while True:
            first = self._queue.get()
            if first is self._sentinel:
                return
            group = [first]
            while len(group) < self._drain_limit:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is self._sentinel:
                    stop_after_group = True
                    break
                group.append(item)

            first_new_row = len(self._vocabulary.domain)
            completions = self._execute_group(group)

            if (
                self._embedding_sink is not None
                and len(self._vocabulary.domain) > first_new_row
            ):
                try:
                    delta = self._vocabulary.graph_delta_since_row(
                        first_new_row
                    )
                    self._embedding_sink.offer_delta(*delta)
                except Exception:
                    try:
                        self._embedding_sink.mark_desynchronized()
                    except Exception:
                        pass

            published_epoch = self._vocabulary.epoch
            for ticket, batch_status, item_statuses, tokens in completions:
                ticket._complete(
                    batch_status,
                    item_statuses,
                    tokens,
                    published_epoch,
                )
            if stop_after_group:
                return

    def _execute_one(self, request):
        ticket, domain, values, offsets = request
        try:
            batch_status, item_statuses, tokens = self._execute_tensor_batch(
                domain, values, offsets
            )
        except Exception:
            batch_status = STATUS_SERVICE_FAILURE
            item_statuses = []
            tokens = []
        return ticket, batch_status, item_statuses, tokens

    def _execute_tensor_batch(self, domain, values, offsets):
        envelope_status, unused_item_count = (
            self._vocabulary.validate_batch_envelope(domain, values, offsets)
        )
        if envelope_status != STATUS_OK:
            return envelope_status, [], []
        item_statuses, fingerprints = preflight_batch(
            domain,
            values,
            offsets,
            max_sequence_atoms=self._vocabulary.max_sequence_atoms,
            fingerprint_mode=self._vocabulary.fingerprint_mode,
            backend=self._tensor_backend,
        )
        return self._vocabulary.intern_preflighted_batch(
            domain, values, offsets, item_statuses, fingerprints
        )

    def _execute_group(self, group):
        """Greedily fuse compatible FIFO requests into flat core batches."""
        completions = []
        cursor = 0
        while cursor < len(group):
            first = group[cursor]
            unused_ticket, domain, values, offsets = first
            envelope_status, item_count = self._vocabulary.validate_batch_envelope(
                domain, values, offsets
            )
            if envelope_status != STATUS_OK:
                completions.append(self._execute_one(first))
                cursor += 1
                continue

            run = [first]
            run_item_counts = [item_count]
            atom_count = len(values)
            next_cursor = cursor + 1
            while next_cursor < len(group):
                candidate = group[next_cursor]
                unused_ticket, next_domain, next_values, next_offsets = candidate
                if next_domain != domain:
                    break
                next_status, next_items = self._vocabulary.validate_batch_envelope(
                    next_domain, next_values, next_offsets
                )
                if next_status != STATUS_OK:
                    break
                if item_count + next_items > self._vocabulary.max_batch_items:
                    break
                if atom_count + len(next_values) > self._fusion_atom_limit:
                    break
                run.append(candidate)
                run_item_counts.append(next_items)
                item_count += next_items
                atom_count += len(next_values)
                next_cursor += 1

            if len(run) == 1:
                completions.append(self._execute_one(first))
                cursor += 1
                continue

            # The run survey already proved exact output extents. Fill those
            # buffers once instead of repeatedly reallocating Python lists;
            # this is the same flat values+offsets ABI the native core sees.
            fused_values = [0] * atom_count
            fused_offsets = [0] * (item_count + 1)
            value_cursor = 0
            offset_cursor = 1
            for unused_ticket, unused_domain, part_values, part_offsets in run:
                base = value_cursor
                part_value_index = 0
                while part_value_index < len(part_values):
                    fused_values[value_cursor] = part_values[part_value_index]
                    value_cursor += 1
                    part_value_index += 1
                part_offset_index = 1
                while part_offset_index < len(part_offsets):
                    fused_offsets[offset_cursor] = (
                        base + part_offsets[part_offset_index]
                    )
                    offset_cursor += 1
                    part_offset_index += 1
            try:
                batch_status, statuses, tokens = self._execute_tensor_batch(
                    domain, fused_values, fused_offsets
                )
            except Exception:
                # Preserve request-level fault isolation if an unexpected core
                # error defeats a fused execution.
                completions.extend(self._execute_one(request) for request in run)
                cursor = next_cursor
                continue

            result_cursor = 0
            for request, result_count in zip(run, run_item_counts):
                ticket = request[0]
                result_end = result_cursor + result_count
                completions.append(
                    (
                        ticket,
                        batch_status,
                        statuses[result_cursor:result_end],
                        tokens[result_cursor:result_end],
                    )
                )
                result_cursor = result_end
            cursor = next_cursor
        return completions
