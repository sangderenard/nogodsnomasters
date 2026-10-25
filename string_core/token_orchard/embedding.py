"""Optional asynchronous AbstractTensor embeddings for evolving token graphs."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import random
import threading
import time

from .tensor_batch import _abstract_tensor_type
from .contract import (
    SEMANTIC_DEPTH_CONTEXT,
    SEMANTIC_DEPTH_POSITION,
    SEMANTIC_DEPTH_QUERY,
    SEMANTIC_DEPTH_TRAINING,
)
from .semantic import SemanticObservationBatch
from .semantic_tensor import (
    embedding_observation_target_batch,
    embedding_ordered_composition_batch,
    embedding_position_batch,
)


EMBEDDING_SCHEMA_VERSION = 3
EMBEDDING_POLICY_SPHERICAL_ABSTRACT_NN_ORDERED_CONTEXT = 4


_ABSTRACT_NN_TAPE_LOCK = threading.RLock()


def _abstract_nn_types():
    """Load Turing's established network, optimizer, and tape mechanisms."""
    abstract_tensor = _abstract_tensor_type()
    from src.common.tensors.abstract_nn import Adam, Identity, Linear, Model, Tanh
    from src.common.tensors.autograd import GradTape, autograd

    return abstract_tensor, Adam, Identity, Linear, Model, Tanh, GradTape, autograd


@dataclass(frozen=True)
class EmbeddingSnapshot:
    """Immutable advisory row-map receipt; never token identity authority."""

    schema_version: int
    policy: int
    epoch: int
    dimension: int
    hidden_dimension: int
    max_depth: int
    tokens: tuple[int, ...]
    positions: tuple[tuple[float, ...], ...]
    residuals: tuple[tuple[float, ...], ...]
    vectors: tuple[tuple[float, ...], ...]
    optimizer_steps: int
    observation_cursor: int
    last_loss: float | None
    desynchronized: bool
    failure: str | None


class ContinuousEmbeddingLearner:
    """Train a shared spherical-position adapter off the token writer.

    Cover targets use AbstractTensor gather/scatter batches. Turing's ordinary
    AbstractNN and Adam train the shared residual map, and one batched forward
    refreshes every resident meaning row after an optimizer step. Python
    manages queueing and the categorical token-to-row topology only.
    A learner begins at epoch zero; bootstrap an existing vocabulary by
    offering ``vocabulary.graph_delta_since_row(0)`` before attaching it.
    """

    def __init__(
        self,
        *,
        dimension=16,
        hidden_dimension=None,
        learning_rate=0.006,
        steps_per_delta=1,
        queue_capacity=64,
        tensor_backend="numpy",
        network_seed=1729,
        max_depth=SEMANTIC_DEPTH_TRAINING,
    ):
        if hidden_dimension is None:
            hidden_dimension = dimension * 2
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or not isinstance(hidden_dimension, int)
            or isinstance(hidden_dimension, bool)
            or hidden_dimension <= 0
            or not isinstance(steps_per_delta, int)
            or isinstance(steps_per_delta, bool)
            or steps_per_delta <= 0
            or not isinstance(queue_capacity, int)
            or isinstance(queue_capacity, bool)
            or queue_capacity <= 0
            or not 0.0 < float(learning_rate) <= 1.0
            or not isinstance(network_seed, int)
            or isinstance(network_seed, bool)
            or not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or max_depth < SEMANTIC_DEPTH_POSITION
            or max_depth > SEMANTIC_DEPTH_QUERY
        ):
            raise ValueError("invalid embedding learner configuration")
        self.dimension = dimension
        self.hidden_dimension = hidden_dimension
        self.learning_rate = float(learning_rate)
        self.steps_per_delta = steps_per_delta
        self.tensor_backend = tensor_backend
        self.network_seed = network_seed
        self.max_depth = max_depth
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._sentinel = object()
        self._admission_lock = threading.Lock()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._epoch = 0
        self._tokens = []
        self._rows = {}
        self._positions = None
        self._residuals = None
        self._vectors = None
        self._model = None
        self._optimizer = None
        self._optimizer_steps = 0
        self._observation_cursor = 0
        self._last_loss = None
        self._desynchronized = False
        self._failure = None
        self._accepting = True
        self._thread = threading.Thread(
            target=self._run,
            name="token-orchard-embedding",
            daemon=False,
        )
        self._thread.start()

    def offer_delta(
        self,
        begin_epoch,
        end_epoch,
        parent_tokens,
        cover_values,
        cover_offsets,
    ):
        """Offer one ordered graph delta without blocking the token writer."""
        parents = tuple(parent_tokens)
        covers = tuple(cover_values)
        offsets = tuple(cover_offsets)
        if (
            end_epoch - begin_epoch != len(parents)
            or len(offsets) != len(parents) + 1
            or not offsets
            or offsets[0] != 0
            or offsets[-1] != len(covers)
            or any(offsets[i] >= offsets[i + 1] for i in range(len(parents)))
            or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or token <= 0
                for token in parents + covers
            )
        ):
            raise ValueError("invalid embedding graph delta")
        with self._admission_lock:
            if not self._accepting or self._desynchronized or self._failure:
                return False
            try:
                self._queue.put_nowait(
                    ("graph", begin_epoch, end_epoch, parents, covers, offsets)
                )
                return True
            except queue.Full:
                # A plain flag write keeps this path nonblocking even while
                # the worker owns the snapshot lock for a tensor update.
                self._desynchronized = True
                return False

    def offer_observations(self, batch):
        """Offer integer live-program facts without blocking their producer."""
        if self.max_depth < SEMANTIC_DEPTH_CONTEXT:
            return False
        if not isinstance(batch, SemanticObservationBatch):
            raise TypeError("observations require SemanticObservationBatch")
        batch.validate()
        with self._admission_lock:
            if not self._accepting or self._desynchronized or self._failure:
                return False
            try:
                self._queue.put_nowait(("observation", batch))
                return True
            except queue.Full:
                self._desynchronized = True
                return False

    def mark_desynchronized(self):
        self._desynchronized = True
        if self._condition.acquire(blocking=False):
            try:
                self._condition.notify_all()
            finally:
                self._condition.release()

    def wait_until_epoch(self, epoch, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (
                self._epoch < epoch
                and not self._desynchronized
                and self._failure is None
            ):
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return (
                self._epoch >= epoch
                and not self._desynchronized
                and self._failure is None
            )

    def wait_until_observation(self, sequence, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (
                self._observation_cursor < sequence
                and not self._desynchronized
                and self._failure is None
            ):
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return (
                self._observation_cursor >= sequence
                and not self._desynchronized
                and self._failure is None
            )

    def snapshot(self):
        with self._lock:
            positions = () if self._positions is None else tuple(
                tuple(float(value) for value in row)
                for row in self._positions.tolist()
            )
            residuals = () if self._residuals is None else tuple(
                tuple(float(value) for value in row)
                for row in self._residuals.tolist()
            )
            vectors = () if self._vectors is None else tuple(
                tuple(float(value) for value in row)
                for row in self._vectors.tolist()
            )
            return EmbeddingSnapshot(
                EMBEDDING_SCHEMA_VERSION,
                EMBEDDING_POLICY_SPHERICAL_ABSTRACT_NN_ORDERED_CONTEXT,
                self._epoch,
                self.dimension,
                self.hidden_dimension,
                self.max_depth,
                tuple(self._tokens),
                positions,
                residuals,
                vectors,
                self._optimizer_steps,
                self._observation_cursor,
                self._last_loss,
                self._desynchronized,
                self._failure,
            )

    def close(self, wait=True):
        with self._admission_lock:
            if not self._accepting:
                should_signal = False
            else:
                self._accepting = False
                should_signal = True
        if should_signal:
            self._queue.put(self._sentinel)
        if wait:
            self._thread.join()

    def _spherical_positions(self, tokens, abstract_tensor):
        """Map token identities to fixed, deterministic points on S^(D-1)."""
        token_tensor = abstract_tensor.tensor(tokens, dtype="int64")
        return embedding_position_batch(token_tensor, self.dimension)

    def _ensure_rows(self, tokens, abstract_tensor):
        new_tokens = []
        for token in tokens:
            if token not in self._rows:
                self._rows[token] = len(self._tokens) + len(new_tokens)
                new_tokens.append(token)
        if not new_tokens:
            return
        new_positions = self._spherical_positions(new_tokens, abstract_tensor)
        self._tokens.extend(new_tokens)
        self._positions = (
            new_positions
            if self._positions is None
            else abstract_tensor.concat(
                (self._positions, new_positions), dim=0
            )
        )
        if self.max_depth >= SEMANTIC_DEPTH_TRAINING:
            self._ensure_model(new_positions)
            self._refresh_vectors()
        else:
            new_residuals = new_positions * 0.0
            self._residuals = (
                new_residuals
                if self._residuals is None
                else abstract_tensor.concat(
                    (self._residuals, new_residuals), dim=0
                )
            )
            self._vectors = self._positions

    def _ensure_model(self, like):
        if self._model is not None:
            return
        (
            abstract_tensor,
            Adam,
            Identity,
            Linear,
            Model,
            Tanh,
            GradTape,
            autograd,
        ) = _abstract_nn_types()
        with _ABSTRACT_NN_TAPE_LOCK:
            previous_tape = autograd.tape
            random_state = random.getstate()
            try:
                autograd.tape = GradTape()
                random.seed(self.network_seed)
                first = Linear(
                    self.dimension,
                    self.hidden_dimension,
                    like=like,
                    init="xavier",
                )
                second = Linear(
                    self.hidden_dimension,
                    self.dimension,
                    like=like,
                    init="xavier",
                )
                with autograd.no_grad():
                    abstract_tensor.copyto(second.W, second.W * 0.0)
                    abstract_tensor.copyto(second.b, second.b * 0.0)
                self._model = Model(
                    layers=[first, second],
                    activations=[Tanh(), Identity()],
                )
                self._optimizer = Adam(
                    self._model.parameters(), lr=self.learning_rate
                )
            finally:
                random.setstate(random_state)
                autograd.tape = previous_tape

    def _refresh_vectors(self):
        if self._model is None or self._positions is None:
            return
        (
            unused_at,
            unused_adam,
            unused_identity,
            unused_linear,
            unused_model,
            unused_tanh,
            GradTape,
            autograd,
        ) = _abstract_nn_types()
        with _ABSTRACT_NN_TAPE_LOCK:
            previous_tape = autograd.tape
            try:
                autograd.tape = GradTape()
                with autograd.no_grad():
                    self._residuals = self._model.forward(self._positions)
                    self._vectors = self._positions + self._residuals
            finally:
                autograd.tape = previous_tape

    def _cover_targets(
        self,
        abstract_tensor,
        parent_index,
        child_index,
        edge_parent_index,
        edge_ordinal,
        edge_parent_length,
        parent_count,
        edge_count,
    ):
        child_vectors = abstract_tensor.gather(
            self._vectors, child_index, dim=0
        )
        meanings = embedding_ordered_composition_batch(
            child_vectors,
            edge_parent_index,
            edge_ordinal,
            edge_parent_length,
            parent_count,
            self.dimension,
        )
        positions = abstract_tensor.gather(
            self._positions, parent_index, dim=0
        )
        return positions, meanings - positions

    def _fit_targets(self, abstract_tensor, positions, target_residuals):
        (
            unused_at,
            unused_adam,
            unused_identity,
            unused_linear,
            unused_model,
            unused_tanh,
            GradTape,
            autograd,
        ) = _abstract_nn_types()
        with _ABSTRACT_NN_TAPE_LOCK:
            previous_tape = autograd.tape
            try:
                step = 0
                while step < self.steps_per_delta:
                    autograd.tape = GradTape()
                    self._model.zero_grad()
                    prediction = self._model.forward(positions)
                    error = prediction - target_residuals
                    loss = (error * error).mean()
                    parameters = self._model.parameters()
                    gradients = list(
                        autograd.grad(
                            loss,
                            parameters,
                            retain_graph=False,
                            allow_unused=False,
                        )
                    )
                    updated = self._optimizer.step(parameters, gradients)
                    for parameter, new_parameter in zip(parameters, updated):
                        abstract_tensor.copyto(parameter, new_parameter)
                    self._optimizer_steps += 1
                    self._last_loss = float(loss.item())
                    self._refresh_vectors()
                    step += 1
            finally:
                autograd.tape = previous_tape

    def _train_network(
        self,
        abstract_tensor,
        parent_index,
        child_index,
        edge_parent_index,
        edge_ordinal,
        edge_parent_length,
        parent_count,
        edge_count,
    ):
        positions, target_residuals = self._cover_targets(
            abstract_tensor,
            parent_index,
            child_index,
            edge_parent_index,
            edge_ordinal,
            edge_parent_length,
            parent_count,
            edge_count,
        )
        self._fit_targets(abstract_tensor, positions, target_residuals)

    def _apply_delta(self, delta):
        unused_tag, begin_epoch, end_epoch, parents, covers, offsets = delta
        with self._condition:
            if self._failure is not None:
                return
            if begin_epoch != self._epoch:
                self._desynchronized = True
                self._condition.notify_all()
                return
            abstract_tensor = _abstract_tensor_type()
            device = "cpu" if self.tensor_backend == "numpy" else None
            with abstract_tensor.use_backend(self.tensor_backend, device):
                self._ensure_rows(covers + parents, abstract_tensor)
                parent_rows = [self._rows[token] for token in parents]
                child_rows = [self._rows[token] for token in covers]
                edge_parents = [0] * len(covers)
                edge_ordinals = [0] * len(covers)
                edge_parent_lengths = [0] * len(covers)
                parent = 0
                while parent < len(parents):
                    edge = offsets[parent]
                    cover_length = offsets[parent + 1] - offsets[parent]
                    while edge < offsets[parent + 1]:
                        edge_parents[edge] = parent
                        edge_ordinals[edge] = edge - offsets[parent]
                        edge_parent_lengths[edge] = cover_length
                        edge += 1
                    parent += 1

                parent_index = abstract_tensor.tensor(
                    parent_rows, dtype="int64"
                )
                child_index = abstract_tensor.tensor(
                    child_rows, dtype="int64"
                )
                edge_parent_index = abstract_tensor.tensor(
                    edge_parents, dtype="int64"
                )
                if self.max_depth >= SEMANTIC_DEPTH_TRAINING:
                    edge_ordinal = abstract_tensor.tensor(
                        edge_ordinals, dtype="int64"
                    )
                    edge_parent_length = abstract_tensor.tensor(
                        edge_parent_lengths, dtype="int64"
                    )
                    self._train_network(
                        abstract_tensor,
                        parent_index,
                        child_index,
                        edge_parent_index,
                        edge_ordinal,
                        edge_parent_length,
                        len(parents),
                        len(covers),
                    )
            self._epoch = end_epoch
            self._condition.notify_all()

    def _apply_observations(self, batch):
        with self._condition:
            if self._failure is not None:
                return
            if batch.begin_sequence != self._observation_cursor:
                self._desynchronized = True
                self._condition.notify_all()
                return
            abstract_tensor = _abstract_tensor_type()
            device = "cpu" if self.tensor_backend == "numpy" else None
            with abstract_tensor.use_backend(self.tensor_backend, device):
                resident = list(batch.subject) + list(batch.object)
                resident.extend(token for token in batch.context if token)
                self._ensure_rows(resident, abstract_tensor)

                unique_subjects = []
                subject_slots = {}
                event_subject = []
                for token in batch.subject:
                    if token not in subject_slots:
                        subject_slots[token] = len(unique_subjects)
                        unique_subjects.append(token)
                    event_subject.append(subject_slots[token])

                subject_rows = abstract_tensor.tensor(
                    [self._rows[token] for token in unique_subjects],
                    dtype="int64",
                )
                object_rows = abstract_tensor.tensor(
                    [self._rows[token] for token in batch.object],
                    dtype="int64",
                )
                context_rows = abstract_tensor.tensor(
                    [
                        self._rows[context] if context else self._rows[obj]
                        for context, obj in zip(batch.context, batch.object)
                    ],
                    dtype="int64",
                )
                event_subject_index = abstract_tensor.tensor(
                    event_subject, dtype="int64"
                )
                kind = abstract_tensor.tensor(batch.kind, dtype="int64")
                position = abstract_tensor.tensor(batch.position, dtype="int64")
                extent = abstract_tensor.tensor(batch.extent, dtype="int64")
                flags = abstract_tensor.tensor(batch.flags, dtype="int64")
                weights = abstract_tensor.tensor(
                    batch.weight_q16, dtype="int64"
                )
                object_vectors = abstract_tensor.gather(
                    self._vectors, object_rows, dim=0
                )
                context_vectors = abstract_tensor.gather(
                    self._vectors, context_rows, dim=0
                )
                has_context = abstract_tensor.tensor(
                    [1 if token else 0 for token in batch.context],
                    dtype="float64",
                )
                target_meanings = embedding_observation_target_batch(
                    object_vectors,
                    context_vectors,
                    event_subject_index,
                    kind,
                    position,
                    extent,
                    flags,
                    weights,
                    has_context,
                    len(unique_subjects),
                    self.dimension,
                )
                subject_positions = abstract_tensor.gather(
                    self._positions, subject_rows, dim=0
                )
                if self.max_depth >= SEMANTIC_DEPTH_TRAINING:
                    self._fit_targets(
                        abstract_tensor,
                        subject_positions,
                        target_meanings - subject_positions,
                    )
            self._observation_cursor = batch.end_sequence
            self._condition.notify_all()

    def _run(self):
        while True:
            work = self._queue.get()
            if work is self._sentinel:
                return
            try:
                if work[0] == "graph":
                    self._apply_delta(work)
                else:
                    self._apply_observations(work[1])
            except Exception as error:
                with self._condition:
                    self._failure = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()
