"""Integer observation ABI and optional semantic SSA product manifest."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import (
    OBSERVATION_FLAGS_SUPPORTED,
    OBSERVATION_KIND_LIMIT,
    OBSERVATION_WEIGHT_MAX,
    SEMANTIC_ABI_VERSION,
    SEMANTIC_DEPTH_COMPOSITION,
    SEMANTIC_DEPTH_CONTEXT,
    SEMANTIC_DEPTH_NONE,
    SEMANTIC_DEPTH_POSITION,
    SEMANTIC_DEPTH_PUBLICATION,
    SEMANTIC_DEPTH_QUERY,
    SEMANTIC_DEPTH_TRAINING,
)


_SEMANTIC_ENTRYPOINTS = (
    "embedding_position_batch",
    "embedding_ordered_composition_batch",
    "embedding_observation_target_batch",
    "embedding_train_step",
    "embedding_publish_epoch",
    "embedding_query_topk_batch",
)

IMPLEMENTED_SEMANTIC_DEPTH = SEMANTIC_DEPTH_CONTEXT


def semantic_entrypoints(max_depth):
    """Return currently implemented compilation roots through context depth.

    Depth is a build choice, not a runtime flag.  This prefix law lets Turing
    pursue only reachable preparation, training, publication, or query code.
    """
    if (
        not isinstance(max_depth, int)
        or isinstance(max_depth, bool)
        or max_depth < SEMANTIC_DEPTH_NONE
        or max_depth > SEMANTIC_DEPTH_QUERY
    ):
        raise ValueError("invalid semantic depth")
    if max_depth > IMPLEMENTED_SEMANTIC_DEPTH:
        raise NotImplementedError(
            "native semantic roots above context depth are not implemented"
        )
    return _SEMANTIC_ENTRYPOINTS[:max_depth]


def semantic_phase_roots(max_depth):
    """Return reserved root names for the complete version-zero depth ABI."""
    if (
        not isinstance(max_depth, int)
        or isinstance(max_depth, bool)
        or max_depth < SEMANTIC_DEPTH_NONE
        or max_depth > SEMANTIC_DEPTH_QUERY
    ):
        raise ValueError("invalid semantic depth")
    return _SEMANTIC_ENTRYPOINTS[:max_depth]


@dataclass(frozen=True)
class SemanticObservationBatch:
    """One contiguous, integer-only batch of live-program graph facts.

    ``subject`` is the token whose usage state is being learned. ``object`` is
    the related token. ``context`` is an optional token for scope, program,
    operation, type, or outcome namespace (zero means absent). ``position``
    is a signed ordinal or source/runtime displacement. ``extent`` is a
    nonnegative width/count. Producers assign contiguous sequence numbers so
    loss and replay are observable.
    """

    abi_version: int
    begin_sequence: int
    kind: tuple[int, ...]
    subject: tuple[int, ...]
    object: tuple[int, ...]
    context: tuple[int, ...]
    position: tuple[int, ...]
    extent: tuple[int, ...]
    flags: tuple[int, ...]
    weight_q16: tuple[int, ...]

    @property
    def count(self):
        return len(self.kind)

    @property
    def end_sequence(self):
        return self.begin_sequence + self.count

    @classmethod
    def from_columns(
        cls,
        begin_sequence,
        kind,
        subject,
        object,
        context,
        position,
        extent,
        flags,
        weight_q16,
    ):
        batch = cls(
            SEMANTIC_ABI_VERSION,
            begin_sequence,
            tuple(kind),
            tuple(subject),
            tuple(object),
            tuple(context),
            tuple(position),
            tuple(extent),
            tuple(flags),
            tuple(weight_q16),
        )
        batch.validate()
        return batch

    def validate(self):
        if self.abi_version != SEMANTIC_ABI_VERSION:
            raise ValueError("unsupported semantic observation ABI")
        if (
            not isinstance(self.begin_sequence, int)
            or isinstance(self.begin_sequence, bool)
            or self.begin_sequence < 0
        ):
            raise ValueError("invalid observation sequence")
        columns = (
            self.kind,
            self.subject,
            self.object,
            self.context,
            self.position,
            self.extent,
            self.flags,
            self.weight_q16,
        )
        count = len(self.kind)
        if count == 0 or any(len(column) != count for column in columns):
            raise ValueError("observation columns must be nonempty and aligned")
        for row in range(count):
            values = tuple(column[row] for column in columns)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise ValueError("observation values must be integers")
            if not 0 < self.kind[row] < OBSERVATION_KIND_LIMIT:
                raise ValueError("invalid observation kind")
            if self.subject[row] <= 0 or self.object[row] <= 0 or self.context[row] < 0:
                raise ValueError("invalid observation token")
            if self.extent[row] < 0:
                raise ValueError("invalid observation extent")
            if (
                self.flags[row] < 0
                or self.flags[row] & ~OBSERVATION_FLAGS_SUPPORTED
            ):
                raise ValueError("invalid observation flags")
            if not 0 < self.weight_q16[row] <= OBSERVATION_WEIGHT_MAX:
                raise ValueError("invalid observation weight")
        return self


def semantic_depth_laws():
    """Expose named depth constants without making them dynamic dispatch."""
    return (
        SEMANTIC_DEPTH_NONE,
        SEMANTIC_DEPTH_POSITION,
        SEMANTIC_DEPTH_COMPOSITION,
        SEMANTIC_DEPTH_CONTEXT,
        SEMANTIC_DEPTH_TRAINING,
        SEMANTIC_DEPTH_PUBLICATION,
        SEMANTIC_DEPTH_QUERY,
    )


_SEMANTIC_STATE_FIELDS = (
    (
        ("token_rows", "int64", 1, "categorical input"),
        ("positions", "float64", 2, "fixed state"),
    ),
    (
        ("cover_values", "int64", 1, "COO input"),
        ("cover_parent", "int64", 1, "COO input"),
        ("cover_ordinal", "int64", 1, "COO input"),
        ("cover_length", "int64", 1, "COO input"),
    ),
    (
        ("observation_columns", "int64", 2, "integer event input"),
        ("observation_cursor", "int64", 0, "receipt state"),
    ),
    (
        ("model_parameters", "float64", 1, "mutable training state"),
        ("adam_first_moment", "float64", 1, "mutable training state"),
        ("adam_second_moment", "float64", 1, "mutable training state"),
        ("optimizer_step", "int64", 0, "receipt state"),
    ),
    (
        ("published_vectors", "float64", 2, "immutable epoch state"),
        ("published_embedding_epoch", "int64", 0, "receipt state"),
    ),
    (
        ("query_token_rows", "int64", 1, "query input"),
        ("query_k", "int64", 0, "query input"),
        ("result_token_rows", "int64", 2, "query output"),
    ),
)

_ABSTRACT_TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


def _span_value(function, parameter, dtype, rank):
    return {
        "function": function,
        "parameter": parameter,
        "storage": "span",
        "dtype": dtype,
        "rank": rank,
        "python_type": _ABSTRACT_TENSOR_TYPE,
    }


def _scalar_value(function, parameter):
    return {
        "function": function,
        "parameter": parameter,
        "storage": "scalar",
        "dtype": "int64",
        "rank": 0,
        "shape": [],
        "python_type": "builtins.int",
    }


_SEMANTIC_PROGRAM_VALUES = (
    (
        _span_value("embedding_position_batch", "tokens", "int64", 1),
        _scalar_value("embedding_position_batch", "dimension"),
    ),
    (
        _span_value(
            "embedding_ordered_composition_batch", "child_vectors", "float64", 2
        ),
        _span_value(
            "embedding_ordered_composition_batch", "edge_parent_index", "int64", 1
        ),
        _span_value(
            "embedding_ordered_composition_batch", "edge_ordinal", "int64", 1
        ),
        _span_value(
            "embedding_ordered_composition_batch", "edge_parent_length", "int64", 1
        ),
        _scalar_value("embedding_ordered_composition_batch", "parent_count"),
        _scalar_value("embedding_ordered_composition_batch", "dimension"),
    ),
    (
        _span_value(
            "embedding_observation_target_batch", "object_vectors", "float64", 2
        ),
        _span_value(
            "embedding_observation_target_batch", "context_vectors", "float64", 2
        ),
        _span_value(
            "embedding_observation_target_batch", "event_subject_index", "int64", 1
        ),
        _span_value("embedding_observation_target_batch", "kind", "int64", 1),
        _span_value("embedding_observation_target_batch", "position", "int64", 1),
        _span_value("embedding_observation_target_batch", "extent", "int64", 1),
        _span_value("embedding_observation_target_batch", "flags", "int64", 1),
        _span_value(
            "embedding_observation_target_batch", "weight_q16", "int64", 1
        ),
        _span_value(
            "embedding_observation_target_batch", "has_context", "float64", 1
        ),
        _scalar_value("embedding_observation_target_batch", "subject_count"),
        _scalar_value("embedding_observation_target_batch", "dimension"),
    ),
)


def semantic_program_abi(max_depth):
    """Return Turing ProgramABI values for implemented preparation roots."""
    semantic_entrypoints(max_depth)
    values = []
    for phase_values in _SEMANTIC_PROGRAM_VALUES[:max_depth]:
        values.extend(dict(value) for value in phase_values)
    return {
        "records": {},
        "bindings": [],
        "values": values,
    }


def semantic_state_abi(max_depth, dimension):
    """Describe state retained by a compiled product at ``max_depth``.

    This is build metadata, not a runtime tagged union.  Fields are appended
    only when their phase is linked, making absence of Adam or query state an
    inspectable contract rather than dead-code-elimination folklore.
    """
    roots = semantic_phase_roots(max_depth)
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
    ):
        raise ValueError("invalid semantic dimension")
    fields = []
    for phase_fields in _SEMANTIC_STATE_FIELDS[:max_depth]:
        fields.extend(phase_fields)
    return {
        "abi_version": SEMANTIC_ABI_VERSION,
        "max_depth": max_depth,
        "dimension": dimension,
        "entrypoints": roots,
        "fields": tuple(fields),
    }
