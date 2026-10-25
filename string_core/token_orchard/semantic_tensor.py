"""AbstractTensor semantic preparation roots intended for SSA composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.common.tensors.abstraction import AbstractTensor

from .contract import OBSERVATION_FLAG_NEGATIVE, OBSERVATION_WEIGHT_ONE


def retention_stale_mask(
    last_notice_tick: AbstractTensor,
    protected: AbstractTensor,
    current_tick: int,
    idle_ticks: int,
) -> AbstractTensor:
    """Select reclaimable rows with one vectorized logical-time operation.

    The result is an integer mask. It is only an eligibility input to a
    generation rotation; it never invalidates tokens or mutates vocabulary
    state. Callers validate nonnegative monotonic ticks and positive idle
    windows at the integer ABI boundary.
    """
    stale = (last_notice_tick + idle_ticks) <= current_tick
    return stale * (1 - protected)


def embedding_position_batch(
    tokens: AbstractTensor,
    dimension: int,
) -> AbstractTensor:
    """Map token IDs to fixed unit-sphere positions as one tensor batch."""
    token_column = tokens.reshape((-1, 1))
    abstract_tensor = type(tokens)
    dimensions = abstract_tensor.arange(
        0, dimension, 1, dtype="int64"
    ).reshape((1, -1))
    mixed = (
        token_column * 1103515245
        + (dimensions + 1) * 12345
        + 1013904223
    ) % 2147483647
    mixed = (mixed * (dimensions * 2 + 3) + 69069) % 2147483647
    coordinates = mixed.to_dtype("float64") / 1073741823.5 - 1.0
    radius = (coordinates * coordinates).sum(dim=1, keepdim=True).sqrt()
    return coordinates / radius


def embedding_ordered_composition_batch(
    child_vectors: AbstractTensor,
    edge_parent_index: AbstractTensor,
    edge_ordinal: AbstractTensor,
    edge_parent_length: AbstractTensor,
    parent_count: int,
    dimension: int,
) -> AbstractTensor:
    """Reduce ordered COO cover edges to distinct parent targets."""
    abstract_tensor = type(child_vectors)
    dimensions = abstract_tensor.arange(
        0, dimension, 1, dtype="int64"
    ).reshape((1, -1))
    ordinal = edge_ordinal.reshape((-1, 1))
    cover_length = edge_parent_length.reshape((-1, 1))
    mixed = (
        (ordinal + 1) * (dimensions * 2 + 3)
        + (cover_length + 1) * (dimensions + 5)
        + 8191
    ) % 104729
    modulation = mixed.to_dtype("float64") / 104728.0 + 0.5
    ordered = child_vectors * modulation
    edge_count = child_vectors.shape[0]
    sums = abstract_tensor.scatter(
        abstract_tensor.zeros((parent_count, dimension), dtype="float64"),
        edge_parent_index,
        ordered,
        dim=0,
        reduce="sum",
    )
    counts = abstract_tensor.scatter(
        abstract_tensor.zeros((parent_count, 1), dtype="float64"),
        edge_parent_index,
        abstract_tensor.full((edge_count, 1), 1.0, dtype="float64"),
        dim=0,
        reduce="sum",
    )
    return sums / counts


def embedding_observation_target_batch(
    object_vectors: AbstractTensor,
    context_vectors: AbstractTensor,
    event_subject_index: AbstractTensor,
    kind: AbstractTensor,
    position: AbstractTensor,
    extent: AbstractTensor,
    flags: AbstractTensor,
    weight_q16: AbstractTensor,
    has_context: AbstractTensor,
    subject_count: int,
    dimension: int,
) -> AbstractTensor:
    """Reduce typed live observations to per-subject meaning targets."""
    abstract_tensor = type(object_vectors)
    kind_column = kind.reshape((-1, 1))
    position_column = position.reshape((-1, 1))
    extent_column = extent.reshape((-1, 1))
    flag_column = flags.reshape((-1, 1))
    weights = weight_q16.to_dtype("float64").reshape((-1, 1)) / float(
        OBSERVATION_WEIGHT_ONE
    )
    dimensions = abstract_tensor.arange(
        0, dimension, 1, dtype="int64"
    ).reshape((1, -1))
    mixed = (
        kind_column * (dimensions * 2 + 3)
        + (position_column + 104729) * (dimensions + 5)
        + (extent_column + 1) * (dimensions * 4 + 7)
        + 12289
    ) % 104729
    modulation = mixed.to_dtype("float64") / 104728.0 + 0.5
    negative = (
        flag_column % (OBSERVATION_FLAG_NEGATIVE * 2)
    ) == OBSERVATION_FLAG_NEGATIVE
    polarity = 1.0 - negative.to_dtype("float64") * 2.0
    event_targets = (
        object_vectors
        + context_vectors * has_context.reshape((-1, 1)) * 0.5
    ) * modulation * polarity * weights
    sums = abstract_tensor.scatter(
        abstract_tensor.zeros((subject_count, dimension), dtype="float64"),
        event_subject_index,
        event_targets,
        dim=0,
        reduce="sum",
    )
    weight_sums = abstract_tensor.scatter(
        abstract_tensor.zeros((subject_count, 1), dtype="float64"),
        event_subject_index,
        weights,
        dim=0,
        reduce="sum",
    )
    return sums / weight_sums
