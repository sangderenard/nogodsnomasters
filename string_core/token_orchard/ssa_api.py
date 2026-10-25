"""Integer-only AbstractTensor laws exported at Token Orchard's SSA boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Turing reads this authored annotation as the tensor-span contract while
    # normal Python avoids importing compiler/runtime machinery at execution.
    from src.common.tensors.abstraction import AbstractTensor

from token_orchard.contract import (
    AGGREGATE_BASE,
    BYTE_ATOM_LIMIT,
    BYTES_EMPTY_TOKEN,
    TEXT_ATOM_BASE,
    TEXT_EMPTY_TOKEN,
    TOKEN_KIND_AGGREGATE,
    TOKEN_KIND_ATOM,
    TOKEN_KIND_EMPTY,
    SSA_API_VERSION,
)


def token_kind_batch(
    tokens: AbstractTensor,
    aggregate_count: int,
) -> AbstractTensor:
    """Classify a runtime-sized token span with branchless tensor arithmetic.

    ``tokens`` is an AbstractTensor at development time and a rank-one signed
    integer span at the native ABI. Invalid tokens naturally retain kind zero.
    """
    aggregate_rows = tokens - AGGREGATE_BASE
    # The equality cases are disjoint, so addition is logical OR. Products
    # are logical AND over zero/one comparison masks. Keeping the law in the
    # ordinary integer arithmetic vocabulary makes the exact same source a
    # Python AbstractTensor program and a C/LLVM-friendly SSA program.
    empty = (tokens == TEXT_EMPTY_TOKEN) + (tokens == BYTES_EMPTY_TOKEN)
    atom = (tokens >= TEXT_ATOM_BASE) * (tokens < BYTE_ATOM_LIMIT)
    aggregate = (aggregate_rows >= 0) * (
        aggregate_rows < aggregate_count
    )
    classified = (
        empty * TOKEN_KIND_EMPTY
        + atom * TOKEN_KIND_ATOM
        + aggregate * TOKEN_KIND_AGGREGATE
    )
    # Carry the input span's runtime extent explicitly through the return.
    # Turing's rank-only ABI then derives the output extent from ``tokens``;
    # native optimization may eliminate the arithmetic identity.
    return classified + tokens * 0
