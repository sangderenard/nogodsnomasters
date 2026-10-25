"""Integer lifecycle contract for generational vocabulary reclamation.

This module does not invalidate a token inside a live vocabulary. It defines
the batched evidence and state needed to choose rows for a future generation.
The writer may reclaim the old generation only after its published epochs have
no pins. Surviving exact sequences are then reinterned into a new vocabulary
identity with an explicit old-token/new-token remap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import (
    REINTERN_COLD,
    REINTERN_CONDITION_OUT,
    REINTERN_POLICY_LIMIT,
    REINTERN_RETAIN_LEARNED,
    RETENTION_ABI_VERSION,
)


@dataclass(frozen=True)
class RetentionNoticeBatch:
    """One logical-time notice for a flat batch of token IDs.

    Time is an explicit monotonic integer supplied by the owning program. No
    wall clock enters identity, compilation, replay, or tests. Duplicate
    tokens are permitted and can be coalesced by the single writer.
    """

    abi_version: int
    notice_tick: int
    tokens: tuple[int, ...]

    @classmethod
    def from_tokens(cls, notice_tick, tokens):
        batch = cls(RETENTION_ABI_VERSION, notice_tick, tuple(tokens))
        batch.validate()
        return batch

    def validate(self):
        if self.abi_version != RETENTION_ABI_VERSION:
            raise ValueError("unsupported retention notice ABI")
        if (
            not isinstance(self.notice_tick, int)
            or isinstance(self.notice_tick, bool)
            or self.notice_tick < 0
        ):
            raise ValueError("invalid retention notice tick")
        if len(self.tokens) == 0:
            raise ValueError("retention notice batch must be nonempty")
        if any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or token <= 0
            for token in self.tokens
        ):
            raise ValueError("invalid retention notice token")
        return self


def retention_state_abi():
    """Describe the optional structure-of-arrays lifecycle state."""
    return {
        "abi_version": RETENTION_ABI_VERSION,
        "clock": "explicit_monotonic_tick",
        "fields": (
            ("last_notice_tick", "uint64", 1, "aggregate-row aligned"),
            ("notice_count", "uint64", 1, "aggregate-row aligned"),
            ("protected", "uint8", 1, "aggregate-row aligned"),
            ("generation_id", "uint64", 0, "vocabulary identity state"),
            ("minimum_pinned_epoch", "uint64", 0, "reader reclamation fence"),
        ),
        "reintern_policies": (
            REINTERN_COLD,
            REINTERN_RETAIN_LEARNED,
            REINTERN_CONDITION_OUT,
        ),
    }


def retention_program_abi():
    """Declare the physical Turing ABI for vectorized stale-row selection."""
    tensor_type = "src.common.tensors.abstraction.AbstractTensor"
    function = "retention_stale_mask"
    return {
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": function,
                "parameter": "last_notice_tick",
                "storage": "span",
                "dtype": "int64",
                "rank": 1,
                "python_type": tensor_type,
            },
            {
                "function": function,
                "parameter": "protected",
                "storage": "span",
                "dtype": "int64",
                "rank": 1,
                "python_type": tensor_type,
            },
            {
                "function": function,
                "parameter": "current_tick",
                "storage": "scalar",
                "dtype": "int64",
                "rank": 0,
                "shape": [],
                "python_type": "builtins.int",
            },
            {
                "function": function,
                "parameter": "idle_ticks",
                "storage": "scalar",
                "dtype": "int64",
                "rank": 0,
                "shape": [],
                "python_type": "builtins.int",
            },
        ],
    }


def validate_reintern_policy(policy):
    if (
        not isinstance(policy, int)
        or isinstance(policy, bool)
        or policy < REINTERN_COLD
        or policy >= REINTERN_POLICY_LIMIT
    ):
        raise ValueError("invalid reinternment policy")
    return policy
