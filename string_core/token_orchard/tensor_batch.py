"""AbstractTensor batch screening shared by the queued writer path."""

from pathlib import Path
import sys

from .contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    FINGERPRINT_CONSTANT,
    STATUS_INVALID_ATOM,
    STATUS_INVALID_DOMAIN,
    STATUS_LIMIT_EXCEEDED,
    STATUS_OK,
    SURROGATE_BEGIN,
    SURROGATE_END,
    UNICODE_LIMIT,
)
from .kernel import (
    FNV_ALT_OFFSET_HIGH,
    FNV_ALT_OFFSET_LOW,
    FNV_OFFSET_HIGH,
    FNV_OFFSET_LOW,
    U32_MODULUS,
    fingerprint_atom_step,
    fnv1a_limb_step,
)


def _abstract_tensor_type():
    """Load the workspace's established tensor substrate, never a stand-in."""
    try:
        from src.common.tensors import AbstractTensor
    except ModuleNotFoundError:
        turing_root = Path(__file__).resolve().parents[2] / "turing"
        if not (turing_root / "src" / "common" / "tensors").is_dir():
            raise RuntimeError(
                "Token Orchard's queued tensor path requires Turing AbstractTensor"
            ) from None
        sys.path.insert(0, str(turing_root))
        from src.common.tensors import AbstractTensor
    return AbstractTensor


def fingerprint_padded_rows(domain, rows, lengths, abstract_tensor):
    """Fingerprint a padded ragged batch with masked whole-lane updates."""
    row_count, padded_atom_count = rows.shape
    first_high = abstract_tensor.full(
        (row_count,), FNV_OFFSET_HIGH, dtype="int64", cls=type(rows)
    )
    first_low = abstract_tensor.full(
        (row_count,), FNV_OFFSET_LOW, dtype="int64", cls=type(rows)
    )
    second_high = abstract_tensor.full(
        (row_count,), FNV_ALT_OFFSET_HIGH, dtype="int64", cls=type(rows)
    )
    second_low = abstract_tensor.full(
        (row_count,), FNV_ALT_OFFSET_LOW, dtype="int64", cls=type(rows)
    )
    domain_column = abstract_tensor.full(
        (row_count,), domain, dtype="int64", cls=type(rows)
    )
    first_high, first_low = fnv1a_limb_step(
        first_high, first_low, domain_column
    )
    second_high, second_low = fnv1a_limb_step(
        second_high, second_low, domain_column
    )
    column = 0
    while column < padded_atom_count:
        atom = rows[:, column]
        active = lengths > column
        next_first_high, next_first_low = fingerprint_atom_step(
            first_high, first_low, atom
        )
        next_second_high, next_second_low = fingerprint_atom_step(
            second_high, second_low, atom
        )
        first_high = abstract_tensor.where(active, next_first_high, first_high)
        first_low = abstract_tensor.where(active, next_first_low, first_low)
        second_high = abstract_tensor.where(
            active, next_second_high, second_high
        )
        second_low = abstract_tensor.where(active, next_second_low, second_low)
        column += 1
    return abstract_tensor.stack(
        (first_high, first_low, second_high, second_low), dim=1
    )


def preflight_batch(
    domain,
    values,
    offsets,
    *,
    max_sequence_atoms,
    fingerprint_mode=0,
    backend="numpy",
):
    """Validate and fingerprint one padded AbstractTensor ragged batch.

    The public ABI stays COO-like flat values plus offsets. One bounded host
    staging pass forms a padded matrix and lengths vector; validation and both
    masked FNV lanes then execute as whole-batch AbstractTensor operations.
    """
    item_count = len(offsets) - 1
    statuses = [STATUS_OK] * item_count
    fingerprints = [None] * item_count
    if domain != DOMAIN_TEXT and domain != DOMAIN_BYTES:
        return [STATUS_INVALID_DOMAIN] * item_count, fingerprints
    if item_count == 0:
        return statuses, fingerprints

    lengths = [0] * item_count
    padded_atom_count = 0
    item_index = 0
    while item_index < item_count:
        length = offsets[item_index + 1] - offsets[item_index]
        if length > max_sequence_atoms:
            statuses[item_index] = STATUS_LIMIT_EXCEEDED
        else:
            lengths[item_index] = length
            if length > padded_atom_count:
                padded_atom_count = length
        item_index += 1

    abstract_tensor = _abstract_tensor_type()
    with abstract_tensor.use_backend(backend, "cpu" if backend == "numpy" else None):
        # Offsets are topology, not string computation. Pack them once in
        # O(atoms) host work. AbstractTensor.searchsorted is intentionally a
        # broadcast compare+sum implementation, which would make this
        # O(items*atoms); the eventual zero-copy ABI should carry COO row and
        # column coordinates directly instead.
        padded_values = [0] * (item_count * padded_atom_count)
        item_index = 0
        while item_index < item_count:
            length = lengths[item_index]
            source = offsets[item_index]
            destination = item_index * padded_atom_count
            column = 0
            while column < length:
                padded_values[destination + column] = values[source + column]
                column += 1
            item_index += 1
        tensor = abstract_tensor.tensor(padded_values, dtype="int64").reshape(
            (item_count, padded_atom_count)
        )
        length_tensor = abstract_tensor.tensor(lengths, dtype="int64")
        if domain == DOMAIN_BYTES:
            valid = ((tensor >= 0) & (tensor <= 255)).all(dim=1)
        else:
            valid = (
                (tensor >= 0)
                & (tensor < UNICODE_LIMIT)
                & ((tensor < SURROGATE_BEGIN) | (tensor >= SURROGATE_END))
            ).all(dim=1)
        valid_rows = [bool(value) for value in valid.tolist()]
        if fingerprint_mode == FINGERPRINT_CONSTANT:
            lane_rows = [[0, 0, 0, 0] for unused_index in range(item_count)]
        else:
            lane_rows = fingerprint_padded_rows(
                domain, tensor, length_tensor, abstract_tensor
            ).tolist()
        item_index = 0
        while item_index < item_count:
            if statuses[item_index] == STATUS_OK:
                if not valid_rows[item_index]:
                    statuses[item_index] = STATUS_INVALID_ATOM
                else:
                    high_a, low_a, high_b, low_b = lane_rows[item_index]
                    fingerprints[item_index] = (
                        int(high_a) * U32_MODULUS + int(low_a),
                        int(high_b) * U32_MODULUS + int(low_b),
                    )
            item_index += 1
    return statuses, fingerprints
