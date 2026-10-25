"""Deterministic portable snapshots for Token Orchard version zero."""

from array import array
import hashlib
import struct
import sys

from .contract import (
    AGGREGATE_BASE,
    BATCH_ABI_VERSION,
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    SCHEMA_VERSION,
    STATUS_INVALID_SNAPSHOT,
    STATUS_OK,
    STATUS_UNSUPPORTED_SCHEMA,
)
from .core import Vocabulary


SNAPSHOT_MAGIC = b"TOKORCH0"
SNAPSHOT_FORMAT_VERSION = 0
HEADER_FORMAT = "<8sHHHHQQQQQ"
SECTION_FORMAT = "<HBBQQ"
DIGEST_BYTES = 32

SECTION_FIELDS = (
    (1, "domain", "B", 1),
    (2, "atom_length", "Q", 8),
    (3, "fingerprint_hi", "Q", 8),
    (4, "fingerprint_lo", "Q", 8),
    (5, "exact_begin", "Q", 8),
    (6, "exact_count", "I", 4),
    (7, "cover_begin", "Q", 8),
    (8, "cover_count", "I", 4),
    (9, "created_epoch", "Q", 8),
    (10, "exact_atoms", "I", 4),
    (11, "cover_tokens", "Q", 8),
)


def _little_endian_bytes(values, width):
    if values.itemsize != width:
        raise ValueError("unexpected native array width")
    copied = array(values.typecode, values)
    if sys.byteorder != "little":
        copied.byteswap()
    return copied.tobytes()


def _array_from_little_endian(typecode, width, payload):
    values = array(typecode)
    if values.itemsize != width:
        raise ValueError("unexpected native array width")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def to_snapshot_bytes(vocabulary):
    """Serialize authoritative arrays to a digested little-endian snapshot."""
    parts = [
        struct.pack(
            HEADER_FORMAT,
            SNAPSHOT_MAGIC,
            SCHEMA_VERSION,
            BATCH_ABI_VERSION,
            SNAPSHOT_FORMAT_VERSION,
            vocabulary.fingerprint_mode,
            vocabulary.epoch,
            vocabulary.max_sequence_atoms,
            vocabulary.max_batch_items,
            vocabulary.max_aggregates,
            len(SECTION_FIELDS),
        )
    ]
    for section_id, field_name, unused_typecode, width in SECTION_FIELDS:
        values = getattr(vocabulary, field_name)
        payload = _little_endian_bytes(values, width)
        parts.append(
            struct.pack(
                SECTION_FORMAT,
                section_id,
                width,
                0,
                len(values),
                len(payload),
            )
        )
        parts.append(payload)
    body = b"".join(parts)
    return body + hashlib.sha256(body).digest()


def snapshot_digest(snapshot_bytes):
    """Return the stored digest only after verifying it."""
    if len(snapshot_bytes) < DIGEST_BYTES:
        return STATUS_INVALID_SNAPSHOT, b""
    body = snapshot_bytes[:-DIGEST_BYTES]
    stored = snapshot_bytes[-DIGEST_BYTES:]
    computed = hashlib.sha256(body).digest()
    if stored != computed:
        return STATUS_INVALID_SNAPSHOT, b""
    return STATUS_OK, stored


def _validate_and_reindex(vocabulary):
    row_count = len(vocabulary.domain)
    if row_count > vocabulary.max_aggregates:
        return False
    aligned_fields = (
        vocabulary.atom_length,
        vocabulary.fingerprint_hi,
        vocabulary.fingerprint_lo,
        vocabulary.exact_begin,
        vocabulary.exact_count,
        vocabulary.cover_begin,
        vocabulary.cover_count,
        vocabulary.created_epoch,
    )
    for values in aligned_fields:
        if len(values) != row_count:
            return False
    if vocabulary.epoch != row_count:
        return False

    expected_exact_begin = 0
    expected_cover_begin = 0
    row = 0
    while row < row_count:
        domain = vocabulary.domain[row]
        if domain != DOMAIN_TEXT and domain != DOMAIN_BYTES:
            return False
        if vocabulary.created_epoch[row] != row + 1:
            return False
        if vocabulary.exact_begin[row] != expected_exact_begin:
            return False
        if vocabulary.cover_begin[row] != expected_cover_begin:
            return False
        exact_count = vocabulary.exact_count[row]
        cover_count = vocabulary.cover_count[row]
        if exact_count < 2 or cover_count == 0:
            return False
        expected_exact_begin += exact_count
        expected_cover_begin += cover_count
        if expected_exact_begin > len(vocabulary.exact_atoms):
            return False
        if expected_cover_begin > len(vocabulary.cover_tokens):
            return False

        atoms = list(
            vocabulary.exact_atoms[
                vocabulary.exact_begin[row] : expected_exact_begin
            ]
        )
        if vocabulary._validate_atoms(domain, atoms) != STATUS_OK:
            return False
        hi, lo = vocabulary._fingerprint(domain, atoms)
        if hi != vocabulary.fingerprint_hi[row] or lo != vocabulary.fingerprint_lo[row]:
            return False
        existing = vocabulary._find_existing(domain, atoms, hi, lo)
        if existing != 0:
            return False

        token = AGGREGATE_BASE + row
        key = (domain, hi, lo, len(atoms))
        vocabulary._candidates.setdefault(key, []).append(token)
        vocabulary._prefix.setdefault((domain, atoms[0]), []).append(token)

        rebuilt = []
        cover_start = vocabulary.cover_begin[row]
        cover_end = cover_start + cover_count
        for child in vocabulary.cover_tokens[cover_start:cover_end]:
            if child >= token or vocabulary.token_domain(child) != domain:
                return False
            status, child_atoms = vocabulary.expand(child)
            if status != STATUS_OK:
                return False
            rebuilt.extend(child_atoms)
        if rebuilt != atoms:
            return False
        row += 1

    return (
        expected_exact_begin == len(vocabulary.exact_atoms)
        and expected_cover_begin == len(vocabulary.cover_tokens)
    )


def from_snapshot_bytes(snapshot_bytes):
    """Validate and load a snapshot; return ``(status, vocabulary_or_none)``."""
    digest_status, unused_digest = snapshot_digest(snapshot_bytes)
    if digest_status != STATUS_OK:
        return digest_status, None
    body = snapshot_bytes[:-DIGEST_BYTES]
    header_size = struct.calcsize(HEADER_FORMAT)
    section_size = struct.calcsize(SECTION_FORMAT)
    if len(body) < header_size:
        return STATUS_INVALID_SNAPSHOT, None
    try:
        header = struct.unpack_from(HEADER_FORMAT, body, 0)
    except struct.error:
        return STATUS_INVALID_SNAPSHOT, None
    (
        magic,
        schema_version,
        batch_version,
        snapshot_version,
        fingerprint_mode,
        epoch,
        max_sequence_atoms,
        max_batch_items,
        max_aggregates,
        section_count,
    ) = header
    if magic != SNAPSHOT_MAGIC:
        return STATUS_INVALID_SNAPSHOT, None
    if (
        schema_version != SCHEMA_VERSION
        or batch_version != BATCH_ABI_VERSION
        or snapshot_version != SNAPSHOT_FORMAT_VERSION
    ):
        return STATUS_UNSUPPORTED_SCHEMA, None
    if section_count != len(SECTION_FIELDS):
        return STATUS_INVALID_SNAPSHOT, None

    try:
        vocabulary = Vocabulary(
            fingerprint_mode=fingerprint_mode,
            max_sequence_atoms=max_sequence_atoms,
            max_batch_items=max_batch_items,
            max_aggregates=max_aggregates,
        )
    except ValueError:
        return STATUS_INVALID_SNAPSHOT, None
    vocabulary.epoch = epoch
    offset = header_size
    try:
        for expected_id, field_name, typecode, expected_width in SECTION_FIELDS:
            if offset + section_size > len(body):
                return STATUS_INVALID_SNAPSHOT, None
            section_id, width, reserved, count, byte_count = struct.unpack_from(
                SECTION_FORMAT, body, offset
            )
            offset += section_size
            if (
                section_id != expected_id
                or width != expected_width
                or reserved != 0
                or byte_count != count * width
                or offset + byte_count > len(body)
            ):
                return STATUS_INVALID_SNAPSHOT, None
            payload = body[offset : offset + byte_count]
            offset += byte_count
            values = _array_from_little_endian(typecode, width, payload)
            setattr(vocabulary, field_name, values)
    except (OverflowError, ValueError, struct.error):
        return STATUS_INVALID_SNAPSHOT, None
    if offset != len(body):
        return STATUS_INVALID_SNAPSHOT, None
    if not _validate_and_reindex(vocabulary):
        return STATUS_INVALID_SNAPSHOT, None
    return STATUS_OK, vocabulary
