"""The authored workload compiled by the live demo, also its Python oracle.

One invocation owns one Vocabulary. Increasing rounds replays a larger
history in a fresh invocation; the UI never claims cross-invocation state.
All operations below call the real core, including its exact collision check.
"""
from token_orchard.core import Vocabulary
from token_orchard.contract import AGGREGATE_BASE


def live_workload(rounds: int, method: int, collision: int):
    vocabulary = Vocabulary(fingerprint_mode=collision)
    failures = 0
    checks = 0
    checksum = 0
    last_token = 0
    last_length = 0
    last_cover = 0
    index = 0
    while index < rounds:
        text = [65 + index, 128578]
        if method == 1:
            status, token = vocabulary.intern(1, text)
            byte_status, byte_token = vocabulary.intern(2, [index // 256, index % 256, 255])
            duplicate_status, duplicate_token = vocabulary.intern(1, text)
        else:
            envelope, statuses, tokens = vocabulary.intern_batch(1, text, [0, 2])
            status = envelope + statuses[0]
            token = tokens[0]
            envelope, statuses, tokens = vocabulary.intern_batch(2, [index // 256, index % 256, 255], [0, 3])
            byte_status = envelope + statuses[0]
            byte_token = tokens[0]
            envelope, statuses, tokens = vocabulary.intern_batch(1, text, [0, 2])
            duplicate_status = envelope + statuses[0]
            duplicate_token = tokens[0]
        checks += 4
        if status != 0 or byte_status != 0 or duplicate_status != 0:
            failures += 1
        if token != AGGREGATE_BASE + 2 * index:
            failures += 1
        if byte_token != AGGREGATE_BASE + 2 * index + 1:
            failures += 1
        if duplicate_token != token:
            failures += 1
        checksum += token - AGGREGATE_BASE + byte_token - AGGREGATE_BASE + 2
        if method >= 2:
            checks += 2
            if vocabulary.token_domain(token) != 1:
                failures += 1
            if vocabulary.token_domain(byte_token) != 2:
                failures += 1
            expanded_status, expanded = vocabulary.expand(token)
            restored_status, restored = vocabulary.intern(1, expanded)
            length_status, measured_length = vocabulary.length(token)
            cover_status, cover = vocabulary.cover(token)
            checks += 4
            if expanded_status != 0 or restored_status != 0 or length_status != 0 or cover_status != 0:
                failures += 1
            if len(expanded) != 2 or measured_length != 2:
                failures += 1
            # Exact re-interning checks the complete expansion through the
            # core's candidate atom comparison, not a hash or length alone.
            if restored != token:
                failures += 1
            if len(cover) < 1:
                failures += 1
            last_length = measured_length
            last_cover = len(cover)
        if method == 3:
            left_status, left = vocabulary.slice(token, 0, 1)
            right_status, right = vocabulary.slice(token, 1, 2)
            joined_status, joined = vocabulary.concat(left, right)
            atom_status, atom = vocabulary.atom_at(token, 0)
            checks += 3
            if left_status != 0 or right_status != 0 or joined_status != 0 or atom_status != 0:
                failures += 1
            if joined != token:
                failures += 1
            if atom != left:
                failures += 1
        last_token = token
        index += 1
    before = vocabulary.epoch
    envelope, statuses, tokens = vocabulary.intern_batch(1, [55296], [0, 1])
    checks += 2
    if envelope != 0 or statuses[0] != 2 or tokens[0] != 0:
        failures += 1
    if vocabulary.epoch != before:
        failures += 1
    fingerprint = 0
    index = 0
    while index < len(vocabulary.domain):
        fingerprint = (fingerprint + vocabulary.fingerprint_hi[index] % 1000000007 + vocabulary.fingerprint_lo[index] % 1000000007) % 1000000007
        index += 1
    return failures, checks, vocabulary.epoch, len(vocabulary.domain), len(vocabulary.exact_atoms), len(vocabulary.cover_tokens), last_token, last_length, last_cover, checksum, fingerprint
