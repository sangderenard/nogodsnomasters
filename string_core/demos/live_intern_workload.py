"""Whole reachable interning core, independently buildable from query probes."""
from token_orchard.core import Vocabulary
from token_orchard.contract import AGGREGATE_BASE


def live_intern_workload(rounds: int, method: int, collision: int):
    vocabulary = Vocabulary(fingerprint_mode=collision)
    failures = 0
    checks = 0
    checksum = 0
    last_token = 0
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
    return failures, checks, vocabulary.epoch, len(vocabulary.domain), len(vocabulary.exact_atoms), len(vocabulary.cover_tokens), last_token, 0, 0, checksum, fingerprint
