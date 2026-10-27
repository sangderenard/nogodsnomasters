"""One real Vocabulary lifetime containing repeated Unicode and byte requests.

The scalar checksum is a harness result, not a replacement interning algorithm.
Persistence here is within one native invocation, not between host invocations.
"""
from token_orchard.core import Vocabulary
from token_orchard.contract import AGGREGATE_BASE


def repeated_manager(rounds: int) -> int:
    vocabulary = Vocabulary()
    checksum = 0
    index = 0
    while index < rounds:
        text = [65 + index, 128578]
        envelope, statuses, tokens = vocabulary.intern_batch(1, text, [0, 2])
        if envelope != 0 or statuses[0] != 0:
            return -1
        text_token = tokens[0]
        if text_token != AGGREGATE_BASE + 2 * index:
            return -2
        checksum += text_token - AGGREGATE_BASE + 1
        envelope, statuses, tokens = vocabulary.intern_batch(
            2, [index // 256, index % 256, 255], [0, 3]
        )
        if envelope != 0 or statuses[0] != 0:
            return -3
        if tokens[0] != AGGREGATE_BASE + 2 * index + 1:
            return -4
        checksum += tokens[0] - AGGREGATE_BASE + 1
        envelope, statuses, tokens = vocabulary.intern_batch(1, text, [0, 2])
        if envelope != 0 or statuses[0] != 0 or tokens[0] != text_token:
            return -5
        checksum += tokens[0] - AGGREGATE_BASE + 1
        index += 1
    # Invalid requests must reject without minting or changing the epoch.
    before = vocabulary.epoch
    envelope, statuses, tokens = vocabulary.intern_batch(1, [55296], [0, 1])
    if envelope != 0 or statuses[0] != 2 or tokens[0] != 0:
        return -6
    envelope, statuses, tokens = vocabulary.intern_batch(2, [256], [0, 1])
    if envelope != 0 or statuses[0] != 2 or tokens[0] != 0:
        return -7
    envelope, statuses, tokens = vocabulary.intern_batch(0, [], [0])
    if envelope != 1:
        return -8
    envelope, statuses, tokens = vocabulary.intern_batch(1, [65], [1, 1])
    if envelope != 4:
        return -9
    envelope, statuses, tokens = vocabulary.intern_batch(1, [65], [False, True])
    if envelope != 4:
        return -14
    if vocabulary.epoch != before or before != 2 * rounds:
        return -10
    if len(vocabulary.domain) != 2 * rounds:
        return -11
    if len(vocabulary.exact_atoms) != 5 * rounds:
        return -12
    if len(vocabulary.cover_tokens) != 5 * rounds:
        return -13
    return checksum + vocabulary.epoch * 1000003 + len(vocabulary.exact_atoms) * 1000033
