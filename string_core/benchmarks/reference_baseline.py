"""Deterministic reference-only Token Orchard throughput and memory receipt."""

import json
import platform
import random
import time
import tracemalloc

from token_orchard.contract import DOMAIN_TEXT, STATUS_OK
from token_orchard.core import Vocabulary


def main():
    random_source = random.Random(424242)
    item_count = 2000
    item_length = 16
    values = []
    offsets = [0]
    for unused in range(item_count):
        for unused_atom in range(item_length):
            values.append(random_source.randrange(32, 127))
        offsets.append(len(values))

    vocabulary = Vocabulary()
    tracemalloc.start()
    start = time.perf_counter()
    batch_status, statuses, tokens = vocabulary.intern_batch(
        DOMAIN_TEXT, values, offsets
    )
    elapsed = time.perf_counter() - start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if batch_status != STATUS_OK or any(status != STATUS_OK for status in statuses):
        raise RuntimeError("reference batch failed")

    result = {
        "claim": "restricted-Python reference baseline; not native performance",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "items": item_count,
        "atoms": len(values),
        "unique_tokens": len(set(tokens)),
        "seconds": elapsed,
        "items_per_second": item_count / elapsed,
        "atoms_per_second": len(values) / elapsed,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "aggregate_rows": len(vocabulary.domain),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
