"""Measure AbstractTensor screening against the scalar fingerprint oracle."""

import json
import random
import time

from token_orchard.contract import DOMAIN_TEXT, STATUS_OK
from token_orchard.core import Vocabulary
from token_orchard.tensor_batch import preflight_batch


def main():
    random_source = random.Random(8675309)
    item_count = 16384
    item_length = 16
    items = [
        [random_source.randrange(32, 127) for unused_atom in range(item_length)]
        for unused_item in range(item_count)
    ]
    values = [atom for item in items for atom in item]
    offsets = [index * item_length for index in range(item_count + 1)]
    vocabulary = Vocabulary()

    # Warm imports, backend registration, and allocator setup out of the timed
    # region. The receipt measures steady batches, not process startup.
    preflight_batch(
        DOMAIN_TEXT,
        values[:item_length],
        [0, item_length],
        max_sequence_atoms=vocabulary.max_sequence_atoms,
        backend="numpy",
    )

    measurements = {}
    for backend in ("numpy", "torch", "nodus"):
        backend_rows = []
        try:
            for measured_items in (256, 1024, 4096, 16384):
                measured_atoms = measured_items * item_length
                started = time.perf_counter()
                statuses, tensor_fingerprints = preflight_batch(
                    DOMAIN_TEXT,
                    values[:measured_atoms],
                    offsets[: measured_items + 1],
                    max_sequence_atoms=vocabulary.max_sequence_atoms,
                    backend=backend,
                )
                tensor_seconds = time.perf_counter() - started

                started = time.perf_counter()
                scalar_fingerprints = [
                    vocabulary._fingerprint(DOMAIN_TEXT, item)
                    for item in items[:measured_items]
                ]
                scalar_seconds = time.perf_counter() - started
                if statuses != [STATUS_OK] * measured_items:
                    raise RuntimeError("tensor preflight rejected valid data")
                if tensor_fingerprints != scalar_fingerprints:
                    raise RuntimeError("tensor and scalar fingerprints diverged")
                backend_rows.append(
                    {
                        "items": measured_items,
                        "atoms": measured_atoms,
                        "tensor_seconds": tensor_seconds,
                        "scalar_seconds": scalar_seconds,
                        "tensor_items_per_second": measured_items / tensor_seconds,
                        "scalar_items_per_second": measured_items / scalar_seconds,
                        "measured_speedup": scalar_seconds / tensor_seconds,
                    }
                )
        except Exception as error:
            measurements[backend] = {"unavailable": str(error)}
        else:
            measurements[backend] = backend_rows

    print(
        json.dumps(
            {
                "claim": "steady AbstractTensor fingerprint preflight; eager backends",
                "item_length": item_length,
                "backends": measurements,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
