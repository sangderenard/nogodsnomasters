"""A program whose string operations share one Token Orchard vocabulary.

The compiler must lower this entire reachable program as one native product.
The imported String namespace is a live-Python spelling, not a request to call
its Python queue, locks, or singleton from the compiled artifact.
"""

from token_orchard import String


def root(payload):
    byte_token = String.bytes(payload)
    batch_tokens = String.bytes_batch((payload, payload))
    status, encoding, receipt_byte_token, text_token = String.decode_receipt(
        payload, String.UTF8
    )
    checked_text_token = 0
    if status == 0:
        checked_text_token = String.decode(payload, String.UTF8)
    return (
        status,
        encoding,
        byte_token,
        receipt_byte_token,
        text_token,
        checked_text_token,
        batch_tokens[0],
        batch_tokens[1],
    )
