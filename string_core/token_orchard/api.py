"""Token-only live Python namespace for the process-global string core."""

from __future__ import annotations

import threading

from .contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    ENCODING_UTF8,
    STATUS_OK,
    STATUS_SERVICE_FAILURE,
)
from .encoding import decode_bytes
from .service import TokenService


class StringCoreError(RuntimeError):
    """A live-Python boundary failure carrying only integer receipts."""

    def __init__(self, status, byte_token=0, text_token=0):
        self.status = int(status)
        self.byte_token = int(byte_token)
        self.text_token = int(text_token)
        super().__init__(f"Token Orchard status {self.status}")


class _GlobalTokenRuntime:
    """Own the one live-Python vocabulary and its genuine writer queue."""

    def __init__(self):
        self.service = TokenService(thread_daemon=True)

    def intern_flat(self, domain, values, offsets):
        status, ticket = self.service.submit_intern(
            domain,
            values,
            offsets,
            block=True,
        )
        if status != STATUS_OK or ticket is None:
            return status, [], [], 0
        return ticket.wait()

    def close(self):
        self.service.close(wait=True)


_RUNTIME_LOCK = threading.Lock()
_RUNTIME = None


def _global_runtime():
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _GlobalTokenRuntime()
        return _RUNTIME


def _byte_values(payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("String ingress accepts bytes-like values, not text")
    return list(bytes(payload))


def _one_token(domain, values):
    batch_status, statuses, tokens, unused_epoch = _global_runtime().intern_flat(
        domain, values, (0, len(values))
    )
    if batch_status != STATUS_OK:
        raise StringCoreError(batch_status)
    if len(statuses) != 1 or len(tokens) != 1:
        raise StringCoreError(STATUS_SERVICE_FAILURE)
    if statuses[0] != STATUS_OK:
        raise StringCoreError(statuses[0])
    return int(tokens[0])


class String:
    """Non-instantiable namespace whose values are always integer tokens.

    The methods are the ergonomic Python boundary. They are not the native
    implementation: compiler ingestion recognizes their exact source identity
    and links the corresponding Token Orchard SSA operation instead of
    pursuing Python queue/lock machinery.
    """

    UTF8 = ENCODING_UTF8

    def __new__(cls, *unused_args, **unused_kwargs):
        raise TypeError("String is a token namespace and cannot be instantiated")

    @staticmethod
    def bytes(payload):
        """Intern one exact bytes-like value and return its integer token."""
        return _one_token(DOMAIN_BYTES, _byte_values(payload))

    @staticmethod
    def bytes_batch(payloads):
        """Intern a batch of bytes-like values through one flat queue request."""
        parts = tuple(_byte_values(payload) for payload in payloads)
        values = []
        offsets = [0]
        for part in parts:
            values.extend(part)
            offsets.append(len(values))
        batch_status, statuses, tokens, unused_epoch = (
            _global_runtime().intern_flat(DOMAIN_BYTES, values, offsets)
        )
        if batch_status != STATUS_OK:
            raise StringCoreError(batch_status)
        for status in statuses:
            if status != STATUS_OK:
                raise StringCoreError(status)
        return tuple(int(token) for token in tokens)

    @staticmethod
    def decode_receipt(payload, encoding=ENCODING_UTF8):
        """Intern exact bytes, then explicitly decode and intern text.

        Return ``(status, encoding, byte_token, text_token)``. Malformed input
        therefore remains available by its byte token while text_token is zero.
        Both successful interning operations pass through the global writer.
        """
        values = _byte_values(payload)
        byte_token = _one_token(DOMAIN_BYTES, values)
        status, scalars = decode_bytes(values, encoding)
        if status != STATUS_OK:
            return int(status), int(encoding), byte_token, 0
        text_token = _one_token(DOMAIN_TEXT, scalars)
        return STATUS_OK, int(encoding), byte_token, text_token

    @staticmethod
    def decode(payload, encoding=ENCODING_UTF8):
        """Return the text-domain token or raise with its byte-token receipt."""
        status, unused_encoding, byte_token, text_token = String.decode_receipt(
            payload, encoding
        )
        if status != STATUS_OK:
            raise StringCoreError(status, byte_token, text_token)
        return text_token


def _reset_global_runtime_for_tests():
    """Close and discard the singleton; never part of the public API."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        _RUNTIME = None
    if runtime is not None:
        runtime.close()
