"""Explicit encoding adapters between byte and text token domains."""

from .contract import (
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    ENCODING_LATIN1,
    ENCODING_UTF16_BE,
    ENCODING_UTF16_LE,
    ENCODING_UTF32_BE,
    ENCODING_UTF32_LE,
    ENCODING_UTF8,
    STATUS_INVALID_ATOM,
    STATUS_INVALID_ENCODING,
    STATUS_INVALID_TOKEN,
    STATUS_OK,
    STATUS_UNSUPPORTED_ENCODING,
    SURROGATE_BEGIN,
    SURROGATE_END,
    UNICODE_LIMIT,
)


def _codec_name(encoding_id):
    if encoding_id == ENCODING_UTF8:
        return "utf-8"
    if encoding_id == ENCODING_UTF16_LE:
        return "utf-16-le"
    if encoding_id == ENCODING_UTF16_BE:
        return "utf-16-be"
    if encoding_id == ENCODING_UTF32_LE:
        return "utf-32-le"
    if encoding_id == ENCODING_UTF32_BE:
        return "utf-32-be"
    if encoding_id == ENCODING_LATIN1:
        return "latin-1"
    return None


def decode_bytes(byte_values, encoding_id):
    """Strictly decode integer bytes; return ``(status, scalar_values)``."""
    codec = _codec_name(encoding_id)
    if codec is None:
        return STATUS_UNSUPPORTED_ENCODING, []
    for value in byte_values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 255:
            return STATUS_INVALID_ATOM, []
    try:
        text = bytes(byte_values).decode(codec, errors="strict")
    except UnicodeError:
        return STATUS_INVALID_ENCODING, []
    scalars = [ord(character) for character in text]
    for scalar in scalars:
        if scalar < 0 or scalar >= UNICODE_LIMIT or SURROGATE_BEGIN <= scalar < SURROGATE_END:
            return STATUS_INVALID_ENCODING, []
    return STATUS_OK, scalars


def encode_scalars(scalars, encoding_id):
    """Strictly encode Unicode scalars; return ``(status, byte_values)``."""
    codec = _codec_name(encoding_id)
    if codec is None:
        return STATUS_UNSUPPORTED_ENCODING, []
    characters = []
    for scalar in scalars:
        if (
            not isinstance(scalar, int)
            or isinstance(scalar, bool)
            or scalar < 0
            or scalar >= UNICODE_LIMIT
            or SURROGATE_BEGIN <= scalar < SURROGATE_END
        ):
            return STATUS_INVALID_ATOM, []
        characters.append(chr(scalar))
    try:
        encoded = "".join(characters).encode(codec, errors="strict")
    except UnicodeError:
        return STATUS_INVALID_ENCODING, []
    return STATUS_OK, list(encoded)


def intern_encoded(vocabulary, byte_values, encoding_id):
    """Intern bytes, strictly decode, then intern text.

    Returns ``(status, encoding_id, byte_token, text_token)``. Invalid encoded
    data remains exactly preserved by ``byte_token`` while ``text_token`` is
    zero, making the failed transform explicit rather than destructive.
    """
    byte_status, byte_token = vocabulary.intern(DOMAIN_BYTES, byte_values)
    if byte_status != STATUS_OK:
        return byte_status, encoding_id, 0, 0
    decode_status, scalars = decode_bytes(byte_values, encoding_id)
    if decode_status != STATUS_OK:
        return decode_status, encoding_id, byte_token, 0
    text_status, text_token = vocabulary.intern(DOMAIN_TEXT, scalars)
    return text_status, encoding_id, byte_token, text_token


def encode_text_token(vocabulary, text_token, encoding_id):
    """Encode a text token and intern its bytes with an explicit receipt."""
    if vocabulary.token_domain(text_token) != DOMAIN_TEXT:
        return STATUS_INVALID_TOKEN, encoding_id, text_token, 0
    status, scalars = vocabulary.expand(text_token)
    if status != STATUS_OK:
        return status, encoding_id, text_token, 0
    encode_status, byte_values = encode_scalars(scalars, encoding_id)
    if encode_status != STATUS_OK:
        return encode_status, encoding_id, text_token, 0
    byte_status, byte_token = vocabulary.intern(DOMAIN_BYTES, byte_values)
    return byte_status, encoding_id, text_token, byte_token
