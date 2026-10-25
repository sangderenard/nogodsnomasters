"""Compiler-targeted integer kernels shared by the Python reference core."""

U32_MODULUS = 4294967296
U24_MODULUS = 16777216
FNV_PRIME_LOW = 435

FNV_OFFSET_HIGH = 0xCBF29CE4
FNV_OFFSET_LOW = 0x84222325
FNV_ALT_OFFSET_HIGH = 0x340D631B
FNV_ALT_OFFSET_LOW = 0x7BDDDCDA


def fnv1a_limb_step(high: int, low: int, byte: int) -> tuple[int, int]:
    """Apply one FNV-1a step to scalar integers or AbstractTensor lanes."""
    mixed_low = low ^ byte
    low_product = mixed_low * FNV_PRIME_LOW
    next_low = low_product % U32_MODULUS
    carry = low_product // U32_MODULUS
    shifted_low = (mixed_low % U24_MODULUS) * 256
    next_high = (
        high * FNV_PRIME_LOW + carry + shifted_low
    ) % U32_MODULUS
    return next_high, next_low


def fingerprint_atom_step(high: int, low: int, atom: int) -> tuple[int, int]:
    """Fold one scalar or AbstractTensor u32 atom lane as four bytes."""
    remaining = atom
    index = 0
    while index < 4:
        byte = remaining % 256
        high, low = fnv1a_limb_step(high, low, byte)
        remaining = remaining // 256
        index += 1
    return high, low


def fingerprint_lanes_span(
    domain: int, atoms, begin: int, end: int
) -> tuple[int, int, int, int]:
    """Fingerprint one half-open atom span without materializing a slice."""
    first_high, first_low = fnv1a_limb_step(
        FNV_OFFSET_HIGH, FNV_OFFSET_LOW, domain
    )
    second_high, second_low = fnv1a_limb_step(
        FNV_ALT_OFFSET_HIGH, FNV_ALT_OFFSET_LOW, domain
    )
    index = begin
    while index < end:
        atom = atoms[index]
        first_high, first_low = fingerprint_atom_step(
            first_high, first_low, atom
        )
        second_high, second_low = fingerprint_atom_step(
            second_high, second_low, atom
        )
        index += 1
    return first_high, first_low, second_high, second_low


def fingerprint_lanes(domain: int, atoms) -> tuple[int, int, int, int]:
    """Return two independent FNV-1a lanes as four u32 limbs."""
    return fingerprint_lanes_span(domain, atoms, 0, len(atoms))
