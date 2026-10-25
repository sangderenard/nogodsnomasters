import unittest

from token_orchard.kernel import (
    FNV_ALT_OFFSET_HIGH,
    FNV_ALT_OFFSET_LOW,
    FNV_OFFSET_HIGH,
    FNV_OFFSET_LOW,
    fingerprint_atom_step,
    fingerprint_lanes,
    fingerprint_lanes_span,
    fnv1a_limb_step,
)


class CompilerKernelTests(unittest.TestCase):
    def naive_step(self, high, low, byte):
        value = (high << 32) | low
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return value >> 32, value & 0xFFFFFFFF

    def test_limb_step_matches_direct_unbounded_python(self):
        cases = [
            (FNV_OFFSET_HIGH, FNV_OFFSET_LOW, 0),
            (FNV_OFFSET_HIGH, FNV_OFFSET_LOW, 255),
            (0, 0, 17),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x80),
        ]
        for high, low, byte in cases:
            self.assertEqual(
                fnv1a_limb_step(high, low, byte),
                self.naive_step(high, low, byte),
            )

    def test_atom_step_is_four_little_endian_byte_steps(self):
        high = FNV_OFFSET_HIGH
        low = FNV_OFFSET_LOW
        atom = 0x10F09F
        expected_high = high
        expected_low = low
        for shift in (0, 8, 16, 24):
            expected_high, expected_low = self.naive_step(
                expected_high, expected_low, (atom >> shift) & 255
            )
        self.assertEqual(
            fingerprint_atom_step(high, low, atom),
            (expected_high, expected_low),
        )

    def test_two_lanes_have_exact_independent_fnv_meaning(self):
        atoms = [0x41, 0x03BB, 0x1F9E0]
        got = fingerprint_lanes(1, atoms)
        expected = []
        for high, low in (
            (FNV_OFFSET_HIGH, FNV_OFFSET_LOW),
            (FNV_ALT_OFFSET_HIGH, FNV_ALT_OFFSET_LOW),
        ):
            high, low = self.naive_step(high, low, 1)
            for atom in atoms:
                for shift in (0, 8, 16, 24):
                    high, low = self.naive_step(
                        high, low, (atom >> shift) & 255
                    )
            expected.extend((high, low))
        self.assertEqual(got, tuple(expected))

    def test_span_kernel_does_not_materialize_or_read_outside_item(self):
        atoms = [999999, 65, 66, 67, 999999]
        self.assertEqual(
            fingerprint_lanes_span(1, atoms, 1, 4),
            fingerprint_lanes(1, [65, 66, 67]),
        )


if __name__ == "__main__":
    unittest.main()
