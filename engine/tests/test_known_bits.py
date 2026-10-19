"""Known bits are proofs: every claimed bit must hold for every input."""
import unittest
from core.expr import Arena


class KnownBitsTests(unittest.TestCase):
    def test_addition_carry_is_not_a_known_zero(self):
        a = Arena()
        x, y = a.opaque('x', 8), a.opaque('y', 8)
        x = a.bin('and', x, a.const(1, 8), 8)
        y = a.bin('and', y, a.const(1, 8), 8)
        value = a.bin('add', x, y, 8)
        self.assertFalse(a.zbits(value) & 2)
        self.assertFalse(a.is_bool(value))

    def test_subtraction_borrow_is_not_a_known_zero(self):
        a = Arena()
        x, y = a.opaque('x', 8), a.opaque('y', 8)
        x = a.bin('and', x, a.const(1, 8), 8)
        y = a.bin('and', y, a.const(1, 8), 8)
        self.assertEqual(0, a.zbits(a.bin('sub', x, y, 8)))

    def test_sar_does_not_invent_a_sign(self):
        a = Arena()
        x = a.opaque('x', 8)
        shifted = a.bin('sar', x, a.const(1, 8), 8)
        self.assertEqual(0, a.zbits(shifted))
        self.assertEqual(0, a.obits(shifted))

    def test_zext_proves_upper_bits_zero(self):
        a = Arena()
        x = a.opaque('x', 8)
        self.assertEqual(0xffffffffffffff00, a.zbits(a.zext(x, 64)))

    def test_all_eight_bit_shift_values_match_integer_arithmetic(self):
        a = Arena()
        x = a.opaque('x', 8)
        for op in ('shl', 'shr', 'sar'):
            for shift in range(10):
                result = a.bin(op, x, a.const(shift, 8), 8)
                for value in range(256):
                    signed = value if value < 128 else value - 256
                    expected = ((value << shift) if op == 'shl' else
                                (value >> shift) if op == 'shr' else (signed >> shift)) & 255
                    actual, exact = a.eval_exact(result, {x: value})
                    self.assertTrue(exact)
                    self.assertEqual(expected, actual, (op, shift, value))

    def test_signed_and_unsigned_comparison_boundaries(self):
        a = Arena()
        x = a.opaque('x', 8)
        for op in ('ult', 'ule', 'slt', 'sle'):
            for constant in (0, 1, 127, 128, 255):
                for reversed_args in (False, True):
                    args = (a.const(constant, 8), x) if reversed_args else (x, a.const(constant, 8))
                    result = a.bin(op, *args, 8)
                    for value in range(256):
                        left, right = (constant, value) if reversed_args else (value, constant)
                        if op.startswith('s'):
                            left = left if left < 128 else left - 256
                            right = right if right < 128 else right - 256
                        expected = int(left < right if op.endswith('lt') else left <= right)
                        actual, exact = a.eval_exact(result, {x: value})
                        self.assertTrue(exact)
                        self.assertEqual(expected, actual, (op, constant, value, reversed_args))

    def test_differing_low_bits_do_not_determine_unknown_high_bits(self):
        a = Arena()
        x, y = a.opaque('x', 8), a.opaque('y', 8)
        even = a.bin('and', x, a.const(254, 8), 8)
        odd = a.bin('or', y, a.const(1, 8), 8)
        result = a.bin('ult', even, odd, 8)
        self.assertIsNone(a.as_const(result))

    def test_claims_hold_for_exhaustive_small_inputs(self):
        a = Arena()
        x, y = a.opaque('x', 8), a.opaque('y', 8)
        for mask_x, mask_y in ((1, 1), (3, 7), (0x81, 0x85), (0x0f, 0xf0)):
            xx = a.bin('and', x, a.const(mask_x, 8), 8)
            yy = a.bin('and', y, a.const(mask_y, 8), 8)
            for op in ('add', 'sub'):
                value = a.bin(op, xx, yy, 8)
                zero, one = a.zbits(value), a.obits(value)
                self.assertEqual(0, zero & one)
                for p in range(256):
                    for q in range(256):
                        expected = (((p & mask_x) + (q & mask_y)) if op == 'add'
                                    else ((p & mask_x) - (q & mask_y))) & 255
                        self.assertEqual(0, expected & zero)
                        self.assertEqual(one, expected & one)


if __name__ == '__main__':
    unittest.main()
