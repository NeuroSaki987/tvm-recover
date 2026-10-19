"""Multiply high halves against Python's independent double-width arithmetic."""
import random
import unittest

from core.expr import Arena
from core.lift64 import Lifter
from core.mach import Machine
from tests.test_state_repair import stub_image


class WideMultiplyTests(unittest.TestCase):
    def test_symbolic_high_halves_match_double_width_arithmetic(self):
        from core.widemul import high_product
        rng = random.Random(0x4D554C)
        for width in (8, 16, 32, 64):
            a = Arena()
            x, y = a.opaque('x', width), a.opaque('y', width)
            unsigned = high_product(a, x, y, width)
            signed = high_product(a, x, y, width, signed=True)
            mask = (1 << width) - 1
            edge = [0, 1, 2, (1 << (width - 1)) - 1, 1 << (width - 1), mask - 1, mask]
            pairs = [(v, w) for v in edge for w in edge]
            pairs += [(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(1500)]
            for xv, yv in pairs:
                pins = {x: a.const(xv, width), y: a.const(yv, width)}
                sx = xv - (1 << width) if xv >> (width - 1) else xv
                sy = yv - (1 << width) if yv >> (width - 1) else yv
                self.assertEqual(((xv * yv) >> width) & mask, a.as_const(a.substitute(unsigned, pins)))
                self.assertEqual(((sx * sy) >> width) & mask, a.as_const(a.substitute(signed, pins)))

    def execute(self, code, rax, rcx, rdx=0x8877665544332211):
        a = Arena()
        m = Machine(a, 0x10000)
        image = stub_image(image_bytes=lambda addr, size: bytes.fromhex(code) if addr == 0x1000 else b'')
        l = Lifter(image, m, a, 0x1000, 0x2000)
        for r, v in [('rax', rax), ('rcx', rcx), ('rdx', rdx)]:
            m.set_reg(r, a.const(v, 64), 64)
        self.assertEqual('next', l.step(0x1000)[0])
        return a, m

    def test_mul64_has_exact_high_result_and_overflow_flags(self):
        a, m = self.execute('48 f7 e1', (1 << 64) - 1, 2)
        self.assertEqual(1, a.as_const(m.reg('rdx', 64)))
        self.assertEqual((1 << 64) - 2, a.as_const(m.reg('rax', 64)))
        self.assertEqual(1, a.as_const(m.flag('cf')))
        self.assertEqual(1, a.as_const(m.flag('of')))

    def test_signed_mul64_sign_extension_does_not_mean_overflow(self):
        a, m = self.execute('48 f7 e9', (1 << 64) - 3, 2)
        self.assertEqual((1 << 64) - 1, a.as_const(m.reg('rdx', 64)))
        self.assertEqual((1 << 64) - 6, a.as_const(m.reg('rax', 64)))
        self.assertEqual(0, a.as_const(m.flag('cf')))
        self.assertEqual(0, a.as_const(m.flag('of')))

    def test_mul8_writes_ax_and_leaves_rdx_untouched(self):
        a, m = self.execute('f6 e1', 0x11223344556677FF, 2)
        self.assertEqual(0x11223344556601FE, a.as_const(m.reg('rax', 64)))
        self.assertEqual(0x8877665544332211, a.as_const(m.reg('rdx', 64)))

    def test_imul_two_operand_overflow_is_exact(self):
        a, m = self.execute('48 0f af c1', 0x7FFFFFFFFFFFFFFF, 2)
        self.assertEqual(1, a.as_const(m.flag('cf')))
        self.assertEqual(1, a.as_const(m.flag('of')))
        # Undefined arithmetic flags cannot be inferred from the low product.
        for flag in ('sf', 'zf', 'af', 'pf'):
            self.assertIsNone(a.as_const(m.flag(flag)))


if __name__ == '__main__':
    unittest.main()
