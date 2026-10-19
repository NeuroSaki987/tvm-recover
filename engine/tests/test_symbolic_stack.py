"""Native instruction fixtures for symbolic stack addresses and x86 ordering."""
import unittest
from types import SimpleNamespace

from core.expr import Arena
from core.lift64 import Lifter
from core.mach import Machine

BASE = 0x1000


class Image:
    imports = {}

    def __init__(self, code):
        self.code = bytes.fromhex(code)

    def image_bytes(self, va, size):
        off = va - BASE
        return self.code[off:off + size] if 0 <= off < len(self.code) else None

    def image_u8(self, va):
        b = self.image_bytes(va, 1)
        return b[0] if b else None

    def section_for_va(self, va):
        return SimpleNamespace(executable=True, writable=False, end=BASE + 0x1000) if BASE <= va < BASE + 0x1000 else None

    def is_loader_bound(self, va, size):
        return False

    def foldable(self, va, size):
        return BASE <= va < BASE + len(self.code)


class SymbolicStackTests(unittest.TestCase):
    def make(self, code):
        a = Arena()
        m = Machine(a, 0x8000)
        p = a.opaque('stack.address', 64)
        m.set_reg('rsp', p, 64)
        return a, m, Lifter(Image(code), m, a, BASE, BASE + 0x1000), p

    def next(self, lifter, ip=BASE):
        kind, result = lifter.step(ip)
        self.assertEqual(kind, 'next', str(result))
        return result

    def test_push_pop_symbolic_register_roundtrip(self):
        a, m, L, p = self.make('50 5b')
        value = m.reg('rax', 64)
        ip = self.next(L)
        self.next(L, ip)
        self.assertEqual(m.reg('rbx', 64), value)
        self.assertEqual(m.reg('rsp', 64), p)

    def test_push_rsp_reads_before_decrement(self):
        a, m, L, p = self.make('54 58')
        ip = self.next(L)
        self.next(L, ip)
        self.assertEqual(m.reg('rax', 64), p)
        self.assertEqual(m.reg('rsp', 64), p)

    def test_push_memory_reads_before_decrement(self):
        a, m, L, p = self.make('ff3424 58')
        value = a.const(0x1234, 64)
        m.store(p, value, 8, 0)
        ip = self.next(L)
        self.next(L, ip)
        self.assertEqual(m.reg('rax', 64), value)

    def test_pop_rsp_destination_overrides_increment(self):
        a, m, L, p = self.make('5c')
        value = a.opaque('new.stack', 64)
        m.store(p, value, 8, 0)
        self.next(L)
        self.assertEqual(m.reg('rsp', 64), value)

    def test_pop_memory_uses_incremented_rsp(self):
        a, m, L, p = self.make('8f0424')
        value = a.const(0x55667788, 64)
        m.store(p, value, 8, 0)
        self.next(L)
        dest = a.bin('add', p, a.const(8, 64), 64)
        self.assertEqual(m.reg('rsp', 64), dest)
        self.assertEqual(m.load(dest, 8, 0, lambda _: None), value)

    def test_operand16_push_pop_preserves_upper_register(self):
        a, m, L, p = self.make('6650 665b')
        m.set_reg('rax', a.const(0x12345678abcd, 64), 64)
        m.set_reg('rbx', a.const(0x9988776655443322, 64), 64)
        ip = self.next(L)
        self.next(L, ip)
        self.assertEqual(a.as_const(m.reg('rbx', 64)), 0x998877665544abcd)
        self.assertEqual(m.reg('rsp', 64), p)

    def test_pushfq_popfq_roundtrip(self):
        a, m, L, p = self.make('9c 9d')
        for name in m.flags:
            m.set_flag(name, a.const(1, 8))
        saved = {name: m.flag(name) for name in m.flags}
        ip = self.next(L)
        m.set_flags_unknown()
        self.next(L, ip)
        self.assertEqual({name: m.flag(name) for name in m.flags}, saved)
        self.assertEqual(m.reg('rsp', 64), p)

    def test_leave_uses_symbolic_rbp_as_stack(self):
        a, m, L, p = self.make('c9')
        bp = a.opaque('frame.address', 64)
        old = a.opaque('old.frame', 64)
        m.set_reg('rbp', bp, 64)
        m.store(bp, old, 8, 0)
        self.next(L)
        self.assertEqual(m.reg('rbp', 64), old)
        self.assertEqual(m.reg('rsp', 64), a.bin('add', bp, a.const(8, 64), 64))

    def test_leave_operand16_pops_only_bp(self):
        a, m, L, p = self.make('66c9')
        bp = 0x1122334455660000
        m.set_reg('rbp', a.const(bp, 64), 64)
        m.store(a.const(bp, 64), a.const(0xabcd, 16), 2, 0)
        self.next(L)
        self.assertEqual(a.as_const(m.reg('rsp', 64)), bp + 2)
        self.assertEqual(a.as_const(m.reg('rbp', 64)), bp | 0xabcd)

    def test_aliasing_store_after_push_keeps_loaded_value_unknown(self):
        a, m, L, p = self.make('50 5b')
        ip = self.next(L)
        m.store(a.opaque('possibly.same.address', 64), a.const(1, 64), 8, 0)
        self.next(L, ip)
        self.assertIsNone(a.as_const(m.reg('rbx', 64)))
        self.assertNotEqual(m.reg('rbx', 64), m.reg('rax', 64))

    def test_ret_imm16_adjusts_after_return_address(self):
        a, m, L, p = self.make('c21000')
        m.store(p, a.const(BASE + 0x50, 64), 8, 0)
        self.assertEqual(self.next(L), BASE + 0x50)
        self.assertEqual(m.reg('rsp', 64), a.bin('add', p, a.const(24, 64), 64))

    def test_unknown_return_remains_explicit_stop(self):
        a, m, L, p = self.make('c3')
        expected = m.load(p, 8, 0, lambda _: None)
        kind, stop = L.step(BASE)
        self.assertEqual(kind, 'stop')
        self.assertEqual(stop.kind, 'return')
        self.assertEqual(stop.dest, expected)
        self.assertEqual(m.reg('rsp', 64), a.bin('add', p, a.const(8, 64), 64))

    def test_concrete_stack_arithmetic_wraps_mod64(self):
        a, m, L, p = self.make('50 5b')
        m.set_reg('rsp', a.const(0, 64), 64)
        ip = self.next(L)
        self.assertEqual(a.as_const(m.reg('rsp', 64)), (1 << 64) - 8)
        self.next(L, ip)
        self.assertEqual(a.as_const(m.reg('rsp', 64)), 0)

    def test_existing_stack_offset_cancels_after_roundtrip(self):
        a, m, L, p = self.make('50 5b')
        initial = a.bin('add', p, a.const(32, 64), 64)
        m.set_reg('rsp', initial, 64)
        ip = self.next(L)
        self.next(L, ip)
        self.assertEqual(m.reg('rsp', 64), initial)


if __name__ == '__main__':
    unittest.main()
