"""AMD64 instruction prefixes that Capstone exposes as mnemonic text."""

import unittest
from types import SimpleNamespace

from core.expr import Arena
from core.lift64 import Lifter
from core.mach import Machine
from tests.test_state_repair import stub_image


class BndPrefixTests(unittest.TestCase):
    def lifter(self, raw: bytes):
        a = Arena()
        m = Machine(a, 0x10000)
        image = stub_image(
            image_bytes=lambda ip, size: raw if ip == 0x1000 else b"",
        )
        return a, m, Lifter(image, m, a, 0x1000, 0x2000)

    def test_bnd_jne_keeps_the_underlying_conditional_branch(self):
        # Capstone spells the MPX-prefixed form as "bnd jne". The prefix does
        # not change which successor JNE selects.
        a, m, lifter = self.lifter(bytes.fromhex("f2 0f 85 02 01 00 00"))

        for zf, expected in ((1, 0x1007), (0, 0x1109)):
            with self.subTest(zf=zf):
                lifter.boxed.clear()
                m.set_flag("zf", a.const(zf, 8))
                self.assertEqual(("next", expected), lifter.step(0x1000))
                self.assertEqual([], lifter.boxed)

    def test_bnd_ret_keeps_the_underlying_return(self):
        a, m, lifter = self.lifter(bytes.fromhex("f2 c3"))
        m.set_reg("rsp", a.const(0x9000, 64), 64)
        m.store(a.const(0x9000, 64), a.const(0x1100, 64), 8, 0x1000)

        self.assertEqual(("next", 0x1100), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(0x9008, a.as_const(m.reg("rsp", 64)))

    def test_repz_ret_keeps_the_underlying_return(self):
        a, m, lifter = self.lifter(bytes.fromhex("f3 c3"))
        m.set_reg("rsp", a.const(0x9000, 64), 64)
        m.store(a.const(0x9000, 64), a.const(0x1100, 64), 8, 0x1000)

        self.assertEqual(("next", 0x1100), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(0x9008, a.as_const(m.reg("rsp", 64)))

    def test_notrack_indirect_jump_keeps_the_underlying_transfer(self):
        a, m, lifter = self.lifter(bytes.fromhex("3e ff 20"))
        m.set_reg("rbx", a.const(0x9000, 64), 64)
        m.store(a.const(0x9000, 64), a.const(0x1100, 64), 8, 0x1000)

        kind, stop = lifter.step(0x1000)
        self.assertEqual("stop", kind)
        self.assertEqual("dispatch", stop.kind)
        self.assertEqual([], lifter.boxed)


class LockCmpxchgTests(unittest.TestCase):
    def test_lock_cmpxchg_updates_destination_accumulator_and_compare_flags(self):
        a = Arena()
        m = Machine(a, 0x10000)
        image = stub_image(
            image_bytes=lambda ip, size: (
                bytes.fromhex("f0 48 0f b1 0d ce 30 11 00")
                if ip == 0x1000 else b""
            ),
        )
        lifter = Lifter(image, m, a, 0x1000, 0x2000)
        target = 0x1140D7

        cases = (
            (0x1111111111111111, 0x1111111111111111, 0x2222222222222222,
             {"cf": 0, "pf": 1, "af": 0, "zf": 1, "sf": 0, "of": 0}),
            (0x10, 0x20, 0x30,
             {"cf": 1, "pf": 1, "af": 0, "zf": 0, "sf": 1, "of": 0}),
        )
        for accumulator_value, memory_value, replacement, expected_flags in cases:
            with self.subTest(accumulator_value=accumulator_value):
                m.set_reg("rax", a.const(accumulator_value, 64), 64)
                m.set_reg("rcx", a.const(replacement, 64), 64)
                m.store(a.const(target, 64), a.const(memory_value, 64), 8, 0x1000)

                self.assertEqual(("next", 0x1009), lifter.step(0x1000))
                self.assertEqual([], lifter.boxed)
                equal = accumulator_value == memory_value
                self.assertEqual(
                    replacement if equal else memory_value,
                    a.as_const(m.load(a.const(target, 64), 8, 0x1009, lambda _: None)),
                )
                self.assertEqual(
                    accumulator_value if equal else memory_value,
                    a.as_const(m.reg("rax", 64)),
                )
                for flag, expected in expected_flags.items():
                    with self.subTest(flag=flag):
                        self.assertEqual(expected, a.as_const(m.flag(flag)))

    def test_hle_cmpxchg_prefixes_keep_the_atomic_semantics(self):
        for hint, raw in (
                ("xacquire", "f2 f0 48 0f b1 0b"),
                ("xrelease", "f3 f0 48 0f b1 0b")):
            with self.subTest(hint=hint):
                a = Arena()
                m = Machine(a, 0x10000)
                encoded = bytes.fromhex(raw)
                image = stub_image(
                    image_bytes=lambda ip, size, encoded=encoded: (
                        encoded if ip == 0x1000 else b""
                    ),
                )
                lifter = Lifter(image, m, a, 0x1000, 0x2000)
                m.set_reg("rax", a.const(0x10, 64), 64)
                m.set_reg("rbx", a.const(0x9000, 64), 64)
                m.set_reg("rcx", a.const(0x20, 64), 64)
                m.store(a.const(0x9000, 64), a.const(0x10, 64), 8, 0x1000)

                self.assertEqual(("next", 0x1006), lifter.step(0x1000))
                self.assertEqual([], lifter.boxed)
                self.assertEqual(
                    0x20,
                    a.as_const(
                        m.load(a.const(0x9000, 64), 8, 0x1006, lambda _: None)
                    ),
                )


class BitTestTests(unittest.TestCase):
    def test_bt_memory_immediate_records_the_selected_bit_in_cf(self):
        a = Arena()
        m = Machine(a, 0x10000)
        image = stub_image(
            image_bytes=lambda ip, size: (
                bytes.fromhex("0f ba 25 64 d8 05 00 01")
                if ip == 0x1000 else b""
            ),
        )
        lifter = Lifter(image, m, a, 0x1000, 0x2000)
        target = 0x5E86C
        m.store(a.const(target, 64), a.const(1 << 1, 32), 4, 0x1000)
        m.set_flag("cf", a.const(0, 8))

        self.assertEqual(("next", 0x1008), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(1, a.as_const(m.flag("cf")))
        for flag in ("pf", "af", "zf", "sf", "of"):
            with self.subTest(flag=flag):
                self.assertIsNone(a.as_const(m.flag(flag)))


class PunpcklBwTests(unittest.TestCase):
    def test_punpcklbw_interleaves_the_low_eight_bytes_of_both_halves(self):
        a = Arena()
        m = Machine(a, 0x10000)
        image = stub_image(
            image_bytes=lambda ip, size: bytes.fromhex("66 0f 60 c0")
            if ip == 0x1000 else b"",
        )
        lifter = Lifter(image, m, a, 0x1000, 0x2000)
        m.xmm[("xmm0", 0)] = a.const(0x0706050403020100, 64)
        m.xmm[("xmm0", 1)] = a.const(0x0F0E0D0C0B0A0908, 64)

        self.assertEqual(("next", 0x1004), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(0x0303020201010000, a.as_const(m.xmm[("xmm0", 0)]))
        self.assertEqual(0x0707060605050404, a.as_const(m.xmm[("xmm0", 1)]))


class Pe32SharedHandlerTests(unittest.TestCase):
    def lifter(self, raw: bytes):
        a = Arena()
        m = Machine(a, 0x10000, bits=32)
        image = SimpleNamespace(
            is64=False,
            imports={},
            image_u8=lambda addr: None,
            image_bytes=lambda ip, size: raw if ip == 0x1000 else b"",
            foldable=lambda addr, size: False,
            section_for_va=lambda addr: SimpleNamespace(
                executable=True, writable=True, end=0x2000,
            ),
            is_loader_bound=lambda addr, size: False,
        )
        return a, m, Lifter(image, m, a, 0x1000, 0x2000)

    def test_i386_lock_cmpxchg_uses_the_same_semantics(self):
        a, m, lifter = self.lifter(bytes.fromhex("f0 0f b1 0b"))
        m.set_reg("eax", a.const(0x10, 32), 32)
        m.set_reg("ebx", a.const(0x9000, 32), 32)
        m.set_reg("ecx", a.const(0x20, 32), 32)
        m.store(a.const(0x9000, 64), a.const(0x10, 64), 4, 0)

        self.assertEqual(("next", 0x1004), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(
            0x20, a.as_const(m.load(a.const(0x9000, 64), 4, 0x1004, lambda _: None))
        )
        self.assertEqual(0x10, a.as_const(m.reg("eax", 32)))
        self.assertEqual(1, a.as_const(m.flag("zf")))

    def test_i386_bt_uses_the_same_bit_test(self):
        a, m, lifter = self.lifter(bytes.fromhex("0f ba 23 01"))
        m.set_reg("ebx", a.const(0x9000, 32), 32)
        m.store(a.const(0x9000, 64), a.const(2, 64), 4, 0)

        self.assertEqual(("next", 0x1004), lifter.step(0x1000))
        self.assertEqual([], lifter.boxed)
        self.assertEqual(1, a.as_const(m.flag("cf")))


if __name__ == "__main__":
    unittest.main()
