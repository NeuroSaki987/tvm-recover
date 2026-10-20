"""Regression coverage for swap cancellation and width-changing NOT expressions."""

import capstone
import pytest

from core.expr import Arena
from core.lift64 import Lifter, _h_xchg
from core.mach import Machine
from core.mba import simplify
from core.state import MachineSnapshot


class Image:
    imports = {}


def swaps(code, rax, rbx):
    arena = Arena()
    machine = Machine(arena, 0x100000)
    machine.set_reg("rax", arena.const(rax, 64))
    machine.set_reg("rbx", arena.const(rbx, 64))
    lifter = Lifter(Image(), machine, arena, 0, 0)
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoder.detail = True
    for insn in decoder.disasm(bytes.fromhex(code), 0x1000):
        _h_xchg(lifter, insn, insn.address, insn.address + insn.size)
    return arena.as_const(machine.reg("rax")), arena.as_const(machine.reg("rbx"))


def test_reversed_operands_restore_each_original_register():
    # xchg rax, rbx ; xchg rbx, rax
    assert swaps("48 87 d8 48 87 c3", 0x11, 0x22) == (0x11, 0x22)


def test_two_32_bit_swaps_still_clear_upper_halves():
    assert swaps("93 93", 0x1234567800000011, 0xabcdef0000000022) == (0x11, 0x22)


def test_high_and_low_byte_swaps_are_different_pairs():
    # xchg al, bl ; xchg ah, bh
    assert swaps("86 d8 86 fc", 0x1122, 0x3344) == (0x3344, 0x1122)


def test_rollback_drops_swaps_from_discarded_execution():
    arena = Arena()
    machine = Machine(arena, 0x100000)
    machine.set_reg("rax", arena.const(0x11, 64))
    machine.set_reg("rbx", arena.const(0x22, 64))
    lifter = Lifter(Image(), machine, arena, 0, 0)
    snapshot = MachineSnapshot(machine)
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoder.detail = True
    insn = next(decoder.disasm(bytes.fromhex("48 87 d8"), 0x1000))
    _h_xchg(lifter, insn, insn.address, insn.address + insn.size)
    snapshot.restore(machine)
    _h_xchg(lifter, insn, insn.address, insn.address + insn.size)
    assert arena.as_const(machine.reg("rax")) == 0x22
    assert arena.as_const(machine.reg("rbx")) == 0x11


@pytest.mark.parametrize("extension, expected", [("zext", 0xff00), ("sext", 0)])
def test_not_after_extension_of_not_keeps_original_semantics(extension, expected):
    arena = Arena()
    value = arena.const(0, 8)
    # Build the unreduced form to exercise the new rebuild path directly.
    inverted = arena._add(("un", "not", value, 8))
    extended = arena._add((extension, inverted, 16))
    expression = arena._add(("un", "not", extended, 16))
    assert arena.as_const(simplify(arena, expression)) == expected
