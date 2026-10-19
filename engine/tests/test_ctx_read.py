"""Acceptance tests for expressing context reads from the propagated register file.

The listing no longer loads a context slot from an absolute frame address: a read is the *name* of the
value the slot holds, or for a narrow read a truncation of it. The failure mode that matters is a
narrow read landing *inside* the slot -- `_guest_slot` accepts byte and word slices, so four bytes at
the slot base are the low half of the register and four bytes four in are the **high** half. Emitting
the same expression for both reports the wrong half, which is a wrong value rather than a missing one.

    python -m tests.test_ctx_read
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.guestctx import GuestContext  # noqa: E402
from core.lower import Lowering  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


@dataclass
class FakeBlock:
    key: str = "A"
    guest_trace: list = field(default_factory=list)


def read_at(ctx_base: int, inside: int, nbytes: int):
    """A block whose single event is a context read at `ctx_base + inside`.

    The payload order is the lifter's and is easy to get wrong: `(site, reg, value, addr, nbytes)`.
    The first version of this test wrote it as `(site, reg, addr, nbytes, value)` and the lowering
    then read the *width* as the address, which surfaced as an `IndexError` out of `as_const` -- the
    test was wrong, not the code, and it is spelled out here so the next reader does not repeat it.
    """
    a = Arena()
    ctx = GuestContext(ctx_base)
    lo = Lowering(a, {}, ctx=ctx, entry={"rax": "guest_rax_in"})
    addr = a.const(ctx_base + inside, 64)
    value = a.opaque("loaded", nbytes * 8)
    block = FakeBlock(guest_trace=[(1, "load", (0x2000, "rax", value, addr, nbytes))])
    lines = lo.events(block)
    return a, lo, lines


def test_full_width_read_is_a_bare_name() -> None:
    _a, lo, lines = read_at(0x8000, 0, 8)
    check("an eight-byte read is the name itself",
          lines == ["guest_rax_in   ; read from ctx @0x2000   [ctx+0x00 = rax]"], str(lines))
    check("and it counts as a state read, not a memory read",
          lo.state_reads == 1 and lo.memory_reads == 0,
          f"{lo.state_reads}/{lo.memory_reads}")


def test_low_half_read_is_a_plain_truncation() -> None:
    _a, _lo, lines = read_at(0x8000, 0, 4)
    check("four bytes at the slot base is the low half",
          lines == ["v1:32 = trunc32(guest_rax_in)   ; read from ctx @0x2000"
                    "   [ctx+0x00 = rax]"], str(lines))


def test_high_half_read_shifts_before_truncating() -> None:
    """The case the first version got wrong, and the reason it is a named check."""
    _a, _lo, lines = read_at(0x8000, 4, 4)
    check("four bytes four into the slot is the HIGH half",
          lines == ["v1:32 = trunc32((guest_rax_in >> 32))   ; read from ctx @0x2000"
                    "   [ctx+0x04 = rax+4]"], str(lines))
    check("so it is not spelled the same as the low half",
          "trunc32(guest_rax_in)" not in (lines[0] if lines else ""), str(lines))


def test_other_in_slot_offsets() -> None:
    _a, _lo, lines = read_at(0x8000, 6, 2)
    check("two bytes six into the slot shifts by 48",
          lines == ["v1:16 = trunc16((guest_rax_in >> 48))   ; read from ctx @0x2000"
                    "   [ctx+0x06 = rax+6]"], str(lines))
    _a, _lo, lines = read_at(0x8000, 1, 1)
    check("one byte one into the slot shifts by 8",
          lines == ["v1:8 = trunc8((guest_rax_in >> 8))   ; read from ctx @0x2000"
                    "   [ctx+0x01 = rax+1]"], str(lines))


def test_the_eight_byte_path_is_guarded_by_the_lifter_not_here() -> None:
    """Why the full-width path does not re-check the in-slot offset, stated as a fact.

    An eight-byte read four bytes into a slot would span two registers, and the lowering would name it
    after one of them -- but the lifter never emits such an event: `Lifter._guest_slot` returns a
    register only when `inside + nbytes <= 8`, so `nbytes == 8` implies `inside == 0`. The guard could
    not live here anyway, because most real `load` events carry a *symbolic* address whose in-slot
    offset is unknown -- requiring it would refuse every one of them, and the listing would go back to
    loading 64 context slots from absolute frame addresses. So the invariant is asserted instead of an
    unreachable input: `reg_at` maps exactly one address per slot, so a slot base is the only address an
    eight-byte read can start at without leaving the slot.
    """
    ctx = GuestContext(0x8000)
    check("a slot base is the only address reg_at maps for rax",
          ctx.reg_at(0x8000) == "rax" and ctx.reg_at(0x8004) is None, str(ctx.reg_at(0x8004)))
    check("and the next slot begins eight bytes later",
          ctx.reg_at(0x8008) == "rbx", str(ctx.reg_at(0x8008)))


def test_a_narrow_read_is_not_spelled_as_the_whole_register() -> None:
    """The property behind all of the above: a narrow read must never print as the register name."""
    for inside in (0, 2, 4, 6):
        _a, _lo, lines = read_at(0x8000, inside, 4 if inside in (0, 4) else 2)
        text = lines[0] if lines else ""
        check(f"offset {inside}: the line is a truncation, not a bare name",
              "trunc" in text, text)


def main() -> int:
    for fn in (test_full_width_read_is_a_bare_name,
               test_low_half_read_is_a_plain_truncation,
               test_high_half_read_shifts_before_truncating,
               test_other_in_slot_offsets,
               test_the_eight_byte_path_is_guarded_by_the_lifter_not_here,
               test_a_narrow_read_is_not_spelled_as_the_whole_register):
        print(f"=== {fn.__name__} ===")
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL CONTEXT-READ CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
