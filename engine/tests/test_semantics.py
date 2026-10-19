"""Regression tests for the semantic defects found in review.

Each case here is a **minimal reproduction** of a defect that shipped, taken from the review that
found it. They are deliberately small and synthetic: they do not need the module, the reference or
a recovery, so they run in milliseconds and can be run after every change. The CFG regression is
the right acceptance test for *recovery* changes; these are the acceptance test for *semantics*,
and the review's point is that the second kind was missing -- several of the defects below survived
`test_fold`, `test_subst` and the full 802/802 graph reproduction because none of those exercise
memory semantics or units.

    python -m tests.test_semantics

Cases, and what each one caught:

1. **Load width units.** `Arena.load` takes a bit width; a rendering path multiplied by 8 and
   printed `load512` for a 64-bit load, because two units were in play in one function (trace
   events carry bytes, arena nodes carry bits).
2. **Mutable reads are not stable.** `load(p); store(p, 5); load(p)` returned the *same* node, so
   the difference folded to zero -- a wrong constant, produced silently.
3. **Reads with no intervening store must still intern together.** The protector's self-cancelling
   dispatch arithmetic depends on it, so the fix for (2) must not cost it.
4. **Narrow stores are partial.** A 4-byte store into an 8-byte slot leaves the upper bytes as they
   were; recording it as a full register update claims a value the machine never holds.
5. **An identity mark is not a licence to name a computation.** A value derived from a guest
   register but not equal to it must render as the expression, not as `guest_rax`.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.mach import Machine  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def no_image(addr, nbytes):
    return None


def test_load_width_in_bits() -> None:
    a = Arena()
    n = a.load(a.opaque("p"), 32)
    check("load node keeps the width it was built with", a.node(n)[2] == 32, str(a.node(n)))
    check("width() returns 32, not the version",
          a.width(n) == 32, f"width()={a.width(n)}")
    n64 = a.load(a.opaque("q"), 64)
    check("a 64-bit load has width 64", a.width(n64) == 64, f"width()={a.width(n64)}")


def test_memory_version() -> None:
    a = Arena()
    m = Machine(a, 0)
    p = a.opaque("p")
    v1 = m.load(p, 8, 0, no_image)
    m.store(p, a.const(5, 64), 8, 0)
    v2 = m.load(p, 8, 0, no_image)
    check("a read after a store is a different value", v1 != v2)
    d = a.bin("sub", v1, v2, 64)
    check("their difference does NOT fold to a constant",
          a.as_const(d) is None, f"folded to {a.as_const(d)}")
    v3 = m.load(p, 8, 0, no_image)
    check("two reads with no store between still intern together", v2 == v3)
    d2 = a.bin("sub", v2, v3, 64)
    check("and their difference still folds to zero", a.as_const(d2) == 0)


def test_partial_write_semantics() -> None:
    a = Arena()
    m = Machine(a, 0)
    slot = 0x1000
    m.store(a.const(slot, 64), a.const(0x1122334455667788, 64), 8, 0)
    m.store(a.const(slot, 64), a.const(0xDEADBEEF, 64), 4, 0)
    got = 0
    for i in range(8):
        byte = m.mem.get(slot + i)
        got |= (byte if isinstance(byte, int) else 0) << (8 * i)
    check("a 4-byte write leaves the upper 4 bytes unchanged",
          got == 0x11223344DEADBEEF, f"slot = {got:#x}")
    check("and it is therefore NOT zext64 of the written value",
          got != 0xDEADBEEF, f"slot = {got:#x}")


def test_identity_is_not_derived() -> None:
    """A value *derived* from a guest register must not render as that register."""
    a = Arena()
    rax = a.opaque("guest_rax", 64)
    plus8 = a.bin("add", rax, a.const(8, 64), 64)
    identity = {rax: "rax"}          # only the register itself is an identity claim
    gv = {rax: "rax", plus8: "rax"}  # the liberal map marks the derived value too
    from core.lower import Lowering
    lo = Lowering(a, gv, None, identity)
    check("the register itself names as guest_rax",
          lo.name(rax) == "guest_rax", lo.name(rax))
    got = lo.name(plus8)
    check("the derived value is lowered as an expression, not as guest_rax",
          got != "guest_rax", got)
    # `name()` returns the operand text and appends the definition to `lo.lines`, so the check is
    # on the emitted line, not on the returned name -- the first version of this test asserted on
    # the return value and failed for that reason rather than for a code defect.
    emitted = " ".join(lo.lines)
    check("and the emitted line shows the addition",
          "guest_rax" in emitted and "8" in emitted, emitted)


def test_frame_separation() -> None:
    """A guest-only address cannot clobber the VM's private frame; a mixed one may.

    This is the escape assumption that closed the `0x140155dc1` coverage gap. The machine is now
    constructed with a **stack base**, because that is what the assumption is about: the frame is
    spared as part of the machine's own stack window, not as "whatever happens to be tracked". The
    frame cell here sits inside that window (`0x7FFFFFFEF000`, one page below the base) exactly as
    the measured one does -- see `probe/frameregion.py`. Both halves still matter: without the first
    the engine loses the frame and stops resolving dispatches, and without the second it *gains*
    precision it is not entitled to.
    """
    a = Arena()
    m = Machine(a, 0x7FFFFFFEF800)
    frame = a.const(0x7FFFFFFEF000, 64)
    m.store(frame, a.const(0xAB, 64), 8, 0)
    check("the VM's frame cell is tracked after a concrete store",
          m.mem.get(0x7FFFFFFEF000) == 0xAB, str(m.mem.get(0x7FFFFFFEF000)))
    m.store(a.init_reg("rcx"), a.const(0xCD, 64), 8, 0)
    check("a guest-only store leaves the frame cell intact",
          m.mem.get(0x7FFFFFFEF000) == 0xAB, str(m.mem.get(0x7FFFFFFEF000)))
    check("and it records the condition it relied on",
          m.isolation_conditions == [(0, 8, 8)], str(m.isolation_conditions))
    check("the spared set is the window, not the whole overlay",
          m.isolation_spared == {0x7FFFFFFEF000 + i for i in range(8)},
          f"{len(m.isolation_spared)} byte(s)")
    mixed = a.bin("add", a.opaque("unknown"), a.load(frame, 64), 64)
    m.store(mixed, a.const(0xEF, 64), 8, 0)
    check("a store whose address reads frame memory does clobber it",
          m.mem.get(0x7FFFFFFEF000) is None, str(m.mem.get(0x7FFFFFFEF000)))


def test_isolation_is_refutable() -> None:
    """The escape assumption is discharged against evidence, not left as a claim.

    `finite_addresses` resolving a guest-pointer store to concrete targets is the case that can
    *decide* it, so it is tested there: a proven target inside the machine's stack window withdraws
    every byte the assumption spared, and the frame cell stops being known.
    """
    a = Arena()
    m = Machine(a, 0x7FFFFFFEF800)
    base = 0x7FFFFFFEF000
    m.store(a.const(base, 64), a.const(0xAB, 64), 8, 0)
    m.store(a.init_reg("rcx"), a.const(0xCD, 64), 8, 0)
    check("the frame cell is spared before any evidence arrives",
          m.mem.get(base) == 0xAB, str(m.mem.get(base)))
    check("and no refutation is counted yet",
          m.isolation_report()["refuted"] == 0, str(m.isolation_report()))
    check("proving a guest address inside the window withdraws the exemption",
          m.refute_isolation(base + 0x80, 0x140000000))
    check("so the spared frame bytes are no longer known",
          m.mem.get(base) is None, str(m.mem.get(base)))
    check("and the withdrawal is counted and logged",
          m.isolation_report()["refuted"] == 1 and len(m.isolation_log) == 1,
          str(m.isolation_report()))
    check("an address outside the window is not a refutation",
          not m.refute_isolation(0x9000, 0x140000000))


def test_symbolic_store_reaches_other_memory() -> None:
    """A store through a guest pointer is not allowed to leave *other* memory stale.

    The review's counterexample, and the one thing the frame separation got wrong: it exempted
    **every** tracked byte rather than the VM's private frame, so a guest write through `rcx` left a
    byte written at `0x9000` untouched -- and a later read of `0x9000` returned the stale `0x11` as a
    *constant*. The escape assumption says a guest pointer cannot name the frame; it says nothing
    about the rest of memory, which is exactly where guest pointers do point.
    """
    a = Arena()
    m = Machine(a, 0)
    m.store(a.const(0x9000, 64), a.const(0x11, 64), 8, 0)
    rcx = a.init_reg("rcx")
    m.store(rcx, a.const(0x22, 64), 8, 0)
    got = m.load(a.const(0x9000, 64), 8, 0, no_image)
    folded = a.as_const(got)
    check("a read after a guest-pointer store is not the stale constant",
          folded != 0x11, f"read = {folded:#x}" if folded is not None else "read = symbolic")
    # The stronger form: when `rcx` *is* `0x9000` the read must be `0x22`. That is the conditional
    # answer the memory model owes, and it is checked by substituting the case into the result.
    if folded is None:
        on_case = a.as_const(a.substitute(got, {rcx: a.const(0x9000, 64)}))
        check("and with rcx = 0x9000 the read resolves to 0x22",
              on_case == 0x22, f"resolved to {on_case:#x}" if on_case is not None else "still symbolic")


def test_uncached_image_byte_is_conditioned_too() -> None:
    """Whether an address was ever cached must not change the answer.

    The first version of the guard wrote it into `mem` at store time, so only bytes that were
    *already cached* received a condition. The review reproduced the consequence: a store through a
    guest pointer followed by a read of an address that still held its image value returned the image
    value, even when the pointer provably equalled that address. Cached and image-initial bytes are
    two branches of one question and now go through one rule.
    """
    a = Arena()
    m = Machine(a, 0)
    target = 0xB000
    image = {target + i: 0x33 for i in range(8)}

    def image_u8(addr):
        return image.get(addr)

    rcx = a.init_reg("rcx")
    m.store(rcx, a.const(0x22, 64), 8, 0)
    got = m.load(a.const(target, 64), 8, 0, image_u8)
    folded = a.as_const(got)
    check("an uncached address is not answered from the image unconditionally",
          folded != 0x33, f"read = {folded:#x}" if folded is not None else "read = symbolic")
    if folded is None:
        on_case = a.as_const(a.substitute(got, {rcx: a.const(target, 64)}))
        check("and with the pointer equal to it the read resolves to the stored value",
              on_case == 0x22, f"resolved to {on_case:#x}" if on_case is not None else "still symbolic")


def test_readonly_bytes_are_not_conditioned() -> None:
    """A store cannot have written a mapped image section, so a read of one stays a constant.

    Without this the guard would turn every constant read out of the image into a `select` over a
    guest pointer, and the engine would lose far more than it gained -- so the rule is checked rather
    than assumed, on the same shape as the previous case with the read-only predicate flipped.
    """
    a = Arena()
    m = Machine(a, 0)
    m.readonly = lambda addr, size: True
    target = 0xB000

    def image_u8(addr):
        return 0x33 if addr < target + 8 else None

    m.store(a.init_reg("rcx"), a.const(0x22, 64), 8, 0)
    got = m.load(a.const(target, 64), 8, 0, image_u8)
    check("a read-only byte is still answered from the image",
          a.as_const(got) == 0x3333333333333333,
          f"read = {a.as_const(got)!r}")


def test_snapshot_restore_does_not_leave_a_stale_identity_map() -> None:
    """A withdrawn census mark must not go on deciding whether a store is isolated.

    The machine holds `guest_marks` -- the census's identity map. `CensusState.restore` used to
    *rebind* `lifter.guest_identity` to a fresh dict, so the machine went on consulting the old one,
    and a mark left by a rejected attempt or a sibling path could still make the memory model treat
    an address as a guest address and spare the frame. Restores now mutate in place, and this is the
    check: after the mark is withdrawn, the store must clobber the frame cell rather than spare it.
    """
    from core.explore import MachineSnapshot

    a = Arena()
    m = Machine(a, 0x7FFFFFFEF800)
    frame = 0x7FFFFFFEF000
    m.store(a.const(frame, 64), a.const(0xAB, 64), 8, 0)

    holder = {"marks": {}}                      # stands in for the lifter's identity map
    m.guest_marks = holder["marks"]
    pointer = a.load(a.opaque("guest_pointer"), 64)
    holder["marks"][pointer] = "rcx"
    check("the marked pointer is a guest address", m.guest_marks is holder["marks"])

    snap = MachineSnapshot(m)
    # The lifter withdraws the mark and the snapshot is restored -- the object must survive.
    holder["marks"].clear()
    snap.restore(m)
    check("the machine still holds the same map object",
          m.guest_marks is holder["marks"],
          f"same object: {m.guest_marks is holder['marks']}")
    m.store(pointer, a.const(0xCD, 64), 8, 0)
    check("with the mark withdrawn the store clobbers the frame instead of sparing it",
          m.mem.get(frame) is None, str(m.mem.get(frame)))


def main() -> int:
    for fn in (test_load_width_in_bits, test_memory_version,
               test_partial_write_semantics, test_identity_is_not_derived,
               test_frame_separation, test_isolation_is_refutable,
               test_symbolic_store_reaches_other_memory,
               test_uncached_image_byte_is_conditioned_too,
               test_readonly_bytes_are_not_conditioned,
               test_snapshot_restore_does_not_leave_a_stale_identity_map):
        print(f"=== {fn.__name__} ===")
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL SEMANTIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
