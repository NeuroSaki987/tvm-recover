"""Acceptance tests for the flags-assembly collapse.

The lowering replaces a hand-built RFLAGS word with one line, and the whole value of that depends on
the recognizer refusing everything it is not certain about. A recognizer that fires on a flags word
built from a call return would hide the one thing the reader needs to see. So the negative cases here
are the important ones.

    python -m tests.test_flags_emit
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.lower import FLAG_BITS, Lowering  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def build_pack(a: Arena, flags, base: int = 0x2):
    """The shape the VM actually emits: OR of `(1 & flag) << bit`, seeded with the fixed bit."""
    acc = a.const(base, 64)
    for flag in flags:
        leaf = a.opaque(f"init_{flag}", 8)
        one = a.bin("and", a.zext(leaf, 64), a.const(1, 64), 64)
        placed = a.bin("shl", one, a.const(FLAG_BITS[flag], 64), 64)
        acc = a.bin("or", acc, placed, 64)
    return acc


def test_a_real_pack_is_recognized() -> None:
    a = Arena()
    node = build_pack(a, ("cf", "pf", "af", "zf", "sf", "of"))
    lo = Lowering(a, {})
    got = lo.flags_pack(node)
    check("a hand-built flags word is recognized", got is not None, str(got))
    if got is None:
        return
    text, positions = got
    check("all six flags are named",
          all(f in text for f in ("cf", "pf", "af", "zf", "sf", "of")), text)
    check("and the bit positions are the architectural ones",
          positions == (0, 2, 4, 6, 7, 11), str(positions))
    check("it renders as one line", text.startswith("pack_flags("), text)


def test_the_collapse_is_equivalent_on_every_assignment() -> None:
    """The compact line is a claim of equivalence, so it is checked exhaustively."""
    a = Arena()
    flags = ("cf", "pf", "af", "zf", "sf", "of")
    node = build_pack(a, flags)
    lo = Lowering(a, {})
    got = lo.flags_pack(node)
    check("recognized", got is not None)
    if got is None:
        return
    _text, positions = got
    leaves = {f: [lf for lf in a.leaves(node)
                  if a.node(lf)[1] == f"init_{f}"][0] for f in flags}
    want_bits = {f: FLAG_BITS[f] for f in flags}
    check("the recognized positions are what the reconstruction uses",
          tuple(sorted(want_bits.values())) == positions, str(positions))
    bad = []
    for mask in range(1 << len(flags)):
        env = {leaves[f]: a.const((mask >> i) & 1, 8) for i, f in enumerate(flags)}
        got_v = a.as_const(a.substitute(node, env))
        want = 0x2
        for i, f in enumerate(flags):
            if (mask >> i) & 1:
                want |= 1 << want_bits[f]
        if got_v != want:
            bad.append((mask, got_v, want))
    check("every one of the 64 assignments reproduces the linear form",
          not bad, str(bad[:3]))


def test_a_non_flag_input_is_refused() -> None:
    """A flags word built from a call return must be emitted in full.

    This is the common case -- most of the flags words the census records depend on something other
    than a flag bit -- and collapsing one would hide the fact that the value is not known.
    """
    a = Arena()
    acc = a.const(0x2, 64)
    for flag in ("cf", "pf"):
        leaf = a.opaque(f"init_{flag}", 8)
        one = a.bin("and", a.zext(leaf, 64), a.const(1, 64), 64)
        acc = a.bin("or", acc, a.bin("shl", one, a.const(FLAG_BITS[flag], 64), 64), 64)
    acc = a.bin("or", acc, a.bin("and", a.opaque("call_ret#6", 64), a.const(0x400, 64), 64), 64)
    lo = Lowering(a, {})
    check("a pack with a call return in it is refused", lo.flags_pack(acc) is None)


def test_a_wrong_bit_position_is_refused() -> None:
    """A recognizer that only counts flags would accept this; the positions are the discriminator."""
    a = Arena()
    acc = a.const(0x2, 64)
    for flag, bit in (("cf", 0), ("pf", 5)):   # pf belongs at bit 2, not 5
        leaf = a.opaque(f"init_{flag}", 8)
        one = a.bin("and", a.zext(leaf, 64), a.const(1, 64), 64)
        acc = a.bin("or", acc, a.bin("shl", one, a.const(bit, 64), 64), 64)
    lo = Lowering(a, {})
    check("a flag at the wrong architectural bit is refused", lo.flags_pack(acc) is None)


def test_a_non_linear_combination_is_refused() -> None:
    """Two flags XORed into one position is not an assembly, and must not be rendered as one."""
    a = Arena()
    cf = a.opaque("init_cf", 8)
    zf = a.opaque("init_zf", 8)
    xored = a.bin("xor", a.zext(cf, 64), a.zext(zf, 64), 64)
    acc = a.bin("or", a.const(0x2, 64), a.bin("shl", xored, a.const(0, 64), 64), 64)
    lo = Lowering(a, {})
    check("a non-linear combination is refused", lo.flags_pack(acc) is None)


def test_an_unusual_base_is_reported_not_hidden() -> None:
    """If the constant seed is not the architectural 0x2, the line says so."""
    a = Arena()
    node = build_pack(a, ("cf", "zf"), base=0x6)
    lo = Lowering(a, {})
    got = lo.flags_pack(node)
    check("a different seed is still recognized", got is not None, str(got))
    if got is not None:
        check("and the seed is printed rather than folded away",
              "| 0x6" in got[0], got[0])


def main() -> int:
    for fn in (test_a_real_pack_is_recognized,
               test_the_collapse_is_equivalent_on_every_assignment,
               test_a_non_flag_input_is_refused,
               test_a_wrong_bit_position_is_refused,
               test_a_non_linear_combination_is_refused,
               test_an_unusual_base_is_reported_not_hidden):
        print(f"=== {fn.__name__} ===")
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL FLAGS-EMISSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
