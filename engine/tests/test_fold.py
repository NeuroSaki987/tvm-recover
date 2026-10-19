"""Unit checks for the folding rules the recovery depends on.

Every one of these started life as a wrong answer somewhere in the walk, and each
is cheap enough to run in a second. They are grouped here because the same
failure mode keeps recurring: an expression that is *semantically* correct but
built in a shape the folder cannot see through. A wrong shape is invisible in
every value-level diagnostic, so it needs a shape-level test.

Usage: python -m tests.test_fold
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import W8, W16, W32, W64, Arena, render  # noqa: E402
from core.mach import FLAG_BITS, FLAGS, Machine  # noqa: E402

FAILS: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}")
    if not ok:
        print(f"        want {want!r}")
        FAILS.append(name)


def main() -> int:
    a = Arena()

    print("constant wrappers must fold")
    check("zext(trunc(0x3d0,16),64)", a.as_const(a.zext(a.trunc(a.const(0x3D0, W64), 16), W64)), 0x3D0)

    x = a.opaque("x", W64)
    print("width algebra")
    check("trunc32(zext64(trunc8(x)))",
          render(a, a.trunc(a.zext(a.trunc(x, 8), W64), W32)) in
          ("zext32(trunc8(x))", "zext32(trunc8((x and 0xff)))"), True)
    check("zext64(trunc8(zext64(y8)))" , render(a, a.zext(a.trunc(a.zext(a.opaque("y", W8), W64), 8), W64)), "zext64(y)")

    print("select must fold on a constant condition, at any width")
    # This is the bug that blocked the project for five rounds: the constant test
    # came after a width test that refused any condition that was not exactly one
    # byte, so a select whose condition had folded to a constant stayed unfolded
    # forever. The shape is a wall and every value-level diagnostic looks fine.
    q64 = a.opaque("q64", W64)
    q32 = a.opaque("q32", W32)
    for val, width in ((1, W64), (0, W64), (1, W32), (0, W8)):
        cond = a.const(val, width)
        got = a.select(cond, a.const(0x11, W64), a.const(0x22, W64), W64)
        check(f"select({val}, {width}-bit condition)",
              a.as_const(got), 0x11 if val else 0x22)

    print("select with equal arms must fold, at any condition width")
    # The same trap, one line further down: the width test returned early, so
    # `cond ? X : X` folded only when the condition happened to be one byte --
    # and the branch it returned from built the identical node anyway. Seventeen
    # dropped store addresses in one function rendered as
    # `trunc16((0xa770 xor ((1 and zext64(flag_af)) ? 0xa7f8 : 0xa7f8)))`: a
    # constant, held symbolic by a select whose arms are the same number.
    for cond in (q64, q32, a.bin("and", q64, a.const(1, W64), W64)):
        got = a.select(cond, q64, q64, W64)
        check(f"select(cond{W64}, X, X) is X", got == q64, True)
        got2 = a.select(cond, a.const(0xA7F8, W64), a.const(0xA7F8, W64), W64)
        check("select(cond, 0xa7f8, 0xa7f8) folds", a.as_const(got2), 0xA7F8)
        # and it must still be a real select when the arms differ
        got3 = a.select(cond, a.const(0xA7F8, W64), a.const(0xA7F9, W64), W64)
        check("select(cond, 0xa7f8, 0xa7f9) stays", a.as_const(got3), None)

    print("comparison normalisation")
    c = a.opaque("c", W32)
    check("(c + -0x40) == 0", render(a, a.bin("eq", a.bin("add", c, a.const(0xFFFFFFC0, W32), W32), a.const(0, W32), W32)), "(c eq 0x40)")

    print("flag round trip: unpack(pack(f)) must return f")
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    for flag in FLAGS:
        original = a.opaque(f"orig_{flag}", W8)
        m.set_flag(flag, original)
        packed = m.pack_flags()
        m.unpack_flags(packed)
        got = a.as_const(m.flag(flag))
        # A flag is 0/1; taking it out of the packed word must give the original
        # bit back, whichever node shape it lands in.
        rendered = render(a, m.flag(flag)) if got is None else f"{got:#x}"
        print(f"  .... {flag:<3} bit {FLAG_BITS[flag]:<3} -> {rendered[:90]}")
        if f"orig_{flag}" not in rendered:
            FAILS.append(f"flag round trip {flag}")

    print("bit projection")
    # bit 6 of a value must not be confused with bit 0.
    v = a.opaque("packed", W64)
    bit6 = a.bin("and", a.bin("shr", v, a.const(6, W64), W64), a.const(1, W64), W64)
    print(f"  .... and(shr(v,6),1) -> {render(a, bit6)[:90]}")
    bit0 = a.bin("and", v, a.const(1, W64), W64)
    print(f"  .... and(v,1)        -> {render(a, bit0)[:90]}")
    if render(a, bit6) == render(a, bit0):
        FAILS.append("bit6 == bit0")

    print("shifts must stay shifts")
    # `shr(v, 6)` is not `v & 6`. A mis-indented rule once made it so.
    sh = a.bin("shr", v, a.const(6, W64), W64)
    rendered = render(a, sh)
    check("shr(v,6) shape", "shr 6" in rendered, True)
    check("shr(v,6) is not a mask", rendered == render(a, a.bin("and", v, a.const(6, W64), W64)), False)

    print("AF and PF must be computed, not opaque")
    from core.lift64 import Lifter  # noqa: E402

    class _NoImg:
        def image_u8(self, va):
            return None

        imports = {}

    L = Lifter(_NoImg(), Machine(a, 0x1000), a, 0, 0)
    # AF is the carry out of bit 3. 0xff + 1 does carry out of bit 3 (the low
    # nibble overflows), so AF is 1 -- the first version of this test asserted 0
    # and was itself wrong.
    for x, y, want_af in ((0x0F, 0x01, 1), (0x0E, 0x01, 0), (0xFF, 0x01, 1), (0x08, 0x08, 1),
                          (0x07, 0x01, 0), (0xF0, 0x10, 0)):
        cx, cy = a.const(x, W64), a.const(y, W64)
        r = a.bin("add", cx, cy, W64)
        L._add_flags(cx, cy, r, W64)
        got = a.as_const(L.m.flag("af"))
        check(f"AF({x:#x}+{y:#x})", got, want_af)
    for value, want_pf in ((0x00, 1), (0x01, 0), (0x03, 1), (0xFF, 1)):
        cv = a.const(value, W64)
        L._zf_sf_pf(cv, W64)
        check(f"PF({value:#x})", a.as_const(L.m.flag("pf")), want_pf)
        check(f"ZF({value:#x})", a.as_const(L.m.flag("zf")), 1 if value == 0 else 0)

    print()
    if FAILS:
        print(f"FAILURES: {len(FAILS)}")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL FOLD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
