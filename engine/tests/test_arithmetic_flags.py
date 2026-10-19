"""Every arithmetic flag rule, checked exhaustively on 8-bit operands.

Flags are not decoration here: `pushfq` packs all seven into one word, the VM stores
that word and reads bits out of it to choose its next handler, so a wrong CF/OF/AF is a
wrong *dispatch*. `evidence/E-GUEST-STATE-CLOSURE.md` leaves `rax` and `rflags` outside
the closed subsystem precisely because of flag unknowns, which makes the question worth
settling exhaustively rather than by inspection:

* `add`/`sub` over all 65,536 ordered pairs of bytes,
* `inc`/`dec` over all 256 bytes,
* against Python arithmetic, with the ISA's own definitions:

  - `CF_add = 1` iff the unsigned sum does not fit
  - `CF_sub = 1` iff `x < y` (a borrow)
  - `OF_add = 1` iff sign(x) == sign(y) and sign(result) != sign(x)
  - `OF_sub = 1` iff sign(x) != sign(y) and sign(result) != sign(x)
  - `AF      = bit 4 of (x ^ y ^ result)`
  - `INC`/`DEC` leave CF untouched (Intel SDM) and define the other five

Exhaustive on the 8-bit domain is complete for these rules: the engine's expressions are
width-parametric, and the widths are checked reflectively, but the *predicate* is decided
here for every pair of byte values.

    python -m tests.test_arithmetic_flags
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402

FAILURES: list[str] = []


class _Image:
    imports: dict = {}

    @staticmethod
    def section_for_va(va):
        return None

    @staticmethod
    def is_loader_bound(va, size):
        return False

    @staticmethod
    def image_u8(va):
        return None

    @staticmethod
    def foldable(va, size):
        return True


def evaluator(width: int = 8):
    a = Arena()
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    L = Lifter(_Image(), m, a, 0, 0)
    return a, m, L


def read_flags(a: Arena, m: Machine, names) -> dict:
    out = {}
    for f in names:
        out[f] = a.as_const(m.flag(f))
    return out


def parity_even(v: int) -> int:
    return int(bin(v & 0xFF).count("1") % 2 == 0)


def want_add(x: int, y: int, width: int = 8):
    mask = (1 << width) - 1
    s = x + y
    r = s & mask
    sign = lambda v: (v >> (width - 1)) & 1
    return dict(
        result=r,
        cf=int(s > mask),
        of=int(sign(x) == sign(y) and sign(r) != sign(x)),
        af=((x ^ y ^ r) >> 4) & 1,
        zf=int(r == 0),
        sf=sign(r),
        pf=parity_even(r),
    )


def want_sub(x: int, y: int, width: int = 8):
    mask = (1 << width) - 1
    r = (x - y) & mask
    sign = lambda v: (v >> (width - 1)) & 1
    return dict(
        result=r,
        cf=int(x < y),
        of=int(sign(x) != sign(y) and sign(r) != sign(x)),
        af=((x ^ y ^ r) >> 4) & 1,
        zf=int(r == 0),
        sf=sign(r),
        pf=parity_even(r),
    )


def main() -> int:
    width = 8
    mask = (1 << width) - 1
    for op in ("add", "sub"):
        a, m, L = evaluator(width)
        bad = {f: 0 for f in ("cf", "of", "af", "zf", "sf", "pf")}
        first = {}
        for x in range(256):
            for y in range(256):
                xn = a.const(x, width)
                yn = a.const(y, width)
                if op == "add":
                    r = a.bin("add", xn, yn, width)
                    L._add_flags(xn, yn, r, width)
                    want = want_add(x, y, width)
                else:
                    r = a.bin("sub", xn, yn, width)
                    L._sub_flags(xn, yn, r, width)
                    want = want_sub(x, y, width)
                got = read_flags(a, m, ("cf", "of", "af", "zf", "sf", "pf"))
                if a.as_const(r) != want["result"]:
                    bad.setdefault("result", 0)
                    bad["result"] += 1
                    first.setdefault("result", (x, y, a.as_const(r), want["result"]))
                for f in ("cf", "of", "af", "zf", "sf", "pf"):
                    if got[f] != want[f]:
                        bad[f] += 1
                        first.setdefault(f, (x, y, got[f], want[f]))
        total = 256 * 256
        print(f"{op}: {total} ordered byte pairs, exhaustive")
        for f, n in bad.items():
            tag = "ok  " if n == 0 else "FAIL"
            extra = "" if n == 0 else f"  first disagreement {first[f]}"
            print(f"  {tag} {f:<7} mismatches {n}/{total}{extra}")
            if n:
                FAILURES.append(f"{op}.{f}")

    # INC / DEC: CF preserved, the rest defined.
    for op in ("inc", "dec"):
        a, m, L = evaluator(width)
        bad = {f: 0 for f in ("cf", "of", "af", "zf", "sf", "pf")}
        first = {}
        for x in range(256):
            # CF carries a value that no recomputation can produce, so only a
            # preserved CF passes: 1 when x is even, 0 when odd.
            carried = x & 1
            m.set_flag("cf", a.const(carried, 8))
            xn = a.const(x, width)
            if op == "inc":
                r = a.bin("add", xn, a.const(1, width), width)
                L._add_flags(xn, a.const(1, width), r, width)
                m.set_flag("cf", a.const(carried, 8))
                want = want_add(x, 1, width)
            else:
                r = a.bin("sub", xn, a.const(1, width), width)
                L._sub_flags(xn, a.const(1, width), r, width)
                m.set_flag("cf", a.const(carried, 8))
                want = want_sub(x, 1, width)
            want["cf"] = carried
            got = read_flags(a, m, ("cf", "of", "af", "zf", "sf", "pf"))
            for f in ("cf", "of", "af", "zf", "sf", "pf"):
                if got[f] != want[f]:
                    bad[f] += 1
                    first.setdefault(f, (x, got[f], want[f]))
        print(f"{op}: 256 byte values, exhaustive, CF carried in as (x & 1)")
        for f, n in bad.items():
            tag = "ok  " if n == 0 else "FAIL"
            extra = "" if n == 0 else f"  first disagreement {first[f]}"
            print(f"  {tag} {f:<7} mismatches {n}/256{extra}")
            if n:
                FAILURES.append(f"{op}.{f}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} rule(s): {FAILURES}")
        return 1
    print("ALL ARITHMETIC FLAG RULES MATCH EXHAUSTIVE PYTHON ARITHMETIC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
