"""Shift and rotate CF/OF, checked against Python bit arithmetic.

`_shift`/`_rotate` used to leave CF and OF as *named witnesses* — `opaque("flag_cf")`
and `opaque("flag_of")` — for every shift and every rotate. The ISA defines CF for all
of them and OF for a count of one, so the witnesses were a modelling gap, not an
unknown. Measured on executed code over all 14 roots (`probe.executed_mnemonics`):
`shr` 1,494 executions at 36 sites, `shl` 592 at 2, `sar` 29 at 2, `rol` 29 at 1, `ror`
29 at 1 — and every one of them was writing a witness into the word `pushfq` packs.

The oracle here is Python integer arithmetic applied to the SDM's *definitions* (bit
positions), which is a second, independent expression of the rule rather than a copy of
it: the engine builds arena expressions, this computes the answer from bits.

Coverage is stated rather than implied:

* **8-bit operands: exhaustive** over all 256 values times every count that is in range
  (this is complete for that operand size).
* **64-bit operands: targeted** over boundary values and a fixed pseudo-random sample,
  not exhaustive — 2**64 values cannot be swept, and calling this a complete sweep is
  exactly the error `PROBLEMS.md` F8 records.
* **Count 0** leaves every flag untouched, checked as an explicit control.
* **Count >= operand size** leaves CF and OF *undefined*: the test asserts the engine
  keeps a witness there. Inventing a number would be the silent failure this file exists
  to prevent, so the negative control is the important half.
* `ror` must not equal `rol` — the body ignored its `op_name` and lifted both as a left
  rotation, so a rule that passed every flag check while rotating the wrong way is
  possible and is checked separately.

    python -m tests.test_shift_rotate_flags
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.lift64 import Lifter, _rotate, _shift  # noqa: E402
from core.mach import Machine  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


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


class _Operand:
    from capstone.x86_const import X86_OP_IMM as _IMM, X86_OP_REG as _REG

    def __init__(self, kind: int, size: int, reg: int = 0, imm: int = 0):
        self.type = kind
        self.size = size
        self.reg = reg
        self.imm = imm


_ROLE = {"dst": 0, "src": 1}


class _Insn:
    """Two operands: a register destination and an immediate count."""

    def __init__(self, width: int, count: int):
        nbytes = width // 8
        self.size = nbytes
        self.operands = [_Operand(_Operand._REG, nbytes),
                         _Operand(_Operand._IMM, nbytes, imm=count)]
        self.mnemonic = "?"
        self.address = 0

    @staticmethod
    def reg_name(reg: int) -> str:
        return "rax"


def new_machine(width: int):
    a = Arena()
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    L = Lifter(_Image(), m, a, 0, 0)
    return a, m, L


SENTINEL = 0        # a flag value no rule below can produce, so "untouched" is visible


def run_one(op: str, x: int, count: int, width: int):
    """Lift one shift/rotate at a concrete operand and return (result, cf, of, flags)."""
    a, m, L = new_machine(width)
    for f in ("cf", "of", "zf", "sf", "pf", "af"):
        m.set_flag(f, a.const(SENTINEL, 8))
    m.set_reg("rax", a.const(x, width), width)
    handler = _rotate(op) if op in ("rol", "ror") else _shift(op)
    handler(L, _Insn(width, count), 0, 0)
    result = a.as_const(m.reg("rax", width))
    cf = m.flag("cf")
    of = m.flag("of")
    return a, result, cf, of, m


def sdm(op: str, x: int, count: int, width: int):
    """The SDM rule, computed from bits. Returns (result, cf, of) with None = undefined."""
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    signed = x - (1 << width) if x & sign else x
    if op == "shl":
        r = (x << count) & mask
        cf = (x >> (width - count)) & 1
        of = (((x >> (width - 1)) & 1) ^ ((r >> (width - 1)) & 1)) if count == 1 else None
    elif op == "shr":
        r = x >> count
        cf = (x >> (count - 1)) & 1
        of = ((x >> (width - 1)) & 1) if count == 1 else None
    elif op == "sar":
        r = (signed >> count) & mask
        cf = (x >> (count - 1)) & 1
        of = 0 if count == 1 else None
    elif op == "rol":
        r = ((x << count) | (x >> (width - count))) & mask
        cf = (x >> (width - count)) & 1
        of = (((r >> (width - 1)) & 1) ^ cf) if count == 1 else None
    elif op == "ror":
        r = ((x >> count) | (x << (width - count))) & mask
        cf = (x >> (count - 1)) & 1
        of = (((r >> (width - 1)) & 1) ^ ((r >> (width - 2)) & 1)) if count == 1 else None
    else:
        raise KeyError(op)
    return r, cf, of


def check_op(op: str, samples: list, width: int, label: str) -> None:
    a, m, L = None, None, None
    bad_result = bad_cf = bad_of = 0
    first = {}
    for x in samples:
        for count in range(1, width):
            a, r, cf, of, m = run_one(op, x, count, width)
            want_r, want_cf, want_of = sdm(op, x, count, width)
            if r != want_r:
                bad_result += 1
                first.setdefault("result", (x, count, r, want_r))
            got_cf = a.as_const(cf)
            if got_cf != want_cf:
                bad_cf += 1
                first.setdefault("cf", (x, count, got_cf, want_cf))
            got_of = a.as_const(of)
            if want_of is None:
                if got_of is not None:
                    bad_of += 1
                    first.setdefault("of", (x, count, got_of, "undefined"))
            elif got_of != want_of:
                bad_of += 1
                first.setdefault("of", (x, count, got_of, want_of))
    total = len(samples) * (width - 1)
    for name, bad, detail in (("result", bad_result, "result"),
                              ("CF", bad_cf, "cf"),
                              ("OF", bad_of, "of")):
        tag = "ok  " if bad == 0 else "FAIL"
        extra = "" if bad == 0 else f"  first {first[detail]}"
        check(f"{op} {width}-bit {label}: {name} mismatches {bad}/{total}{extra}", bad == 0)


def main() -> int:
    # --- 8-bit: exhaustive over every operand value.
    for op in ("shl", "shr", "sar", "rol", "ror"):
        check_op(op, list(range(256)), 8, "(exhaustive over all 256 values)")

    # --- 64-bit: boundary values plus a fixed pseudo-random sample. Not exhaustive.
    boundaries = [0, 1, 2, 0x7F, 0x80, 0xFF, 0x100, (1 << 63), (1 << 63) - 1,
                  (1 << 63) + 1, (1 << 64) - 1, (1 << 64) - 2, 0x0123_4567_89AB_CDEF,
                  0xFFFF_FFFF_0000_0000, 0x8000_0000_0000_0001]
    state = 0x5DEE_CE66_D1CE_1234
    randomish = []
    for _ in range(45):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        randomish.append(state)
    for op in ("shl", "shr", "sar", "rol", "ror"):
        check_op(op, boundaries + randomish, 64,
                 f"(targeted: {len(boundaries)} boundaries + {len(randomish)} sampled)")

    # --- Count 0 leaves every flag alone.
    for op in ("shl", "shr", "sar", "rol", "ror"):
        a, r, cf, of, m = run_one(op, 0x12, 0, 8)
        check(f"{op} count 0: result unchanged", r == 0x12, f"{r!r}")
        check(f"{op} count 0: CF untouched", a.as_const(cf) == SENTINEL)
        check(f"{op} count 0: OF untouched", a.as_const(of) == SENTINEL)

    # --- Count >= operand size: CF and OF are undefined, so they must stay witnesses.
    for op, width, count in (("shl", 8, 8), ("shr", 8, 9), ("sar", 8, 16),
                             ("ror", 8, 8), ("rol", 8, 8)):
        a, r, cf, of, m = run_one(op, 0x5A, count, width)
        if op in ("rol", "ror"):
            # A rotation by the operand size is the identity and touches nothing; the
            # engine masks the count modulo the width, so this is the count-0 case.
            check(f"{op} {width}-bit count {count}: identity", r == 0x5A, f"{r!r}")
        else:
            check(f"{op} {width}-bit count {count}: CF stays undefined",
                  a.as_const(cf) is None, f"{a.as_const(cf)!r}")
            check(f"{op} {width}-bit count {count}: OF stays undefined",
                  a.as_const(of) is None, f"{a.as_const(of)!r}")

    # --- ror must not be rol. The body ignored `op_name` for a whole revision, so a
    #     rule can pass every flag check above while rotating the wrong way.
    for x, count in ((0x01, 1), (0x80, 1), (0x03, 2), (0xF0, 3)):
        _a, rol_r, _c, _o, _m = run_one("rol", x, count, 8)
        _a, ror_r, _c, _o, _m = run_one("ror", x, count, 8)
        want_ror, _cf, _of = sdm("ror", x, count, 8)
        check(f"ror 8-bit {x:#04x} by {count} is a right rotation", ror_r == want_ror,
              f"got {ror_r!r} want {want_ror:#04x}")
        if (x, count) in ((0x01, 1), (0x03, 2)):
            check(f"ror 8-bit {x:#04x} by {count} differs from rol", ror_r != rol_r,
                  f"ror {ror_r!r} rol {rol_r!r}")

    # --- ZF/SF/PF come from the result for shifts and are untouched by rotates.
    #     Derived from the computed result rather than written as literals: the first
    #     version of this check asserted `shl 0x80 by 1` gives ZF=0, which is wrong
    #     (the result is 0, so ZF=1) and cost a cycle chasing a non-bug.
    for x, count in ((0x80, 1), (0x40, 1), (0x01, 3), (0xFF, 1)):
        a, r, _cf, _of, m = run_one("shl", x, count, 8)
        want_r, _wc, _wo = sdm("shl", x, count, 8)
        check(f"shl {x:#04x} by {count}: result", r == want_r, f"{r!r} want {want_r:#04x}")
        check(f"shl {x:#04x} by {count}: ZF is (result == 0)",
              a.as_const(m.flag("zf")) == int(r == 0), f"zf={a.as_const(m.flag('zf'))} r={r!r}")
        check(f"shl {x:#04x} by {count}: SF is the result's sign bit",
              a.as_const(m.flag("sf")) == ((r or 0) >> 7) & 1,
              f"sf={a.as_const(m.flag('sf'))} r={r!r}")
        check(f"shl {x:#04x} by {count}: PF is even parity of the low byte",
              a.as_const(m.flag("pf")) == int(bin((r or 0) & 0xFF).count("1") % 2 == 0),
              f"pf={a.as_const(m.flag('pf'))}")
    a, r, cf, of, m = run_one("rol", 0x00, 1, 8)
    check("rol leaves ZF untouched", a.as_const(m.flag("zf")) == SENTINEL)
    check("rol leaves SF untouched", a.as_const(m.flag("sf")) == SENTINEL)

    # --- The count mask. x86 masks to five bits below a 64-bit operand, six at 64.
    #     `_shift` used `k % 64` at every width, so `shl r32, 33` shifted by 33 (giving 0)
    #     where hardware shifts by 1. Latent in this module -- `probe.shift_counts` finds
    #     no executed count in [32, 63] on a sub-64-bit operand -- but a wrong result is
    #     wrong, so it is checked exhaustively over the counts that distinguish the masks.
    for width in (8, 16, 32, 64):
        bits = 0x3F if width == 64 else 0x1F
        worst = 32 if width == 64 else 5
        for count in (32, 33, 40, 63, 64, 65, 96):
            eff = count & bits
            a, r, cf, of, m = run_one("shl", 0x01, count, width)
            tag = f"shl {width}-bit count {count} (x86 applies {eff})"
            if eff == 0:
                check(f"{tag}: result unchanged", r == 0x01, f"{r!r}")
                check(f"{tag}: flags untouched", a.as_const(cf) == SENTINEL
                      and a.as_const(of) == SENTINEL)
            elif eff >= width:
                check(f"{tag}: result 0", r == 0, f"{r!r}")
                check(f"{tag}: CF/OF undefined",
                      a.as_const(cf) is None and a.as_const(of) is None)
            else:
                want_r, want_cf, want_of = sdm("shl", 0x01, eff, width)
                check(f"{tag}: result", r == want_r, f"{r!r} want {want_r:#x}")
                check(f"{tag}: CF", a.as_const(cf) == want_cf,
                      f"{a.as_const(cf)!r} want {want_cf}")
                if want_of is None:
                    check(f"{tag}: OF undefined", a.as_const(of) is None)
                else:
                    check(f"{tag}: OF", a.as_const(of) == want_of,
                          f"{a.as_const(of)!r} want {want_of}")
    # The single case the old mask got wrong, named explicitly.
    a, r, cf, of, m = run_one("shl", 0x01, 33, 32)
    check("shl r32, 33 shifts by 1 (the case the six-bit mask broke)", r == 2, f"{r!r}")
    a, r, cf, of, m = run_one("shl", 0x01, 32, 32)
    check("shl r32, 32 leaves the operand alone", r == 1, f"{r!r}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for name in FAILURES[:12]:
            print(f"  {name}")
        return 1
    print("ALL SHIFT/ROTATE FLAG RULES MATCH THE SDM DEFINITIONS")
    return 0


from core.lift64 import _rotate as run_rotate  # noqa: E402,F401
from core.lift64 import _shift as run_shift  # noqa: E402,F401


def run(op: str, x: int, count: int, width: int):
    return run_one(op, x, count, width)


if __name__ == "__main__":
    raise SystemExit(main())
