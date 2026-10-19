"""The `int3` guest-exit trap: recognised by shape, and refused without the shape.

Two builds spell the VM's "leave the virtualized function" stub differently:

* `SGuardSvc64.exe` ends it with `call qword ptr [slot]` through a runtime-bound pointer
  (`Lifter._runtime_bound_slot`);
* `SGuardAgent64.dll` ends it with a **bare `int3`**.

On `SGuardAgent64.dll` four blocks at handler `0x1800b30d7` run exactly 372 instructions and
stop at a single `0xCC`, and the reference terminates those same four as `return (372 steps)`
— four for four at the same step count. Classifying them `return` is what this rule does.

**The mnemonic alone must never be the test.** `int3` in ordinary code is a breakpoint or a
patch point; calling it a return would invent a function exit wherever a trap appears. So the
rule requires the three instructions after the `int3` to be the VM re-entry prologue:

    lea rsp, [rsp - IMM]
    mov [rsp + 0x10], rbp
    mov rbp, rsp

The negative controls here are the point of the file: bare `int3`, `int3` followed by
something else, a prologue based on `rbp` instead of `rsp`, a truncated prologue, and the
whole rule switched off must all stay `unsupported`.

    python -m tests.test_int3_exit
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402

FAILURES: list[str] = []

BASE = 0x1000


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- the bytes under test, assembled by hand and quoted here so the encoding is auditable.
INT3 = bytes.fromhex("cc")
# lea rsp, [rsp - 0xa70]
LEA_RSP = bytes.fromhex("488da42490f5ffff")
# lea rsp, [rbp - 0xa70]   -- same instruction, wrong base
LEA_RBP = bytes.fromhex("488da590f5ffff")
# mov [rsp + 0x10], rbp
MOV_SLOT_RBP = bytes.fromhex("48896c2410")
# mov rbp, rsp
MOV_RBP_RSP = bytes.fromhex("4889e5")
# nop, and a second, different trap
NOP = bytes.fromhex("90")
HLT = bytes.fromhex("f4")


class _Image:
    """A flat byte blob at one base, with the surface `Lifter` uses."""

    imports: dict = {}

    def __init__(self, blob: bytes):
        self.blob = blob

    def image_bytes(self, va: int, size: int) -> bytes | None:
        off = va - BASE
        if off < 0 or off >= len(self.blob):
            return None
        return self.blob[off:off + size]

    def image_u8(self, va: int) -> int | None:
        b = self.image_bytes(va, 1)
        return b[0] if b else None

    @staticmethod
    def section_for_va(va):
        return None

    @staticmethod
    def is_loader_bound(va, size):
        return False

    @staticmethod
    def foldable(va, size):
        return False


def run(blob: bytes, trap_enabled: bool = True):
    a = Arena()
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    L = Lifter(_Image(blob), m, a, BASE, BASE + len(blob))
    if not trap_enabled:
        L.guest_exit_trap = False
    return L.step(BASE)


PROLOGUE = LEA_RSP + MOV_SLOT_RBP + MOV_RBP_RSP


def main() -> int:
    print("the trap is recognised by shape")
    kind, stop = run(INT3 + PROLOGUE)
    check("int3 + full re-entry prologue -> stop", kind == "stop", f"{kind}")
    check("  kind is `return`", getattr(stop, "kind", None) == "return", f"{getattr(stop, 'kind', None)}")
    check("  the reason is named",
          "int3 guest-exit trap" in (getattr(stop, "text", "") or ""), f"{getattr(stop, 'text', '')}")
    check("  no successor is claimed", getattr(stop, "dest", 1) is None)

    print()
    print("negative controls -- every one must stay `unsupported`")
    negatives = [
        ("bare int3, nothing after it", INT3),
        ("int3 then a nop", INT3 + NOP),
        ("int3 then a different trap", INT3 + HLT),
        ("int3 then a prologue on the wrong base", INT3 + LEA_RBP + MOV_SLOT_RBP + MOV_RBP_RSP),
        ("int3 then a truncated prologue (2 of 3)", INT3 + LEA_RSP + MOV_SLOT_RBP),
        ("int3 then only the frame adjustment", INT3 + LEA_RSP + NOP + NOP),
        ("a real trap with no int3: hlt + prologue", HLT + PROLOGUE),
    ]
    for label, blob in negatives:
        kind, stop = run(blob)
        check(label, getattr(stop, "kind", None) == "unsupported",
              f"{getattr(stop, 'kind', None)}")

    print()
    print("the switch turns the rule off")
    kind, stop = run(INT3 + PROLOGUE, trap_enabled=False)
    check("int3 + prologue with guest_exit_trap=False", getattr(stop, "kind", None) == "unsupported",
          f"{getattr(stop, 'kind', None)}")

    print()
    print("a different frame size still matches (0xaa0, the other build's frame)")
    lea_aa0 = bytes.fromhex("488da42460f5ffff")          # lea rsp, [rsp - 0xaa0]
    kind, stop = run(INT3 + lea_aa0 + MOV_SLOT_RBP + MOV_RBP_RSP)
    check("frame 0xaa0 prologue matches", getattr(stop, "kind", None) == "return",
          f"{getattr(stop, 'kind', None)}")

    print()
    print("the prologue's own slot displacement is not what is matched")
    slot_08 = bytes.fromhex("48896c2408")                # mov [rsp + 0x08], rbp
    kind, stop = run(INT3 + LEA_RSP + slot_08 + MOV_RBP_RSP)
    check("mov [rsp+0x08],rbp still matches", getattr(stop, "kind", None) == "return",
          f"{getattr(stop, 'kind', None)}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for name in FAILURES:
            print(f"  {name}")
        return 1
    print("ALL INT3 GUEST-EXIT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
