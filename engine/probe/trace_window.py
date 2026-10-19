"""Instruction-level trace of one address range, with register provenance.

Written when a single site diverges from the reference and the aggregate numbers
cannot say why. It prints, for each instruction in the window, the concrete value
of a few registers before and after, plus how many loads the evaluation has had
to give up on. That is enough to tell "the VM really did compute this" from "the
lifter read a value it should not have been able to read".

Usage: python -m probe.trace_window <module> <root_va> <lo_va> <hi_va>
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

WATCH = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11")


def main() -> int:
    path, root, lo, hi = sys.argv[1], int(sys.argv[2], 0), int(sys.argv[3], 0), int(sys.argv[4], 0)
    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    a = Arena()
    m = Machine(a, 0x7FFF_FFFE_FDA0)
    L = Lifter(img, m, a, vm.va, vm.end)

    ip = root
    steps = 0
    while steps < 400_000:
        if lo <= ip < hi:
            size, insn, _ = L.decode(ip)
            text = f"{insn.mnemonic} {insn.op_str}" if insn is not None else "<undecodable>"
            print(f"{ip:#x}  {text}")
            # Compare the *Refs*, not their constant values. Comparing
            # `as_const` makes every symbolic write look like "no change", since
            # both sides are None -- which made this readout silently meaningless
            # for exactly the registers it was being used to inspect.
            pre = {r: m.reg(r, 64) for r in WATCH}
            k, p = L.step(ip)
            steps += 1
            post = {r: m.reg(r, 64) for r in WATCH}
            changed = [r for r in WATCH if pre[r] != post[r]]
            for r in changed:
                pc, nc = a.as_const(pre[r]), a.as_const(post[r])
                pv = f"{pc:#x}" if pc is not None else render(a, pre[r], limit=20)[:50]
                nv = f"{nc:#x}" if nc is not None else render(a, post[r], limit=20)[:50]
                print(f"        {r}: {pv} -> {nv}")
            if not changed:
                print("        (no watched register changed)")
            print(f"        unmapped {len(m.unmapped_loads)}  symbolic-loads {len(m.sym_loads)}"
                  f"  sym-stores {len(m.sym_stores)}")
        else:
            k, p = L.step(ip)
            steps += 1
        if k != "next":
            print(f"STOP {p}")
            break
        ip = p
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
