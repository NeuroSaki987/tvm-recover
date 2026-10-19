"""Explain why evaluation stopped: the last instructions, and the predicate.

A stop is either correct (the VM really does branch on guest data here) or a
bug. The two are told apart by reading the predicate: if its leaves reduce to
concrete instructions the engine failed to model, it is a bug; if they reduce to
`init_*` (guest entry values) or `call_ret#N` (guest call results), the stop is
the real thing.

Usage: python -m probe.stop_debug <module> <root_va> [--steps N] [--back N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from core.expr import Arena, render  # noqa: E402
from core.lift64 import Lifter  # noqa: E402
from core.mach import Machine  # noqa: E402
from core.peimg import PeImage  # noqa: E402

STACK_BASE = 0x7FFF_FFFE_FDA0


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    steps, back = 400_000, 24
    for x in sys.argv[1:]:
        if x.startswith("--steps="):
            steps = int(x.split("=", 1)[1], 0)
        if x.startswith("--back="):
            back = int(x.split("=", 1)[1], 0)
    path, root = args[0], int(args[1], 0)

    img = PeImage(path)
    vm = img.section_by_name(".tvm0")
    arena = Arena()
    mach = Machine(arena, STACK_BASE)
    L = Lifter(img, mach, arena, vm.va, vm.end)

    ip = root
    stop = None
    for _ in range(steps):
        kind, payload = L.step(ip)
        if kind == "next":
            ip = payload
        else:
            stop = payload
            break

    print(f"steps {L.steps}  DAG {len(arena)}  dispatches {len(L.dispatches)}")
    if stop is None:
        print("no stop within budget")
        return 0
    print(f"STOP {stop.kind} at {stop.site:#x} {stop.text}")

    print()
    print(f"last {back} executed instructions:")
    for a in L.trace[-back:]:
        size, insn, _ = L.decode(a)
        text = f"{insn.mnemonic} {insn.op_str}" if insn is not None else "<undecodable>"
        print(f"  {a:#x}  {text}")

    if stop.pred is not None:
        print()
        print(f"predicate: {render(arena, stop.pred)}")
        leaves = arena.leaves(stop.pred)
        print(f"leaves ({len(leaves)}):")
        for lf in leaves[:40]:
            print(f"  {lf}: {render(arena, lf)}")
    if stop.dest is not None:
        print()
        print(f"destination: {render(arena, stop.dest)}")
        for lf in arena.leaves(stop.dest)[:40]:
            print(f"  leaf {lf}: {render(arena, lf)}")

    print()
    print("concrete context slots around rbp:")
    import core.mach as _m  # noqa: E402

    rbp = arena.as_const(mach.reg("rbp", 64))
    print(f"  rbp = {rbp:#x}" if rbp is not None else "  rbp = symbolic")
    print(f"  rsp = {mach.concrete_rsp()}")
    if rbp is not None:
        for off in range(0, 0x90, 8):
            val = mach.load(arena.const(rbp + off, 64), 8, 0, img.image_u8)
            c = arena.as_const(val)
            tag = ""
            if c is not None and vm.va <= c < vm.end:
                tag = "   <-- in .tvm0"
            print(f"  [rbp+{off:#04x}] = {render(arena, val)}{('  = ' + hex(c)) if c is not None else ''}{tag}")
    del _m
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
